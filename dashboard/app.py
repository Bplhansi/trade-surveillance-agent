"""Streamlit review dashboard. Plan: docs/dashboard-plan.md.

Run (after `python -m dashboard.prepare --name validation`):
    .\\.venv\\Scripts\\python.exe -m streamlit run dashboard/app.py

Settings (environment variables): TSA_DATA (default data), TSA_NAME (default validation),
TSA_DB (default <TSA_DATA>/decisions.sqlite).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from dashboard.review import alerts_per_minute, false_alarm_rate, price_window, queue_order
from dashboard.store import DecisionStore
from sim.event_array import read_event_array

DATA = Path(os.environ.get("TSA_DATA", "data"))
NAME = os.environ.get("TSA_NAME", "validation")
DB = Path(os.environ.get("TSA_DB", DATA / "decisions.sqlite"))
LABELS = {"spoofing": "Spoofing", "wash_trade": "Wash trade", "pump_and_dump": "Pump and dump"}


@st.cache_data
def load_rows(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


@st.cache_resource
def load_events(path: str):
    return read_event_array(path)


def fmt_time(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).strftime("%H:%M:%S.%f")[:-3]


st.set_page_config(page_title="Trade surveillance review", layout="wide")
st.title("Trade surveillance review")
st.caption("Simulated data only. This is a learning project, not a compliance tool.")

review_path = DATA / f"{NAME}.review.jsonl"
if not review_path.exists():
    st.error(f"No review file at {review_path}. Run: python -m dashboard.prepare --name {NAME}")
    st.stop()

rows = queue_order(load_rows(str(review_path)))
events = load_events(str(DATA / f"{NAME}.events.bin"))
store = DecisionStore(DB)
decisions = store.latest()
engine_path = DATA / f"{NAME}.engine.json"
engine = json.loads(engine_path.read_text()) if engine_path.exists() else {}

# --- live metrics ----------------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Alerts per minute", f"{alerts_per_minute(len(rows), events):.2f}",
          help="Alerts per minute of simulated market time")
rate = false_alarm_rate({k: v for k, v in decisions.items() if k in {r['alert_id'] for r in rows}})
c2.metric("False alarm rate", "—" if rate is None else f"{rate:.0%}",
          help="Dismissed / decided, from your decisions")
c3.metric("Engine p50 latency", f"{engine['latency_p50_ns']:,} ns" if engine else "—")
c4.metric("Engine p99 latency", f"{engine['latency_p99_ns']:,} ns" if engine else "—")
c5.metric("Left to review", sum(r["alert_id"] not in decisions for r in rows))

# --- queue -----------------------------------------------------------------------------
left, right = st.columns([2, 3])
with left:
    st.subheader("Queue (highest score first)")
    queue = pd.DataFrame([{
        "alert": r["alert_id"], "pattern": LABELS[r["alert"]["rule"]],
        "score": r["alert"].get("model_score"), "trader": r["alert"]["trader_id"],
        "symbol": r["alert"]["symbol_id"], "found by": r["alert"].get("source", "rules"),
        "decision": decisions.get(r["alert_id"], ""),
    } for r in rows])
    hide_done = st.checkbox("Hide decided alerts")
    shown = queue[queue["decision"] == ""] if hide_done else queue
    picked = st.dataframe(shown, hide_index=True, width="stretch", height=520,
                          on_select="rerun", selection_mode="single-row",
                          column_config={"score": st.column_config.NumberColumn(format="%.2f")})
    selected = picked.selection.rows if picked else []
    alert_id = int(shown.iloc[selected[0]]["alert"]) if selected else (
        int(shown.iloc[0]["alert"]) if len(shown) else None)

# --- one alert -------------------------------------------------------------------------
with right:
    if alert_id is None:
        st.info("Nothing left to review.")
        st.stop()
    row = next(r for r in rows if r["alert_id"] == alert_id)
    a, case, result = row["alert"], row["case"], row["note"]
    st.subheader(f"Alert {alert_id}: {LABELS[a['rule']]} — trader {a['trader_id']}, "
                 f"symbol {a['symbol_id']}")
    st.caption(f"At {fmt_time(a['timestamp_ns'])} UTC, {a['window']} window, found by "
               f"{a.get('source', 'rules')}"
               + (f", model score {a['model_score']:.2f}" if a.get("model_score") else ""))

    trades, flagged = price_window(events, a["symbol_id"], a["timestamp_ns"], 60,
                                   case["event_ids"])
    line = alt.Chart(trades).mark_line(color="#8a8f98").encode(
        x=alt.X("time:T", title="time (UTC)"), y=alt.Y("price:Q", scale=alt.Scale(zero=False)))
    points = alt.Chart(flagged).mark_point(size=90, filled=True).encode(
        x="time:T", y="price:Q",
        color=alt.Color("kind:N", title="this trader's events"),
        shape="kind:N", tooltip=["event_id", "kind", "price", "quantity"])
    st.altair_chart(line + points, width="stretch")

    note = result["note"]
    by = f"LLM ({result['model']})" if result["source"] == "llm" else "template"
    st.markdown(f"**Case note** · written by {by} · "
                + ("evidence check passed ✅" if result["checks_passed"] else "evidence check FAILED ❌"))
    # Note text is shown with st.text (never parsed as markdown), so nothing in a note can
    # become a link or an image, even if it slipped past the evidence checker.
    st.text(note["summary"])
    st.markdown("**Evidence**")
    st.text("\n".join(f"- {item}" for item in note["evidence"]))
    st.markdown("**Suggested next check:**")
    st.text(note["next_check"])
    if result["rejected_reasons"] and result["source"] == "template":
        st.caption("LLM note not used: " + "; ".join(result["rejected_reasons"]))
    with st.expander("Statistics behind the alert"):
        st.dataframe(pd.DataFrame(case["stats"].items(), columns=["statistic", "value"]),
                     hide_index=True)
        st.caption(f"Event IDs (rows in {NAME}.events.bin): {case['event_ids']}"
                   + (" (older ones dropped)" if case["event_ids_truncated"] else ""))

    # --- decision ------------------------------------------------------------------------
    comment = st.text_input("Comment (optional)", key=f"comment-{alert_id}", max_chars=500)
    b1, b2, _ = st.columns([1, 1, 3])
    if b1.button("Accept (escalate)", type="primary", key=f"acc-{alert_id}"):
        store.record(alert_id, "accept", comment)
        st.rerun()
    if b2.button("Dismiss", key=f"dis-{alert_id}"):
        store.record(alert_id, "dismiss", comment)
        st.rerun()
    history = store.history(alert_id)
    if history:
        st.caption("Decisions: " + "; ".join(
            f"{h['decision']} at {h['decided_at'][:19]} UTC" + (f" ({h['comment']})" if h["comment"] else "")
            for h in history))

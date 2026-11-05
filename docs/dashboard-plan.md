# Review dashboard plan

**Who uses it:** one surveillance reviewer working through alerts.
**The question it answers:** "Is this alert real, and what did I decide?"

## Data flow

```text
tsa_replay (alerts.jsonl + printed speed summary)
  -> python -m dashboard.prepare  (cases + notes + engine stats -> data/<name>.review.jsonl,
                                   data/<name>.engine.json)
  -> streamlit run dashboard/app.py   (reads those files + the events file)
  -> data/decisions.sqlite            (accept / dismiss, with UTC timestamp)
```

`prepare` gets notes from the note service: directly in Python, or over HTTP when
`--service-url` is given (used by Docker Compose).

## Screen layout

| Area | Shows | Guide item |
| --- | --- | --- |
| Top row | Alerts per minute (simulated time), false alarm rate (dismissed / decided), engine p50 / p99 latency, alerts left to review | 5 |
| Left | Queue: alerts sorted by score (highest first), with pattern, trader, symbol and decision so far | 1 |
| Right, top | Price chart for the alert's symbol around the alert, trades as a line, this trader's flagged events marked | 2 |
| Right, middle | Case note (summary, evidence, next check), its source (LLM or template), and the raw statistics | 3 |
| Right, bottom | Accept / Dismiss buttons and an optional comment; the latest decision is shown | 4 |

## Choices

- **Score for sorting:** the model score (every alert has one when the engine runs with
  `--model`); alerts without one sort last, then by alert number.
- **False alarm rate:** from the reviewer's own decisions only (dismissed / decided), not from
  the simulator's labels, so it is what a real desk would see. Empty until a decision is made.
- **Alerts per minute:** alerts / minutes of simulated market time in the events file.
- **Engine latency:** read from the replay summary saved by `prepare`, never typed in by hand.
- **Decisions are append-only:** each click adds a row; the latest row per alert counts. Nothing
  is overwritten, so the history is an audit trail.

## Tests

Plain functions (queue order, price window, metrics, summary parsing, SQLite store) are
tested with pytest. The Streamlit page itself only arranges them on screen.

"""Tests for the FastAPI case note service."""

from fastapi.testclient import TestClient

from service.app import create_app
from service.notes import NoteService

CASE = {
    "alert_id": 7, "pattern": "wash_trade", "trader_id": 42, "symbol_id": 3,
    "source": "rules", "model_score": None, "window_seconds": 60,
    "window_start_ns": 1_000_000_000_000, "window_end_ns": 1_060_000_000_000,
    "stats": {"trades": 8, "buy_qty": 800, "sell_qty": 800, "net_qty": 0, "self_trades": 4},
    "event_ids": [10, 11, 12], "event_ids_truncated": False,
}


def client() -> TestClient:
    return TestClient(create_app(NoteService(None)))


def test_health() -> None:
    response = client().get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "llm": "off"}


def test_post_note_returns_a_checked_note() -> None:
    response = client().post("/v1/notes", json=CASE)

    assert response.status_code == 200
    body = response.json()
    assert body["alert_id"] == 7
    assert body["source"] == "template"
    assert set(body["note"]) == {"summary", "evidence", "next_check"}
    assert body["checks_passed"] is True


def test_invalid_case_is_rejected_with_422() -> None:
    bad = dict(CASE, pattern="insider_trading")

    assert client().post("/v1/notes", json=bad).status_code == 422


def test_oversized_case_is_rejected() -> None:
    bad = dict(CASE, event_ids=list(range(1000)))

    assert client().post("/v1/notes", json=bad).status_code == 422

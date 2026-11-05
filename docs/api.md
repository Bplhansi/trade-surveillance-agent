# Case note service API

A small HTTP service that turns one alert into a short case note for a human reviewer.
The LLM only **explains** an alert; it never decides anything. Code: `service/`.

Run it locally (only on your own machine, it has no login):

```powershell
.\.venv\Scripts\python.exe -m uvicorn service.app:app --host 127.0.0.1 --port 8000
```

Interactive docs: http://127.0.0.1:8000/docs

## How a note is made

1. The engine writes alerts (`<name>.alerts.jsonl`).
2. `python -m service.case` turns each alert into a **case**: the alert, its statistics, and
   the IDs (row numbers in the events file) of this trader's events in the alert window.
3. `POST /v1/notes` sends the case to a free local LLM (Ollama) and asks for the fixed
   note shape as a JSON schema.
4. The **evidence checker** (`service/validate.py`) rejects the note if any number in it is
   not in the case, or if it contains a link or HTML.
5. If anything goes wrong (no LLM configured, timeout, refusal, bad JSON, failed check), a **template
   note** built only from the case is returned instead. Every alert always gets a note.

## Endpoints

### `GET /health`

```json
{"status": "ok", "llm": "on"}
```

`llm` is `"off"` when no API key is configured (template notes only).

### `POST /v1/notes`

Request body: a **Case**.

| Field | Type | Rules |
| --- | --- | --- |
| `alert_id` | int | ≥ 0 |
| `pattern` | string | `spoofing`, `wash_trade` or `pump_and_dump` |
| `trader_id`, `symbol_id` | int | ≥ 0 |
| `source` | string | `rules` (default) or `model` |
| `model_score` | number or null | model score, if the model scored the event |
| `window_seconds` | int | `1` or `60` |
| `window_start_ns`, `window_end_ns` | int | nanoseconds since 1970 |
| `stats` | object | up to 40 entries; names like `large_cancels` (lowercase, digits, `_`) |
| `event_ids` | int list | up to 50, oldest first |
| `event_ids_truncated` | bool | true if older events were dropped |

Unknown fields are rejected.

Response `200`: a **NoteResult**.

```json
{
  "alert_id": 7,
  "note": {
    "summary": "Trader 42 on symbol 3 was flagged for possible spoofing ...",
    "evidence": ["large_cancels = 4", "large_cancel_ratio = 1"],
    "next_check": "Check whether the cancelled orders sat at the best price ..."
  },
  "source": "llm",
  "model": "qwen2.5:3b",
  "checks_passed": true,
  "rejected_reasons": []
}
```

- `source`: `llm` or `template`.
- `rejected_reasons`: why an LLM note was not used (for example
  `"summary: number 9 is not in the alert data"`).
- `checks_passed`: the returned note passed the evidence checker.

Errors: `422` if the case does not match the rules above (FastAPI's standard error body).
LLM failures are **not** errors; they produce a template note.

## Configuration

| Variable | Meaning |
| --- | --- |
| `LLM_PROVIDER` | `ollama` to write notes with a local model. Unset: template notes only. |
| `LLM_MODEL` | Optional. Default `qwen2.5:3b`. |
| `OLLAMA_URL` | Optional. Default `http://127.0.0.1:11434` (Docker: `http://host.docker.internal:11434`). |

All settings go in `.env` (ignored by Git; see `.env.example`). Limits: 300 s timeout per
note, temperature 0 (the same case gives the same note).

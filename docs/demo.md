# Two-minute demo script

Record the screen while following this script (for example with the Windows Snipping Tool's
screen recorder or OBS). Aim for about 2 minutes.

## Before recording

1. `docker compose up --build` and wait for "Done. Open the dashboard" in the logs.
2. Optional, for LLM-written notes: start Ollama and set `LLM_PROVIDER=ollama` in `.env`.
3. Open http://localhost:8501 in a browser window sized to fill the screen.

## Script

| Time | Show | Say |
| --- | --- | --- |
| 0:00–0:15 | README top | "This watches simulated trading, flags spoofing, wash trades and pump-and-dump, and writes checked case notes for a reviewer. Simulated data only." |
| 0:15–0:35 | Terminal: `docker compose logs pipeline` | "One command simulates the market, trains the model, runs the C++ engine with rules and an ONNX model, and writes a note for every alert." |
| 0:35–0:50 | Dashboard top row | "Alerts per minute, the false alarm rate from my decisions, and the engine's measured latency." |
| 0:50–1:20 | Click the top alert in the queue | "Highest score first. The chart shows the price, and this trader's own orders and cancels. Here is the note: every number in it is checked against the alert, or a template is used instead." |
| 1:20–1:40 | Open "Statistics behind the alert", type a comment, click **Accept** | "The evidence behind the alert, then my decision, stored in SQLite with a timestamp." |
| 1:40–2:00 | README results table | "On the held-out test set, rules plus model reach 0.98 precision and 0.88 recall, scored once with frozen settings." |

Save the video as `docs/demo.mp4`, or upload it, and link it at the top of the README.

# Commands and project decisions

All commands are for PowerShell on Windows, run from the project folder.

## Commands

```powershell
# Configure and build the C++ engine (Release)
cmake -S engine -B engine/build -DCMAKE_BUILD_TYPE=Release
cmake --build engine/build --config Release

# Run C++ tests
ctest --test-dir engine/build -C Release --output-on-failure

# Replay events through the engine and print throughput / latency (5 runs)
.\engine\build\Release\tsa_replay.exe data\train.events.bin --repeat 5

# Run the baseline rules, write alerts, then grade them (precision / recall per pattern)
.\engine\build\Release\tsa_replay.exe data\validation.events.bin --rules config\rules.ini --alerts data\validation.alerts.jsonl
.\.venv\Scripts\python.exe -m ml.evaluate_alerts --data data --name validation --alerts data\validation.alerts.jsonl

# Set up Python (once), then run Python tests
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest

# Export feature rows, train models (validation), and score test ONCE
.\engine\build\Release\tsa_replay.exe data\train.events.bin --rules config\rules.ini --features data\train.features.bin
.\.venv\Scripts\python.exe -m ml.train --data data --out models
.\.venv\Scripts\python.exe -m ml.train --data data --out models --test

# Export the model to ONNX and run rules + model in C++ (configure once with the venv Python
# so the parity test can train its tiny model)
.\.venv\Scripts\python.exe -m ml.export_onnx --model models\gradient_boosting.joblib --out models\gradient_boosting.onnx
cmake -S engine -B engine/build -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$PWD\.venv\Scripts\python.exe"
.\engine\build\Release\tsa_replay.exe data\validation.events.bin --rules config\rules.ini --model models\gradient_boosting.onnx --alerts data\validation.alerts.jsonl

# Case notes: build cases from alerts, then start the note service (http://127.0.0.1:8000/docs)
.\.venv\Scripts\python.exe -m service.case --data data --name validation --alerts data\validation.alerts.jsonl --out data\validation.cases.jsonl
.\.venv\Scripts\python.exe -m uvicorn service.app:app --host 127.0.0.1 --port 8000

# Review dashboard: run engine + notes for one dataset, then open http://localhost:8501
.\.venv\Scripts\python.exe -m dashboard.prepare --name validation
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py

# One-command demo (needs Docker Desktop with WSL 2), then open http://localhost:8501
docker compose up --build

# Python test coverage
.\.venv\Scripts\python.exe -m pytest --cov=sim --cov=ml --cov=service --cov=dashboard

# Generate the train / validation / test datasets into data/, then draw the chart
.\.venv\Scripts\python.exe -m sim.generate --out data
.\.venv\Scripts\python.exe -m sim.plot --data data --name train --out docs/sim.png
```

## Decisions

- C++ tests use **GoogleTest** (fetched by CMake).
- All C++ engine headers go in `engine/include/tsa/` and are included as `"tsa/<name>.hpp"`
  (for example the event format is `engine/include/tsa/event.hpp`). Code is in the
  `tsa` namespace.
- Python uses a local virtual environment in `.venv/` (not committed).
- Event arrays in Python use `sim/event_array.py`. Join/reorder them with `concat_events` /
  `take_events`, never plain `np.concatenate` or `events[idx]`: those leave the padding bytes
  uninitialised and make files non-reproducible.
- Detection is graded **per injected case** (`ml/evaluate_alerts.py`, ±10 s grace), not per
  event. Tune on **validation** only; score the **test** set once, with frozen settings, and
  log every tuning change in `docs/results.md`.
- Wash-trade rule also requires `min_self_trades = 1` (added after validation
  iteration 1; the guide's plain balance rule is `min_self_trades = 0`).
- The model runs in C++ through **ONNX Runtime 1.31.1** (prebuilt, fetched by CMake with a
  pinned SHA-256). Rules run first; the model only scores events that pass the `[model]` gate
  in `config/rules.ini`. Python pins `protobuf==6.33.6` (protobuf 7 breaks skl2onnx).
- Case notes use a **free local Ollama model** (`LLM_PROVIDER=ollama`, `qwen2.5:3b`), so no
  paid API or key is needed. A note is only used if every number in it is in
  the case (`service/validate.py`); otherwise a template note is used.

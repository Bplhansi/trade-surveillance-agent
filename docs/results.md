# Results

Only measured numbers go in this file. Each section says how the numbers were produced so
they can be re-run.

## Detection quality

Scored on the **test set** (seed 44, traders 251–500, gentler patterns), run once with frozen
settings. Precision and recall are per *case* (see "How alerts are graded" below). Latency is
per event with the rules on (`data/train.events.bin`, median of 5 runs, see Engine speed).

| Method | Precision | Recall | False alarms per 10,000 events | Median latency | 99th percentile latency |
| --- | --- | --- | --- | --- | --- |
| Rules only | 0.971 | 0.853 | 0.04 | 300 ns | 1,600 ns |
| Isolation Forest | 0.706 | 0.353 | 0.20 | not measured (not deployed) | not measured |
| Gradient boosted model (Python, every event scored) | 1.000 | 0.588 | 0.00 | not deployed this way | not deployed this way |
| Gradient boosted model in C++, gated (Step 8) | 1.000 | 0.441 | 0.00 | not measured separately | not measured separately |
| **Rules + model in C++ (Step 8)** | **0.984** | **0.882** | 0.04 | 1,000 ns | 71,600 ns |

### Step 6: rules only, per pattern (2026-11-05)

Test set, run once (`config/rules.ini` SHA-256 `ee56d919…74547`):

| Pattern | Precision | Recall | Cases caught | Alerts | False alarms | False alarms per 10k events |
| --- | --- | --- | --- | --- | --- | --- |
| Spoofing | 1.000 | 0.867 | 13 / 15 | 13 | 0 | 0.00 |
| Wash trade | 1.000 | 0.800 | 12 / 15 | 12 | 0 | 0.00 |
| Pump and dump | 0.900 | 1.000 | 4 / 4 | 10 | 1 | 0.04 |
| **Overall** | **0.971** | **0.853** | **29 / 34** | 35 | 1 | 0.04 |

Validation set (used for tuning, so optimistic): precision 1.000 and recall 1.000 for every
pattern (34 / 34 cases, 37 alerts, 0 false alarms). Only 4 pump-and-dump cases in each of
validation and test, so that row in particular is a very small sample.

**How alerts are graded** (`ml/evaluate_alerts.py`): an alert is correct if its rule, trader
and symbol match an injected case and its time is inside the case (±10 s grace).
Recall = cases with at least one correct alert / all cases. Precision = correct alerts / all
alerts. A cooldown (60 s per rule + trader + symbol) stops one case producing hundreds of
alerts.

**Tuning log (validation only):**

| Iteration | Change | Validation result | Why |
| --- | --- | --- | --- |
| 1 | Starting thresholds, chosen by reasoning before any run | Wash precision 0.625 (9 false alarms), everything else 1.000 | 8 of the 9 were innocent traders just over the volume/trade-count limits; 1 was a pump-and-dump trader trading after the pump. All 9 had **zero** self-trades; all 15 real cases had 1 or more. |
| 2 | Added `wash_trade.min_self_trades = 1` (trading with yourself is the definition of a wash trade) | 1.000 / 1.000 for every pattern | Frozen here; test run once. |

**What the rules miss on test** (looked at after the one test run; thresholds *not* changed):

- **2 spoofing cases:** almost all of the spoofer's orders were under the 1,000-share "large"
  size (1–2 orders at or above it), so the rule never counted enough large cancels.
- **3 wash trades:** the self-trades were detected (4–6 pairs each), but the same trader's
  normal trading in the same minute pushed their overall buy/sell imbalance to 6–10%, over
  the 5% limit. The rule looks at all of a trader's trades, not just the self-trades.
- **1 false alarm (pump and dump):** a real pump-and-dump trader, alerted again 99 s after
  the labelled case ended (outside the 10 s grace period).

These are the gaps the Step 7 model has to beat.

**Re-run:**

```powershell
.\engine\build\Release\tsa_replay.exe data\test.events.bin --rules config\rules.ini --alerts data\test.alerts.jsonl
.\.venv\Scripts\python.exe -m ml.evaluate_alerts --data data --name test --alerts data\test.alerts.jsonl
```

### Step 7: anomaly models (2026-11-05)

**Features:** the C++ engine exports one row of 33 numbers per event
(`tsa_replay --features`, list in `engine/src/features.cpp`): the event's type, side and
size, plus the trader + symbol's rolling 1 s and 60 s statistics *after* the event (past and
present only). No IDs or timestamps. Python trains on exactly the numbers C++ computes.

**Models** (`ml/train.py`, scikit-learn 1.9.1, seed 0, trained on `train` only):

- *Isolation Forest* (200 trees): unsupervised, never sees labels; raises pattern-agnostic
  `anomaly` alerts, which count as correct if they match a case of **any** pattern (the
  gradient boosted model must name the right pattern, a stricter test).
- *Gradient boosting* (HistGradientBoostingClassifier, 4 classes, 200 rounds,
  `class_weight="balanced"`, no early stopping). Score = −log10 P(normal).

Thresholds were picked on validation (best case-level F1; middle of tied thresholds), alerts
use the same 60 s cooldown as the rules, and the test set was scored once with
`python -m ml.train --test` (which refuses to run twice). An `mle-reviewer` leakage review
before the test run found no leakage; its fixes (tie-breaking, no hidden early-stopping
split, run-once guard, provenance in `models/model_card.json`) were applied first.

| Model | Validation precision / recall | **Test** precision / recall | Test false alarms per 10k |
| --- | --- | --- | --- |
| Rules only (Step 6) | 1.000 / 1.000 | **0.971 / 0.853** | 0.04 |
| Isolation Forest | 0.833 / 0.588 | 0.706 / 0.353 | 0.20 |
| Gradient boosting | 1.000 / 1.000 | **1.000 / 0.588** | 0.00 |

Gradient boosting on test, per pattern: spoofing 1.000 / 0.467 (7 / 15), wash trade
1.000 / 0.600 (9 / 15), pump and dump 1.000 / 1.000 (4 / 4). Isolation Forest on test caught
12 / 15 spoofing, 0 / 15 wash trades, 0 / 4 pump and dumps.

**Verdict: the model does not beat the rules on the test set.** It has fewer false alarms
(none) but misses many more cases. Why (analysed after the single test run, nothing re-tuned):

- **Distribution shift.** The test set's patterns are deliberately smaller: median spoofing
  order 1,290 shares vs 3,430 in train; median wash-trade size 240 vs 650. The model learned
  what suspicious looked like *at training sizes*.
- **The threshold could not see it coming.** Validation comes from the same distribution as
  train, so every case was easy and the chosen threshold (6.92) was strict. 12 of the 14
  missed test cases still scored far above normal (2.1–6.8), just under that threshold.
- **The rules generalise better** because their thresholds were set by hand with margin, not
  fitted to training sizes.

What would help next (not done, to keep the test set honest): train on more varied pattern
sizes, use size-relative features (order size / the trader's usual size), or a validation set
drawn from a shifted distribution so threshold choice can see the shift.

**PyTorch (skipped):** the guide says to move to a small PyTorch model only if it clearly
beats these. Both models already score 1.000 / 1.000 on validation, so there is no room to
clearly beat them there, and a network on the same features would face the same shift.

**Re-run:** `.\engine\build\Release\tsa_replay.exe data\train.events.bin --rules config\rules.ini --features data\train.features.bin`
(and the same for `validation`, `test`), then `.\.venv\Scripts\python.exe -m ml.train`
and, once only, `... -m ml.train --test`.

### Step 8: the model inside the C++ engine (2026-11-05)

The gradient boosting model is exported to ONNX (`ml/export_onnx.py`) and run by ONNX Runtime
1.31.1 inside `tsa_replay` (`--model models\gradient_boosting.onnx`). Each event is first
checked by the rules; if no rule fires, the model scores it, but only if it passes a cheap
**gate** (a Cancel with at least 1 large cancel in the last second, or a Trade with at least
2 trades in the last 60 s). The threshold (6.918) is the one chosen in Step 7, not re-tuned.

**C++ gives the same answers as Python.** On the validation set, max |C++ score − Python
score| = 3.88e-6 (mean 1.12e-6); a CTest test checks this on every build (tolerance 1e-3).

**Choosing the gate (validation only).** Every gate tried kept validation at 1.000 / 1.000,
so the gate was chosen for speed with some margin:

| Gate | Share of events scored | Validation precision / recall |
| --- | --- | --- |
| None (every event) | 100% | 1.000 / 1.000 |
| Any Cancel or Trade | 53.5% | 1.000 / 1.000 |
| **Chosen: large cancel ≥ 1 in 1 s, or trades ≥ 2 in 60 s** | **20.2%** | 1.000 / 1.000 |
| Stricter | 4.3% | 1.000 / 1.000 |

**Test set (scored once, frozen settings):**

| Method | Precision | Recall | Cases caught | False alarms |
| --- | --- | --- | --- | --- |
| Rules only | 0.971 | 0.853 | 29 / 34 | 1 |
| Model only (C++, gated) | 1.000 | 0.441 | 15 / 34 | 0 |
| **Rules + model** | **0.984** | **0.882** | **30 / 34** | 1 |

Rules + model is the best result so far: the model adds one case the rules missed and no
extra false alarms. Honest note: the gate costs the model on test. Ungated Python scoring
caught 20 / 34 (0.588); gated C++ catches 15 / 34 (0.441). The smaller test patterns pass
the gate less often. This was found after the test run and is not re-tuned.

## Case notes (Step 9, 2026-11-05)

All 85 validation alerts (rules + model, Step 8 settings) were turned into cases
(`python -m service.case`) and sent to the note service.

| Measure | Template notes (no LLM) |
| --- | --- |
| Alerts that got a note | **85 / 85** |
| Notes passing the evidence checker | **85 / 85 (100%)** |
| Service time per note (in-process, 425 requests) | p50 10.4 ms, p95 15.7 ms, p99 22.0 ms |

The checker caught a real bug in the first version: template notes printed values like
`0.333333` with 6 significant digits, while the checker accepts at most 4 decimals, so
74 / 85 failed. The service also reported `checks_passed` without running the check on
template notes. Both are fixed, with regression tests.

**Security review (Step 9b).** The LLM has no tools and no authority: it cannot change
alerts, scores or decisions, and its text is checked before use. Added after the review:
statistic names must be plain identifiers, so free text from a caller cannot reach the
prompt that way, and notes containing links or HTML are rejected. Settings are read only from
the environment (`.env`, ignored by Git). The service has no login, so run it on 127.0.0.1 only.

**Local LLM notes (Ollama, qwen2.5:3b):** being measured on all 85 validation alerts; the
results (share of LLM notes that pass the checker, time per note) will be added here.

## Engine speed

### Step 4 baseline: replay only, no rules yet (2026-11-05)

What is measured:

- **Latency** = time for `state.update(e)` + `rules.check(e, state)` on one event, measured
  with `std::chrono::steady_clock`. The rules checker is still empty (Step 6 adds real rules),
  so this is the floor that later steps build on.
- **Throughput** = events / wall time for the whole run, *including* reading the file.

| Metric | Median of 5 runs | Range over the 5 runs |
| --- | --- | --- |
| Throughput | **4,469,213 events/s** | 3,163,641 – 7,686,927 |
| Latency p50 (median) | **100 ns** (one clock tick, see note) | 100 – 200 |
| Latency p99 | **500 ns** | 200 – 600 |
| Latency mean | 140 ns | 79 – 198 |
| Latency p95 | 300 ns | 100 – 400 |
| Latency max | 1,629,400 ns (≈1.6 ms) | 1,394,000 – 2,892,800 |

Events per run: 1,006,697 (`data/train.events.bin`, seed 42).

**Notes on reading these numbers**

- **Clock resolution:** on this machine `steady_clock` ticks every **100 ns**, so single-event
  times are rounded to 0, 100, 200 … ns. The p50 of 100 ns means "about one tick"; the mean
  (140 ns), averaged over a million events, is the more precise figure.
- **Timer overhead:** one `steady_clock::now()` call costs about **24.5 ns**, and two are made
  per event, so roughly 50 ns of each latency figure is the measuring itself.
- **Run-to-run spread:** later runs were faster (laptop CPU raising its clock speed under
  load), hence the wide throughput range. The median run is reported, not the best.
- **Max:** the ~1.6 ms worst case is consistent with the operating system briefly pausing
  the process; p99 is the better measure of slow events.

**Machine:** 11th Gen Intel Core i5-1135G7 @ 2.40 GHz (4 cores / 8 threads), 8 GB RAM,
Windows 11 Home, on battery/AC not controlled. Compiler: MSVC 14.44, Release, `/O2` (MSVC's
highest optimisation level; Linux CI builds use `-O3`).

### Step 5: + rolling statistics (2026-11-05)

Same file, machine and build as above. The engine now also updates per-trader, per-symbol
rolling statistics (1 s and 60 s ring-buffer windows) on every event. Rules are still empty.

| Metric | Median of 5 runs | Range over the 5 runs | Change vs Step 4 |
| --- | --- | --- | --- |
| Throughput | **3,737,209 events/s** | 2,052,249 – 3,774,152 | −16% |
| Latency p50 (median) | **200 ns** (two clock ticks) | 200 – 300 | +100 ns |
| Latency p99 | **600 ns** | 600 – 1,000 | +100 ns |
| Latency mean | 214 ns | 212 – 348 | +74 ns |
| Latency p95 | 400 ns | 400 – 700 | +100 ns |
| Latency max | 1,709,400 ns (≈1.7 ms) | 1,508,700 – 3,228,300 | — |

So the rolling statistics cost roughly **75 ns per event** on average. Timer overhead
measured 16.4 ns per `now()` call in this session. Runs 2–5 were very consistent; run 1 was
slower, likely from first-time memory setup for each new trader/symbol pair plus CPU
warm-up (not separately measured).

After the code review, every event is also validated (type, side and symbol_id ranges) as it
is read. Re-measured with 5 runs: 3,649,260 events/s (range 3,453,318 – 3,733,196), mean
218 ns, p50 200 ns, p99 600 ns. So the validation cost is within run-to-run noise.

### Step 6: + baseline rules (2026-11-05)

Same file, machine and build. The three rules now run on every event (`--rules
config/rules.ini`, alerts not written to a file during timing).

| Metric | Median of 5 runs | Range over the 5 runs | Change vs Step 5 (after review) |
| --- | --- | --- | --- |
| Throughput | **1,798,062 events/s** | 1,765,527 – 1,820,762 | −51% |
| Latency p50 (median) | **300 ns** | 300 – 300 | +100 ns |
| Latency p99 | **1,600 ns** | 1,500 – 1,700 | +1,000 ns |
| Latency mean | 498 ns | 492 – 512 | +280 ns |
| Latency p95 | 1,100 ns | 1,100 – 1,100 | +700 ns |
| Latency max | 1,703,400 ns (≈1.7 ms) | 1,687,100 – 1,761,500 | — |

Alerts on the training set: 157 per run. The rules cost roughly **280 ns per event** on
average. Likely cause (not separately measured): every Trade event sums the 60-bucket window
twice, once for pump and dump and once for wash trading. A candidate for Step 8.

### Step 8: + model scoring (2026-11-05)

Training set, 5 runs, rules + gated model (`--rules config\rules.ini --model models\gradient_boosting.onnx --repeat 5`):

| Metric | Value | Change vs Step 6 |
| --- | --- | --- |
| Throughput | **101,918 events/s** (range 100,546 – 102,470) | about 18× slower |
| Latency p50 (median) | **1,000 ns** | +700 ns |
| Latency p95 | 47,300 ns | — |
| Latency p99 | **71,600 ns** | +70,000 ns |
| Latency mean | 9,709 ns | — |
| Latency max | 2,525,400 ns (≈2.5 ms) | — |

The model scored 207,632 events; 354 alerts per run. Each model call costs roughly 45 µs.
Most events (about 80%) skip the model, which is why the median stays at 1 µs.

**Trying to make model calls cheaper (performance-optimizer agent).** A micro-benchmark of
20,000 single calls (34–39 µs each) tried: sequential execution, IoBinding, memory pattern
off, memory arena off, basic graph optimisation, spinning off, a model rebuilt with the newer
`ai.onnx.ml` opset 5 tree operator, and 2/4/8 threads. None helped enough to keep. The best,
2 threads, was about 10% faster but uses a second core on every call; opset 5 was about 30%
slower. The cost is the 800 trees themselves (about 5,000–8,000 tree-node visits per call),
which session settings cannot change. A second replay run on the unchanged code measured
96,852 events/s, which shows the run-to-run noise. **Follow-up:** a smaller model (fewer or
shallower trees), a stricter gate, or scoring several events per call. All of these change
the model, so they need a new validation run.

**Re-run:**

```powershell
.\.venv\Scripts\python.exe -m sim.generate --out data
cmake -S engine -B engine/build -DCMAKE_BUILD_TYPE=Release
cmake --build engine/build --config Release
.\engine\build\Release\tsa_replay.exe data\train.events.bin --repeat 5
.\engine\build\Release\tsa_replay.exe data\train.events.bin --rules config\rules.ini --repeat 5
```

### Step 11: inside Docker, from a clean machine (2026-11-05)

Measured by the `docker-demo` CI job (GitHub-hosted `ubuntu-latest` runner; its CPU was not
recorded), GCC -O3, after `docker compose up --build` built everything from scratch.

**Reproducible from nothing:** the container re-simulated the data, retrained the model and
got the same threshold (6.91822), the same 85 validation alerts and the same 354 training-set
alerts as the Windows build; all 85 notes passed the evidence check.

Training set, 5 runs, rules + model:

| Metric | Docker (Linux, GCC -O3) | Windows laptop (MSVC /O2, Step 8) |
| --- | --- | --- |
| Throughput | **104,746 events/s** (range 100,831 – 105,293) | 101,918 events/s |
| Latency p50 | **902 ns** | 1,000 ns |
| Latency p95 | 44,894 ns | 47,300 ns |
| Latency p99 | **59,641 ns** | 71,600 ns |
| Latency mean | 9,495 ns | 9,709 ns |
| Events scored by the model | 207,632 | 207,632 |
| Alerts | 354 | 354 |

**Memory safety:** all C++ tests also run under AddressSanitizer + UndefinedBehaviorSanitizer
(`sanitizers` CI job): no errors.

**Test coverage (Python, `pytest --cov`):** 98% of lines in `sim/`, `ml/`, `service/` and
`dashboard/` (135 tests). C++: 113 tests; line coverage not measured.

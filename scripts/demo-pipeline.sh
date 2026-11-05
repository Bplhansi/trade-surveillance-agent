#!/bin/sh
# Builds everything the dashboard needs, from nothing (runs inside Docker).
# Steps that already ran are skipped, so restarting the demo is quick.
set -eu
cd /app

if [ ! -f data/validation.events.bin ]; then
    echo "== 1/5 Simulating market data (train / validation / test)"
    python -m sim.generate --out data
fi

for name in train validation; do
    if [ ! -f "data/$name.features.bin" ]; then
        echo "== 2/5 Exporting features for $name"
        tsa_replay "data/$name.events.bin" --rules config/rules.ini --features "data/$name.features.bin"
    fi
done

if [ ! -f models/gradient_boosting.joblib ]; then
    echo "== 3/5 Training the models (train + validation only; the test set is not used)"
    python -m ml.train --data data --out models
fi

if [ ! -f models/gradient_boosting.onnx ]; then
    echo "== 4/5 Exporting the model to ONNX"
    python -m ml.export_onnx --model models/gradient_boosting.joblib --out models/gradient_boosting.onnx
fi

echo "== 5/5 Running the engine (rules + model) and writing case notes"
python -m dashboard.prepare --name validation --engine /usr/local/bin/tsa_replay \
    --service-url "${NOTES_URL:-http://notes:8000}"
echo "== Done. Open the dashboard: http://localhost:8501"

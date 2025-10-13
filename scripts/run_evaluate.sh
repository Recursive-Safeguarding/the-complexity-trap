#!/bin/bash

BASE_DIR="./trajectories/500"
DATASET_NAME="princeton-nlp/SWE-bench_Verified"
MAX_WORKERS=40

for dir in "$BASE_DIR"/*; do
    [ -d "$dir" ] || continue

    PREDICTIONS_PATH="$dir/preds.json"
    if [ -f "$PREDICTIONS_PATH" ]; then
        RUN_ID=$(basename "$dir")
        echo "Running evaluation for: $RUN_ID"
        python -m swebench.harness.run_evaluation \
            --dataset_name "$DATASET_NAME" \
            --predictions_path "$PREDICTIONS_PATH" \
            --max_workers "$MAX_WORKERS" \
            --timeout 900 \
            --run_id "$RUN_ID" \
            --report_dir "./reports/"
    else
        echo "No preds.json found in $dir, skipping."
    fi
done

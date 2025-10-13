#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
gpus_count=$(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c | awk '{print $1+1}')
model_name="Qwen/Qwen3-8B"

port=$(comm -23 <(seq 1024 65535) <(ss -tuln | awk 'NR>1 {print $4}' | cut -d':' -f2 | sort -n) | shuf | head -n 1)

log_dir=logs/run_batch
vllm_log_file="$log_dir/$(basename "$0").vllm.log"
run_log_file="$log_dir/$(basename "$0").run.log"
eval_log_file="$log_dir/$(basename "$0").eval.log"

mkdir -p $log_dir

#########################################################

model_path=$model_name

python3 -m vllm.entrypoints.openai.api_server \
    --data_parallel_size $gpus_count \
    --enforce_eager \
    --gpu_memory_utilization 0.9 \
    --model $model_path \
    --tokenizer $model_path \
    --served-model-name $model_name \
    --enable_prefix_caching \
    --seed 41 \
    --port $port > $vllm_log_file 2>&1 &

vllm_pid=$!

#########################################################

timeout_minutes=20
start_time=$(date +%s)
timeout_seconds=$((timeout_minutes * 60))

echo "Waiting for vLLM server to initialize (timeout: ${timeout_minutes} minutes)..." >> $run_log_file

while [ $(($(date +%s) - start_time)) -lt $timeout_seconds ]; do
    if ! ps -p $vllm_pid > /dev/null; then
        echo "vLLM server process exited with an error" >> $run_log_file
        exit 1
    fi

    if [ -f "$vllm_log_file" ] && grep -q "Application startup complete." "$vllm_log_file"; then
        echo "vLLM server initialized successfully" >> $run_log_file
        break
    fi
    sleep 1
done

if [ $(($(date +%s) - start_time)) -ge $timeout_seconds ]; then
    echo "Server initialization timed out after ${timeout_minutes} minutes" >> $run_log_file
    kill $vllm_pid
    exit 1
fi

#########################################################

{
    sweagent run-batch \
        --config config/swesmith_infer_raw.yaml \
        --agent.model.name hosted_vllm/$model_name \
        --num_workers $((gpus_count * 15)) \
        --agent.model.temperature 0.0 \
        --agent.model.api_base http://localhost:$port/v1/ \
        --agent.model.per_instance_call_limit 75 \
        --agent.model.per_instance_cost_limit 0 \
        --agent.model.total_cost_limit 0 \
        --agent.model.retry.retries 3 \
        --agent.model.retry.min_wait 5 \
        --agent.model.retry.max_wait 60 \
        --agent.model.completion_kwargs='{"chat_template_kwargs":{"enable_thinking":false}}' \
        --instances.type swe_bench \
        --instances.filter "$(paste -sd, config/dataset_filters/verified-50.txt)" \
        --instances.subset verified \
        --instances.split test \
        --instances.shuffle False >> $run_log_file 2>&1
} ; kill $vllm_pid


#########################################################
# Run evaluation
#########################################################

BASE_DIR=$(awk '/Find output files at/ {path=""; while ((getline line) > 0) { if (line ~ /^[[:space:]]/) { sub(/^[[:space:]]+/, "", line); sub(/[[:space:]]+$/, "", line); path = path line } else { print path; exit } } print path; exit }' "$run_log_file")
DATASET_NAME="princeton-nlp/SWE-bench_Verified"
MAX_WORKERS=16

PREDICTIONS_PATH="$BASE_DIR/preds.json"

RUN_ID=$(basename "$BASE_DIR")
echo "Running evaluation for: $RUN_ID"
taskset -c 0-15 python -m swebench.harness.run_evaluation \
    --dataset_name "$DATASET_NAME" \
    --predictions_path "$PREDICTIONS_PATH" \
    --max_workers "$MAX_WORKERS" \
    --timeout 900 \
    --run_id "$RUN_ID" \
    --report_dir "./reports/"  >> $eval_log_file 2>&1

#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
gpus_count=$(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c | awk '{print $1+1}')
model_name="smith_qwen3-8b-full-lr5e-6-warmup100-adamw-torch____ft_xml_0p9"

find_available_port() {
    local port
    for port in $(seq 1024 65535 | shuf); do
        if ! ss -tuln | grep -q ":$port "; then
            echo $port
            return 0
        fi
    done
    echo "No available ports found" >&2
    return 1
}

port=$(find_available_port)

log_dir=logs/run_batch/$(basename "$0" .sh)/$(date +%Y-%m-%d_%H-%M-%S)
sglang_log_file="$log_dir/sglang.log"
run_log_file="$log_dir/run.log"
eval_log_file="$log_dir/eval.log"

mkdir -p $log_dir

#########################################################
##################### SGLang server #####################
#########################################################

model_path="/mnt/shared-fs/slinko/projects/LLaMA-Factory/saves/Qwen3-8B/full/${model_name}"  # if model is global, then use model_path=$model_path

python -m sglang.launch_server \
    --model-path $model_path \
    --served-model-name $model_name \
    --port $port \
    --dp $gpus_count \
    > $sglang_log_file 2>&1 &

sglang_pid=$!

#########################################################
##################### Wait for SGLang server ############
#########################################################

timeout_minutes=20
start_time=$(date +%s)
timeout_seconds=$((timeout_minutes * 60))

echo "Waiting for SGLang server to initialize (timeout: ${timeout_minutes} minutes)..." >> $run_log_file

while [ $(($(date +%s) - start_time)) -lt $timeout_seconds ]; do
    if ! ps -p $sglang_pid > /dev/null; then
        echo "SGLang server process exited with an error" >> $run_log_file
        exit 1
    fi

    if [ -f "$sglang_log_file" ] && grep -q "Uvicorn running on" "$sglang_log_file"; then
        echo "SGLang server initialized successfully" >> $run_log_file
        break
    fi
    sleep 1
done

if [ $(($(date +%s) - start_time)) -ge $timeout_seconds ]; then
    echo "Server initialization timed out after ${timeout_minutes} minutes" >> $run_log_file
    kill $sglang_pid
    exit 1
fi

#########################################################
##################### Run SWE-AGENT #####################
#########################################################
TRAJECTORIES_DIR=$log_dir/trajectories

{
    sweagent run-batch \
        --config config/swesmith_infer.yaml \
        --agent.model.name openai/$model_name \
        --num_workers $((gpus_count * 30)) \
        --agent.model.temperature 0.0 \
        --agent.model.api_base http://localhost:$port/v1/ \
        --agent.model.per_instance_call_limit 75 \
        --agent.model.per_instance_cost_limit 0 \
        --agent.model.total_cost_limit 0 \
        --agent.model.retry.retries 3 \
        --agent.model.retry.min_wait 5 \
        --agent.model.retry.max_wait 60 \
        --output_dir $TRAJECTORIES_DIR \
        --instances.type swe_bench \
        --instances.subset verified \
        --instances.split test \
        --instances.shuffle False >> $run_log_file 2>&1
} ; kill $sglang_pid


#########################################################
##################### Run evaluation ####################
#########################################################

DATASET_NAME="princeton-nlp/SWE-bench_Verified"
MAX_WORKERS=16

PREDICTIONS_PATH=$TRAJECTORIES_DIR/preds.json

RUN_ID=$(basename "$TRAJECTORIES_DIR")
echo "Running evaluation for: $RUN_ID"
taskset -c 0-15 python -m swebench.harness.run_evaluation \
    --dataset_name "$DATASET_NAME" \
    --predictions_path "$PREDICTIONS_PATH" \
    --max_workers "$MAX_WORKERS" \
    --timeout 900 \
    --run_id "$RUN_ID" \
    --report_dir $log_dir/reports  \
    --output_dir $log_dir/run_evaluation  >> $eval_log_file 2>&1

#########################################################
################ Calculate statistics ###################
#########################################################

python scripts/calculate_mean_traj_length.py $TRAJECTORIES_DIR >> $eval_log_file 2>&1

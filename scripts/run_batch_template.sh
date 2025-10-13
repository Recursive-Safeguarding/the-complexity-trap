#!/bin/bash

# $1: First argument is the run identifier or name

export CUDA_VISIBLE_DEVICES=2
model_name="Qwen/Qwen2.5-7B-Instruct"
port=8124
log_dir=logs/run_batch
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

run_id="${1:-unspecified_run}"
vllm_log_file="$log_dir/vllm_$(basename "$0")-${TIMESTAMP}-${run_id}.log"
run_log_file="$log_dir/swe_agent-$(basename "$0")-${TIMESTAMP}-${run_id}.log"

mkdir -p $log_dir

#########################################################

vllm serve $model_name \
    --tensor_parallel_size $(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c | awk '{print $1+1}') \
    --enforce_eager \
    --gpu_memory_utilization 0.7 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --rope-scaling '{"factor": 4.0, "original_max_position_embeddings": 32768, "rope_type": "yarn"}' \
    --enable_prefix_caching \
    --seed 41 \
    --port $port > $vllm_log_file 2>&1 &

vllm_pid=$!

#########################################################

timeout_minutes=9
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

# Input and output token limits for Qwen2.5

sweagent run-batch \
  --config config/default_no_demo.yaml \
  --agent.model.name hosted_vllm/$model_name \
  --agent.model.is_local_model true \
  --agent.model.per_instance_cost_limit 0 \
  --agent.model.total_cost_limit 0 \
  --num_workers 10 \
  --agent.model.temperature 0.8 \
  --agent.model.api_base http://0.0.0.0:$port/v1/ \
  --agent.model.per_instance_call_limit 70.00 \
  --agent.model.max_input_tokens $((128 * 1024 - 8 * 1024)) \
  --agent.model.max_output_tokens $((8 * 1024)) \
  --instances.type swe_bench \
  --instances.subset verified \
  --instances.split test \
  --instances.shuffle True >> $run_log_file 2>&1

#########################################################
#!/bin/bash

# $1: First argument is the run identifier or name
# $2: Second argument is an optional user-specified port that vLLM will use

export CUDA_VISIBLE_DEVICES=2,6
model_name="Qwen/Qwen3-32B"
log_dir=logs/run_batch
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

run_id="${1:-unspecified_run}"
vllm_log_file="$log_dir/vllm_$(basename "$0")-${TIMESTAMP}-${run_id}.log"
run_log_file="$log_dir/swe_agent-$(basename "$0")-${TIMESTAMP}-${run_id}.log"

mkdir -p $log_dir

#########################################################

find_free_port() {
    local port_candidate
    local min_port=49152
    local max_port=65535
    local max_attempts=100
    local attempt_num=0

    echo "Attempting to find a free port..." >&2
    while [ "$attempt_num" -lt "$max_attempts" ]; do
        port_candidate=$(shuf -i "${min_port}-${max_port}" -n 1)

        if ! ss -lntu | awk 'NR>1 {print $5}' | sed 's/.*://' | grep -qw "$port_candidate"; then
            echo "Found free port: $port_candidate" >&2
            echo "$port_candidate"
            return 0
        fi
        attempt_num=$((attempt_num + 1))
    done

    echo "Error: Could not find a free port after $max_attempts attempts." >&2
    return 1
}

#########################################################
# Determine the port to use
if [ -n "$2" ]; then
    requested_port="$2"
    echo "User requested port: $requested_port"
    if ! ss -lntu | awk 'NR>1 {print $5}' | sed 's/.*://' | grep -qw "$requested_port"; then
        port="$requested_port"
        echo "Requested port $port is free. Using it."
    else
        echo "Requested port $requested_port is occupied. Finding a random available port instead."
        port=$(find_free_port)
        if [ $? -ne 0 ]; then
            exit 1
        fi
    fi
else
    echo "No port specified by user. Finding a random available port."
    port=$(find_free_port)
    if [ $? -ne 0 ]; then
        exit 1
    fi
fi

echo "vLLM will use port: $port"

#########################################################

vllm serve $model_name \
    --tensor_parallel_size $(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c | awk '{print $1+1}') \
    --enforce_eager \
    --gpu_memory_utilization 0.95 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --enable-reasoning \
    --reasoning-parser deepseek_r1 \
    --max_num_seqs 1 \
    --rope-scaling '{"factor": 4.0, "original_max_position_embeddings": 32768, "rope_type": "yarn"}' \
    --enable_prefix_caching \
    --seed 41 \
    --port $port > $vllm_log_file 2>&1 &

vllm_pid=$!

#########################################################

# Clean up vLLM server if script is interrupted to prevent unneeded GPU usage
cleanup() { 
    echo "Script interrupted or exiting. Cleaning up vLLM server (PID: $vllm_pid)..." >&2
    if [ -n "$vllm_pid" ] && ps -p "$vllm_pid" > /dev/null; then
        kill "$vllm_pid"
        wait "$vllm_pid" 2>/dev/null 
        echo "vLLM server stopped." >&2
    else
        echo "vLLM server (PID: $vllm_pid) not found or already stopped." >&2
    fi
}
trap cleanup SIGINT SIGTERM EXIT

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

DEBUG_MODE=true 

# Input and output token limits for Qwen2.5
common_args=(
  --config config/default_no_demo_checkpoint_same_model_openhands_N=21_M=10.yaml
  --agent.model.name hosted_vllm/$model_name
  --agent.model.is_local_model True
  --agent.model.api_base http://0.0.0.0:$port/v1/
  --agent.model.per_instance_cost_limit 0
  --agent.model.total_cost_limit 0
  --agent.model.completion_kwargs '{"timeout": "600"}'
  --agent.model.use_reasoning True
  --num_workers 1
  --agent.model.temperature 0.8
  --agent.model.per_instance_call_limit 10.00
  --agent.model.max_input_tokens $((128 * 1024 - 8 * 1024))
  --agent.model.max_output_tokens $((8 * 1024))
  --instances.type swe_bench
  --instances.subset verified
  --instances.split test
  --instances.slice :1
)

if [ "$DEBUG_MODE" = true ]; then
    echo "!!!! ACTION-REQUIRED: Starting sweagent in debug mode - attach debugger to proceed !!!!" >> "$run_log_file"
    python -Xfrozen_modules=off -m debugpy --listen 5779 --wait-for-client -m sweagent run-batch "${common_args[@]}" >> "$run_log_file" 2>&1
else
    sweagent run-batch "${common_args[@]}" >> "$run_log_file" 2>&1
fi

#########################################################
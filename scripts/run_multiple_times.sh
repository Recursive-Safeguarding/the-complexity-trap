#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 <script_path> <num_runs>"
    echo "Example: $0 scripts/2025-09-02-run-batch-local_qwen2.5-32b-10k.sh 5"
    exit 1
fi

script_path="$1"
num_runs="$2"

if [ ! -f "$script_path" ]; then
    echo "Error: Script file '$script_path' not found"
    exit 1
fi

if ! [[ "$num_runs" =~ ^[0-9]+$ ]] || [ "$num_runs" -lt 1 ]; then
    echo "Error: Number of runs must be a positive integer"
    exit 1
fi

if [ ! -x "$script_path" ]; then
    echo "Making script executable..."
    chmod +x "$script_path"
fi

log_dir=logs/run_batch/$(basename "$script_path")/
date_str=$(date +%Y-%m-%d_%H-%M-%S)
summary_log="$log_dir/$date_str.summary.log"

mkdir -p "$log_dir"

echo "Starting multiple runs of: $script_path" | tee -a "$summary_log"
echo "Number of runs: $num_runs" | tee -a "$summary_log"
echo "Start time: $(date)" | tee -a "$summary_log"
echo "Log directory: $log_dir" | tee -a "$summary_log"
echo "----------------------------------------" | tee -a "$summary_log"

successful_runs=0
failed_runs=0

for run_num in $(seq 1 "$num_runs"); do
    echo "Starting run $run_num/$num_runs at $(date)" | tee -a "$summary_log"

    run_start_time=$(date +%s)

    if "$script_path"; then
        run_duration=$(($(date +%s) - run_start_time))
        echo "Run $run_num completed successfully in ${run_duration}s" | tee -a "$summary_log"
        ((successful_runs++))
    else
        run_duration=$(($(date +%s) - run_start_time))
        echo "Run $run_num failed after ${run_duration}s" | tee -a "$summary_log"

        if [ $run_num -gt 1 ]; then
            echo "Attempting retry for run $run_num (20 second delay)..." | tee -a "$summary_log"
            sleep 20

            retry_start_time=$(date +%s)
            if "$script_path"; then
                retry_status="succeeded"
                ((successful_runs++))
            else
                retry_status="failed"
                ((failed_runs++))
            fi
            retry_duration=$(($(date +%s) - retry_start_time))
            total_duration=$(($(date +%s) - run_start_time))
            retry_duration_fmt=$(printf "%d:%02d:%02d" $((retry_duration/3600)) $(( (retry_duration%3600)/60 )) $((retry_duration%60)))
            total_duration_fmt=$(printf "%d:%02d:%02d" $((total_duration/3600)) $(( (total_duration%3600)/60 )) $((total_duration%60)))
            echo "Run $run_num retry $retry_status in ${retry_duration_fmt} (total: ${total_duration_fmt})" | tee -a "$summary_log"
        else
            ((failed_runs++))
        fi
    fi

    echo "----------------------------------------" | tee -a "$summary_log"
done

echo "All runs completed at $(date)" | tee -a "$summary_log"
echo "Summary:" | tee -a "$summary_log"
echo "  Total runs: $num_runs" | tee -a "$summary_log"
echo "  Successful: $successful_runs" | tee -a "$summary_log"
echo "  Failed: $failed_runs" | tee -a "$summary_log"
echo "  Success rate: $((successful_runs * 100 / num_runs))%" | tee -a "$summary_log"

exit $([ $failed_runs -gt 0 ] && echo 1 || echo 0)

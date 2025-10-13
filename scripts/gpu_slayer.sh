#!/bin/bash

LOG_FILE="logs/gpu_slayer.log"
USER_TO_MONITOR="$USER"
CHECK_INTERVAL=600  # 10 mins
SLEEP_BETWEEN_CHECKS=1  
declare -A zero_usage_counts 

echo "$(date): GPU monitoring started." >> "$LOG_FILE"

while true; do
    declare -A current_pids

    while read -r pid gpu_usage; do
        if [[ -z "$pid" || "$pid" == "-" ]]; then
            continue
        fi

        process_user=$(ps -o user= -p "$pid" 2>/dev/null)

        if [[ "$process_user" == "$USER_TO_MONITOR" ]]; then
            current_pids["$pid"]=1

            if [[ "$gpu_usage" == "0" ]]; then
                ((zero_usage_counts[$pid]++))
            else
                zero_usage_counts[$pid]=0
            fi

            if [[ "${zero_usage_counts[$pid]}" -ge "$CHECK_INTERVAL" ]]; then
                echo "$(date): Killing process $pid (GPU usage: $gpu_usage%) after $CHECK_INTERVAL seconds of 0% GPU" >> "$LOG_FILE"
                kill -9 "$pid"
                unset zero_usage_counts[$pid]
            else
                echo "$(date): No kill for process $pid (GPU usage: $gpu_usage%, count: ${zero_usage_counts[$pid]}/$CHECK_INTERVAL)" >> "$LOG_FILE"
            fi
        fi
    done < <(nvidia-smi pmon -c 1 | awk 'NR>2 {print $2, ($4 == "-" ? "0" : $4)}')

    for pid in "${!zero_usage_counts[@]}"; do
        if [[ -z "${current_pids[$pid]}" ]]; then
            unset zero_usage_counts[$pid]
        fi
    done

    sleep "$SLEEP_BETWEEN_CHECKS"
done
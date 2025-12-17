#!/usr/bin/env bash
# Benchmark script for AWS Bedrock models - TPS measurement
#
# Usage:
#   AWS_BEARER_TOKEN_BEDROCK=<token> ./scripts/bedrock_tps_estimation.sh
#
# Optional:
#   MODEL=qwen.qwen3-32b-v1:0 REGION=eu-west-2 MAX_TOKENS=500 ./scripts/bedrock_tps_estimation.sh
#   # Auto-load .env (repo root) if present (default behavior):
#   ./scripts/bedrock_tps_estimation.sh
#   # Or specify a custom env file:
#   ENV_FILE=/path/to/.env ./scripts/bedrock_tps_estimation.sh
#
# Notes:
# - Uses bearer-token auth (Authorization: Bearer ...), same as the compaction repo TPS script.
# - Requires: curl, jq, bc

set -euo pipefail

load_dotenv() {
  # If a .env file exists, load it so AWS_BEARER_TOKEN_BEDROCK/AWS_DEFAULT_REGION are picked up.
  local env_file=""

  if [ -n "${ENV_FILE:-}" ]; then
    env_file="${ENV_FILE}"
  else
    local script_dir repo_root
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    repo_root="$(cd "${script_dir}/.." && pwd)"

    if [ -f "${repo_root}/.env" ]; then
      env_file="${repo_root}/.env"
    elif [ -f ".env" ]; then
      env_file=".env"
    fi
  fi

  if [ -n "${env_file}" ] && [ -f "${env_file}" ]; then
    # Export variables defined in the env file for the remainder of the script.
    set -a
    # shellcheck disable=SC1090
    source "${env_file}"
    set +a
  fi
}

load_dotenv

if ! command -v curl >/dev/null 2>&1; then
  echo "Error: curl is required"
  exit 1
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "Error: jq is required (install jq and retry)"
  exit 1
fi
if ! command -v bc >/dev/null 2>&1; then
  echo "Error: bc is required (install bc and retry)"
  exit 1
fi

if [ -z "${AWS_BEARER_TOKEN_BEDROCK:-}" ]; then
  echo "Error: AWS_BEARER_TOKEN_BEDROCK environment variable not set"
  exit 1
fi

MODEL="${MODEL:-qwen.qwen3-32b-v1:0}"
REGION="${REGION:-${AWS_DEFAULT_REGION:-eu-west-2}}"
ENDPOINT="https://bedrock-runtime.${REGION}.amazonaws.com/model/${MODEL}/converse"

WARMUP=${WARMUP:-3}
ITERATIONS=${ITERATIONS:-3}
MAX_TOKENS=${MAX_TOKENS:-500}

prompts=(
  "Explain the concept of entropy in thermodynamics and its relationship to the second law."
  "Describe how neural networks learn through backpropagation, including the role of gradients."
  "What are the key differences between TCP and UDP protocols, and when would you use each?"
  "Explain the CAP theorem in distributed systems and its practical implications."
  "How does garbage collection work in modern programming languages like Java or Go?"
  "Describe the process of photosynthesis at the molecular level."
  "What is the difference between symmetric and asymmetric encryption, with examples?"
  "Explain how compilers optimize code through techniques like loop unrolling and inlining."
)

call_api() {
  local prompt="$1"
  curl -s -X POST "$ENDPOINT" \
    -H "Authorization: Bearer $AWS_BEARER_TOKEN_BEDROCK" \
    -H "Content-Type: application/json" \
    -d "{\"messages\": [{\"role\": \"user\", \"content\": [{\"text\": \"$prompt\"}]}], \"inferenceConfig\": {\"maxTokens\": $MAX_TOKENS}}"
}

echo "Benchmarking $MODEL (region: $REGION)"
echo "Config: warmup=$WARMUP, prompts=${#prompts[@]}, iterations_per_prompt=$ITERATIONS, max_tokens=$MAX_TOKENS"
echo ""

echo -n "Warming up"
for ((i=1; i<=WARMUP; i++)); do
  call_api "Say hello briefly." > /dev/null
  echo -n "."
done
echo " done"
echo ""

tps_values=()
total_tokens=0
total_latency=0

printf "%-12s %10s %8s %8s\n" "Prompt" "Latency" "Tokens" "TPS"
echo "-------------------------------------------"

for ((iter=1; iter<=ITERATIONS; iter++)); do
  for pi in "${!prompts[@]}"; do
    prompt="${prompts[$pi]}"
    response=$(call_api "$prompt")

    latency=$(echo "$response" | jq -r '.metrics.latencyMs')
    output_tokens=$(echo "$response" | jq -r '.usage.outputTokens')

    if [ "$latency" = "null" ] || [ "$output_tokens" = "null" ]; then
      echo "P$((pi+1))/$iter: ERROR"
      continue
    fi

    tps=$(echo "scale=2; $output_tokens / ($latency / 1000)" | bc)
    tps_values+=("$tps")
    total_tokens=$((total_tokens + output_tokens))
    total_latency=$((total_latency + latency))

    printf "P%d/iter%d      %7d ms %6d   %6.1f\n" "$((pi+1))" "$iter" "$latency" "$output_tokens" "$tps"
  done
done

echo ""

# Calculate stats
n=${#tps_values[@]}
if [ $n -eq 0 ]; then
  echo "No successful measurements"
  exit 1
fi

sorted=($(printf '%s\n' "${tps_values[@]}" | sort -n))
mid=$((n / 2))
if [ $((n % 2)) -eq 0 ]; then
  median=$(echo "scale=1; (${sorted[$mid-1]} + ${sorted[$mid]}) / 2" | bc)
else
  median=${sorted[$mid]}
fi

sum=$(printf '%s\n' "${tps_values[@]}" | awk '{s+=$1} END {print s}')
mean=$(echo "scale=1; $sum / $n" | bc)

variance=$(printf '%s\n' "${tps_values[@]}" | awk -v mean="$mean" '{s+=($1-mean)^2} END {print s/NR}')
stddev=$(echo "scale=1; sqrt($variance)" | bc)

min=${sorted[0]}
max=${sorted[$((n-1))]}

# Percentiles (P10, P90)
p10_idx=$(echo "$n * 0.1 / 1" | bc)
p90_idx=$(echo "$n * 0.9 / 1" | bc)
[ $p10_idx -lt 0 ] && p10_idx=0
[ $p90_idx -ge $n ] && p90_idx=$((n-1))
p10=${sorted[$p10_idx]}
p90=${sorted[$p90_idx]}

agg_tps=$(echo "scale=1; $total_tokens / ($total_latency / 1000)" | bc)

echo "Results (n=$n samples across ${#prompts[@]} prompts):"
echo "  Median:     $median TPS"
echo "  Mean:       $mean TPS (±$stddev)"
echo "  P10-P90:    $p10 - $p90 TPS"
echo "  Range:      $min - $max TPS"
echo "  Aggregate:  $agg_tps TPS ($total_tokens tokens / $(echo "scale=1; $total_latency/1000" | bc)s)"

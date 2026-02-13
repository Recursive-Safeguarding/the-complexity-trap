#!/bin/bash
# Extract compaction trigger statistics from run directory
# Usage: ./scripts/extract_trigger_stats.sh <run_directory>

set -euo pipefail

if [ $# -eq 0 ]; then
    echo "Usage: $0 <run_directory>"
    echo "Example: $0 trajectories/root/glm-4.7__od_glm-4.7_la_lf-0p0001_lt-40000__mini__2026-02-10_00-06-03/"
    exit 1
fi

RUN_DIR="$1"

if [ ! -d "$RUN_DIR" ]; then
    echo "Error: Directory $RUN_DIR not found"
    exit 1
fi

echo "==================================================================="
echo "Compaction Trigger Statistics"
echo "==================================================================="
echo "Run: $(basename $RUN_DIR)"
echo ""

# Count total instances
TOTAL_INSTANCES=$(ls -1 "$RUN_DIR" | grep -E '^django|^astropy|^sympy|^matplotlib|^scikit' | wc -l)
echo "Total instances: $TOTAL_INSTANCES"
echo ""

# Count instances with triggers
INSTANCES_WITH_TRIGGERS=0
TOTAL_TRIGGERS=0

echo "Analyzing instance trajectories..."
echo ""

for instance_dir in "$RUN_DIR"/*; do
    if [ ! -d "$instance_dir" ]; then
        continue
    fi

    instance_name=$(basename "$instance_dir")

    # Skip non-instance directories
    if [[ ! "$instance_name" =~ ^(django|astropy|sympy|matplotlib|scikit) ]]; then
        continue
    fi

    # Look for .debug.log file
    debug_log=$(find "$instance_dir" -name "*.debug.log" -type f 2>/dev/null | head -1)

    if [ -z "$debug_log" ]; then
        continue
    fi

    # Count trigger messages in debug log
    TRIGGERS=$(grep -c "triggering compaction" "$debug_log" 2>/dev/null || echo "0")
    # Remove any whitespace/newlines
    TRIGGERS=$(echo "$TRIGGERS" | tr -d '\n\r ')

    if [ "$TRIGGERS" -gt 0 ] 2>/dev/null; then
        INSTANCES_WITH_TRIGGERS=$((INSTANCES_WITH_TRIGGERS + 1))
        TOTAL_TRIGGERS=$((TOTAL_TRIGGERS + TRIGGERS))
        echo "  $instance_name: $TRIGGERS triggers"
    fi
done

echo ""
echo "==================================================================="
echo "Summary:"
echo "==================================================================="
echo "  Instances analyzed: $TOTAL_INSTANCES"
echo "  Instances with triggers: $INSTANCES_WITH_TRIGGERS"
echo "  Trigger rate: $(awk "BEGIN {printf \"%.1f%%\", 100.0 * $INSTANCES_WITH_TRIGGERS / $TOTAL_INSTANCES}")"
echo "  Total triggers: $TOTAL_TRIGGERS"
if [ "$INSTANCES_WITH_TRIGGERS" -gt 0 ]; then
    echo "  Average triggers per triggered instance: $(awk "BEGIN {printf \"%.1f\", $TOTAL_TRIGGERS / $INSTANCES_WITH_TRIGGERS}")"
fi
echo ""

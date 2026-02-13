#!/usr/bin/env python3
"""Verify all experiment results from local trajectory logs (no WandB dependency).

Reads trajectory files directly from trajectories/root/ to produce ground-truth
tables of solve rates, token usage, and turn counts.

Usage:
    python scripts/verify_results_from_logs.py
"""
import json
import sys
from pathlib import Path
from math import sqrt

TRAJ_ROOT = Path("trajectories/root")

# exact directory names for each evaluated run
RUNS = [
    ("glm-4.7__raw__mini__2025-12-29_06-13-58",
     "raw", "—", "baseline"),
    ("glm-4.7__obs_n=10__mini__2025-12-29_06-13-58",
     "observation_masking", "—", "periodic"),
    ("glm-4.7__od_glm-4.7_la_lf-0p85_lt-0__mini__2026-02-08_14-53-12",
     "on_demand (self)", "glm-4.7", "on_demand"),
    ("glm-4.7__obs_n-10_la_lf-0p85_lt-0__mini__2026-02-08_17-43-39",
     "observation_masking", "—", "limit-aware"),
    ("glm-4.7__sum_glm-4.7_n-21_k-10__mini__2026-02-04_11-04-40",
     "llm_summary (self)", "glm-4.7", "periodic"),
    ("glm-4.7__hyb_minimax-m2.1_o=10_s=21_k=10__mini__2026-01-01_09-07-45",
     "hybrid", "minimax", "periodic"),
    ("glm-4.7__od_minimax-m2.1_la_lf-0p85_lt-0__mini__2026-02-08_14-46-08",
     "on_demand", "minimax", "on_demand"),
    ("glm-4.7__sum_minimax-m2.1_n-21_k-10__mini__2026-02-04_11-04-39",
     "llm_summary", "minimax", "periodic"),
]


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return max(0, center - margin), min(1, center + margin)


def analyze_run(run_dir):
    results_file = run_dir / "results.json"
    if not results_file.exists():
        return None

    with open(results_file) as f:
        results = json.load(f)

    resolved_ids = set(results.get("resolved_ids", []))

    instances = []
    instance_dirs = sorted(d for d in run_dir.iterdir() if d.is_dir())

    for idir in instance_dirs:
        instance_id = idir.name
        traj_files = list(idir.glob("*.traj"))
        if not traj_files:
            continue

        try:
            with open(traj_files[0]) as f:
                traj = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        info = traj.get("info", {})
        model_stats = info.get("model_stats", {})
        tokens = model_stats.get("tokens", {})
        history = traj.get("history", [])

        raw_input = tokens.get("raw_input", 0) or 0
        cached_input = tokens.get("cached_input", 0) or 0
        output = tokens.get("output", 0) or 0
        cost = model_stats.get("instance_cost", 0) or 0
        # api_calls = total LLM calls (includes retries); matches WandB n_turns
        n_turns = model_stats.get("api_calls", 0) or 0
        if n_turns == 0:
            n_turns = sum(1 for h in history if h.get("role") == "assistant")

        total_input = raw_input + cached_input
        avg_input = total_input / n_turns if n_turns > 0 else 0
        peak_estimate = max(0, 2 * avg_input - 5000) if n_turns > 0 else 0

        instances.append({
            "instance_id": instance_id,
            "resolved": instance_id in resolved_ids,
            "n_turns": n_turns,
            "total_input": total_input,
            "output": output,
            "cost": cost,
            "peak_estimate": peak_estimate,
        })

    return {
        "resolved": len([i for i in instances if i["resolved"]]),
        "n_instances": len(instances),
        "instances": instances,
        "dir": run_dir.name,
    }


def percentile(vals, p):
    """Simple percentile (nearest-rank method)."""
    s = sorted(vals)
    k = int(len(s) * p)
    return s[min(k, len(s) - 1)]


def print_box(headers, rows, col_widths=None):
    if col_widths is None:
        col_widths = []
        for i, h in enumerate(headers):
            w = len(str(h))
            for row in rows:
                w = max(w, len(str(row[i])))
            col_widths.append(w + 2)

    def line(left, mid, right, fill="─"):
        return "  " + left + mid.join(fill * w for w in col_widths) + right

    def data(vals):
        cells = []
        for v, w in zip(vals, col_widths):
            cells.append(f" {str(v):<{w-1}}")
        return "  │" + "│".join(cells) + "│"

    print(line("┌", "┬", "┐"))
    print(data(headers))
    print(line("├", "┼", "┤"))
    for row in rows:
        print(data(row))
    print(line("└", "┴", "┘"))


def main():
    if not TRAJ_ROOT.exists():
        print(f"ERROR: {TRAJ_ROOT} not found")
        sys.exit(1)

    all_results = []
    all_instances = {}

    for dirname, config, summarizer, trigger in RUNS:
        run_dir = TRAJ_ROOT / dirname
        if not run_dir.exists():
            print(f"  WARNING: {dirname} not found")
            continue

        data = analyze_run(run_dir)
        if data is None:
            print(f"  WARNING: no results.json in {dirname}")
            continue

        n = data["n_instances"]
        k = data["resolved"]
        ci_lo, ci_hi = wilson_ci(k, n)
        costs = [i["cost"] for i in data["instances"] if i["cost"] > 0]
        avg_cost = sum(costs) / len(costs) if costs else 0

        all_results.append({
            "config": config, "summarizer": summarizer, "trigger": trigger,
            "resolved": k, "total": n, "solve_rate": k / n if n else 0,
            "ci_lo": ci_lo, "ci_hi": ci_hi, "avg_cost": avg_cost,
            "dir": data["dir"],
        })
        all_instances[config] = data["instances"]

    if not all_results:
        print("No results found.")
        sys.exit(1)

    raw_rate = next((r["solve_rate"] for r in all_results if r["config"] == "raw"), 0)
    all_results.sort(key=lambda r: (-r["solve_rate"], r["config"]))

    # === Table 1: Solve rates ===
    print()
    print("  GLM-4.7 on SWE-bench Verified (from local trajectory logs)")
    print()

    headers = ["Configuration", "Summarizer", "Trigger", "Solve Rate (95% CI)", "Cost", "vs Raw"]
    rows = []
    for r in all_results:
        rate = f"{r['solve_rate']*100:.1f}% [{r['ci_lo']*100:.1f}%, {r['ci_hi']*100:.1f}%]"
        cost = f"${r['avg_cost']:.2f}"
        delta = "—" if r["config"] == "raw" else f"{(r['solve_rate'] - raw_rate)*100:+.1f}%"
        rows.append([r["config"], r["summarizer"], r["trigger"], rate, cost, delta])
    print_box(headers, rows)

    print()
    print("  NOTE: Costs are agent-model only (summary model costs not tracked in .traj files)")
    print()
    print("  Source directories:")
    for r in all_results:
        n = r["total"]
        k = r["resolved"]
        print(f"    {r['config']:30s} {k:2d}/{n:2d}  {r['dir']}")

    # === Table 2: Peak context thresholds ===
    raw_insts = all_instances.get("raw", [])
    if raw_insts:
        peaks = [i["peak_estimate"] for i in raw_insts]
        n = len(peaks)
        med = percentile(peaks, 0.5)
        mean = sum(peaks) / n

        print()
        print(f"  Peak context (estimated)")
        print(f"  RAW baseline median: {med/1000:.1f}k tokens (mean: {mean/1000:.1f}k)")
        print()

        rows = []
        for val, label in [(40000,"40k"),(60000,"60k"),(80000,"80k"),(100000,"100k"),(170000,"170k (current 85%)")]:
            pct = sum(1 for p in peaks if p > val) / n * 100
            rows.append([label, f"{pct:.0f}%"])
        print_box(["Threshold", "% of raw instances that exceed it"], rows)

    # === Table 3: Turn count thresholds ===
    if raw_insts:
        turns = [i["n_turns"] for i in raw_insts]
        n = len(turns)
        med = percentile(turns, 0.5)
        q25 = percentile(turns, 0.25)
        q75 = percentile(turns, 0.75)

        print()
        print(f"  Turn count")
        print(f"  RAW baseline median: {med} turns (IQR: [{q25}, {q75}])")
        print()

        rows = []
        for val, label in [(10,"M=10  (obs window)"),(21,"N=21  (sum trigger)"),(43,"N=43  (hybrid)")]:
            pct = sum(1 for t in turns if t > val) / n * 100
            rows.append([label, f"{pct:.0f}%"])
        print_box(["Threshold", "% of raw instances that exceed it"], rows)

    # === Table 4: Turn count by strategy ===
    LABELS = {
        "raw": "Raw",
        "observation_masking": "Masking (periodic)",
        "observation_masking (limit-aware)": "Masking (limit-aware)",
        "on_demand (self)": "On-demand (self)",
        "on_demand": "On-demand (minimax)",
        "llm_summary (self)": "Summary (periodic, self)",
        "llm_summary": "Summary (periodic, minimax)",
        "hybrid": "Hybrid (periodic, minimax)",
    }
    ORDER = ["raw", "observation_masking", "observation_masking (limit-aware)",
             "on_demand (self)", "on_demand",
             "llm_summary (self)", "llm_summary", "hybrid"]

    print()
    print("  Turn count by strategy")
    print()
    headers = ["Strategy", "Median", "IQR", "Min", "Max"]
    rows = []
    for cfg in ORDER:
        insts = all_instances.get(cfg, [])
        if not insts:
            continue
        turns = sorted(i["n_turns"] for i in insts)
        rows.append([LABELS.get(cfg, cfg), str(percentile(turns, 0.5)),
                     f"[{percentile(turns, 0.25)}, {percentile(turns, 0.75)}]",
                     str(turns[0]), str(turns[-1])])
    print_box(headers, rows)

    # === Table 5: Peak context by strategy ===
    print()
    print("  Peak context by strategy (estimated)")
    print()
    headers = ["Strategy", "Median (k)", "IQR (k)", "Min (k)", "Max (k)"]
    rows = []
    for cfg in ORDER:
        insts = all_instances.get(cfg, [])
        if not insts:
            continue
        peaks = sorted(i["peak_estimate"] / 1000 for i in insts)
        rows.append([LABELS.get(cfg, cfg), f"{percentile(peaks, 0.5):.1f}",
                     f"[{percentile(peaks, 0.25):.1f}, {percentile(peaks, 0.75):.1f}]",
                     f"{peaks[0]:.1f}", f"{peaks[-1]:.1f}"])
    print_box(headers, rows)

    print()
    print(f"  Total instances analyzed: {sum(len(v) for v in all_instances.values())}")
    print(f"  Runs verified: {len(all_results)}")


if __name__ == "__main__":
    main()

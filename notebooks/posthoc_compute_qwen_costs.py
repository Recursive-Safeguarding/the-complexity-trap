# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
import os, json
from pathlib import Path
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, Optional, Any

os.chdir("..")

# %%
base_dir = 'trajectories/lindenbauer/main_experiments'
experiments = os.listdir(base_dir)

# %% [markdown]
# # Posthoc compute qwen costs

# %%
# Unit: $/1M tokens

qwen_costs = {
    'coder': {
        'scale': 1000000,
        '<32k': {
            'input': 1,
            'output': 5,
            'cached_input': 0.4
        },
        '32k-128k': {
            'input': 1.8,
            'output': 9,
            'cached_input': 0.72
        },
        '128k-256k': {
            'input': 3,
            'output': 15,
            'cached_input': 1.2
        },
        '256k-1M': {
            'input': 6,
            'output': 60,
            'cached_input': 2.4
        },
    },
    'qwen3-32b': {
        'scale': 1000000,
        'any': {
            'input': 0.7,
            'output': 2.8,
            'cached_input': 0.7
        },
    },
    'qwen3-32b-thinking': {
        'scale': 1000000,
        'any': {
            'input': 0.7,
            'output': 8.4,
            'cached_input': 0.7
        },
    }
}

# %%
# Compute costs for Qwen3 experiments post-hoc (concurrent per-experiment)
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def _process_experiment_costs(experiment: str) -> tuple[str, float, float]:
    model: str | None = None
    lower_name = experiment.lower()
    if 'coder' in lower_name:
        model = 'coder'
    elif 'thinking' in lower_name and 'qwen3' in lower_name:
        model = 'qwen3-32b-thinking'
    elif 'thinking' not in lower_name and 'qwen3' in lower_name:
        model = 'qwen3-32b'
    else:
        return experiment, 0.0, 0.0

    dir_path = Path(base_dir) / experiment
    traj_files = list(dir_path.rglob('*.traj'))
    experiment_cost: float = 0.0
    experiment_summary_cost: float = 0.0

    for trajectory_file in traj_files:
        try:
            trajectory = json.loads(Path(trajectory_file).read_text())
        except Exception:
            continue

        agent_cost: float = 0.0
        summary_cost: float = 0.0
        for i, turn in enumerate(trajectory['trajectory']):
            if 'turn_statistics' not in turn or turn['turn_statistics'] is None:
                if 'exit' not in turn['response'].lower() and 'exit' not in turn['observation'].lower():
                    print(f'Warning: No turn statistics found for {trajectory_file} turn {i}.')
                    print(turn)
                continue

            if 'tokens' not in turn['turn_statistics']:
                if 'exit' not in turn['response'].lower() and 'exit' not in turn['observation'].lower():
                    print(f'Warning: No turn statistics found for tokens in {trajectory_file} turn {i}.')
                continue

            if ('raw_input' not in turn['turn_statistics']['tokens'] or
                'cached_input' not in turn['turn_statistics']['tokens'] or
                'output' not in turn['turn_statistics']['tokens']):
                if 'exit' not in turn['response'].lower() and 'exit' not in turn['observation'].lower():
                    print(f'Warning: No detailed turn statistics found for tokens (I/O) in {trajectory_file} turn {i}.')
                continue

            total_current_context = turn['turn_statistics']['tokens']['raw_input'] + turn['turn_statistics']['tokens']['cached_input']

            cost_tier = 'any'
            if 'coder' in lower_name:
                if total_current_context < 32000:
                    cost_tier = '<32k'
                elif total_current_context >= 32000 and total_current_context < 128000:
                    cost_tier = '32k-128k'
                elif total_current_context >= 128000 and total_current_context < 256000:
                    cost_tier = '128k-256k'
                elif total_current_context >= 256000 and total_current_context < 1000000:
                    cost_tier = '256k-1M'
                else:
                    print('ERROR: Unexpected context window, should not be possible.')
                    continue

            turn['turn_statistics']['cost'] = (turn['turn_statistics']['tokens']['raw_input'] / qwen_costs[model]['scale']) * qwen_costs[model][cost_tier]['input'] + \
                                              (turn['turn_statistics']['tokens']['cached_input'] / qwen_costs[model]['scale']) * qwen_costs[model][cost_tier]['cached_input'] + \
                                              (turn['turn_statistics']['tokens']['output'] / qwen_costs[model]['scale']) * qwen_costs[model][cost_tier]['output']
            if 'internal_reasoning' in turn['turn_statistics']['tokens']:
                turn['turn_statistics']['cost'] += (turn['turn_statistics']['tokens']['internal_reasoning'] / qwen_costs[model]['scale']) * qwen_costs[model][cost_tier]['output']
            agent_cost += turn['turn_statistics']['cost']

        if trajectory['summaries'] is not None:
            for summary in trajectory['summaries']:
                summary['statistics']['cost'] = (summary['statistics']['tokens']['raw_input'] / qwen_costs[model]['scale']) * qwen_costs[model][cost_tier]['input'] + \
                                                (summary['statistics']['tokens']['cached_input'] / qwen_costs[model]['scale']) * qwen_costs[model][cost_tier]['cached_input'] + \
                                                (summary['statistics']['tokens']['output'] / qwen_costs[model]['scale']) * qwen_costs[model][cost_tier]['output']
                if 'internal_reasoning' in summary['statistics']['tokens']:
                    summary['statistics']['cost'] += (summary['statistics']['tokens']['internal_reasoning'] / qwen_costs[model]['scale']) * qwen_costs[model][cost_tier]['output']
                summary_cost += summary['statistics']['cost']
            trajectory['info']['summary_model_stats']['instance_cost'] = summary_cost

        trajectory['info']['agent_model_stats']['instance_cost'] = agent_cost
        trajectory['info']['model_stats']['instance_cost'] = agent_cost + summary_cost
        experiment_cost += trajectory['info']['model_stats']['instance_cost']
        experiment_summary_cost += summary_cost

        try:
            Path(trajectory_file).write_text(json.dumps(trajectory, indent=2))
        except Exception:
            continue

    return experiment, experiment_cost, experiment_summary_cost

with ThreadPoolExecutor(max_workers=6) as executor:
    futures = {executor.submit(_process_experiment_costs, experiment): experiment for experiment in experiments}
    for future in as_completed(futures):
        experiment, experiment_cost, experiment_summary_cost = future.result()
        if experiment_cost == 0.0 and experiment_summary_cost == 0.0 and ('qwen3' not in experiment.lower() and 'coder' not in experiment.lower()):
            print(f'Skipping {experiment}')
            continue
        print('='*20)
        print(f'Costs for experiment {experiment} were {experiment_cost}$. (Summaries: {experiment_summary_cost})')
        print('='*20)


# %% [markdown]
# # Main Figure - Cost vs Solve rate facette plot

# %%
# Construct experiment results dictionary with cost and results structure
experiment_results = {}

for experiment in experiments:
    experiment_results[experiment] = {
        'cost': 0,
        'results': None
    }


# %%
def load_swe_bench_results(experiment_name: str) -> dict | None:
    """Load SWE-bench results JSON file for the given experiment."""
    swe_bench_dir = Path("auxiliary-data/swe-agent-eval-results")
    potential_files = list(swe_bench_dir.glob(f"*{experiment_name}.json"))
    
    if not potential_files:
        return None
    
    # If multiple files match, take the first one (could be enhanced with better matching logic)
    results_file = potential_files[0]
    try:
        return json.loads(results_file.read_text())
    except (json.JSONDecodeError, FileNotFoundError):
        return None

def compute_experiment_cost(experiment_name: str, base_directory: str) -> float:
    """Compute total cost for an experiment by summing instance costs from all trajectory files."""
    experiment_dir = Path(base_directory) / experiment_name
    traj_files = list(experiment_dir.rglob("*.traj"))
    
    total_cost = 0.0
    
    for traj_file in traj_files:
        try:
            trajectory_data = json.loads(traj_file.read_text())
            instance_cost = trajectory_data.get("info", {}).get("model_stats", {}).get("instance_cost", 0)
            total_cost += instance_cost
        except (json.JSONDecodeError, FileNotFoundError, KeyError):
            # Skip files that can't be read or don't have the expected structure
            continue
    
    return total_cost

def process_single_experiment(experiment: str) -> Tuple[str, dict[str, Any] | None, float]:
    """Process a single experiment: load results and compute cost."""
    results = load_swe_bench_results(experiment)
    cost = compute_experiment_cost(experiment, base_dir)
    return experiment, results, cost


# Infer number of instances (denominator) from experiment name
def _infer_experiment_denominator(experiment_name: str) -> int:
    name = experiment_name.lower()
    if "verified-500" in name:
        return 500
    
    if "verified-50" in name:
        return 50
    
    # Fallbacks if format differs
    if name.endswith("-500") or "-500-" in name:
        return 500
    if name.endswith("-50") or "-50-" in name:
        return 50
    
    return 500

# %%
# Process all experiments concurrently
print(f"Processing {len(experiments)} experiments with concurrent workers...")

with ThreadPoolExecutor(max_workers=4) as executor:
    # Submit all experiments for processing
    future_to_experiment = {
        executor.submit(process_single_experiment, experiment): experiment 
        for experiment in experiments
    }
    
    # Process completed futures with progress bar
    with tqdm(total=len(experiments), desc="Processing experiments") as pbar:
        for future in as_completed(future_to_experiment):
            experiment, results, cost = future.result()
            
            # Update experiment_results dictionary
            experiment_results[experiment]["results"] = results
            experiment_results[experiment]["cost"] = cost
            
            print(f"Processed {experiment}: Cost=${cost:.2f}, "
                  f"Results={'Found' if results else 'Not found'}")
            
            pbar.update(1)

print(f"\nCompleted processing {len(experiments)} experiments.")


# %%
for experiment in experiments:
    denom = _infer_experiment_denominator(experiment)
    experiment_results[experiment]['solve_rate'] = (experiment_results[experiment]['results']['resolved_instances'] / denom) * 100

# %%
for experiment in experiments:
    print(f'{experiment}: {experiment_results[experiment]["cost"]}$ for {experiment_results[experiment]["solve_rate"]} instances.')

# %%
for experiment in experiments:
    denom = _infer_experiment_denominator(experiment)
    experiment_results[experiment]['instance_cost'] = experiment_results[experiment]['cost'] / denom


# %% [markdown]
# ## Plot

# %%
# Main Plot for Paper
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
from collections import defaultdict

plt.style.use('default')
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.0
})

strategies = [
    ('Raw Agent', 'raw', 'black', 'o'),
    ('Observation Masking', 'baseline_N', '#6b57ff', '^'),
    ('LLM-Summary', 'summaries', '#ff318c', 's'),
    # New strategy: Hybrid (green diamonds)
    ('Hybrid', 'hybrid', '#1a7f37', 'D'),
]
# Optional per-experiment annotation configuration for Hybrid points.
# Keyed by subplot label -> experiment substring -> annotation config.
# Configure 'xy' (text position) and 'text' (annotation label). The arrow points
# from the text position to the actual data point for that experiment.
from typing import Any as _Any

HYBRID_ANNOTATIONS: dict[str, dict[str, dict[str, _Any]]] = {
    # Example (leave empty by default; user can fill in):
    '(c) Qwen3-Coder 480B': {
        'N_43_M_10_obs_masking_M_10_combined': {
            'xy': (0.2, 35.0),
            'text': 'LLM: N=43, M=10\nMASK: W=10',
        },
        'N_21_M_10_obs_masking_M_10_combined': {
            'xy': (0.75, 45.0),
            'text': 'LLM: N=21, M=10\nMASK: W=10',
        },
    },
    'Qwen3-Coder 480B': {
        'N_43_M_10_obs_masking_M_10_combined': {
            'xy': (0.2, 35.0),
            'text': 'LLM: N=43, M=10\nMASK: W=10',
        },
        'N_21_M_10_obs_masking_M_10_combined': {
            'xy': (0.75, 45.0),
            'text': 'LLM: N=21, M=10\nMASK: W=10',
        },
    },
}

def _find_hybrid_annotation_config(experiment_name: str, subplot_name: str) -> dict[str, _Any] | None:
    group_cfg = HYBRID_ANNOTATIONS.get(subplot_name, {})
    for key, cfg in group_cfg.items():
        if key in experiment_name:
            return cfg
    return None

# Group experiments by model type, including Qwen3-32B thinking as a separate group
experiment_groups = defaultdict(list)
for experiment in experiments:
    if 'gemini_2.5_flash_thinking' in experiment:
        group_key = '(b) Gemini 2.5 Flash (Thinking)'
    elif 'gemini_2.5_flash' in experiment:
        group_key = '(a) Gemini 2.5 Flash'
    elif 'Qwen3-Coder_480B' in experiment:
        group_key = '(c) Qwen3-Coder 480B'
    elif 'qwen3-32b' in experiment and 'thinking' not in experiment:
        group_key = '(d) Qwen3-32B'
    elif 'Qwen3-32B_thinking' in experiment:
        group_key = '(e) Qwen3-32B (Thinking)'
    else:
        parts = experiment.split('-')
        group_key = f"{parts[1]}-{parts[2]}"
    experiment_groups[group_key].append(experiment)

def get_strategy_info(experiment_name):
    name_lower = experiment_name.lower()
    # Robust Hybrid detection to cover legacy names (e.g., "combined") or
    # experiments that mix summaries and masking (contain both summaries and N_*).
    if (
        'hybrid' in name_lower
        or 'combined' in name_lower
        or (
            ('summary' in name_lower or 'summaries' in name_lower)
            and ('baseline_n' in name_lower or '_n_' in name_lower or ' n_' in name_lower)
        )
    ):
        return 'Hybrid', '#1a7f37', 'D'

    for label, identifier, color, marker in strategies:
        if identifier in experiment_name:
            return label, color, marker
    return None, None, None

# Define the order and row/col placement for the subplots (remove the duplicate)
subplot_order = [
    '(a) Gemini 2.5 Flash',
    '(b) Gemini 2.5 Flash (Thinking)',
    '(c) Qwen3-Coder 480B',
    '(d) Qwen3-32B',
    '(e) Qwen3-32B (Thinking)'
]

# Map group names to experiment_groups keys
group_name_map = {}
for key in experiment_groups:
    if key.startswith('(a)'):
        group_name_map['(a) Gemini 2.5 Flash'] = key
    elif key.startswith('(b)'):
        group_name_map['(b) Gemini 2.5 Flash (Thinking)'] = key
    elif key.startswith('(c)'):
        group_name_map['(c) Qwen3-Coder 480B'] = key
    elif key.startswith('(d)'):
        group_name_map['(d) Qwen3-32B'] = key
    elif key.startswith('(e)'):
        group_name_map['(e) Qwen3-32B (Thinking)'] = key

# 2 rows, 3 columns, but only 5 subplots (last one in bottom right will be empty)
fig, axes = plt.subplots(2, 3, figsize=(10.5, 7), constrained_layout=False)
axes = axes.reshape(2, 3)

legend_elements = []
for label, _, color, marker in strategies:
    legend_elements.append(Line2D([0], [0], marker=marker, color='w',
                                  markerfacecolor=color, markersize=8,
                                  label=label, linestyle='None',
                                  markeredgecolor='black', markeredgewidth=0.5))

# --- Consistent x-axis and y-axis across all tiles ---
all_costs = []
all_solve_rates = []
for group_experiments in experiment_groups.values():
    for exp in group_experiments:
        all_costs.append(experiment_results[exp]['instance_cost'])
        all_solve_rates.append(experiment_results[exp]['solve_rate'])
if all_costs:
    global_xmin = min(all_costs)
    global_xmax = max(all_costs)
    x_margin = (global_xmax - global_xmin) * 0.1 if global_xmax > global_xmin else 0.1 * (global_xmax if global_xmax else 1)
    global_xlim = (global_xmin - x_margin, global_xmax + x_margin)
else:
    global_xlim = (0, 1)

if all_solve_rates:
    global_ymin = min(all_solve_rates)
    global_ymax = max(all_solve_rates)
    y_margin = (global_ymax - global_ymin) * 0.1 if global_ymax > global_ymin else 0.1 * (global_ymax if global_ymax else 1)
    global_ylim = (global_ymin - y_margin, global_ymax + y_margin)
else:
    global_ylim = (0, 100)

# Plot each group in the specified order and row/col
for idx, subplot_name in enumerate(subplot_order):
    row, col = divmod(idx, 3)
    ax = axes[row, col]
    group_key = group_name_map.get(subplot_name)
    group_name = subplot_name
    if group_key is None:
        ax.axis('off')
        continue
    group_experiments = experiment_groups[group_key]
    costs = [experiment_results[exp]['instance_cost'] for exp in group_experiments]
    solve_rates = [experiment_results[exp]['solve_rate'] for exp in group_experiments]

    for experiment in group_experiments:
        cost = experiment_results[experiment]['instance_cost']
        solve_rate = experiment_results[experiment]['solve_rate']
        strategy_label, color, marker = get_strategy_info(experiment)
        if strategy_label:
            ax.scatter(cost, solve_rate, c=color, marker=marker, s=80,
                       alpha=0.9, edgecolors='black', linewidth=0.5, zorder=5)
            # Optional annotations for Hybrid points
            if strategy_label == 'Hybrid':
                ann_cfg = _find_hybrid_annotation_config(experiment, group_name)
                if ann_cfg is not None:
                    xytext = tuple(ann_cfg.get('xy', (cost, solve_rate)))  # text position
                    text = str(ann_cfg.get('text', ''))
                    ax.annotate(
                        text,
                        xy=(cost, solve_rate),
                        xytext=xytext,
                        textcoords='data',
                        fontsize=8,
                        ha='left', va='bottom',
                        color=color,
                        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor=color, linewidth=0.8),
                        arrowprops=dict(arrowstyle='-', color=color, lw=0.8),
                        zorder=10,
                    )

    ax.set_title(group_name, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # Set consistent x-axis and y-axis across all tiles
    ax.set_xlim(global_xlim)
    ax.set_ylim(global_ylim)

    ax.tick_params(direction='in', length=3, width=0.5)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.1f}'))

    # Remove per-tile y-axis label, will use fig.text for global label
    ax.set_ylabel('')

# Turn off the unused subplot (bottom right)
axes[1, 2].axis('off')

plt.subplots_adjust(left=0.08, bottom=0.18, right=0.97, top=0.88, wspace=0.15, hspace=0.35)

fig.text(0.525, 0.13, 'Instance Cost (USD) ', ha='center', va='center', fontweight='bold', fontsize=12)
fig.text(0.015, 0.5, 'Solve Rate (%) ', ha='center', va='center', fontweight='bold', fontsize=12, rotation='vertical')

fig.legend(handles=legend_elements,
           loc='lower center',
           bbox_to_anchor=(0.52, 0.06),
           ncol=4,
           frameon=False,
           columnspacing=1.5)

plt.show()

fig.savefig('report_data/figure1_cost_vs_solve_rate.pdf', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
fig.savefig('report_data/figure1_cost_vs_solve_rate.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')

print("Figure saved as 'figure1_cost_vs_solve_rate.pdf' and 'figure1_cost_vs_solve_rate.png'")


# %%
# Qwen3-Coder 480B single scatter plot (same formatting as main figure, no panel prefix)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

plt.style.use('default')
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.0
})

fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.5), constrained_layout=False)

group_key = group_name_map.get('(c) Qwen3-Coder 480B')
group_experiments = experiment_groups.get(group_key, []) if group_key is not None else []

for experiment in group_experiments:
    cost = experiment_results[experiment]['instance_cost']
    solve_rate = experiment_results[experiment]['solve_rate']
    strategy_label, color, marker = get_strategy_info(experiment)
    if strategy_label:
        ax.scatter(cost, solve_rate, c=color, marker=marker, s=80,
                   alpha=0.9, edgecolors='black', linewidth=0.5, zorder=5)
        if strategy_label == 'Hybrid':
            ann_cfg = _find_hybrid_annotation_config(experiment, 'Qwen3-Coder 480B')
            if ann_cfg is not None:
                xytext = tuple(ann_cfg.get('xy', (cost, solve_rate)))
                text = str(ann_cfg.get('text', ''))
                ax.annotate(
                    text,
                    xy=(cost, solve_rate),
                    xytext=xytext,
                    textcoords='data',
                    fontsize=8,
                    ha='left', va='bottom',
                    color=color,
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor=color, linewidth=0.8),
                    arrowprops=dict(arrowstyle='-', color=color, lw=0.8),
                    zorder=10,
                )

ax.set_title('Qwen3-Coder 480B', fontweight='bold', pad=10)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_axisbelow(True)

# Keep axis limits consistent with main figure
ax.set_xlim(global_xlim)
ax.set_ylim(global_ylim)

ax.tick_params(direction='in', length=3, width=0.5)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.1f}'))
ax.set_xlabel('Instance Cost (USD)', fontweight='bold', fontsize=12)
ax.set_ylabel('Solve Rate (%)', fontweight='bold', fontsize=12)

# Legend (same style as main figure, single row with 4 strategies)
legend_elements = []
for label, _, color, marker in strategies:
    legend_elements.append(Line2D([0], [0], marker=marker, color='w',
                                  markerfacecolor=color, markersize=8,
                                  label=label, linestyle='None',
                                  markeredgecolor='black', markeredgewidth=0.5))
# Use a figure-level legend anchored below and increase bottom margin to make space
fig.legend(handles=legend_elements,
           loc='lower center',
           bbox_to_anchor=(0.5, 0.02),
           ncol=4,
           frameon=False)

plt.subplots_adjust(left=0.12, bottom=0.22, right=0.95, top=0.90)
plt.show()
Path('report_data').mkdir(parents=True, exist_ok=True)
fig.savefig('report_data/qwen3_coder_single_scatter.pdf', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
fig.savefig('report_data/qwen3_coder_single_scatter.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Qwen3-Coder 480B single scatter saved as 'qwen3_coder_single_scatter.pdf' and 'qwen3_coder_single_scatter.png'")

# %%
openhands_df.experiment.unique()

# %%
# OpenHands scatter plot
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
from collections import defaultdict

# Load OpenHands data from CSV
openhands_csv_path = Path("auxiliary-data/experiment_instance_costs_openhands.csv")
openhands_df = pd.read_csv(openhands_csv_path)

# Aggregate per-experiment metrics
openhands_data = {}
for experiment_name, group in openhands_df.groupby('experiment'):
    total_cost = group['cost'].sum()  # Total cost across all instances
    summary_cost = group['summary_cost'].sum()  # Total summary cost across all instances
    n_instances = len(group)
    n_solved = group['outcome'].sum()
    
    openhands_data[experiment_name] = {
        'instance_cost': total_cost / n_instances,  # Average cost per instance
        'summary_instance_cost': summary_cost / n_instances,  # Average summary cost per instance
        'solve_rate': n_solved / n_instances,  # Solve rate as proportion
        'total_cost': total_cost,
        'total_summary_cost': summary_cost,
        'n_instances': n_instances,
        'n_solved': n_solved
    }

# Create single scatter plot following the same style as the main figure
plt.style.use('default')
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.0
})

# Use the same strategy definitions as the main figure
strategies_openhands = [
    ('Raw Agent', 'raw_agent', 'black', 'o'),
    ('Observation Masking', 'observation_masking', '#6b57ff', '^'),
    ('LLM-Summary', 'llm_summary', '#ff318c', 's'),
]

def get_strategy_info_openhands(experiment_name):
    """Extract strategy from OpenHands experiment name."""
    exp_lower = experiment_name.lower()
    
    # Check for LLM-Summary strategy
    if 'llm_summary' in exp_lower or 'llm-summary' in exp_lower:
        return 'LLM-Summary', '#ff318c', 's'
    
    # Check for Observation Masking strategy  
    elif ('observation_masking' in exp_lower or 
          ('baseline' in exp_lower and 'raw' not in exp_lower)):
        return 'Observation Masking', '#6b57ff', '^'
    
    # Check for Raw Agent strategy
    elif ('raw_agent' in exp_lower or 'raw-agent' in exp_lower or
          ('baseline' in exp_lower and 'raw' in exp_lower)):
        return 'Raw Agent', 'black', 'o'
    
    # Default fallback - assume it's observation masking if it has maxiter and no other indicators
    elif 'maxiter' in exp_lower and 'llm_summary' not in exp_lower:
        return 'Observation Masking', '#6b57ff', '^'
    
    return None, None, None

# Optional: per-experiment annotations for OpenHands. Keyed by experiment-name substring.
# Each config supports keys: 'xy' (text position tuple) and 'text' (annotation label).
from typing import Any as _Any
OPENHANDS_ANNOTATIONS: dict[str, dict[str, _Any]] = {
    'gemini-2.5-flash_maxiter_250_N_v0.43.0-no-hint-llm_summary_N_21_M_10-verified_50-summarizer_for_eval-run_1': {},
    'gemini-2.5-flash_maxiter_250_N_v0.43.0-no-hint-raw_agent-verified_50-run_1': {},
    'gemini-2.5-flash_maxiter_250_N_v0.43.0-no-hint-observation_masking-M_10-verified_50-observation_masking_for_eval-run_1': {
        'xy': (1.3, 29),
        'text': 'M=10',
    },
    'gemini-2.5-flash_maxiter_250_N_v0.43.0-no-hint-observation_masking-M_58-verified_50-observation_masking_for_eval-run_1': {
        'xy': (1.4, 43),
        'text': 'M=58',
    }
}

# Optional: filter which OpenHands experiments to include in the plot. If empty, include all.
# Match by substring anywhere in the experiment name.
OPENHANDS_EXPERIMENT_FILTER: list[str] = [k for k in OPENHANDS_ANNOTATIONS.keys()]

def _find_openhands_annotation_config(experiment_name: str) -> dict[str, _Any] | None:
    for key, cfg in OPENHANDS_ANNOTATIONS.items():
        if key in experiment_name:
            return cfg
    return None

# Create single scatter plot
fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.5), constrained_layout=False)

"""Build iterable of (experiment_name, data) honoring optional filter."""
from typing import Any as _PAny
if OPENHANDS_EXPERIMENT_FILTER:
    iter_items: list[tuple[str, dict[str, _PAny]]] = [
        (exp_name, data)
        for exp_name, data in openhands_data.items()
        if any(frag in exp_name for frag in OPENHANDS_EXPERIMENT_FILTER)
    ]
else:
    iter_items = list(openhands_data.items())

# Collect OpenHands data for axis limit calculations (post-filter)
openhands_costs: list[float] = []
openhands_solve_rates: list[float] = []

# Plot OpenHands data points
for experiment_name, data in iter_items:
    instance_cost = (data['instance_cost'] + data['summary_instance_cost'])
    solve_rate = data['solve_rate'] * 100  # Convert to percentage to match main figure
    strategy_label, color, marker = get_strategy_info_openhands(experiment_name)
    
    openhands_costs.append(instance_cost)
    openhands_solve_rates.append(solve_rate)
    
    if strategy_label:
        ax.scatter(instance_cost, solve_rate, c=color, marker=marker, s=80,
                   alpha=0.9, edgecolors='black', linewidth=0.5, zorder=5)
        # Optional annotations per experiment
        ann_cfg = _find_openhands_annotation_config(experiment_name)
        if ann_cfg is not None:
            xytext = tuple(ann_cfg.get('xy', (instance_cost, solve_rate)))
            text = str(ann_cfg.get('text', ''))
            if text:
                ax.annotate(
                    text,
                    xy=(instance_cost, solve_rate),
                    xytext=xytext,
                    textcoords='data',
                    fontsize=8,
                    ha='left', va='bottom',
                    color=color,
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor=color, linewidth=0.8),
                    arrowprops=dict(arrowstyle='-', color=color, lw=0.8),
                    zorder=10,
                )

# Calculate axis limits to match main figure with OpenHands data considerations
# Use the same y-axis limits as the main figure
ax.set_ylim(global_ylim)

# For x-axis: use same minimum as main figure, but extend max if OpenHands data requires it
openhands_xmax = max(openhands_costs) if openhands_costs else 0
main_fig_xmax = global_xlim[1]
x_margin = (global_xlim[1] - global_xlim[0]) * 0.1 if global_xlim[1] > global_xlim[0] else 0.1

if openhands_xmax > main_fig_xmax:
    # Extend x-axis to accommodate OpenHands data
    extended_xlim = (global_xlim[0], openhands_xmax + x_margin)
    ax.set_xlim(extended_xlim)
else:
    # Use the same x-axis limits as main figure
    ax.set_xlim(global_xlim)

ax.set_title('OpenHands Results (Gemini 2.5 Flash)', fontweight='bold', pad=10)
ax.set_xlabel('Instance Cost (USD)', fontweight='bold', fontsize=12)
ax.set_ylabel('Solve Rate (%)', fontweight='bold', fontsize=12)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_axisbelow(True)
ax.tick_params(direction='in', length=3, width=0.5)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.1f}'))

# Create legend with same style as main figure
legend_elements = []
for label, _, color, marker in strategies_openhands:
    legend_elements.append(Line2D([0], [0], marker=marker, color='w',
                                  markerfacecolor=color, markersize=8,
                                  label=label, linestyle='None',
                                  markeredgecolor='black', markeredgewidth=0.5))

ax.legend(handles=legend_elements, loc='best', frameon=False)

plt.subplots_adjust(left=0.12, bottom=0.12, right=0.95, top=0.90)
plt.show()

# Save the plot
Path("report_data").mkdir(parents=True, exist_ok=True)
fig.savefig('report_data/openhands_scatter_plot.pdf', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
fig.savefig('report_data/openhands_scatter_plot.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')

print("OpenHands scatter plot saved as 'openhands_scatter_plot.pdf' and 'openhands_scatter_plot.png'")

# %%
# Create a summary table for the paper
import pandas as pd

# Prepare data for the table
table_data = []

for group_name, group_experiments in experiment_groups.items():
    for experiment in group_experiments:
        cost = experiment_results[experiment]['cost']
        solve_rate = experiment_results[experiment]['solve_rate']
        strategy_label, _, _ = get_strategy_info(experiment)
        denom = _infer_experiment_denominator(experiment)
        
        table_data.append({
            'Model': group_name.replace('\n', ' '),
            'Strategy': strategy_label,
            'Cost per Instance (USD)': f"${cost / denom:.4f}",
            'Total Cost (USD)': f"${cost:.2f}",
            'Solve Rate (%)': f"{solve_rate:.1f}%"
        })

# Create DataFrame and display
summary_df = pd.DataFrame(table_data)
summary_df = summary_df.sort_values(['Model', 'Strategy'])

print("Summary Table for Paper:")
print("=" * 80)
print(summary_df.to_string(index=False))

# Calculate efficiency metrics
print("\n\nEfficiency Analysis:")
print("=" * 50)

for group_name, group_experiments in experiment_groups.items():
    print(f"\n{group_name.replace(chr(10), ' ')}:")
    
    group_data = {}
    for experiment in group_experiments:
        cost = experiment_results[experiment]['cost']
        solve_rate = experiment_results[experiment]['solve_rate']
        strategy_label, _, _ = get_strategy_info(experiment)
        
        if strategy_label:
            group_data[strategy_label] = {
                'cost': cost,
                'solve_rate': solve_rate,
                'efficiency': solve_rate / cost if cost > 0 else 0
            }
    
    # Find the most efficient strategy (highest solve rate per dollar)
    if group_data:
        best_strategy = max(group_data.items(), key=lambda x: x[1]['efficiency'])
        print(f"  Most efficient: {best_strategy[0]} ({best_strategy[1]['efficiency']:.3f} solved instances per $)")
        
        # Compare to raw baseline if available
        if 'Raw Agent' in group_data:
            raw_efficiency = group_data['Raw Agent']['efficiency']
            for strategy, data in group_data.items():
                if strategy != 'Raw Agent':
                    improvement = ((data['efficiency'] - raw_efficiency) / raw_efficiency) * 100
                    print(f"  {strategy} vs Raw: {improvement:+.1f}% efficiency change")

# Export table to CSV for LaTeX/paper use
#summary_df.to_csv('experiment_results_summary.csv', index=False)
print(f"\nTable exported to 'experiment_results_summary.csv'")


# %% [markdown]
# # Trajectory elongation and summary costs

# %%
# Collect per-instance costs and turns across all experiments (embarrassingly parallel per experiment)
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

def _collect_instance_metrics_for_experiment(experiment_name: str, base_directory: str) -> list[tuple[str, str, float, int, float]]:
    """Return list of (experiment, instance_id, instance_cost, turns, summary_costs) for a single experiment."""
    experiment_dir = Path(base_directory) / experiment_name
    traj_files = list(experiment_dir.rglob("*.traj"))

    results: list[tuple[str, str, float, int, float]] = []
    for traj_file in traj_files:
        try:
            trajectory_data = json.loads(Path(traj_file).read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            continue

        instance_cost = float(trajectory_data.get("info", {}).get("model_stats", {}).get("instance_cost", 0.0))
        turns = int(len(trajectory_data.get("trajectory", [])))
        summary_cost_total: float = 0.0
        summaries = trajectory_data.get("summaries", None)
        if summaries is not None:
            try:
                for summary in summaries:
                    summary_cost_total += float(summary.get("statistics", {}).get("cost", 0.0))
            except Exception:
                summary_cost_total = 0.0
        instance_id = Path(traj_file).stem
        results.append((experiment_name, instance_id, instance_cost, turns, summary_cost_total))

    return results


def _flatten(list_of_lists: Iterable[list[tuple[str, str, float, int, float]]]) -> list[tuple[str, str, float, int, float]]:
    flat: list[tuple[str, str, float, int, float]] = []
    for chunk in list_of_lists:
        flat.extend(chunk)
    return flat

per_experiment_futures: dict = {}
with ThreadPoolExecutor(max_workers=15) as executor:
    per_experiment_futures = {
        executor.submit(_collect_instance_metrics_for_experiment, experiment, base_dir): experiment
        for experiment in experiments
    }

collected: list[list[tuple[str, str, float, int, float]]] = []
for future in as_completed(per_experiment_futures):
    try:
        collected.append(future.result())
    except Exception:
        continue

flat_records = _flatten(collected)

# Build the MultiIndex DataFrame: index=(experiment, instance_id); columns=[instance_cost, turns, summary_costs]
index_tuples: list[tuple[str, str]] = []
rows: list[dict[str, float | int]] = []
for experiment_name, instance_id, instance_cost, turns, summary_costs in flat_records:
    index_tuples.append((experiment_name, instance_id))
    rows.append({"instance_cost": instance_cost, "turns": turns, "summary_costs": summary_costs})

import pandas as pd  # Safe to re-import in notebook context
per_instance_df = pd.DataFrame(
    rows,
    index=pd.MultiIndex.from_tuples(index_tuples, names=["experiment", "instance_id"]),
).sort_index()

print(f"Per-instance metrics collected: {per_instance_df.shape[0]} rows across {len(experiments)} experiments.")


# %%
per_instance_df.index.get_level_values("experiment").unique()

# %% [markdown]
# ## Summary Costs

# %%
mask = per_instance_df.index.get_level_values("experiment").str.contains("Qwen3-32B_thinking") \
        & per_instance_df.index.get_level_values("experiment").str.contains("turn-summaries")
per_instance_df.loc[mask, :].describe()

# %%
mask = per_instance_df.index.get_level_values("experiment").str.contains("grazie-gemini_2.5_flash-agent") \
        & per_instance_df.index.get_level_values("experiment").str.contains("turn-summaries")
per_instance_df.loc[mask, :].describe()

# %%
mask = per_instance_df.index.get_level_values("experiment").str.contains("grazie-gemini_2.5_flash_thinking-agent") \
        & per_instance_df.index.get_level_values("experiment").str.contains("turn-summaries")
per_instance_df.loc[mask, :].describe()

# %%
mask = per_instance_df.index.get_level_values("experiment").str.contains("local-qwen3-32b-") \
        & per_instance_df.index.get_level_values("experiment").str.contains("turn-summaries")
per_instance_df.loc[mask, :].describe()

# %%
mask = per_instance_df.index.get_level_values("experiment").str.contains("Qwen3-Coder_480B_A35B") \
        & per_instance_df.index.get_level_values("experiment").str.contains("turn-summaries")
per_instance_df.loc[mask, :].describe()

# %% [markdown]
# ## Trajectory Elongation Boxplots

# %%
mask = per_instance_df.index.get_level_values("experiment").str.contains("qwen3-32b") \
        & per_instance_df.index.get_level_values("experiment").str.contains("turn-summaries")
per_instance_df.loc[mask, :].describe()

# %%
mask = per_instance_df.index.get_level_values("experiment").str.contains("qwen3-32b") \
        & per_instance_df.index.get_level_values("experiment").str.contains("N_1")
per_instance_df.loc[mask, :].describe()

# %%
mask = per_instance_df.index.get_level_values("experiment").str.contains("qwen3-32b") \
        & per_instance_df.index.get_level_values("experiment").str.contains("raw")
per_instance_df.loc[mask, :].describe()

# %%
mask = per_instance_df.index.get_level_values("experiment").str.contains("Coder") \
        & per_instance_df.index.get_level_values("experiment").str.contains("turn-summaries")
per_instance_df.loc[mask, :].describe()

# %%
mask = per_instance_df.index.get_level_values("experiment").str.contains("Coder") \
        & per_instance_df.index.get_level_values("experiment").str.contains("N_1")
per_instance_df.loc[mask, :].describe()

# %%
mask = per_instance_df.index.get_level_values("experiment").str.contains("Coder") \
        & per_instance_df.index.get_level_values("experiment").str.contains("raw")
per_instance_df.loc[mask, :].describe()

# %%
mask = per_instance_df.index.get_level_values("experiment").str.contains("Qwen3-32B_thinking") \
        & per_instance_df.index.get_level_values("experiment").str.contains("turn-summaries")
per_instance_df.loc[mask, :].describe()

# %%
mask = per_instance_df.index.get_level_values("experiment").str.contains("Qwen3-32B_thinking") \
        & per_instance_df.index.get_level_values("experiment").str.contains("N_1")
per_instance_df.loc[mask, :].describe()

# %%
mask = per_instance_df.index.get_level_values("experiment").str.contains("Qwen3-32B_thinking") \
        & per_instance_df.index.get_level_values("experiment").str.contains("raw")
per_instance_df.loc[mask, :].describe()

# %%
mask = per_instance_df.index.get_level_values("experiment").str.contains("gemini_2.5_flash-agent-t_0.8") \
        & per_instance_df.index.get_level_values("experiment").str.contains("turn-summaries")
per_instance_df.loc[mask, :].describe()

# %%
mask = per_instance_df.index.get_level_values("experiment").str.contains("gemini_2.5_flash-agent-t_0.8") \
        & per_instance_df.index.get_level_values("experiment").str.contains("N_1")
per_instance_df.loc[mask, :].describe()

# %%
mask = per_instance_df.index.get_level_values("experiment").str.contains("gemini_2.5_flash-agent-t_0.8") \
        & per_instance_df.index.get_level_values("experiment").str.contains("raw")
per_instance_df.loc[mask, :].describe()

# %%
per_instance_df.info()

# %%
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Two-panel boxplot: Turns per trajectory
# Panel (a): Qwen3-32B (thinking) — Raw Agent vs Observation Masking vs LLM-Summary 
# Panel (b): Gemini 2.5 Flash — Raw Agent vs Observation Masking vs LLM-Summary

qwen_experiments_map: dict[str, str] = {
    "Raw Agent": "local-Qwen3-Coder_480B_A35B_Instruct_FP8-baseline_raw.0-verified-500",
    "Observation Masking": "local-Qwen3-Coder_480B_A35B_Instruct_FP8-baseline_N_1_M_10.0-verified-500",
    "LLM-Summary": "local-Qwen3-Coder_480B_A35B_Instruct_FP8-t_0.8-turn-summaries-t_0-N_21_M_10_openhands.0-verified-500",
}

gemini_experiments_map: dict[str, str] = {
    "Raw Agent": "gemini_2.5_flash-agent-t_0.8-baseline_raw.0-verified-500",
    "Observation Masking": "gemini_2.5_flash-agent-t_0.8-baseline_N_1_M_10.0-verified-500",
    "LLM-Summary": "gemini_2.5_flash-agent-t_0.8-turn-summaries-t_0-N_21_M_10_openhands.0-verified-500",
}

strategy_color_map: dict[str, str] = {s[0]: s[2] for s in strategies}

def _collect_turns_for_map(
    mapping: dict[str, str], experiment_keys: list[str]
) -> tuple[list[np.ndarray], list[str], list[str]]:
    data: list[np.ndarray] = []
    labels: list[str] = []
    colors: list[str] = []
    available_experiments = set(per_instance_df.index.get_level_values("experiment").unique())

    for label in experiment_keys:
        exp_name = mapping.get(label)
        if exp_name is None:
            continue
        if exp_name not in available_experiments:
            print(f"Warning: Experiment '{exp_name}' not found in per_instance_df. Skipping '{label}'.")
            continue
        turns_series = per_instance_df.xs(exp_name, level="experiment")["turns"]
        data.append(np.asarray(turns_series.to_numpy()))
        labels.append(label)
        colors.append(strategy_color_map.get(label, "#666666"))
    return data, labels, colors

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=False)

# Panel (a)
ax = axes[0]
data_a, labels_a, colors_a = _collect_turns_for_map(
    qwen_experiments_map, ["Raw Agent", "Observation Masking", "LLM-Summary"]
)
positions_a = np.arange(1, len(data_a) + 1) * 2  # Increase spacing between boxes
bplot_a = ax.boxplot(
    data_a,
    positions=positions_a,
    patch_artist=True,
    showmeans=True,
    widths=0.7,
    meanprops={"marker": "*", "markerfacecolor": "black", "markeredgecolor": "white", "markersize": 6},
    boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 0.8},
    whiskerprops={"color": "black", "linewidth": 0.8},
    capprops={"color": "black", "linewidth": 0.8},
    medianprops={"color": "black", "linewidth": 1.0},
    flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "#666666", "markeredgecolor": "#666666", "alpha": 0.6},
)
for patch, facecolor in zip(bplot_a["boxes"], colors_a):
    patch.set_facecolor(facecolor)
    patch.set_alpha(0.6)
ax.set_title("(a) Qwen3-Coder 480B", fontweight="bold", pad=8)
ax.set_ylabel("Turns per trajectory")
ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
ax.set_axisbelow(True)
ax.tick_params(direction="in", length=3, width=0.5)
ax.set_xticks(positions_a)
ax.set_xticklabels(labels_a, rotation=15, ha="right", rotation_mode="anchor")

# Panel (b)
ax = axes[1]
data_b, labels_b, colors_b = _collect_turns_for_map(
    gemini_experiments_map, ["Raw Agent", "Observation Masking", "LLM-Summary"]
)
positions_b = np.arange(1, len(data_b) + 1) * 2  # Increase spacing between boxes
bplot_b = ax.boxplot(
    data_b,
    positions=positions_b,
    patch_artist=True,
    showmeans=True,
    widths=0.7,
    meanprops={"marker": "*", "markerfacecolor": "black", "markeredgecolor": "white", "markersize": 6},
    boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 0.8},
    whiskerprops={"color": "black", "linewidth": 0.8},
    capprops={"color": "black", "linewidth": 0.8},
    medianprops={"color": "black", "linewidth": 1.0},
    flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "#666666", "markeredgecolor": "#666666", "alpha": 0.6},
)
for patch, facecolor in zip(bplot_b["boxes"], colors_b):
    patch.set_facecolor(facecolor)
    patch.set_alpha(0.6)
ax.set_title("(b) Gemini 2.5 Flash", fontweight="bold", pad=8)
ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
ax.set_axisbelow(True)
ax.tick_params(direction="in", length=3, width=0.5)
ax.set_xticks(positions_b)
ax.set_xticklabels(labels_b, rotation=15, ha="right", rotation_mode="anchor")

plt.subplots_adjust(left=0.1, right=0.98, top=0.88, bottom=0.2, wspace=0.25)
plt.show()

Path("report_data").mkdir(parents=True, exist_ok=True)
fig.savefig("report_data/trajectory_lengths_boxplots.pdf", dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
fig.savefig("report_data/trajectory_lengths_boxplots.png", dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
print("Figure saved as 'trajectory_lengths_boxplots.pdf' and 'trajectory_lengths_boxplots.png'")

# %% [markdown]
# # Bootstrapped CIs

# %% [markdown]
# This section computes nonparametric bootstrapped confidence intervals (CIs) for
# per-experiment solve rate and mean instance cost. For each experiment folder,
# we build a per-instance dataset with fields: outcome (is the instance_id in the
# resolved_ids of the SWE-bench results for that experiment?) and cost (the
# instance_cost recorded in the trajectory JSON). We then apply simple bootstrap
# resampling (sampling n instances with replacement, B times) and compute the
# empirical percentile interval [2.5%, 97.5%].
#
# Intuition: Bootstrapping simulates “re-running” the experiment by resampling
# instances. This gives a distribution over the metric (solve rate, mean cost)
# without assuming Normality. The 95% CI is the central interval of that
# distribution, indicating plausible values for the true metric given the data.

# %%
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Callable, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

BOOTSTRAP_CI_PERSISTENCE_FILE = "auxiliary-data/bootstrapped_cis_openhands.csv"
INPUT_FILE = "auxiliary-data/experiment_instance_costs_openhands.csv"

def bootstrap_ci(
    metric_fn: Callable[[np.ndarray], float],
    n: int,
    B: int = 10_000,
    alpha: float = 0.05,
    seed: int | None = 0,
) -> tuple[float, float]:
    """Compute a nonparametric bootstrap percentile CI for a statistic.

    metric_fn receives an index array of length n (sampled with replacement)
    and returns the statistic for that resample. We repeat B times and take the
    empirical (alpha/2, 1 - alpha/2) quantiles.
    """
    rng = np.random.default_rng(seed)
    stats = np.empty(B, dtype=float)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        stats[b] = float(metric_fn(idx))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def _collect_outcome_cost_for_experiment(
    experiment_name: str, base_directory: str
) -> list[tuple[str, str, float, float]]:
    """Return list of (experiment, instance_id, outcome, cost) for one experiment.

    outcome is 1.0 if the instance_id (parent dir of the .traj file) is in the
    resolved_ids for that experiment, else 0.0.
    cost is the per-instance cost recorded in the trajectory JSON.
    """
    results = load_swe_bench_results(experiment_name)
    if results is None:
        return []

    resolved_ids: set[str] = set(results.get("resolved_ids", []))
    experiment_dir = Path(base_directory) / experiment_name
    traj_files = list(experiment_dir.rglob("*.traj"))

    rows: list[tuple[str, str, float, float]] = []
    for traj_file in traj_files:
        try:
            trajectory_data = json.loads(Path(traj_file).read_text())
        except Exception:
            continue

        instance_id = Path(traj_file).parent.name  # parent folder of the .traj
        instance_cost = float(
            trajectory_data.get("info", {})
            .get("model_stats", {})
            .get("instance_cost", 0.0)
        )
        outcome = 1.0 if instance_id in resolved_ids else 0.0
        rows.append((experiment_name, instance_id, outcome, instance_cost))

    return rows


# Only load/build per-instance data if INPUT_FILE doesn't exist or is empty
per_instance_boot_df = pd.DataFrame()
if Path(INPUT_FILE).exists():
    try:
        per_instance_boot_df = pd.read_csv(INPUT_FILE)
    except Exception:
        per_instance_boot_df = pd.DataFrame()

if per_instance_boot_df.empty:
    print("No per-instance data found. Building from scratch based on SWE-agent main experiments.")

    # Build per-instance dataset across all experiments (concurrently)
    per_instance_records: list[list[tuple[str, str, float, float]]] = []
    with ThreadPoolExecutor(max_workers=15) as executor:
        future_map = {
            executor.submit(_collect_outcome_cost_for_experiment, experiment, base_dir): experiment
            for experiment in experiments
        }
        for future in as_completed(future_map):
            try:
                per_instance_records.append(future.result())
            except Exception:
                continue

    # Flatten and create DataFrame
    flat_rows: list[tuple[str, str, float, float]] = []
    for chunk in per_instance_records:
        flat_rows.extend(chunk)

    per_instance_boot_df = pd.DataFrame(
        flat_rows, columns=["experiment", "instance_id", "outcome", "cost"]
    )

    print(
        f"Collected {len(per_instance_boot_df)} per-instance rows across "
        f"{per_instance_boot_df['experiment'].nunique()} experiments for bootstrapping."
    )

# Compute point estimates and bootstrap CIs per experiment
summary_rows: list[dict[str, Any]] = []
B = 10_000
ALPHA = 0.05

for exp_name, grp in per_instance_boot_df.groupby("experiment"):
    y = grp["outcome"].to_numpy(dtype=float)
    c = grp["cost"].to_numpy(dtype=float)
    n = int(y.shape[0])
    if n == 0:
        continue

    solve_rate = float(y.mean())
    mean_cost = float(c.mean())

    solve_rate_lo, solve_rate_hi = bootstrap_ci(
        lambda idx, arr=y: float(arr[idx].mean()), n=n, B=B, alpha=ALPHA, seed=0
    )
    mean_cost_lo, mean_cost_hi = bootstrap_ci(
        lambda idx, arr=c: float(arr[idx].mean()), n=n, B=B, alpha=ALPHA, seed=1
    )

    summary_rows.append(
        {
            "experiment": exp_name,
            "n_instances": n,
            "solve_rate": solve_rate,
            "solve_rate_lo": solve_rate_lo,
            "solve_rate_hi": solve_rate_hi,
            "solve_rate_lo_delta": solve_rate_lo - solve_rate,
            "solve_rate_hi_delta": solve_rate_hi - solve_rate,
            "mean_cost": mean_cost,
            "mean_cost_lo": mean_cost_lo,
            "mean_cost_hi": mean_cost_hi,
            "mean_cost_lo_delta": mean_cost_lo - mean_cost,
            "mean_cost_hi_delta": mean_cost_hi - mean_cost,
        }
    )

boot_summary_df = pd.DataFrame(summary_rows).sort_values("experiment")

# Pretty-print and persist results
print("\nBootstrapped CIs (solve rate as %, costs in USD):")
display_df = boot_summary_df.copy()
display_df["solve_rate"] = (display_df["solve_rate"] * 100).round(2)
display_df["solve_rate_lo"] = (display_df["solve_rate_lo"] * 100).round(2)
display_df["solve_rate_hi"] = (display_df["solve_rate_hi"] * 100).round(2)
display_df["solve_rate_lo_delta"] = (display_df["solve_rate_lo_delta"] * 100).round(2)
display_df["solve_rate_hi_delta"] = (display_df["solve_rate_hi_delta"] * 100).round(2)
display_df["mean_cost"] = display_df["mean_cost"].round(4)
display_df["mean_cost_lo"] = display_df["mean_cost_lo"].round(4)
display_df["mean_cost_hi"] = display_df["mean_cost_hi"].round(4)
display_df["mean_cost_lo_delta"] = display_df["mean_cost_lo_delta"].round(4)
display_df["mean_cost_hi_delta"] = display_df["mean_cost_hi_delta"].round(4)
print(display_df.to_string(index=False))

Path("report_data").mkdir(parents=True, exist_ok=True)
boot_summary_df.to_csv(BOOTSTRAP_CI_PERSISTENCE_FILE, index=False)
print(f"\nSaved bootstrapped CI summary to '{BOOTSTRAP_CI_PERSISTENCE_FILE}'")

# %% [markdown]
# # Paired bootstrap CI for differences vs Raw
#
# Computes paired bootstrap 95% CIs for the difference in solve rate and mean
# instance cost between each non-Raw strategy and the Raw baseline, per model
# group. We load the list of experiments from the saved summary CSV to avoid
# re-running earlier aggregation, but we must read per-instance data for pairing.

# %%
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

PAIRED_BOOTSTRAP_CI_PERSISTENCE_FILE = "auxiliary-data/paired_ci_diffs_vs_raw_openhands.csv"

def paired_bootstrap_ci_diff_mean(
    a: np.ndarray,
    b: np.ndarray,
    B: int = 10_000,
    alpha: float = 0.05,
    seed: int | None = 0,
) -> tuple[float, float, float, float]:
    """Return (diff, lo, hi, p) for mean(a) - mean(b) via paired bootstrap."""
    assert a.shape == b.shape
    n = int(a.shape[0])
    rng = np.random.default_rng(seed)
    stats = np.empty(B, dtype=float)
    for i in range(B):
        idx = rng.integers(0, n, size=n)
        stats[i] = float(a[idx].mean() - b[idx].mean())
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    diff = float(a.mean() - b.mean())
    p = 2 * min(float(np.mean(stats >= 0.0)), float(np.mean(stats <= 0.0)))
    return diff, float(lo), float(hi), p


def _align_arrays(
    df_a: pd.DataFrame, df_b: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    merged = df_a[["instance_id", "outcome", "cost"]].merge(
        df_b[["instance_id", "outcome", "cost"]], on="instance_id", suffixes=("_a", "_b")
    )
    y_a = merged["outcome_a"].to_numpy(dtype=float)
    y_b = merged["outcome_b"].to_numpy(dtype=float)
    c_a = merged["cost_a"].to_numpy(dtype=float)
    c_b = merged["cost_b"].to_numpy(dtype=float)
    return y_a, y_b, c_a, c_b


# Load the saved bootstrapped summary to get the experiment list
summary_path = Path(BOOTSTRAP_CI_PERSISTENCE_FILE)
summary_loaded = pd.read_csv(summary_path)
available_experiments = set(summary_loaded["experiment"].astype(str).tolist())


def _extract_strategy_from_name(experiment_name: str) -> str | None:
    """Extract strategy from experiment name using specified rules."""
    exp_lower = experiment_name.lower()
    
    if 'raw' in exp_lower:
        return 'Raw Agent'
    elif 'summary' in exp_lower or 'summaries' in exp_lower:
        return 'LLM-Summary'
    elif ('baseline' in exp_lower and 'raw' not in exp_lower) or 'observation_masking' in exp_lower:
        return 'Observation Masking'
    
    return None


def _extract_model_group_from_name(experiment_name: str) -> str | None:
    """Extract model group from experiment name."""
    exp_lower = experiment_name.lower()
    
    if 'gemini' in exp_lower and '2.5' in exp_lower and 'flash' in exp_lower:
        if 'thinking' in exp_lower:
            return 'Gemini 2.5 Flash (Thinking)'
        else:
            return 'Gemini 2.5 Flash'
    elif 'qwen3' in exp_lower and 'coder' in exp_lower:
        return 'Qwen3-Coder 480B'
    elif 'qwen3' in exp_lower and '32b' in exp_lower:
        if 'thinking' in exp_lower:
            return 'Qwen3-32B (Thinking)'
        else:
            return 'Qwen3-32B'
    
    return None


def _build_needed_experiments() -> dict[str, dict[str, str]]:
    """Map group -> {strategy_label -> experiment_name} for available experiments."""
    mapping: dict[str, dict[str, str]] = {}
    
    # Group experiments by model group and strategy
    for exp_name in available_experiments:
        strategy = _extract_strategy_from_name(exp_name)
        model_group = _extract_model_group_from_name(exp_name)
        
        if strategy is None or model_group is None:
            continue
            
        if model_group not in mapping:
            mapping[model_group] = {}
            
        mapping[model_group][strategy] = exp_name
    
    # Only keep groups that have Raw Agent and at least one other strategy
    filtered_mapping = {}
    for group_key, strategies in mapping.items():
        if "Raw Agent" in strategies and ("Observation Masking" in strategies or "LLM-Summary" in strategies):
            filtered_mapping[group_key] = strategies
    
    return filtered_mapping


group_to_strategy_map = _build_needed_experiments()
unique_needed_experiments: set[str] = set()
for choices in group_to_strategy_map.values():
    unique_needed_experiments.update(choices.values())


# Load per-instance data just for needed experiments (try INPUT_FILE first, then concurrent collection)
exp_to_df: dict[str, pd.DataFrame] = {}

# Try to load from INPUT_FILE first
paired_per_instance_df = pd.DataFrame()
if Path(INPUT_FILE).exists():
    try:
        paired_per_instance_df = pd.read_csv(INPUT_FILE)
    except Exception:
        paired_per_instance_df = pd.DataFrame()

if not paired_per_instance_df.empty:
    print(f"Loading per-instance data from {INPUT_FILE}")
    # Group by experiment and filter to only needed experiments
    for exp_name in unique_needed_experiments:
        exp_data = paired_per_instance_df[paired_per_instance_df['experiment'] == exp_name]
        if not exp_data.empty:
            exp_to_df[exp_name] = exp_data[["experiment", "instance_id", "outcome", "cost"]].sort_values("instance_id")
else:
    print("No cached per-instance data found. Collecting from scratch (concurrent)...")
    # Fall back to concurrent data collection
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {
            executor.submit(_collect_outcome_cost_for_experiment, exp, base_dir): exp
            for exp in sorted(unique_needed_experiments)
        }
        for f in as_completed(futures):
            exp_name = futures[f]
            try:
                rows = f.result()
            except Exception:
                rows = []
            df = pd.DataFrame(rows, columns=["experiment", "instance_id", "outcome", "cost"]).sort_values(
                "instance_id"
            )
            exp_to_df[exp_name] = df

print(f"Collected data for {len(exp_to_df)} experiments")

# Compute paired diffs vs Raw per group and strategy
diff_rows: list[dict[str, Any]] = []
for group_key, choices in group_to_strategy_map.items():
    raw_exp = choices.get("Raw Agent")
    if raw_exp is None:
        continue
    raw_df = exp_to_df.get(raw_exp)
    if raw_df is None or raw_df.empty:
        continue

    for other_label in ["Observation Masking", "LLM-Summary"]:
        other_exp = choices.get(other_label)
        if other_exp is None:
            continue
        other_df = exp_to_df.get(other_exp)
        if other_df is None or other_df.empty:
            continue

        y_a, y_b, c_a, c_b = _align_arrays(other_df, raw_df)
        if y_a.size == 0:
            continue

        sr_diff, sr_lo, sr_hi, sr_p = paired_bootstrap_ci_diff_mean(
            y_a, y_b, B=10_000, alpha=0.05, seed=0
        )
        mc_diff, mc_lo, mc_hi, mc_p = paired_bootstrap_ci_diff_mean(
            c_a, c_b, B=10_000, alpha=0.05, seed=1
        )

        diff_rows.append(
            {
                "group": group_key,
                "strategy": other_label,
                "experiment_other": other_exp,
                "experiment_raw": raw_exp,
                "n_common": int(y_a.size),
                "solve_rate_diff": float(sr_diff),
                "solve_rate_lo": float(sr_lo),
                "solve_rate_hi": float(sr_hi),
                "solve_rate_p": float(sr_p),
                "mean_cost_diff": float(mc_diff),
                "mean_cost_lo": float(mc_lo),
                "mean_cost_hi": float(mc_hi),
                "mean_cost_p": float(mc_p),
            }
        )

ci_diff_df = pd.DataFrame(diff_rows).sort_values(["group", "strategy"])\
    .reset_index(drop=True)

print("\nPaired bootstrap CI for differences vs Raw (solve rate and mean cost):")
pretty = ci_diff_df.copy()
pretty["solve_rate_diff_pct"] = (pretty["solve_rate_diff"] * 100).round(2)
pretty["solve_rate_lo_pct"] = (pretty["solve_rate_lo"] * 100).round(2)
pretty["solve_rate_hi_pct"] = (pretty["solve_rate_hi"] * 100).round(2)
pretty["mean_cost_diff"] = pretty["mean_cost_diff"].round(4)
pretty["mean_cost_lo"] = pretty["mean_cost_lo"].round(4)
pretty["mean_cost_hi"] = pretty["mean_cost_hi"].round(4)
cols = [
    "group",
    "strategy",
    "n_common",
    "solve_rate_diff_pct",
    "solve_rate_lo_pct",
    "solve_rate_hi_pct",
    "solve_rate_p",
    "mean_cost_diff",
    "mean_cost_lo",
    "mean_cost_hi",
    "mean_cost_p",
]
print(pretty[cols].to_string(index=False))

Path("report_data").mkdir(parents=True, exist_ok=True)
ci_diff_df.to_csv(PAIRED_BOOTSTRAP_CI_PERSISTENCE_FILE, index=False)
print(f"\nSaved paired CI diffs to '{PAIRED_BOOTSTRAP_CI_PERSISTENCE_FILE}'")


# %%

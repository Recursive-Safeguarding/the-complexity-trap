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

# %%
os.chdir("/path/to/project/root") # TODO
os.getcwd()

# %% [markdown]
# # Construct SWE-bench Lite 50 without any Verified samples

# %%
import pandas as pd

# %%
splits = {'dev': 'data/dev-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
swe_bench_lite_dev = pd.read_parquet("hf://datasets/princeton-nlp/SWE-bench_Lite/" + splits["dev"])
swe_bench_lite_test = pd.read_parquet("hf://datasets/princeton-nlp/SWE-bench_Lite/" + splits["test"])

swe_bench_verified = pd.read_parquet("hf://datasets/princeton-nlp/SWE-bench_Verified/data/test-00000-of-00001.parquet")

splits = {'dev': 'data/dev-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet', 'train': 'data/train-00000-of-00001.parquet'}
swe_bench = pd.read_parquet("hf://datasets/princeton-nlp/SWE-bench/" + splits["test"])

# %%
swe_bench_lite_filter = open('config/dataset_filters/lite-50.txt','r').read().split('|')

# %%
in_lite_dev_not_in_verified = swe_bench_lite_dev[~swe_bench_lite_dev['instance_id'].isin(swe_bench_verified['instance_id'])]

# %%
len(in_lite_dev_not_in_verified)

# %%
in_lite_test_not_in_verified = swe_bench_lite_test[~swe_bench_lite_test['instance_id'].isin(swe_bench_verified['instance_id'])]

# %%
len(in_lite_test_not_in_verified)

# %%
valid_from_50 = in_lite_test_not_in_verified[in_lite_test_not_in_verified['instance_id'].isin(swe_bench_lite_filter)]

# %%
valid_from_50.info()

# %%
valid_from_50_dev = in_lite_dev_not_in_verified[in_lite_dev_not_in_verified['instance_id'].isin(swe_bench_lite_filter)]

# %%
valid_from_50_dev.info()

# %%
new_lite_50 = in_lite_test_not_in_verified['instance_id'].sample(n=50)

# %%
len(new_lite_50)

# %%
new_lite_50_str = '|'.join(new_lite_50)


# %%
new_lite_50_str

# %%
with open('config/dataset_filters/lite-50.txt','w') as f:
    f.write(new_lite_50_str)

# %% [markdown]
# ## Create Verified-150

# %%
swe_bench_verified = pd.read_parquet("hf://datasets/princeton-nlp/SWE-bench_Verified/data/test-00000-of-00001.parquet")
swe_bench_verified_filter = open('config/dataset_filters/verified-50.txt','r').read().split('|')

# %%
unused_instances = swe_bench_verified[~swe_bench_verified['instance_id'].isin(swe_bench_verified_filter)]
len(unused_instances)

# %%
swe_bench_verified_filter = list(swe_bench_verified_filter) + unused_instances.sample(100)['instance_id'].tolist()
len(swe_bench_verified_filter)

# %%
verified_150_str = '|'.join(swe_bench_verified_filter)
with open('config/dataset_filters/verified-150.txt','w') as f:
    f.write(verified_150_str)

# %% [markdown]
# # Eval Hyperparameter N 

# %% [markdown]
# ## Overall system results

# %%
import glob
import json
import pandas as pd

# Get all JSON files in the current directory
base_path = 'auxiliary-data/swe-agent-eval-results'
json_files = glob.glob(os.path.join(base_path, '*gpt4.1*.json')) \
    + glob.glob(os.path.join(base_path, '*gemini_2.5_flash*.json')) \
    + glob.glob(os.path.join(base_path, '*qwen*.json')) \
    + glob.glob(os.path.join(base_path, '*Qwen*.json'))

# Create a dictionary to store dataframes
results = {}

# Load each JSON file into a dataframe
for json_file in json_files:
    config = json_file.split('-')[-3]
    model = ''
    if 'gpt' in json_file:
        model = '-'.join(json_file.split('grazie-')[1].split('-')[:2]) if 'mini' in json_file else '-'.join(json_file.split('grazie-')[1].split('-')[:1])
    elif 'gemini' in json_file:
        experiment = json_file.split('/')[-1]
        if experiment.startswith('_'):
            model = experiment.split('_.')[1].split('-')[0]
        elif experiment.startswith('gemini'):
            model = experiment.split('-')[0]
        else:
            raise ValueError(f'Unexpected gemini results file name :{json_file}')
    elif 'qwen' in json_file.lower():
        model = '-'.join(json_file.split('local-')[1].split('-')[:2])
    else:
        raise ValueError(f"Unknown model in {json_file}")
    
    # Read JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    results[f'{model}-{config}'] = data

print(results)

# %%
config_groups = {}
for k in sorted(results.keys()):
    base_key = '.'.join(k.split('.')[0:2])  # Remove the last .N suffix
    if base_key not in config_groups:
        config_groups[base_key] = []
    config_groups[base_key].append(k)

# Print the groups for verification
for base_key, configs in config_groups.items():
    print(f"{base_key}:")
    for config in sorted(configs):
        print(f"  {config}")


# %%
import matplotlib.pyplot as plt
import numpy as np
import re
import seaborn as sns
from pathlib import Path
import pandas as pd
import matplotlib as mpl
from typing import Any, cast
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['axes.titlesize'] = 18
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 12

def config_sort_value(s: str) -> int | float:
    if 'baseline' in s.lower() or 'raw' in s.lower():
        return float('inf')
    match = re.search(r'N_(\d+)', s)
    return int(match.group(1)) if match else 0

label_to_color = {
    'Raw Agent': '#303030',
    'Observation Masking': '#6b57ff',
    'LLM-Summary': '#ff318c',
}

def plot_results_of(
    configs_or_title: list[dict[str, str]] | dict[str, list[str]] | str,
    maybe_config_groups: dict[str, list[str]] | None = None,
    *,
    plot_name: str = 'configuration_results.png',
    y_step: int = 2,
    title: str | None = None,
    bar_order: dict[str, int] | None = None,
) -> None:
    """
    Plot mean solve-rate (with std) per configuration.

    Preferred usage: pass a list of dicts with keys 'config', 'strategy', 'label'.

    Backward-compatible usage:
      - plot_results_of(config_groups_dict, plot_name='x.png', y_step=10)
      - plot_results_of('Some Title', config_groups_dict, plot_name='x.png', y_step=10)
    """
    # Normalize inputs to a list of spec dicts
    specs: list[dict[str, str]]
    config_groups_local: dict[str, list[str]]
    if isinstance(configs_or_title, str) and isinstance(maybe_config_groups, dict):
        # Legacy: (title, config_groups)
        title = configs_or_title
        config_groups_local = maybe_config_groups
        specs = [
            {
                'config': k,
                'strategy': ('Raw Agent' if 'raw' in k else ('Observation Masking' if 'baseline' in k else 'LLM-Summary')),
                'label': k,
            }
            for k in sorted(config_groups_local.keys(), key=config_sort_value)
        ]
    elif isinstance(configs_or_title, dict):
        # Legacy: (config_groups)
        config_groups_local = configs_or_title
        specs = [
            {
                'config': k,
                'strategy': ('Raw Agent' if 'raw' in k else ('Observation Masking' if 'baseline' in k else 'LLM-Summary')),
                'label': k,
            }
            for k in sorted(config_groups_local.keys(), key=config_sort_value)
        ]
    else:
        # New API: list of spec dicts
        from typing import cast as _cast
        specs = sorted(_cast(list[dict[str, str]], configs_or_title), key=lambda d: config_sort_value(d['config']))
        config_groups_local = config_groups  # use global mapping built above

    # Optional explicit bar ordering across strategies, and numeric sort within strategies by N/M
    if bar_order is not None:
        def _nm_sort_key(cfg: str) -> tuple[int, int, int]:
            n_match = re.search(r'N_(\d+)', cfg)
            m_match = re.search(r'M_(\d+)', cfg)
            n_val = int(n_match.group(1)) if n_match else None
            m_val = int(m_match.group(1)) if m_match else None
            if n_val is not None and m_val is not None:
                return (0, n_val, m_val)
            if n_val is not None:
                return (0, n_val, -1)
            if m_val is not None:
                return (1, m_val, -1)
            return (2, -1, -1)

        def _key(d: dict[str, str]) -> tuple[int, tuple[int, int, int], str]:
            return (
                bar_order.get(d.get('strategy', ''), 999),
                _nm_sort_key(d.get('config', '')),
                d.get('label', ''),
            )

        specs = sorted(specs, key=_key)

    # Collect per-run solve rates
    rows: list[dict[str, str | float]] = []
    for spec in specs:
        runs = config_groups_local.get(spec['config'], [])  # type: ignore[union-attr]
        for run in runs:
            rows.append({
                'label': spec['label'],
                'strategy': spec['strategy'],
                'rate': (results[run]['resolved_instances'] / results[run]['completed_instances']) * 100.0,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return

    df_any = cast(Any, df)
    agg = cast(Any, df_any.groupby(['label', 'strategy'])['rate'].agg(['mean', 'std']).reset_index())
    # Preserve user-provided label order
    label_order = {spec['label']: i for i, spec in enumerate(specs)}
    agg_df = cast(Any, agg.sort_values(by='label', key=lambda s: s.map(label_order))).copy()
    # Robust to single-run groups: std is NaN when count == 1
    agg_df['std'] = agg_df['std'].fillna(0.0)

    colors = [label_to_color.get(strategy, '#333333') for strategy in agg_df['strategy']]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(agg_df))
    ax.bar(x, agg_df['mean'], yerr=agg_df['std'], color=colors, capsize=6)

    max_y = int(np.nanmax(agg_df['mean'] + agg_df['std'])) if len(agg_df) else 0
    for y in range(0, max(100, max_y) + 1, y_step):
        ax.axhline(y=y, color='gray', linestyle='--', alpha=0.3)

    ax.set_xlabel('Configuration')
    ax.set_ylabel('Solve Rate (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(agg_df['label'], rotation=30, ha='right')
    ax.set_ylim(0, max(100, max_y))
    if title:
        ax.set_title(title)

    plt.tight_layout()
    figures_dir = Path('report_data')
    figures_dir.mkdir(exist_ok=True)
    plt.savefig(figures_dir / plot_name, bbox_inches='tight')
    plt.show()


# %%
plot_specs_main = [
    {'config': 'gpt4.1-mini-N_21_static_summary_M_10_openhands', 'strategy': 'LLM-Summary', 'label': 'LLM-Summary (N=21,M=10)'},
    {'config': 'gpt4.1-mini-baseline_M_5', 'strategy': 'Observation Masking', 'label': 'Observation Masking (M=5)'},
    {'config': 'gpt4.1-mini-baseline_raw', 'strategy': 'Raw Agent', 'label': 'Raw Agent'},
]

plot_results_of(
    plot_specs_main,
    plot_name='preliminary_experiments_solve_rate.png',
    y_step=10,
    title='Preliminary Experiments on SWE-bench Lite-50',
    bar_order={'Raw Agent': 0, 'Observation Masking': 1, 'LLM-Summary': 2},
)

# %% [markdown]
# ### Main hyperparameter sweep for number of turns
# Rationale for this plot: Provide an overview of performance on Lite-50 across different summarization intervals to find a promising interval (N) for further investigation.

# %%
plot_specs_n_sweep = [
    {'config': 'gpt4.1-mini-N_3', 'strategy': 'LLM-Summary', 'label':  'LLM-Summary (N=3,M=0)'},
    {'config': 'gpt4.1-mini-N_7', 'strategy': 'LLM-Summary', 'label': 'LLM-Summary (N=7,M=0)'},
    {'config': 'gpt4.1-mini-N_13', 'strategy': 'LLM-Summary', 'label': 'LLM-Summary (N=13,M=0)'},
    {'config': 'gpt4.1-mini-N_21', 'strategy': 'LLM-Summary', 'label': 'LLM-Summary (N=21,M=0)'},
    {'config': 'gpt4.1-mini-N_34', 'strategy': 'LLM-Summary', 'label': 'LLM-Summary (N=34,M=0)'},
    {'config': 'gpt4.1-mini-baseline_M_5', 'strategy': 'Observation Masking', 'label': 'Observation Masking (M=5)'},
    {'config': 'gpt4.1-mini-baseline_raw', 'strategy': 'Raw Agent', 'label': 'Raw Agent'},
]

plot_results_of(
    plot_specs_n_sweep,
    plot_name='main_configuration_results.png',
    y_step=10,
    bar_order={'Raw Agent': 0, 'Observation Masking': 1, 'LLM-Summary': 2},
)

# %% [markdown]
# ### Ablation: Generalization to SWE-bench Verified
# Rationale: We know that naive summarization by omission is a strong baseline, the best configuration we found on our validation split is using a checkpoint critic that adds a new summary everytime we trigger one (instead of a static one). Let's check whether our findings generalize.

# %%
swe_bench_verified_150_filter = open('config/dataset_filters/verified-150.txt','r').read().split('|')

# %%
verified_500_configs = [
    'gpt4.1-mini-baseline_M_5_v500',
    'gpt4.1-mini-baseline_raw_v500',
    'gpt4.1-mini-gpt_4_1_N_21_checkpoint_critic_v500'
    
]
verified_500_config_groups = {config: config_groups[config] for config in verified_500_configs}

plot_results_of('Ablation: Generalization of Checkpoint Critic on Verified-500', verified_500_config_groups, plot_name='verified_500_ablation_configuration_results.png', y_step=10)

# %% [markdown]
# ### Ablation - Findings vs OpenHands setup on SWE-agent scaffold
# Rationale: We have a setup that works decently for us, but is different from OpenHands. How does our setup compare to theirs?
#
# NOTE: This is now integrated in the initial prompt engineering improvements plot. Further explorations of the config will only be shown on Verified-150 further down.

# %%
openhands_plot_specs = [
    {'config': 'gpt4.1-mini-baseline_M_5', 'strategy': 'Observation Masking', 'label': 'Observation Masking (M=5)'},
    {'config': 'gpt4.1-mini-baseline_raw', 'strategy': 'Raw Agent', 'label': 'Raw Agent'},
    {'config': 'gpt4.1-mini-N_21_static_summary_M_10_openhands', 'strategy': 'LLM-Summary', 'label': 'LLM-Summary (N=21, M=10, OpenHands)'},
    {'config': 'gpt4.1-mini-N_21_static_checkpoint_critic_M_10', 'strategy': 'LLM-Summary', 'label': 'LLM-Summary (N=21, M=10, Checkpoint Critic)'},
    {'config': 'gpt4.1-mini-N_21_static_checkpoint_critic', 'strategy': 'LLM-Summary', 'label': 'LLM-Summary (N=21, M=0, Checkpoint Critic)'},
]

plot_results_of(
    openhands_plot_specs,
    plot_name='joint_critic_and_summarization_vs_openhands_ablation_configuration_results.png',
    y_step=10,
    bar_order={'Raw Agent': 0, 'Observation Masking': 1, 'LLM-Summary': 2},
)


# %% [markdown]
# ### Ablation: Generalization to SWE-bench Verified - Negative testing
# Rationale: Our first experiment does not show generalization, how do the negative tests perform on our test split? OpenHands contacts mentioned that they used strong models throught their experiments, perhaps we need to just use a stronger agent action generation model to reproduce their findings?

# %%
def downsample_to_verified_150(base_run_id_with_suffix: str, original_sample_size: int, results: dict, config_groups: dict):
    run_id_150 = base_run_id_with_suffix.replace(str(original_sample_size), '150')
    results[run_id_150] = {}
    for k in results[base_run_id_with_suffix].keys():
        if 'ids' not in k:
            continue

        results[run_id_150][k] = [instance for instance in results[base_run_id_with_suffix][k] if instance in swe_bench_verified_150_filter]
        results[run_id_150]['_'.join(k.split('_')[:-1]) + '_instances'] = len(results[run_id_150][k])

    config_groups['.'.join(run_id_150.split('.')[:-1])] = [run_id_150]

downsample_to_verified_150('gpt4.1-mini-baseline_raw_v500.0', 500, results, config_groups)
downsample_to_verified_150('gpt4.1-mini-baseline_M_5_v500.0', 500, results, config_groups)
downsample_to_verified_150('gpt4.1-mini-gpt_4_1_N_21_checkpoint_critic_v500.0', 500, results, config_groups)

# %% [markdown]
# ### Ablation: Baseline Tail Length - v150
# Rationale: We want to validate that the performance just monotoneously improves with increased tail length when using environment omission summarization (as this approaches the raw baseline). We do this to inform our tail length for the further experiments.

# %%
plot_specs_n_sweep = [
    {'config': 'gpt4.1-mini-baseline_M_5_v150', 'strategy': 'Observation Masking', 'label':  'Observation Masking (M=5)'},
    {'config': 'gpt4.1-mini-baseline_M_10_v150', 'strategy': 'Observation Masking', 'label': 'Observation Masking (M=10)'},
    {'config': 'gpt4.1-mini-baseline_M_20_v150', 'strategy': 'Observation Masking', 'label': 'Observation Masking (M=20)'},
    {'config': 'gpt4.1-mini-baseline_raw_v150', 'strategy': 'Raw Agent', 'label': 'Raw Agent'},
]

plot_results_of(
    plot_specs_n_sweep,
    plot_name='baseline_tail_length_ablation_configuration_results.png',
    y_step=10,
    bar_order={'Raw Agent': 0, 'Observation Masking': 1, 'LLM-Summary': 2},
)

# %% [markdown]
# ### Ablation: Prompt & Setup
# Rationale: So we know that the tail length is very important. We have a summarization frequency that worked well in our experiments so far. We only tried OpenHands with their setup. Is our configuration (N,M) the driving force of performance? Do our and OpenHands' approach differ in performance behaviour in any way when varying between
# - no tail: N=21, M=0
# - our config: N=21, M=10
# - their setup: N=10, M=10
# - their setup + config values: N=21, M=20 

# %%
openhands_vs_checkpoint_critic_v150_specs = [
    {'config': 'gpt4.1-mini-baseline_raw_v150', 'strategy': 'Raw Agent', 'label': 'Raw Agent'},
    {'config': 'gpt4.1-mini-baseline_M_10_v150', 'strategy': 'Observation Masking', 'label': 'Observation Masking (M=10)'},
    {'config': 'gpt4.1-mini-N_10_M_10_openhands_v150', 'strategy': 'LLM-Summary', 'label': 'LLM Summary (N=10,M=10)'},
    {'config': 'gpt4.1-mini-N_21_M_0_openhands_v150', 'strategy': 'LLM-Summary', 'label': 'LLM Summary (N=21,M=0)'},
    {'config': 'gpt4.1-mini-N_21_M_10_openhands_v150', 'strategy': 'LLM-Summary', 'label': 'LLM Summary (N=21,M=10)'},
]

plot_results_of(
    openhands_vs_checkpoint_critic_v150_specs,
    plot_name='openhands_vs_checkpoint_critic_v150_ablation_configuration_results.png',
    y_step=10,
    bar_order={'Raw Agent': 0, 'Observation Masking': 1, 'LLM-Summary': 2},
)

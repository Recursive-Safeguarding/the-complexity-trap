#!/usr/bin/env python3
"""
Orchestrate SWE-agent experiments with concurrent evaluations.

This script manages the execution of multiple experiments with a maximum of 2 concurrent
experiments and 4 concurrent evaluations at any time. Evaluations start immediately
when experiments complete, reducing overall wall time.
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from queue import Queue
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Thread, Event
import yaml

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

# Logger will be configured dynamically
logger = logging.getLogger(__name__)

# Global shutdown event
shutdown_event = Event()

# Global list to track all active subprocesses for cleanup
active_subprocesses = []
active_subprocesses_lock = Lock()

def signal_handler(signum, frame):
    """Handle shutdown signals by setting shutdown event and cleaning up resources."""
    with log_lock:
        logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
    
    shutdown_event.set()
    
    # Terminate all active subprocesses
    with active_subprocesses_lock:
        for process in active_subprocesses[:]:  # Create a copy to iterate over
            try:
                if process.poll() is None:  # Process is still running
                    with log_lock:
                        logger.info(f"🔪 Terminating subprocess PID {process.pid}")
                    process.terminate()
                    # Give it a moment to terminate gracefully
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        logger.warning(f"⚡ Force killing subprocess PID {process.pid}")
                        process.kill()
                        process.wait()
            except Exception as e:
                with log_lock:
                    logger.error(f"❌ Error terminating subprocess: {e}")
    
    with log_lock:
        logger.info("🏁 Cleanup completed, exiting...")
    sys.exit(0)

def register_subprocess(process: subprocess.Popen):
    """Register a subprocess for cleanup tracking."""
    with active_subprocesses_lock:
        active_subprocesses.append(process)

def unregister_subprocess(process: subprocess.Popen):
    """Unregister a subprocess from cleanup tracking."""
    with active_subprocesses_lock:
        try:
            active_subprocesses.remove(process)
        except ValueError:
            pass  # Process already removed


class ExperimentScheduler:
    """Manages experiment scheduling and resource limits."""
    
    def __init__(
        self, 
        executor: ThreadPoolExecutor,
        max_local_experiments: int = 1,
        max_remote_experiments: int = 1,
        completed_experiments_queue: Optional[Queue] = None
    ):
        """
        Initialize the experiment scheduler.
        
        Args:
            executor: ThreadPoolExecutor for running experiments
            max_local_experiments: Maximum number of local experiments to run concurrently
            max_remote_experiments: Maximum number of remote experiments to run concurrently
            completed_experiments_queue: Queue to put completed experiment run_ids for evaluation
        """
        self.executor = executor
        self.futures = {}  # run_id -> (future, process, is_local_model)
        self.waiting_experiments = []  # (experiment, run_id) waiting to be scheduled
        self.running_local = 0
        self.running_remote = 0
        self.max_local_experiments = max_local_experiments
        self.max_remote_experiments = max_remote_experiments
        self.completed_experiments_queue = completed_experiments_queue
        
    def can_start_experiment(self, is_local_model: bool) -> bool:
        """Check if we can start an experiment of the given type."""
        if shutdown_event.is_set():
            return False
        if is_local_model:
            return self.running_local < self.max_local_experiments
        else:
            return self.running_remote < self.max_remote_experiments
    
    def _start_single_experiment(self, experiment: "ExperimentConfig", run_id: str) -> bool:
        """Start a single experiment if resources allow. Returns True if started."""
        if shutdown_event.is_set():
            return False
            
        is_local_model = experiment.agent_model.is_local_model
        
        if not self.can_start_experiment(is_local_model):
            return False
        
        # Start the experiment
        process = run_sweagent_with(experiment, run_id)
        future = self.executor.submit(process.wait)
        self.futures[run_id] = (future, process, is_local_model)
        
        # Update resource counts
        if is_local_model:
            self.running_local += 1
        else:
            self.running_remote += 1
        
        model_type = "local" if is_local_model else "remote"
        with log_lock:
            logger.info(f"🚀 Started {model_type} experiment {run_id} (running: {self.running_local} local, {self.running_remote} remote)")
        return True
    
    def load_from_queue_and_start_experiments(self, experiment_queue: Queue) -> None:
        """Load new experiments from queue and start as many as possible."""
        if shutdown_event.is_set():
            return
            
        newly_queued = []
        while not experiment_queue.empty() and not shutdown_event.is_set():
            try:
                experiment, run_id = experiment_queue.get_nowait()
                self.waiting_experiments.append((experiment, run_id))
                newly_queued.append((experiment, run_id))
            except:
                break
        
        # Now try to start as many waiting experiments as possible
        still_waiting = []
        for experiment, run_id in self.waiting_experiments:
            if shutdown_event.is_set():
                break
            if self._start_single_experiment(experiment, run_id):
                # Started successfully, don't add back to waiting list
                continue
            else:
                # Couldn't start, keep in waiting list
                still_waiting.append((experiment, run_id))
                
                # Log only for newly queued experiments
                if (experiment, run_id) in newly_queued:
                    is_local_model = experiment.agent_model.is_local_model
                    model_type = "local" if is_local_model else "remote"
                    with log_lock:
                        logger.info(f"⏳ {model_type.title()} experiment {run_id} queued (running: {self.running_local} local, {self.running_remote} remote)")
        
        self.waiting_experiments = still_waiting
    
    def handle_completed_experiments(self, completed_experiments: Dict[str, int], total_experiments: int) -> None:
        """Process any completed experiments and update resource counts."""
        for run_id, (future, process, is_local_model) in list(self.futures.items()):
            if not future.done():
                continue
            
            # Process completed experiment
            return_code = future.result()
            completed_experiments['count'] += 1
            
            # Unregister the subprocess from cleanup tracking
            unregister_subprocess(process)
            
            # Update resource counts
            if is_local_model:
                self.running_local -= 1
            else:
                self.running_remote -= 1
            
            model_type = "local" if is_local_model else "remote"
            
            if return_code != 0:
                with log_lock:
                    logger.error(f"❌ {model_type.title()} experiment {run_id} failed with return code {return_code}")
            else:
                with log_lock:
                    logger.info(f"✅ {model_type.title()} experiment {run_id} completed successfully")
                
                # Check if preds.json was created and queue for evaluation
                if not shutdown_event.is_set():
                    preds_path = Path(f"trajectories/lindenbauer/main_experiments/{run_id}/preds.json")
                    if preds_path.exists() and self.completed_experiments_queue is not None:
                        self.completed_experiments_queue.put(run_id)
                        with log_lock:
                            logger.info(f"📤 Queued {run_id} for evaluation")
                    elif not preds_path.exists():
                        with log_lock:
                            logger.warning(f"⚠️  No preds.json found for {run_id}")
            
            with log_lock:
                logger.info(f"📈 Experiment progress: {completed_experiments['count']}/{total_experiments} completed")
            
            del self.futures[run_id]
    
    def has_active_work(self, experiment_queue: Queue) -> bool:
        """Check if there's any active work (running experiments, waiting experiments, or queued experiments)."""
        if shutdown_event.is_set():
            return bool(self.futures)  # Only check running experiments during shutdown
        return bool(self.futures) or bool(self.waiting_experiments) or not experiment_queue.empty()
    
    def wait_for_completion(self, timeout: int = 86400) -> None:
        """Wait for at least one experiment to complete."""
        if self.futures:
            try:
                # Use shorter timeout during shutdown
                actual_timeout = 1 if shutdown_event.is_set() else timeout
                next(as_completed([future for future, _, _ in self.futures.values()], timeout=actual_timeout))
            except:
                pass
        else:
            # Use shorter sleep during shutdown
            sleep_time = 1 if shutdown_event.is_set() else 10
            time.sleep(sleep_time)
    
    def terminate_all_experiments(self) -> None:
        """Terminate all running experiments during shutdown."""
        with log_lock:
            logger.info("🛑 Terminating all running experiments...")
        
        for run_id, (future, process, is_local_model) in self.futures.items():
            try:
                if process.poll() is None:  # Process is still running
                    with log_lock:
                        logger.info(f"🔪 Terminating experiment {run_id} (PID {process.pid})")
                    process.terminate()
                    # Try to wait briefly for graceful termination
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        with log_lock:
                            logger.warning(f"⚡ Force killing experiment {run_id} (PID {process.pid})")
                        process.kill()
                        process.wait()
                
                unregister_subprocess(process)
            except Exception as e:
                with log_lock:
                    logger.error(f"❌ Error terminating experiment {run_id}: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the scheduler."""
        return {
            "running_local": self.running_local,
            "running_remote": self.running_remote,
            "waiting_experiments": len(self.waiting_experiments),
            "active_experiments": len(self.futures),
            "max_local": self.max_local_experiments,
            "max_remote": self.max_remote_experiments,
            "shutdown_requested": shutdown_event.is_set()
        }


@dataclass
class ModelConfig:
    """Configuration for a model (agent or summary)."""
    name: str
    per_instance_call_limit: int
    per_instance_cost_limit: int
    is_local_model: bool = False
    use_reasoning: Optional[bool] = None
    api_base: Optional[str] = None
    max_input_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    total_cost_limit: Optional[float] = None
    temperature: Optional[float] = None
    api_key: Optional[str] = None
    completion_kwargs: Optional[str] = None

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    runs: int
    run_id: str
    config: str
    agent_model: ModelConfig
    summary_model: Optional[ModelConfig] = None
    num_workers: int = 35
    extra_args: Dict[str, Any] = field(default_factory=dict)
    
    def get_run_id_for_iteration(self, iteration: int) -> str:
        """Generate run_id for a specific iteration."""
        base_parts = self.run_id.split('-')
        if len(base_parts) >= 3:
            base_parts[-3] += f".{iteration}"
            return '-'.join(base_parts)
        else:
            # Fallback: append iteration to the end
            return f"{self.run_id}-{iteration}"

port = 65482

LOCAL_EXPERIMENTS = [
    ExperimentConfig(
        runs=1,
        run_id="local-Qwen3-32B_thinking-baseline_raw-collect_reasoning-retry_errors-verified-500",
        config="config/default_no_demo_raw.yaml",
        num_workers=19,
        agent_model=ModelConfig(
            name="hosted_vllm/Qwen/Qwen3-32B",
            is_local_model=True,
            api_base=f'http://0.0.0.0:{port}/v1/',
            max_input_tokens=128 * 1024 - 8 * 1024,
            max_output_tokens=8 * 1024,
            temperature=0.8,
            use_reasoning=True,
            per_instance_cost_limit=0,
            per_instance_call_limit=250,
            total_cost_limit=0.0,
            completion_kwargs='{"timeout": "600"}'
        ),
        extra_args={
            "instances.filter": "sympy__sympy-17318|sympy__sympy-17630|sympy__sympy-17655|sympy__sympy-18189|sympy__sympy-18199|sympy__sympy-18211|sympy__sympy-18698|sympy__sympy-18763|sympy__sympy-19040|sympy__sympy-19346|sympy__sympy-19495|sympy__sympy-19637|sympy__sympy-19783|sympy__sympy-19954|sympy__sympy-20154|sympy__sympy-20428|sympy__sympy-20438|sympy__sympy-20590|sympy__sympy-20801|sympy__sympy-20916|sympy__sympy-21379|sympy__sympy-21596|sympy__sympy-21612|sympy__sympy-21847|sympy__sympy-21930|sympy__sympy-22080|sympy__sympy-22456|sympy__sympy-22714|sympy__sympy-22914|sympy__sympy-23262|sympy__sympy-23413|sympy__sympy-23534|sympy__sympy-23824|sympy__sympy-23950|sympy__sympy-24066|sympy__sympy-24213|sympy__sympy-24443|sympy__sympy-24539|sympy__sympy-24562|sympy__sympy-24661"
        }
    ),
    ExperimentConfig(
        runs=1,
        run_id="local-Qwen3-32B_thinking-baseline_N_1_M_10-collect_reasoning-verified-500",
        config="config/default_no_demo_N=1_M=10.yaml",
        num_workers=19,
        agent_model=ModelConfig(
            name="hosted_vllm/Qwen/Qwen3-32B",
            is_local_model=True,
            api_base=f'http://0.0.0.0:{port}/v1/',
            max_input_tokens=128 * 1024 - 8 * 1024,
            max_output_tokens=8 * 1024,
            temperature=0.8,
            use_reasoning=True,
            per_instance_cost_limit=0,
            per_instance_call_limit=250,
            total_cost_limit=0.0,
            completion_kwargs='{"timeout": "600"}'
        )
    ),
    ExperimentConfig(
        runs=1,
        run_id="local-Qwen3-32B_thinking-t_0.8-turn-summaries-t_0-N_21_M_10_openhands-collect_reasoning-verified-500",
        config="config/default_no_demo_checkpoint_same_model_openhands_N=21_M=10.yaml",
        num_workers=25,
        agent_model=ModelConfig(
            name="hosted_vllm/Qwen/Qwen3-32B",
            is_local_model=True,
            api_base=f'http://0.0.0.0:{port}/v1/',
            max_input_tokens=128 * 1024 - 8 * 1024,
            max_output_tokens=8 * 1024,
            temperature=0.8,
            use_reasoning=True,
            per_instance_cost_limit=0,
            per_instance_call_limit=250,
            total_cost_limit=0.0,
            completion_kwargs='{"timeout": "600"}'
        ),
        summary_model=ModelConfig(
            name="hosted_vllm/Qwen/Qwen3-32B",
            is_local_model=True,
            temperature=0.0,
            api_base=f'http://0.0.0.0:{port}/v1/',
            max_input_tokens=128 * 1024 - 8 * 1024,
            max_output_tokens=8 * 1024,
            use_reasoning=True,
            per_instance_cost_limit=0,
            per_instance_call_limit=0,
            total_cost_limit=0.0,
            completion_kwargs='{"timeout": "600"}'
        )
    ),
]


def build_bedrock_repro_experiments(
    *,
    runs: int = 1,
    num_workers: int = 1,
    instances_slice: str | None = None,
    summarizer_model: str = "bedrock-nova-pro",
) -> list[ExperimentConfig]:
    """Paper-style 4-pack (raw/masking/summary/hybrid) on AWS Bedrock.

    Main model: Qwen3 32B on Bedrock
    Summarizer: Amazon Nova Pro on Bedrock (temporary Gemini Flash substitute).
    """
    # Import here so the local suite doesn't require this module to exist.
    from sweagent.utils.model_config import get_model_args

    agent_args = get_model_args("bedrock-qwen3-32b")
    summarizer_args = get_model_args(summarizer_model)

    agent_model = ModelConfig(
        name=agent_args["name"],
        temperature=0.8,
        per_instance_cost_limit=0,
        per_instance_call_limit=250,
        total_cost_limit=0.0,
        max_input_tokens=agent_args.get("max_input_tokens"),
        max_output_tokens=agent_args.get("max_output_tokens"),
        completion_kwargs='{"timeout": "600"}',
    )

    summary_model = ModelConfig(
        name=summarizer_args["name"],
        temperature=0.0,
        per_instance_cost_limit=0,
        per_instance_call_limit=0,
        total_cost_limit=0.0,
        max_input_tokens=summarizer_args.get("max_input_tokens"),
        max_output_tokens=summarizer_args.get("max_output_tokens"),
        completion_kwargs='{"timeout": "600"}',
    )

    base_extra_args: dict[str, Any] = {}
    if instances_slice:
        base_extra_args["instances.slice"] = instances_slice

    return [
        ExperimentConfig(
            runs=runs,
            run_id="bedrock-qwen3-32b-agent-t_0.8-baseline_raw-verified-500",
            config="config/default_no_demo_raw.yaml",
            num_workers=num_workers,
            agent_model=agent_model,
            extra_args=dict(base_extra_args),
        ),
        ExperimentConfig(
            runs=runs,
            run_id="bedrock-qwen3-32b-agent-t_0.8-baseline_N_1_M_10-verified-500",
            config="config/default_no_demo_N=1_M=10.yaml",
            num_workers=num_workers,
            agent_model=agent_model,
            extra_args=dict(base_extra_args),
        ),
        ExperimentConfig(
            runs=runs,
            run_id="bedrock-qwen3-32b-agent-t_0.8-turn-summaries-t_0-N_21_M_10_openhands-summarizer_haiku45-verified-500",
            config="config/default_no_demo_checkpoint_same_model_openhands_N=21_M=10.yaml",
            num_workers=num_workers,
            agent_model=agent_model,
            summary_model=summary_model,
            extra_args=dict(base_extra_args),
        ),
        ExperimentConfig(
            runs=runs,
            run_id="bedrock-qwen3-32b-agent-t_0.8-hybrid-N_21_M_10_masking_M_10_openhands-summarizer_haiku45-verified-500",
            config="config/default_no_demo_checkpoint_same_model_openhands_N=21_M=10_masking_M=10.yaml",
            num_workers=num_workers,
            agent_model=agent_model,
            summary_model=summary_model,
            extra_args=dict(base_extra_args),
        ),
    ]


# Global experiments list used by the worker threads. This is overridden in main()
# based on CLI flags (e.g., to switch from local vLLM to Bedrock reproduction).
experiments = LOCAL_EXPERIMENTS

# Shared queue for passing completed experiments to evaluation workers
completed_experiments_queue = Queue()

# Lock for coordinating logging
log_lock = Lock()
        
def _validate_model_config(model_config: ModelConfig, model_type: str) -> None:
    """
    Validate model configuration.
    
    Args:
        model_config: ModelConfig object to validate
        model_type: Type of model for error messages (e.g., "agent", "summary")
    """
    if model_config.is_local_model and model_config.api_base is None:
        raise ValueError(f"Local {model_type} model specified but no API base provided")

def _add_model_config_to_cmd(cmd: List[str], model_config: ModelConfig, prefix: str, exclude_attrs: Optional[List[str]] = None) -> None:
    """
    Add model configuration parameters to command list.
    
    Args:
        cmd: Command list to extend
        model_config: ModelConfig object with parameters
        prefix: Command prefix (e.g., "--agent.model" or "--agent.summary_model")
        exclude_attrs: List of attribute names to skip (already added to cmd)
    """
    exclude_attrs = exclude_attrs or []
    
    for attr_name, attr_value in vars(model_config).items():
        if attr_value is not None and attr_name not in exclude_attrs:
            flag = f"--{prefix}.{attr_name}"
            
            # Handle boolean values by converting to lowercase strings
            if isinstance(attr_value, bool):
                cmd.extend([flag, str(attr_value).lower()])
            else:
                cmd.extend([flag, str(attr_value)])

def run_sweagent_with(experiment_config: ExperimentConfig, run_id: str) -> subprocess.Popen:
    """
    Run SWE-agent with the specified parameters from experiment configuration.
    
    Args:
        experiment_config: ExperimentConfig object containing all experiment parameters
        run_id: Unique identifier for this run
    
    Returns:
        subprocess.Popen: The subprocess handle
    """
    _validate_model_config(experiment_config.agent_model, "agent")
    if experiment_config.summary_model is not None:
        _validate_model_config(experiment_config.summary_model, "summary")
    
    cmd = [
        "sweagent", "run-batch",
        "--config", experiment_config.config,
        "--agent.model.name", experiment_config.agent_model.name,
        "--agent.model.per_instance_call_limit", str(experiment_config.agent_model.per_instance_call_limit),
        "--agent.model.per_instance_cost_limit", str(experiment_config.agent_model.per_instance_cost_limit),
        "--agent.type", "default",
        "--instances.type", "swe_bench",
        "--instances.subset", "verified",
        "--instances.split", "test",
        "--num_workers", str(experiment_config.num_workers),
        "--output_dir", f"trajectories/lindenbauer/main_experiments/{run_id}"
    ]
    
    # Add remaining agent model configuration (excluding already added attributes)
    _add_model_config_to_cmd(
        cmd, 
        experiment_config.agent_model, 
        "agent.model", 
        exclude_attrs=["name", "per_instance_call_limit", "per_instance_cost_limit"]
    )
    
    # Add summary model configuration if present
    if experiment_config.summary_model is not None:
        _add_model_config_to_cmd(cmd, experiment_config.summary_model, "agent.summary_model")
    
    # Add extra arguments if provided
    if experiment_config.extra_args:
        for key, value in experiment_config.extra_args.items():
            if key.startswith('--'):
                cmd.extend([key, str(value)])
            else:
                cmd.extend([f"--{key}", str(value)])
    
    with log_lock:
        logger.info(f"Starting experiment {run_id} with command: {' '.join(cmd)}")
    
    # Create output directory
    output_dir = Path(f"trajectories/lindenbauer/main_experiments/{run_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Start process with output redirection
    log_file = output_dir / "experiment.log"
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=Path.cwd(),
            env=os.environ.copy()
        )
    
    # Register the subprocess for cleanup tracking
    register_subprocess(process)
    
    return process

def run_eval(run_id: str) -> subprocess.Popen:
    """
    Run evaluation for a completed experiment.
    
    Args:
        run_id: The run ID to evaluate
        
    Returns:
        subprocess.Popen: The subprocess handle
    """
    
    # Get absolute paths to avoid confusion
    current_dir = Path.cwd().absolute()
    swe_bench_dir = current_dir.parent / "SWE-bench"
    predictions_path = current_dir / "trajectories" / "lindenbauer" / "main_experiments" / run_id / "preds.json"
    
    shell_script = f"""#!/bin/bash
set -e

# Check if SWE-bench directory exists
if [ ! -d "{swe_bench_dir}" ]; then
    echo "ERROR: SWE-bench directory does not exist at {swe_bench_dir}"
    exit 1
fi

# Check if predictions file exists
if [ ! -f "{predictions_path}" ]; then
    echo "ERROR: Predictions file does not exist at {predictions_path}"
    exit 1
fi

# Change to SWE-bench directory and activate virtual environment
cd "{swe_bench_dir}"
if [ ! -d ".venv" ]; then
    echo "ERROR: Virtual environment .venv does not exist in $(pwd)"
    exit 1
fi
source .venv/bin/activate

# Run evaluation
python -m swebench.harness.run_evaluation \\
    --dataset_name "princeton-nlp/SWE-bench_Verified" \\
    --predictions_path "{predictions_path}" \\
    --max_workers "35" \\
    --run_id "{run_id}"
"""
    
    with log_lock:
        logger.info(f"Starting evaluation for {run_id}")
    
    # Create evaluation log file in the current project directory
    eval_log_file = Path(f"trajectories/lindenbauer/main_experiments/{run_id}/evaluation.log")
    eval_log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Start evaluation process using bash with the shell script
    with open(eval_log_file, 'w') as f:
        process = subprocess.Popen(
            ["bash", "-c", shell_script],
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=Path.cwd(),
            env=os.environ.copy()
        )
    
    # Register the subprocess for cleanup tracking
    register_subprocess(process)
    
    return process

def get_filter_string() -> str:
    """Get the filter string for verified-150 instances."""
    try:
        filter_file = Path("config/dataset_filters/verified-150.txt")
        if filter_file.exists():
            with open(filter_file, 'r') as f:
                filters = [line.strip() for line in f if line.strip()]
            return ','.join(filters)
        else:
            with log_lock:
                logger.warning(f"Filter file {filter_file} not found, using empty filter")
            return ""
    except Exception as e:
        with log_lock:
            logger.error(f"Error reading filter file: {e}")
        return ""

def check_existing_evaluations() -> List[str]:
    """Check for existing evaluations and queue any that are ready but not yet evaluated."""
    ready_for_eval = []
    
    for experiment in experiments:
        for run in range(experiment.runs):
            run_id = experiment.get_run_id_for_iteration(run)
            preds_path = Path(f"trajectories/lindenbauer/main_experiments/{run_id}/preds.json")
            
            if preds_path.exists():
                # Check if evaluation results already exist
                eval_result_path = Path(f"{run_id}.{run_id}.json")
                
                if eval_result_path.exists():
                    with log_lock:
                        logger.info(f"Skipping {run_id} - evaluation results already exist at {eval_result_path}")
                else:
                    ready_for_eval.append(run_id)
                    with log_lock:
                        logger.info(f"Found existing preds.json for {run_id}, queueing for evaluation")
    
    return ready_for_eval

def experiment_worker(
    experiment_queue: Queue, 
    total_experiments: int, 
    completed_experiments: Dict[str, int],
    max_local_experiments: int = 1,
    max_remote_experiments: int = 1
) -> None:
    """
    Worker function for running experiments with clean resource management.
    
    Args:
        experiment_queue: Queue containing (experiment, run_id) tuples to process
        total_experiments: Total number of experiments for progress tracking
        completed_experiments: Dict with 'count' key to track completed experiments
        max_local_experiments: Maximum number of local experiments to run concurrently
        max_remote_experiments: Maximum number of remote experiments to run concurrently
    """
    with ThreadPoolExecutor(max_workers=max_local_experiments + max_remote_experiments) as executor:
        scheduler = ExperimentScheduler(
            executor=executor,
            max_local_experiments=max_local_experiments,
            max_remote_experiments=max_remote_experiments,
            completed_experiments_queue=completed_experiments_queue
        )
        
        with log_lock:
            logger.info(f"🎯 Started experiment worker (max local: {max_local_experiments}, max remote: {max_remote_experiments})")
        
        try:
            while scheduler.has_active_work(experiment_queue) and not shutdown_event.is_set():
                scheduler.handle_completed_experiments(completed_experiments, total_experiments)
                scheduler.load_from_queue_and_start_experiments(experiment_queue)
                
                status = scheduler.get_status()
                with log_lock:
                    logger.debug(f"📊 Scheduler status: {status}")
                
                scheduler.wait_for_completion()
            
            # If shutdown was requested, terminate all running experiments
            if shutdown_event.is_set():
                scheduler.terminate_all_experiments()
                # Wait a bit for experiments to finish terminating
                final_wait_count = 0
                while scheduler.futures and final_wait_count < 10:
                    scheduler.handle_completed_experiments(completed_experiments, total_experiments)
                    time.sleep(0.5)
                    final_wait_count += 1
        
        except Exception as e:
            with log_lock:
                logger.error(f"❌ Error in experiment worker: {e}")
            # Ensure cleanup happens even on exception
            if scheduler:
                scheduler.terminate_all_experiments()
        
        with log_lock:
            if shutdown_event.is_set():
                logger.info("🛑 Experiment worker terminated due to shutdown signal")
            else:
                logger.info("🏁 Experiment worker finished - no more work to do")

def evaluation_worker(total_evaluations: int, completed_evaluations: Dict[str, int], experiments_finished: Dict[str, bool]):
    """Worker function for running evaluations."""
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}
        
        try:
            while (not experiments_finished['value'] or not completed_experiments_queue.empty() or futures) and not shutdown_event.is_set():
                # Start new evaluations if we have capacity
                while len(futures) < 1 and not shutdown_event.is_set():
                    try:
                        run_id = completed_experiments_queue.get(timeout=1)
                        
                        eval_result_path = Path(f"{run_id}.{run_id}.json")
                        if eval_result_path.exists():
                            with log_lock:
                                logger.info(f"Skipping {run_id} - evaluation results already exist")
                            continue
                        
                        process = run_eval(run_id)
                        future = executor.submit(process.wait)
                        futures[run_id] = (future, process)
                        with log_lock:
                            logger.info(f"🔬 Started evaluation {run_id}")
                    except:
                        break
                
                # Check for completed evaluations
                for run_id, (future, process) in list(futures.items()):
                    if future.done():
                        return_code = future.result()
                        completed_evaluations['count'] += 1
                        
                        # Unregister the subprocess from cleanup tracking
                        unregister_subprocess(process)
                        
                        if return_code != 0:
                            with log_lock:
                                logger.error(f"❌ Evaluation {run_id} failed with return code {return_code}")
                        else:
                            with log_lock:
                                logger.info(f"✅ Evaluation {run_id} completed successfully")
                        
                        with log_lock:
                            logger.info(f"📊 Evaluation progress: {completed_evaluations['count']}/{total_evaluations} completed")
                        del futures[run_id]
                
                # Sleep briefly to avoid busy waiting (shorter during shutdown)
                sleep_time = 1 if shutdown_event.is_set() else 10
                time.sleep(sleep_time)
            
            # If shutdown was requested, terminate all running evaluations
            if shutdown_event.is_set() and futures:
                with log_lock:
                    logger.info("🛑 Terminating all running evaluations...")
                
                for run_id, (future, process) in futures.items():
                    try:
                        if process.poll() is None:  # Process is still running
                            with log_lock:
                                logger.info(f"🔪 Terminating evaluation {run_id} (PID {process.pid})")
                            process.terminate()
                            # Try to wait briefly for graceful termination
                            try:
                                process.wait(timeout=2)
                            except subprocess.TimeoutExpired:
                                with log_lock:
                                    logger.warning(f"⚡ Force killing evaluation {run_id} (PID {process.pid})")
                                process.kill()
                                process.wait()
                        
                        unregister_subprocess(process)
                    except Exception as e:
                        with log_lock:
                            logger.error(f"❌ Error terminating evaluation {run_id}: {e}")
        
        except Exception as e:
            with log_lock:
                logger.error(f"❌ Error in evaluation worker: {e}")
        
        with log_lock:
            if shutdown_event.is_set():
                logger.info("🛑 Evaluation worker terminated due to shutdown signal")
            else:
                logger.info("🏁 Evaluation worker finished")

def run_concurrent_orchestration(*, max_local_experiments: int = 1, max_remote_experiments: int = 1):
    """Main orchestration function that runs experiments and evaluations concurrently."""
    logger.info("🎬 Starting concurrent experiment and evaluation orchestration")
    
    # Check for existing experiments that are ready for evaluation
    existing_ready = check_existing_evaluations()
    for run_id in existing_ready:
        completed_experiments_queue.put(run_id)
    
    experiment_queue = Queue()
    total_experiments = 0
    total_evaluations = len(existing_ready)
    
    for experiment in experiments:
        for run in range(experiment.runs):
            run_id = experiment.get_run_id_for_iteration(run)
            
            # Check if this experiment is already completed
            preds_path = Path(f"trajectories/lindenbauer/main_experiments/{run_id}/preds.json")
            if preds_path.exists():
                logger.info(f"Experiment {run_id} already completed, skipping")
                continue
                
            experiment_queue.put((experiment, run_id))
            total_experiments += 1
            total_evaluations += 1  # Each experiment will eventually need evaluation
    
    if total_experiments == 0:
        logger.info("No new experiments to run")
        if existing_ready:
            logger.info(f"Only running evaluations for {len(existing_ready)} existing experiments")
        else:
            logger.info("Nothing to do")
            return
    
    logger.info(f"📋 Planning to run {total_experiments} experiments and {total_evaluations} evaluations")
    
    # Shared counters
    completed_experiments = {'count': 0}
    completed_evaluations = {'count': 0}
    experiments_finished = {'value': False}
    
    # Start experiment worker in a separate thread
    experiment_thread = Thread(
        target=experiment_worker, 
        args=(experiment_queue, total_experiments, completed_experiments, max_local_experiments, max_remote_experiments),
        daemon=False  # Changed from daemon=True to allow proper cleanup
    )
    experiment_thread.start()
    
    # Start evaluation worker in a separate thread
    evaluation_thread = Thread(
        target=evaluation_worker, 
        args=(total_evaluations, completed_evaluations, experiments_finished),
        daemon=False  # Changed from daemon=True to allow proper cleanup
    )
    evaluation_thread.start()
    
    try:
        # Wait for experiment thread to finish
        while experiment_thread.is_alive():
            if shutdown_event.is_set():
                break
            experiment_thread.join(timeout=1)
        
        if not shutdown_event.is_set():
            experiments_finished['value'] = True
            logger.info("🏁 All experiments completed")
        
        # Wait for evaluation thread to finish
        while evaluation_thread.is_alive():
            if shutdown_event.is_set():
                break
            evaluation_thread.join(timeout=1)
        
        if not shutdown_event.is_set():
            logger.info("🏆 All evaluations completed")
            logger.info(f"🎉 Orchestration complete! Ran {completed_experiments['count']} experiments and {completed_evaluations['count']} evaluations")
        else:
            logger.info("🛑 Orchestration interrupted by shutdown signal")
    
    except KeyboardInterrupt:
        # This shouldn't happen since we have signal handlers, but just in case
        logger.info("🛑 Orchestration interrupted by KeyboardInterrupt")
        shutdown_event.set()
    
    finally:
        # Ensure we wait for threads to finish (with timeout)
        if experiment_thread.is_alive():
            logger.info("⏳ Waiting for experiment worker to finish...")
            experiment_thread.join(timeout=10)
            if experiment_thread.is_alive():
                logger.warning("⚠️  Experiment worker did not finish in time")
        
        if evaluation_thread.is_alive():
            logger.info("⏳ Waiting for evaluation worker to finish...")
            evaluation_thread.join(timeout=10)
            if evaluation_thread.is_alive():
                logger.warning("⚠️  Evaluation worker did not finish in time")
        
        experiments_finished['value'] = True
        
        # Final cleanup of any remaining subprocesses
        with active_subprocesses_lock:
            if active_subprocesses:
                logger.warning(f"🧹 Cleaning up {len(active_subprocesses)} remaining subprocesses...")
                for process in active_subprocesses[:]:
                    try:
                        if process.poll() is None:
                            process.terminate()
                            try:
                                process.wait(timeout=2)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()
                    except Exception as e:
                        logger.error(f"❌ Error in final cleanup: {e}")
                active_subprocesses.clear()
        
        logger.info("🧹 Cleanup completed")

def main():
    """Main function to run the orchestrator."""
    parser = argparse.ArgumentParser(description="Orchestrate SWE-agent experiments with concurrent evaluations")
    parser.add_argument(
        "--suite",
        choices=["local", "bedrock_repro"],
        default="local",
        help="Experiment suite to run (default: local).",
    )
    parser.add_argument(
        "--instances-slice",
        default=None,
        help="Optional SWE-bench slice override (e.g. ':10' for a sanity run). Only applied for bedrock_repro.",
    )
    parser.add_argument(
        "--summarizer-model",
        default="bedrock-nova-pro",
        help=(
            "Model preset key to use for the summarizer in the bedrock_repro suite "
            "(default: bedrock-nova-pro)."
        ),
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Repeat each experiment N times (default: 1). Only applied for bedrock_repro.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Parallel workers per experiment (default: 1). Only applied for bedrock_repro.",
    )
    parser.add_argument(
        "--max-local-experiments",
        type=int,
        default=1,
        help="Max concurrent local experiments (default: 1).",
    )
    parser.add_argument(
        "--max-remote-experiments",
        type=int,
        default=1,
        help="Max concurrent remote experiments (default: 1).",
    )
    parser.add_argument(
        "--log-level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
        default="INFO",
        help="Set the logging level"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("orchestrate_concurrent.log"),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )
    
    logger.info("🎬 Concurrent orchestration mode - experiments and evaluations will run simultaneously")
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Select suite
    global experiments
    if args.suite == "bedrock_repro":
        experiments = build_bedrock_repro_experiments(
            runs=args.runs,
            num_workers=args.num_workers,
            instances_slice=args.instances_slice,
            summarizer_model=args.summarizer_model,
        )
        logger.info(
            f"📋 Selected suite: bedrock_repro (experiments={len(experiments)}, runs={args.runs}, "
            f"instances_slice={args.instances_slice or 'FULL'}, num_workers={args.num_workers}, "
            f"summarizer_model={args.summarizer_model})"
        )
    else:
        experiments = LOCAL_EXPERIMENTS
        logger.info(f"📋 Selected suite: local (experiments={len(experiments)})")

    run_concurrent_orchestration(
        max_local_experiments=args.max_local_experiments,
        max_remote_experiments=args.max_remote_experiments,
    )

if __name__ == "__main__":
    main()

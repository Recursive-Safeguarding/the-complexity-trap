"""WandB hook for live experiment logging."""

from __future__ import annotations

import json
import queue
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any

from sweagent.agent.hooks.abstract import AbstractAgentHook
from sweagent.run.hooks.abstract import RunHook

if TYPE_CHECKING:
    from sweagent.types import AgentInfo, AgentRunResult, StepOutput


class WandBAgentHook(AbstractAgentHook):
    """Agent-level hook for per-step live logging to WandB."""

    def __init__(self, wandb_hook: "WandBHook"):
        self._wandb_hook = wandb_hook
        self._step = 0
        # Instance-level accumulators (reset per instance)
        self._instance_cost = 0.0
        self._instance_tokens_in = 0
        self._instance_tokens_out = 0
        self._instance_tokens_raw_input = 0
        self._instance_tokens_cached_input = 0
        self._instance_tokens_internal_reasoning = 0
        self._instance_api_calls = 0
        self._instance_inference_time = 0.0
        self._instance_execution_time = 0.0

    def on_step_done(self, *, step: "StepOutput", info: "AgentInfo"):
        if not self._wandb_hook._run:
            return

        self._step += 1
        self._wandb_hook._global_step += 1

        turn_stats = step.turn_statistics
        if turn_stats:
            step_cost = turn_stats.cost or 0
            tokens = turn_stats.tokens
            tokens_raw_input = tokens.raw_input if tokens else 0
            tokens_cached_input = tokens.cached_input if tokens else 0
            tokens_in = tokens_raw_input + tokens_cached_input
            tokens_out = tokens.output if tokens else 0
            tokens_internal_reasoning = (tokens.internal_reasoning or 0) if tokens else 0
            inference_time = turn_stats.inference_time or 0

            # Execution time is in seconds, convert to ms
            execution_time = (step.execution_time or 0) * 1000

            # Handle NaN/Inf
            import math
            if not math.isfinite(inference_time):
                inference_time = 0
            if not math.isfinite(execution_time):
                execution_time = 0

            # Update instance accumulators
            self._instance_cost += step_cost
            self._instance_tokens_in += tokens_in
            self._instance_tokens_out += tokens_out
            self._instance_tokens_raw_input += tokens_raw_input
            self._instance_tokens_cached_input += tokens_cached_input
            self._instance_tokens_internal_reasoning += tokens_internal_reasoning
            self._instance_api_calls += 1
            self._instance_inference_time += inference_time
            self._instance_execution_time += execution_time

            # Cumulative update (locked)
            with self._wandb_hook._cumulative_lock:
                cumul = self._wandb_hook._cumulative
                cumul["cost"] += step_cost
                cumul["tokens_in"] += tokens_in
                cumul["tokens_out"] += tokens_out
                cumul["tokens_raw_input"] += tokens_raw_input
                cumul["tokens_cached_input"] += tokens_cached_input
                cumul["tokens_internal_reasoning"] += tokens_internal_reasoning
                cumul["api_calls"] += 1
                cumul["inference_time"] += inference_time
                cumul["execution_time"] += execution_time
                cumul_total_in = cumul["tokens_raw_input"] + cumul["tokens_cached_input"]
                cumul_cached = cumul["tokens_cached_input"]

            # Compute cache hit rates
            step_cache_hit_rate = tokens_cached_input / tokens_in if tokens_in else 0
            instance_total_in = self._instance_tokens_raw_input + self._instance_tokens_cached_input
            instance_cache_hit_rate = self._instance_tokens_cached_input / instance_total_in if instance_total_in else 0
            cumul_cache_hit_rate = cumul_cached / cumul_total_in if cumul_total_in else 0

            # Use safe logging to prevent crashes on broken WandB connection
            self._wandb_hook._safe_log({
                # Step identifiers
                "step": self._step,
                "global_step": self._wandb_hook._global_step,

                # Per-step metrics
                "step/cost": step_cost,
                "step/tokens_in": tokens_in,
                "step/tokens_out": tokens_out,
                "step/tokens_raw_input": tokens_raw_input,
                "step/tokens_cached_input": tokens_cached_input,
                "step/tokens_internal_reasoning": tokens_internal_reasoning,
                "step/inference_time_ms": inference_time,
                "step/execution_time_ms": execution_time,
                "step/cache_hit_rate": step_cache_hit_rate,

                # Instance running totals (reset per instance)
                "instance/cost": self._instance_cost,
                "instance/tokens_in": self._instance_tokens_in,
                "instance/tokens_out": self._instance_tokens_out,
                "instance/tokens_raw_input": self._instance_tokens_raw_input,
                "instance/tokens_cached_input": self._instance_tokens_cached_input,
                "instance/tokens_internal_reasoning": self._instance_tokens_internal_reasoning,
                "instance/api_calls": self._instance_api_calls,
                "instance/inference_time_ms": self._instance_inference_time,
                "instance/execution_time_ms": self._instance_execution_time,
                "instance/cache_hit_rate": instance_cache_hit_rate,

                # Cumulative totals (across all instances)
                "cumulative/cost": cumul["cost"],
                "cumulative/tokens_in": cumul["tokens_in"],
                "cumulative/tokens_out": cumul["tokens_out"],
                "cumulative/tokens_raw_input": cumul["tokens_raw_input"],
                "cumulative/tokens_cached_input": cumul["tokens_cached_input"],
                "cumulative/tokens_internal_reasoning": cumul["tokens_internal_reasoning"],
                "cumulative/api_calls": cumul["api_calls"],
                "cumulative/inference_time_ms": cumul["inference_time"],
                "cumulative/execution_time_ms": cumul["execution_time"],
                "cumulative/cache_hit_rate": cumul_cache_hit_rate,
            })


class WandBHook(RunHook):
    """Logs metrics to WandB as each instance completes."""

    def __init__(
        self,
        project: str = "the-complexity-trap",
        entity: str | None = None,
        group: str | None = None,
        tags: list[str] | None = None,
        config: dict[str, Any] | None = None,
        name: str | None = None,
        defer_finish: bool = False,
        batch_eval_interval: int = 0,
        dataset_subset: str = "verified",
        model_name: str = "unknown",
        output_dir: Path | None = None,
        run_final_eval: bool = False,
        eval_lock_path: Path | str | None = None,
        eval_timeout: int = 900,
        eval_backend: str = "docker",  # "docker" or "sbcli"
    ):
        self._project = project
        self._entity = entity
        self._group = group
        self._name = name
        self._tags = tags or []
        self._config = config or {}
        self._defer_finish = defer_finish
        self._run = None
        self._log_failures = 0  # Counter for consecutive WandB log failures (3-strike rule)
        self._instances: list[dict[str, Any]] = []
        self._global_step = 0
        self._current_instance_id: str | None = None

        # Batch evaluation settings
        self._batch_eval_interval = batch_eval_interval
        self._dataset_subset = dataset_subset
        self._model_name = model_name
        self._output_dir = output_dir
        self._submissions: dict[str, dict] = {}  # {instance_id: prediction_data}
        self._submissions_lock = Lock()  # Thread safety for multi-worker access
        self._eval_executor: Any = None  # ThreadPoolExecutor, lazy import
        self._last_eval_count = 0

        # Final evaluation settings (run Docker eval in on_end)
        self._run_final_eval = run_final_eval
        self._eval_lock_path = Path(eval_lock_path) if eval_lock_path else None
        self._eval_timeout = eval_timeout
        self._eval_backend = eval_backend
        self._lock_file: Any = None  # File handle for fcntl lock

        self._cumulative = {
            "cost": 0.0,
            "tokens_in": 0,
            "tokens_out": 0,
            "tokens_raw_input": 0,
            "tokens_cached_input": 0,
            "tokens_internal_reasoning": 0,
            "api_calls": 0,
            "inference_time": 0.0,
            "execution_time": 0.0,
        }
        self._cumulative_lock = Lock()
        self._metrics_lock = Lock()  # Protects _instances, _totals, _exit_status_counts, _repo_counts, _turn_counts
        self._totals = {
            "n_instances": 0,
            "n_submitted": 0,
            "total_cost": 0.0,
            "total_agent_cost": 0.0,
            "total_summary_cost": 0.0,
            "total_rloop_cost": 0.0,
            "total_turns": 0,
            "total_api_calls": 0,
            "total_summary_api_calls": 0,
            "total_rloop_api_calls": 0,
            "total_raw_input_tokens": 0,
            "total_cached_input_tokens": 0,
            "total_output_tokens": 0,
            "total_internal_reasoning_tokens": 0,
            # Summary model token breakdown
            "total_summary_raw_input_tokens": 0,
            "total_summary_cached_input_tokens": 0,
            "total_summary_output_tokens": 0,
            # Visualization metrics
            "total_patch_lines": 0,
            "total_duration_ms": 0.0,
        }
        self._exit_status_counts: dict[str, int] = {}
        self._repo_counts: dict[str, int] = {}
        self._turn_counts: list[int] = []

    def _extract_repo(self, instance_id: str | None) -> str:
        """Extract repo from instance_id (e.g., 'django__django-12345' -> 'django')."""
        if not instance_id:
            return "unknown"
        try:
            # Coerce to string to handle non-string iterables
            instance_id = str(instance_id)
            if "__" in instance_id:
                repo = instance_id.split("__")[0]
            elif "-" in instance_id:
                repo = instance_id.split("-")[0]
            else:
                repo = instance_id
            return repo if repo else "unknown"
        except (TypeError, ValueError):
            return "unknown"

    def _count_patch_lines(self, submission: str | None) -> int:
        """Count non-empty lines in patch."""
        if not submission:
            return 0
        try:
            return len([line for line in submission.splitlines() if line.strip()])
        except (TypeError, AttributeError):
            return 0  # Non-string input

    def _compute_instance_duration(self, trajectory: list | None) -> float:
        """Sum execution_time from trajectory steps. Returns ms (input assumed in seconds)."""
        import math

        if not trajectory:
            return 0.0
        total = 0.0
        for step in trajectory:
            try:
                if isinstance(step, dict):
                    raw = step.get("execution_time")
                elif hasattr(step, "execution_time"):
                    raw = step.execution_time
                elif hasattr(step, "model_dump"):
                    raw = step.model_dump().get("execution_time")
                else:
                    raw = None
                if raw is not None:
                    val = float(raw)
                    if math.isfinite(val):  # Skip nan/inf
                        total += val
            except (TypeError, ValueError, AttributeError):
                pass  # Skip invalid values silently
        return total * 1000  # Convert s to ms

    def _safe_log(self, metrics: dict[str, Any]) -> bool:
        """Log to WandB, return False on failure."""
        if not self._run:
            return False
        try:
            import wandb
            wandb.log(metrics)
            self._log_failures = 0  # Reset on success
            return True
        except Exception as e:
            self._log_failures += 1
            print(f"WARNING: WandB log failed ({self._log_failures}/3): {e}")
            if self._log_failures >= 3:
                print("ERROR: WandB logging disabled after 3 consecutive failures")
                self._run = None
            return False

    def _drain_pending_eval_results(self) -> None:
        """Process queued eval results."""
        if not hasattr(self, "_pending_eval_results"):
            return
        while True:
            try:
                result = self._pending_eval_results.get_nowait()
            except queue.Empty:
                break
            if self._run:
                self._safe_log(result)

    def _shutdown_eval_executor(self) -> None:
        """Clean up executor."""
        if self._eval_executor is None:
            # Still drain any results even without executor
            self._drain_pending_eval_results()
            return
        try:
            self._eval_executor.shutdown(wait=True, cancel_futures=False)
        except KeyboardInterrupt:
            # On Ctrl+C, cancel remaining futures and re-raise
            print("Interrupted - canceling pending batch evaluations...")
            self._eval_executor.shutdown(wait=False, cancel_futures=True)
            self._eval_executor = None
            raise
        except Exception as e:
            print(f"WARNING: Batch eval executor shutdown failed: {e}")
        finally:
            self._eval_executor = None
        # Drain any remaining results after executor is done
        self._drain_pending_eval_results()

    def _categorize_exit_status(self, exit_status: str) -> str:
        """Map exit status to category. Extracts reason from "submitted (reason)" patterns."""
        if not exit_status:
            return "unknown"
        status_lower = exit_status.lower()

        if status_lower.startswith("submitted"):
            if "(" in status_lower and ")" in status_lower:
                # "submitted (exit_cost)" -> categorize "exit_cost"
                reason = status_lower.split("(")[1].split(")")[0].strip()
                return self._categorize_exit_status(reason)
            return "submitted"

        if "cost" in status_lower:
            return "exit_cost"
        if "context" in status_lower:
            return "exit_context"
        if "timeout" in status_lower or "execution_time" in status_lower:
            return "exit_timeout"
        if "format" in status_lower:
            return "exit_format"
        if "forfeit" in status_lower:
            return "exit_forfeit"
        if "api" in status_lower:
            return "exit_api"
        if "environment" in status_lower:
            return "exit_environment"
        if "command" in status_lower:
            return "exit_command"
        if "error" in status_lower:
            return "exit_error"
        return "other"

    def on_start(self):
        try:
            import wandb

            self._run = wandb.init(
                project=self._project,
                entity=self._entity,
                group=self._group,
                tags=self._tags,
                config=self._config,
                name=self._name,
            )

            # In sweep mode, wandb.init() joins an existing run so the name param
            # is ignored. Must set explicitly. See wandb docs/community.
            if self._name and self._run:
                self._run.name = self._name

            # Define custom x-axes for metric groups to ensure proper plotting
            # Step-level metrics use global_step (monotonically increasing across all instances)
            wandb.define_metric("global_step")
            wandb.define_metric("step", step_metric="global_step")
            wandb.define_metric("step/*", step_metric="global_step")
            wandb.define_metric("instance/*", step_metric="global_step")
            wandb.define_metric("cumulative/*", step_metric="global_step")

            # summary="last" for optimizer
            wandb.define_metric("n_instances")
            wandb.define_metric("n_submitted", step_metric="n_instances", summary="last")
            wandb.define_metric("submission_rate", step_metric="n_instances", summary="last")
            wandb.define_metric("cache_hit_rate", step_metric="n_instances", summary="last")
            wandb.define_metric("avg_*", step_metric="n_instances", summary="last")
            wandb.define_metric("total_*", step_metric="n_instances", summary="last")
            wandb.define_metric("exit/*", step_metric="n_instances", summary="last")
            wandb.define_metric("repo/*", step_metric="n_instances", summary="last")
            wandb.define_metric("turn_*", step_metric="n_instances", summary="last")
            wandb.define_metric("summary_cost_fraction", step_metric="n_instances", summary="last")
            wandb.define_metric("rloop_cost_fraction", step_metric="n_instances", summary="last")

            # Evaluation metrics (logged after sb-cli evaluation)
            # Use step_metric="n_instances" for x-axis alignment and summary="last"
            # to avoid inflated values from early batch evals
            wandb.define_metric("eval_pass_rate", step_metric="n_instances", summary="last")
            wandb.define_metric("eval_coverage", step_metric="n_instances", summary="last")
            wandb.define_metric("solve_rate", step_metric="n_instances", summary="last")
            wandb.define_metric("n_resolved", step_metric="n_instances", summary="last")
            wandb.define_metric("n_evaluated", step_metric="n_instances", summary="last")

        except ImportError:
            print("WARNING: wandb not installed, skipping WandB logging")
            self._run = None
        except Exception as e:
            print(f"WARNING: WandB init failed: {e}")
            # Reset to prevent crashes from partial init failure
            self._run = None

        # Initialize batch evaluation executor if enabled (only if WandB is active)
        if self._batch_eval_interval > 0 and self._run:
            from concurrent.futures import ThreadPoolExecutor
            self._eval_executor = ThreadPoolExecutor(max_workers=1)
            self._pending_eval_results: queue.Queue[dict] = queue.Queue()  # Thread-safe queue

    def on_agent_created(self, *, agent):
        if self._run:
            agent.add_hook(WandBAgentHook(self))

    def on_instance_start(self, *, index: int, env, problem_statement):
        """Store instance_id as fallback for single-instance mode.

        In multi-worker batch mode, instance_id is threaded through result.info
        to avoid race conditions. This fallback supports run_single.py usage.
        """
        self._current_instance_id = getattr(problem_statement, "id", None)

    def on_instance_completed(self, *, result: "AgentRunResult"):
        if not self._run:
            return

        info = result.info
        trajectory = result.trajectory

        def _to_dict(obj):
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            return obj if isinstance(obj, dict) else {}

        model_stats = _to_dict(info.get("model_stats", {}))
        agent_stats = _to_dict(info.get("agent_model_stats")) or model_stats
        summary_stats = _to_dict(info.get("summary_model_stats")) or {}
        rloop_stats = _to_dict(info.get("rloop_model_stats")) or {}
        agent_tokens = agent_stats.get("tokens", {})
        summary_tokens = summary_stats.get("tokens", {}) if summary_stats else {}

        agent_cost = agent_stats.get("instance_cost", 0) or 0
        summary_cost = summary_stats.get("instance_cost", 0) or 0
        rloop_cost = rloop_stats.get("instance_cost", 0) or 0
        n_turns = len(trajectory)

        # Handle exit status - can be "submitted", "submitted (exit_cost)", etc.
        exit_status = info.get("exit_status", "") or ""
        submitted = exit_status.startswith("submitted")

        # Categorize exit status for distribution tracking
        exit_category = self._categorize_exit_status(exit_status)

        # Agent token breakdown
        raw_input = agent_tokens.get("raw_input", 0) or 0
        cached_input = agent_tokens.get("cached_input", 0) or 0
        output_tokens = agent_tokens.get("output", 0) or 0
        internal_reasoning = agent_tokens.get("internal_reasoning", 0) or 0

        # Summary token breakdown
        summary_raw_input = summary_tokens.get("raw_input", 0) or 0
        summary_cached_input = summary_tokens.get("cached_input", 0) or 0
        summary_output = summary_tokens.get("output", 0) or 0

        # Cache hit rate for this instance
        total_input = raw_input + cached_input
        cache_hit_rate = cached_input / total_input if total_input else 0

        # Review score (if available from retry loop)
        review = info.get("review", {}) or {}
        review_score = review.get("accept") if isinstance(review.get("accept"), (int, float)) else None

        # Get instance_id from result.info (threaded through from run_batch, avoids race)
        # Falls back to _current_instance_id for single-instance mode compatibility
        instance_id = info.get("instance_id") or self._current_instance_id or "unknown"
        repo = self._extract_repo(instance_id)
        submission = info.get("submission")
        patch_lines = self._count_patch_lines(submission)
        instance_duration = self._compute_instance_duration(trajectory)
        tokens_per_turn = total_input / n_turns if n_turns else 0

        metrics = {
            "instance_id": instance_id,
            "repo": repo,
            "exit_status": exit_status or "unknown",
            "exit_category": exit_category,
            "submitted": submitted,
            "n_turns": n_turns,
            # Cost metrics
            "total_cost": agent_cost + summary_cost + rloop_cost,
            "agent_cost": agent_cost,
            "summary_cost": summary_cost,
            "rloop_cost": rloop_cost,
            # API call counts
            "agent_api_calls": agent_stats.get("api_calls", 0) or 0,
            "summary_api_calls": (summary_stats.get("api_calls", 0) or 0) if summary_stats else 0,
            "rloop_api_calls": (rloop_stats.get("api_calls", 0) or 0) if rloop_stats else 0,
            # Agent token breakdown
            "raw_input_tokens": raw_input,
            "cached_input_tokens": cached_input,
            "output_tokens": output_tokens,
            "internal_reasoning_tokens": internal_reasoning,
            "cache_hit_rate": cache_hit_rate,
            # Summary token breakdown
            "summary_raw_input_tokens": summary_raw_input,
            "summary_cached_input_tokens": summary_cached_input,
            "summary_output_tokens": summary_output,
            # Visualization metrics
            "patch_lines": patch_lines,
            "instance_duration_ms": instance_duration,
            "tokens_per_turn": tokens_per_turn,
            # Review score (if retry loop used)
            "review_score": review_score,
        }

        # Thread-safe update of all shared metrics state
        with self._metrics_lock:
            self._instances.append(metrics)

            # Track exit status distribution
            self._exit_status_counts[exit_category] = self._exit_status_counts.get(exit_category, 0) + 1

            # Track repo distribution
            self._repo_counts[repo] = self._repo_counts.get(repo, 0) + 1

            # Track turn counts for distribution
            self._turn_counts.append(n_turns)

            # Update running totals
            self._totals["n_instances"] += 1
            self._totals["n_submitted"] += int(submitted)
            self._totals["total_cost"] += metrics["total_cost"]
            self._totals["total_agent_cost"] += metrics["agent_cost"]
            self._totals["total_summary_cost"] += metrics["summary_cost"]
            self._totals["total_rloop_cost"] += metrics["rloop_cost"]
            self._totals["total_turns"] += n_turns
            self._totals["total_api_calls"] += metrics["agent_api_calls"]
            self._totals["total_summary_api_calls"] += metrics["summary_api_calls"]
            self._totals["total_rloop_api_calls"] += metrics["rloop_api_calls"]
            self._totals["total_raw_input_tokens"] += raw_input
            self._totals["total_cached_input_tokens"] += cached_input
            self._totals["total_output_tokens"] += output_tokens
            self._totals["total_internal_reasoning_tokens"] += internal_reasoning
            self._totals["total_summary_raw_input_tokens"] += summary_raw_input
            self._totals["total_summary_cached_input_tokens"] += summary_cached_input
            self._totals["total_summary_output_tokens"] += summary_output
            self._totals["total_patch_lines"] += patch_lines
            self._totals["total_duration_ms"] += instance_duration

            n = self._totals["n_instances"]
            total_raw = self._totals["total_raw_input_tokens"]
            total_cached = self._totals["total_cached_input_tokens"]
            total_input_all = total_raw + total_cached

            # Build exit status distribution metrics (prefixed for WandB grouping)
            exit_dist = {f"exit/{k}": v for k, v in self._exit_status_counts.items()}

            # Build repo distribution metrics (prefixed for WandB grouping)
            repo_dist = {f"repo/{k}": v for k, v in self._repo_counts.items()}

            # Turn statistics
            turn_std = 0.0
            turn_median = 0.0
            turn_min = 0
            turn_max = 0
            if self._turn_counts:
                import statistics
                turn_std = statistics.stdev(self._turn_counts) if len(self._turn_counts) > 1 else 0.0
                turn_median = statistics.median(self._turn_counts)
                turn_min = min(self._turn_counts)
                turn_max = max(self._turn_counts)

            # Cost fractions for live plotting
            total_cost = self._totals["total_cost"]
            summary_cost_fraction = self._totals["total_summary_cost"] / total_cost if total_cost else 0
            rloop_cost_fraction = self._totals["total_rloop_cost"] / total_cost if total_cost else 0

            live = {
                **self._totals,
                **exit_dist,
                **repo_dist,
                "submission_rate": self._totals["n_submitted"] / n if n else 0,
                "cache_hit_rate": total_cached / total_input_all if total_input_all else 0,
                "avg_cost": self._totals["total_cost"] / n if n else 0,
                "avg_turns": self._totals["total_turns"] / n if n else 0,
                "avg_api_calls": self._totals["total_api_calls"] / n if n else 0,
                "avg_tokens_per_turn": total_input_all / self._totals["total_turns"] if self._totals["total_turns"] else 0,
                "avg_patch_lines": self._totals["total_patch_lines"] / n if n else 0,
                "avg_duration_ms": self._totals["total_duration_ms"] / n if n else 0,
                # Cost fractions (for line plots over n_instances)
                "summary_cost_fraction": summary_cost_fraction,
                "rloop_cost_fraction": rloop_cost_fraction,
                # Turn statistics
                "turn_std": turn_std,
                "turn_median": turn_median,
                "turn_min": turn_min,
                "turn_max": turn_max,
            }

        # Log outside the lock (WandB has its own thread safety)
        self._safe_log(live)

        # Track submission for batch evaluation (thread-safe)
        if submitted and submission:
            with self._submissions_lock:
                self._submissions[instance_id] = {
                    "instance_id": instance_id,
                    "model_name_or_path": self._model_name,
                    "model_patch": submission,
                }

        # Log any pending batch eval results from background thread (thread-safe)
        self._drain_pending_eval_results()

        # Skip batch eval scheduling if WandB failed mid-call
        if not self._run:
            return

        # Trigger batch eval every N instances (if enabled and has submissions)
        with self._submissions_lock:
            n_submissions = len(self._submissions)
        if (self._batch_eval_interval > 0 and
            self._eval_executor is not None and
            n_submissions >= self._last_eval_count + self._batch_eval_interval):
            self._trigger_batch_eval(n_submissions)

    def _trigger_batch_eval(self, n_submitted: int):
        """Submit batch eval to background executor."""
        self._last_eval_count = n_submitted
        # Copy submissions under lock to avoid race conditions
        with self._submissions_lock:
            submissions_copy = list(self._submissions.values())
        if not submissions_copy:
            return
        # Pass n_instances for correct solve_rate denominator (read under lock)
        with self._metrics_lock:
            n_instances_seen = self._totals["n_instances"]
        self._eval_executor.submit(
            self._run_batch_eval, submissions_copy, n_submitted, n_instances_seen
        )

    def _run_batch_eval(self, submissions: list[dict], n_submitted: int, n_instances_seen: int):
        """Run SWE-bench eval on submissions."""
        if self._eval_backend != "docker":
            # Batch eval is best-effort live logging. Avoid accidentally running Docker
            # evaluation when the user selected sb-cli (VPS risk, quota/cost risk).
            print(f"Skipping batch eval: eval_backend={self._eval_backend}")
            return

        import tempfile
        from sweagent.utils.sbcli import run_docker_evaluation

        preds_path = None
        try:
            # write partial predictions to temp file
            with tempfile.NamedTemporaryFile(
                mode='w', suffix="_partial_preds.json", delete=False
            ) as tmp:
                preds_path = Path(tmp.name)
                json.dump(submissions, tmp, indent=2)

            run_id = f"batch_eval_{n_submitted}"
            print(f"Running batch eval ({n_submitted} submissions, {n_instances_seen} instances)...")

            success, error, results_path = run_docker_evaluation(
                preds_path=preds_path,
                subset=self._dataset_subset,
                run_id=run_id,
                output_dir=None,  # don't copy, just get report path
                max_workers=4,
                per_instance_timeout=0,  # no per-instance timeout for batch
                n_instances=len(submissions),
            )

            if not success:
                print(f"WARNING: Batch eval failed: {error}")
            elif results_path:
                self._update_wandb_with_partial_results(results_path, n_submitted, n_instances_seen)

        except Exception as e:
            print(f"WARNING: Batch eval failed (non-fatal): {e}")
        finally:
            if preds_path:
                preds_path.unlink(missing_ok=True)

    def _update_wandb_with_partial_results(self, results_path: Path, n_submitted: int, n_instances_seen: int):
        """Parse results and queue for main-thread logging (thread-safe)."""
        from sweagent.utils.sbcli import parse_eval_results

        try:
            result = parse_eval_results(results_path)
            if result is None:
                print(f"WARNING: Failed to parse batch eval results from {results_path}")
                return

            n_resolved = result["n_resolved"]
            # Use n_submitted as fallback if parse_eval_results couldn't determine n_evaluated
            n_evaluated = result["n_evaluated"] or n_submitted

            # solve_rate uses n_instances_seen as denominator (paper's definition)
            # eval_pass_rate uses n_evaluated as denominator (pass rate among evaluated)
            # eval_coverage uses n_instances_seen as denominator (submission rate)
            solve_rate = n_resolved / n_instances_seen if n_instances_seen else 0
            eval_pass_rate = n_resolved / n_evaluated if n_evaluated else 0
            eval_coverage = n_evaluated / n_instances_seen if n_instances_seen else 0

            # Queue results for main thread to log (WandB is not thread-safe)
            # Include n_instances for x-axis alignment (step_metric="n_instances")
            if hasattr(self, '_pending_eval_results'):
                self._pending_eval_results.put({
                    "n_instances": n_instances_seen,
                    "solve_rate": solve_rate,
                    "eval_pass_rate": eval_pass_rate,
                    "eval_coverage": eval_coverage,
                    "n_resolved": n_resolved,
                    "n_evaluated": n_evaluated,
                })
            print(f"Batch eval: {n_resolved}/{n_evaluated} passed, solve_rate={solve_rate:.1%} ({n_instances_seen} instances)")
        except Exception as e:
            print(f"WARNING: Failed to parse batch eval results: {e}")

    def _acquire_eval_lock(self) -> bool:
        """Acquire file lock for VPS serialization (prevents concurrent Docker evals)."""
        if not self._eval_lock_path:
            return True
        try:
            import fcntl
            self._eval_lock_path.parent.mkdir(parents=True, exist_ok=True)
            self._lock_file = open(self._eval_lock_path, 'a')
            try:
                try:
                    fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    print("Waiting for eval lock (another evaluation in progress)...")
                    fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX)
            except Exception:
                self._lock_file.close()
                self._lock_file = None
                raise
            return True
        except Exception as e:
            print(f"WARNING: Failed to acquire eval lock: {e}")
            return False

    def _release_eval_lock(self) -> None:
        """Release file lock."""
        if self._lock_file:
            try:
                import fcntl
                fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
                self._lock_file.close()
            except Exception:
                pass
            finally:
                self._lock_file = None

    _EVAL_MAX_WORKERS = 4  # workers for final Docker evaluation

    def _execute_final_eval(self, run_path: str | None = None) -> bool:
        """Run SWE-bench evaluation on all submissions. Updates WandB via API if run_path given."""
        from sweagent.utils.sbcli import run_docker_evaluation

        if not self._submissions:
            print("No submissions to evaluate")
            return False

        if not self._output_dir:
            print("WARNING: output_dir not set, skipping final evaluation")
            return False

        if not self._acquire_eval_lock():
            print("WARNING: Could not acquire eval lock, skipping evaluation")
            return False

        try:
            preds_path = self._output_dir / "preds.json"
            preds_path.write_text(json.dumps(list(self._submissions.values()), indent=2))

            print(f"Running final evaluation ({len(self._submissions)} submissions)...")

            success, error, results_path = run_docker_evaluation(
                preds_path=preds_path,
                subset=self._dataset_subset,
                run_id=self._output_dir.name,
                output_dir=self._output_dir,
                max_workers=self._EVAL_MAX_WORKERS,
                per_instance_timeout=self._eval_timeout,
                log_file=self._output_dir / "evaluation.log",
                n_instances=len(self._submissions),
            )

            if not success:
                print(f"WARNING: Final eval failed: {error}")
                return False

            self._update_finished_run_via_api(results_path, run_path)
            print(f"Final evaluation complete, results saved to {results_path}")
            return True

        except Exception as e:
            print(f"WARNING: Final evaluation failed: {e}")
            return False
        finally:
            self._release_eval_lock()

    def _update_finished_run_via_api(self, results_path: Path, run_path: str | None) -> bool:
        """Update finished WandB run with eval metrics via API."""
        from sweagent.utils.sbcli import parse_eval_results

        if not run_path:
            print("WARNING: No run_path provided, skipping WandB update")
            return False

        try:
            import wandb

            result = parse_eval_results(results_path)
            if result is None:
                print(f"WARNING: Failed to parse results from {results_path}")
                return False

            n_resolved = result["n_resolved"]
            # Read totals under lock for thread safety
            with self._metrics_lock:
                n_submitted_fallback = self._totals.get("n_submitted", 0)
                n_instances = self._totals.get("n_instances", 0)
            n_evaluated = result["n_evaluated"] or n_submitted_fallback

            eval_pass_rate = n_resolved / n_evaluated if n_evaluated else 0
            eval_coverage = n_evaluated / n_instances if n_instances else 0
            solve_rate = n_resolved / n_instances if n_instances else 0

            api = wandb.Api()
            run = api.run(run_path)
            run.summary.update({
                "n_resolved": n_resolved,
                "n_evaluated": n_evaluated,
                "eval_pass_rate": eval_pass_rate,
                "eval_coverage": eval_coverage,
                "solve_rate": solve_rate,
                "eval_complete": True,
            })
            run.update()

            print(f"WandB updated via API: {n_resolved}/{n_evaluated} passed ({eval_pass_rate:.1%}), solve_rate: {solve_rate:.1%}")
            return True

        except Exception as e:
            print(f"WARNING: Failed to update WandB via API: {e}")
            return False

    def _execute_sbcli_eval(self, run_path: str | None = None) -> bool:
        """Submit evaluation to sb-cli cloud service.

        Falls back to Docker on failure, except for guardrail-triggered errors (which
        indicate a likely misconfiguration and should not be "fixed" by running a
        potentially expensive Docker eval).
        """
        from sweagent.utils.sbcli import run_sbcli_evaluation

        if not self._submissions:
            print("No submissions to evaluate")
            return False

        if not self._output_dir:
            print("WARNING: output_dir not set, skipping sb-cli evaluation")
            return False

        # write predictions file
        preds_path = self._output_dir / "preds.json"
        preds_path.write_text(json.dumps(list(self._submissions.values()), indent=2))

        print(f"Submitting to sb-cli ({len(self._submissions)} predictions)...")
        success, error, results_path = run_sbcli_evaluation(
            preds_path=preds_path,
            subset=self._dataset_subset,
            run_id=self._output_dir.name,
            output_dir=self._output_dir,
        )

        if not success:
            if isinstance(error, str) and error.startswith("guardrail:"):
                print(f"WARNING: sb-cli {error}")
                print("🛑 Guardrail triggered; not falling back to Docker.")
                return False
            print(f"WARNING: sb-cli {error}, falling back to Docker")
            return self._execute_final_eval(run_path=run_path)

        self._update_finished_run_via_api(results_path, run_path)
        print(f"sb-cli evaluation complete, results saved to {results_path}")
        return True

    def on_end(self):
        # Always shutdown executor first to avoid thread leak (even if WandB failed)
        self._shutdown_eval_executor()

        # capture eval decision before any WandB operations might fail
        should_run_final_eval = self._run_final_eval and self._submissions
        run_path = None

        if not self._run:
            # no WandB run, but still run final eval if requested
            if should_run_final_eval:
                if self._eval_backend == "sbcli":
                    self._execute_sbcli_eval(run_path=None)
                else:
                    self._execute_final_eval(run_path=None)
            return

        try:
            import wandb
            import statistics

            # Snapshot all shared state under lock for thread safety
            with self._metrics_lock:
                totals_copy = dict(self._totals)
                exit_status_copy = dict(self._exit_status_counts)
                repo_counts_copy = dict(self._repo_counts)
                turn_counts_copy = list(self._turn_counts)
                instances_copy = list(self._instances)

            n = totals_copy["n_instances"]
            raw = totals_copy["total_raw_input_tokens"]
            cached = totals_copy["total_cached_input_tokens"]
            total_input = raw + cached

            total_all_api_calls = (
                totals_copy["total_api_calls"] +
                totals_copy["total_summary_api_calls"] +
                totals_copy["total_rloop_api_calls"]
            )

            # Exit status distribution for final summary
            exit_dist = {f"exit/{k}": v for k, v in exit_status_copy.items()}

            # Repo distribution for final summary
            repo_dist = {f"repo/{k}": v for k, v in repo_counts_copy.items()}

            # Turn statistics
            turn_std = 0.0
            turn_median = 0.0
            turn_min = 0
            turn_max = 0
            if turn_counts_copy:
                turn_std = statistics.stdev(turn_counts_copy) if len(turn_counts_copy) > 1 else 0.0
                turn_median = statistics.median(turn_counts_copy)
                turn_min = min(turn_counts_copy)
                turn_max = max(turn_counts_copy)

            final = {
                **totals_copy,
                **exit_dist,
                **repo_dist,
                "submission_rate": totals_copy["n_submitted"] / n if n else 0,
                "cache_hit_rate": cached / total_input if total_input else 0,
                "avg_cost": totals_copy["total_cost"] / n if n else 0,
                "avg_turns": totals_copy["total_turns"] / n if n else 0,
                "avg_api_calls": totals_copy["total_api_calls"] / n if n else 0,
                "avg_tokens_per_turn": total_input / totals_copy["total_turns"] if totals_copy["total_turns"] else 0,
                "avg_patch_lines": totals_copy["total_patch_lines"] / n if n else 0,
                "avg_duration_ms": totals_copy["total_duration_ms"] / n if n else 0,
                "summary_cost_fraction": (
                    totals_copy["total_summary_cost"] / totals_copy["total_cost"]
                    if totals_copy["total_cost"] else 0
                ),
                "rloop_cost_fraction": (
                    totals_copy["total_rloop_cost"] / totals_copy["total_cost"]
                    if totals_copy["total_cost"] else 0
                ),
                "summary_api_fraction": (
                    totals_copy["total_summary_api_calls"] / total_all_api_calls
                    if total_all_api_calls else 0
                ),
                "rloop_api_fraction": (
                    totals_copy["total_rloop_api_calls"] / total_all_api_calls
                    if total_all_api_calls else 0
                ),
                # Turn statistics
                "turn_std": turn_std,
                "turn_median": turn_median,
                "turn_min": turn_min,
                "turn_max": turn_max,
            }

            try:
                wandb.summary.update(final)
            except Exception as e:
                print(f"WARNING: WandB summary update failed: {e}")
                self._run = None  # Consistent with _safe_log() pattern

            # Log turn distribution histogram (using snapshot from above)
            if turn_counts_copy and self._run:
                try:
                    wandb.log({"turn_distribution": wandb.Histogram(turn_counts_copy)})
                except Exception as e:
                    print(f"WARNING: WandB histogram logging failed: {e}")
                    self._run = None

            if instances_copy and self._run:
                try:
                    cols = list(instances_copy[0].keys())
                    table = wandb.Table(
                        columns=cols,
                        data=[[row.get(c) for c in cols] for row in instances_copy],
                    )
                    wandb.log({"instances": table})
                except Exception as e:
                    print(f"WARNING: WandB table logging failed: {e}")
                    self._run = None

            # capture run path before finishing (run.path can be tuple in some versions)
            if self._run and hasattr(self._run, 'entity') and self._run.entity:
                run_path = f"{self._run.entity}/{self._run.project}/{self._run.id}"

            # finish before eval to avoid connection timeout
            if not self._defer_finish:
                try:
                    wandb.finish()
                except Exception as e:
                    print(f"WARNING: WandB finish failed: {e}")
                self._run = None

            if should_run_final_eval:
                if self._eval_backend == "sbcli":
                    self._execute_sbcli_eval(run_path=run_path)
                else:
                    self._execute_final_eval(run_path=run_path)

        except Exception as e:
            print(f"WARNING: WandB on_end failed: {e}")
            self._run = None

    def finalize(self):
        """Finish the WandB run. Call this after update_with_evaluation_results() if defer_finish=True."""
        if not self._run:
            return
        try:
            import wandb
            wandb.finish()
        except Exception as e:
            print(f"WARNING: WandB finalize failed: {e}")
        finally:
            self._run = None

    def update_with_evaluation_results(self, results_path: Path) -> bool:
        """Parse results.json and log eval metrics to WandB."""
        from sweagent.utils.sbcli import parse_eval_results

        if not results_path.exists():
            print(f"WARNING: Results file not found: {results_path}")
            return False

        result = parse_eval_results(results_path)
        if result is None:
            print(f"WARNING: Failed to parse results from {results_path}")
            return False

        n_resolved = result["n_resolved"]
        # Read totals under lock for thread safety
        with self._metrics_lock:
            n_submitted_fallback = self._totals.get("n_submitted", 0)
            n_instances = self._totals.get("n_instances", 0)
        n_evaluated = result["n_evaluated"] or n_submitted_fallback

        eval_pass_rate = n_resolved / n_evaluated if n_evaluated else 0
        eval_coverage = n_evaluated / n_instances if n_instances else 0
        solve_rate = n_resolved / n_instances if n_instances else 0

        if not self._run:
            print(f"INFO: No active WandB run, but evaluation results: {n_resolved}/{n_evaluated} resolved")
            return False

        try:
            import wandb

            wandb.summary.update({
                "n_resolved": n_resolved,
                "n_evaluated": n_evaluated,
                "eval_pass_rate": eval_pass_rate,
                "eval_coverage": eval_coverage,
                "solve_rate": solve_rate,
                "eval_complete": True,
            })
            print(f"WandB updated: {n_resolved}/{n_evaluated} passed ({eval_pass_rate:.1%}), coverage: {eval_coverage:.1%}, solve_rate: {solve_rate:.1%}")
            return True
        except Exception as e:
            print(f"WARNING: Failed to update WandB with evaluation results: {e}")
            self._run = None
            return False

"""WandB hook for live experiment logging."""

from __future__ import annotations

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

            # Update global cumulative accumulators
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

            # Compute cache hit rates
            step_cache_hit_rate = tokens_cached_input / tokens_in if tokens_in else 0
            instance_total_in = self._instance_tokens_raw_input + self._instance_tokens_cached_input
            instance_cache_hit_rate = self._instance_tokens_cached_input / instance_total_in if instance_total_in else 0
            cumul_total_in = cumul["tokens_raw_input"] + cumul["tokens_cached_input"]
            cumul_cache_hit_rate = cumul["tokens_cached_input"] / cumul_total_in if cumul_total_in else 0

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
        project: str = "complexity-trap",
        entity: str | None = None,
        group: str | None = None,
        tags: list[str] | None = None,
        config: dict[str, Any] | None = None,
        name: str | None = None,
    ):
        self._project = project
        self._entity = entity
        self._group = group
        self._name = name
        self._tags = tags or []
        self._config = config or {}
        self._run = None
        self._instances: list[dict[str, Any]] = []
        self._global_step = 0
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
        }
        self._exit_status_counts: dict[str, int] = {}

    def _safe_log(self, metrics: dict[str, Any]) -> bool:
        """Safely log metrics to WandB, handling connection failures gracefully.

        Returns True if logging succeeded, False otherwise.
        On failure, disables further WandB logging by setting _run to None.
        """
        if not self._run:
            return False
        try:
            import wandb
            wandb.log(metrics)
            return True
        except Exception as e:
            print(f"WARNING: WandB log failed: {e}")
            self._run = None
            return False

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

            # Instance-level summary metrics use n_instances as x-axis
            wandb.define_metric("n_instances")
            wandb.define_metric("n_submitted", step_metric="n_instances")
            wandb.define_metric("submission_rate", step_metric="n_instances")
            wandb.define_metric("cache_hit_rate", step_metric="n_instances")
            wandb.define_metric("avg_*", step_metric="n_instances")
            wandb.define_metric("total_*", step_metric="n_instances")
            wandb.define_metric("exit/*", step_metric="n_instances")

        except ImportError:
            print("WARNING: wandb not installed, skipping WandB logging")
            self._run = None
        except Exception as e:
            print(f"WARNING: WandB init failed: {e}")
            # CRITICAL: Reset _run to None to prevent subsequent wandb.log() calls
            # from crashing the run. This can happen if wandb.init() partially
            # succeeded but define_metric() or save() failed (e.g., broken pipe).
            self._run = None

    def on_agent_created(self, *, agent):
        if self._run:
            agent.add_hook(WandBAgentHook(self))

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

        metrics = {
            "instance_id": info.get("instance_id", "unknown"),
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
            "summary_api_calls": summary_stats.get("api_calls", 0) if summary_stats else 0,
            "rloop_api_calls": rloop_stats.get("api_calls", 0) if rloop_stats else 0,
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
            # Review score (if retry loop used)
            "review_score": review_score,
        }
        self._instances.append(metrics)

        # Track exit status distribution
        self._exit_status_counts[exit_category] = self._exit_status_counts.get(exit_category, 0) + 1

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

        n = self._totals["n_instances"]
        total_raw = self._totals["total_raw_input_tokens"]
        total_cached = self._totals["total_cached_input_tokens"]
        total_input_all = total_raw + total_cached

        # Build exit status distribution metrics (prefixed for WandB grouping)
        exit_dist = {f"exit/{k}": v for k, v in self._exit_status_counts.items()}

        live = {
            **self._totals,
            **exit_dist,
            "submission_rate": self._totals["n_submitted"] / n if n else 0,
            "cache_hit_rate": total_cached / total_input_all if total_input_all else 0,
            "avg_cost": self._totals["total_cost"] / n if n else 0,
            "avg_turns": self._totals["total_turns"] / n if n else 0,
            "avg_api_calls": self._totals["total_api_calls"] / n if n else 0,
            "avg_tokens_per_turn": total_input_all / self._totals["total_turns"] if self._totals["total_turns"] else 0,
        }
        self._safe_log(live)

    def on_end(self):
        if not self._run:
            return

        try:
            import wandb

            n = self._totals["n_instances"]
            raw = self._totals["total_raw_input_tokens"]
            cached = self._totals["total_cached_input_tokens"]
            total_input = raw + cached

            total_all_api_calls = (
                self._totals["total_api_calls"] +
                self._totals["total_summary_api_calls"] +
                self._totals["total_rloop_api_calls"]
            )

            # Exit status distribution for final summary
            exit_dist = {f"exit/{k}": v for k, v in self._exit_status_counts.items()}

            final = {
                **self._totals,
                **exit_dist,
                "submission_rate": self._totals["n_submitted"] / n if n else 0,
                "cache_hit_rate": cached / total_input if total_input else 0,
                "avg_cost": self._totals["total_cost"] / n if n else 0,
                "avg_turns": self._totals["total_turns"] / n if n else 0,
                "avg_api_calls": self._totals["total_api_calls"] / n if n else 0,
                "avg_tokens_per_turn": total_input / self._totals["total_turns"] if self._totals["total_turns"] else 0,
                "summary_cost_fraction": (
                    self._totals["total_summary_cost"] / self._totals["total_cost"]
                    if self._totals["total_cost"] else 0
                ),
                "rloop_cost_fraction": (
                    self._totals["total_rloop_cost"] / self._totals["total_cost"]
                    if self._totals["total_cost"] else 0
                ),
                "summary_api_fraction": (
                    self._totals["total_summary_api_calls"] / total_all_api_calls
                    if total_all_api_calls else 0
                ),
                "rloop_api_fraction": (
                    self._totals["total_rloop_api_calls"] / total_all_api_calls
                    if total_all_api_calls else 0
                ),
            }

            try:
                wandb.summary.update(final)
            except Exception as e:
                print(f"WARNING: WandB summary update failed: {e}")

            if self._instances and self._run:
                try:
                    cols = list(self._instances[0].keys())
                    table = wandb.Table(
                        columns=cols,
                        data=[[row.get(c) for c in cols] for row in self._instances],
                    )
                    wandb.log({"instances": table})
                except Exception as e:
                    print(f"WARNING: WandB table logging failed: {e}")

            try:
                wandb.finish()
            except Exception as e:
                print(f"WARNING: WandB finish failed: {e}")

        except Exception as e:
            print(f"WARNING: WandB on_end failed: {e}")
            self._run = None

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
        self._instance_cost = 0.0
        self._instance_tokens_in = 0
        self._instance_tokens_out = 0
        self._instance_api_calls = 0

    def on_step_done(self, *, step: "StepOutput", info: "AgentInfo"):
        if not self._wandb_hook._run:
            return

        import wandb

        self._step += 1
        self._wandb_hook._global_step += 1

        turn_stats = step.turn_statistics
        if turn_stats:
            step_cost = turn_stats.cost or 0
            tokens = turn_stats.tokens
            tokens_in = (tokens.raw_input + tokens.cached_input) if tokens else 0
            tokens_out = tokens.output if tokens else 0
            inference_time = turn_stats.inference_time or 0

            self._instance_cost += step_cost
            self._instance_tokens_in += tokens_in
            self._instance_tokens_out += tokens_out
            self._instance_api_calls += 1

            self._wandb_hook._cumulative["cost"] += step_cost
            self._wandb_hook._cumulative["tokens_in"] += tokens_in
            self._wandb_hook._cumulative["tokens_out"] += tokens_out
            self._wandb_hook._cumulative["api_calls"] += 1

            wandb.log({
                "step": self._step,
                "global_step": self._wandb_hook._global_step,
                "step/cost": step_cost,
                "step/tokens_in": tokens_in,
                "step/tokens_out": tokens_out,
                "step/inference_time_ms": inference_time,
                "instance/cost": self._instance_cost,
                "instance/tokens_in": self._instance_tokens_in,
                "instance/tokens_out": self._instance_tokens_out,
                "instance/api_calls": self._instance_api_calls,
                "cumulative/cost": self._wandb_hook._cumulative["cost"],
                "cumulative/tokens_in": self._wandb_hook._cumulative["tokens_in"],
                "cumulative/tokens_out": self._wandb_hook._cumulative["tokens_out"],
                "cumulative/api_calls": self._wandb_hook._cumulative["api_calls"],
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
    ):
        self._project = project
        self._entity = entity
        self._group = group
        self._tags = tags or []
        self._config = config or {}
        self._run = None
        self._instances: list[dict[str, Any]] = []
        self._global_step = 0
        self._cumulative = {
            "cost": 0.0,
            "tokens_in": 0,
            "tokens_out": 0,
            "api_calls": 0,
        }
        self._totals = {
            "n_instances": 0,
            "n_submitted": 0,
            "total_cost": 0.0,
            "total_agent_cost": 0.0,
            "total_summary_cost": 0.0,
            "total_turns": 0,
            "total_api_calls": 0,
            "total_raw_input_tokens": 0,
            "total_cached_input_tokens": 0,
            "total_output_tokens": 0,
        }

    def on_start(self):
        try:
            import wandb

            self._run = wandb.init(
                project=self._project,
                entity=self._entity,
                group=self._group,
                tags=self._tags,
                config=self._config,
            )
        except ImportError:
            print("WARNING: wandb not installed, skipping WandB logging")
        except Exception as e:
            print(f"WARNING: WandB init failed: {e}")

    def on_agent_created(self, *, agent):
        if self._run:
            agent.add_hook(WandBAgentHook(self))

    def on_instance_completed(self, *, result: AgentRunResult):
        if not self._run:
            return

        import wandb

        info = result.info
        trajectory = result.trajectory

        def _to_dict(obj):
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            return obj if isinstance(obj, dict) else {}

        model_stats = _to_dict(info.get("model_stats", {}))
        agent_stats = _to_dict(info.get("agent_model_stats")) or model_stats
        summary_stats = _to_dict(info.get("summary_model_stats")) or {}
        agent_tokens = agent_stats.get("tokens", {})

        agent_cost = agent_stats.get("instance_cost", 0) or 0
        summary_cost = summary_stats.get("instance_cost", 0) or 0
        n_turns = len(trajectory)
        submitted = info.get("exit_status") == "submitted"

        metrics = {
            "instance_id": info.get("instance_id", "unknown"),
            "exit_status": info.get("exit_status", "unknown"),
            "submitted": submitted,
            "n_turns": n_turns,
            "total_cost": agent_cost + summary_cost,
            "agent_cost": agent_cost,
            "summary_cost": summary_cost,
            "agent_api_calls": agent_stats.get("api_calls", 0) or 0,
            "summary_api_calls": summary_stats.get("api_calls", 0) if summary_stats else 0,
            "raw_input_tokens": agent_tokens.get("raw_input", 0) or 0,
            "cached_input_tokens": agent_tokens.get("cached_input", 0) or 0,
            "output_tokens": agent_tokens.get("output", 0) or 0,
        }
        self._instances.append(metrics)

        self._totals["n_instances"] += 1
        self._totals["n_submitted"] += int(submitted)
        self._totals["total_cost"] += metrics["total_cost"]
        self._totals["total_agent_cost"] += metrics["agent_cost"]
        self._totals["total_summary_cost"] += metrics["summary_cost"]
        self._totals["total_turns"] += n_turns
        self._totals["total_api_calls"] += metrics["agent_api_calls"]
        self._totals["total_raw_input_tokens"] += metrics["raw_input_tokens"]
        self._totals["total_cached_input_tokens"] += metrics["cached_input_tokens"]
        self._totals["total_output_tokens"] += metrics["output_tokens"]

        n = self._totals["n_instances"]
        live = {
            **self._totals,
            "submission_rate": self._totals["n_submitted"] / n if n else 0,
            "avg_cost": self._totals["total_cost"] / n if n else 0,
            "avg_turns": self._totals["total_turns"] / n if n else 0,
        }
        wandb.log(live)

    def on_end(self):
        if not self._run:
            return

        import wandb

        n = self._totals["n_instances"]
        raw = self._totals["total_raw_input_tokens"]
        cached = self._totals["total_cached_input_tokens"]

        final = {
            **self._totals,
            "submission_rate": self._totals["n_submitted"] / n if n else 0,
            "avg_cost": self._totals["total_cost"] / n if n else 0,
            "avg_turns": self._totals["total_turns"] / n if n else 0,
            "avg_api_calls": self._totals["total_api_calls"] / n if n else 0,
            "cache_hit_rate": cached / (raw + cached) if (raw + cached) else 0,
            "summary_cost_fraction": (
                self._totals["total_summary_cost"] / self._totals["total_cost"]
                if self._totals["total_cost"] else 0
            ),
        }
        wandb.summary.update(final)

        if self._instances:
            cols = list(self._instances[0].keys())
            table = wandb.Table(
                columns=cols,
                data=[[row.get(c) for c in cols] for row in self._instances],
            )
            wandb.log({"instances": table})

        wandb.finish()

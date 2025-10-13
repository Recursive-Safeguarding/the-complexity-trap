"""This file has types/dataclass definitions that are used in the SWE agent
for exchanging data between different modules/functions/classes.
They oftentimes cannot be defined in the same file where they are used
because of circular dependencies.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel as PydanticBaseModel
from typing_extensions import TypedDict

class StepOutput(PydanticBaseModel):
    thought: str = ""
    action: str = ""
    output: str = ""
    observation: str = ""
    execution_time: float = 0.0
    done: bool = False
    exit_status: int | str | None = None
    submission: str | None = None
    state: dict[str, str] = {}
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_ids: list[str] | None = None
    """State of the environment at the end of the step"""
    turn_statistics: TurnStatistics | None = None
    extra_info: dict[str, Any] = {}

    def to_template_format_dict(self) -> dict[str, str | int | float | bool | None]:
        """Used for formatting (error) prompt templates"""
        out = {}
        for k, v in self.model_dump().items():
            if k in ("tool_calls", "tool_call_ids", "state"):
                continue
            out[k] = v
        out |= self.state
        return out


class TrajectoryStep(TypedDict):
    action: str
    observation: str
    response: str
    state: dict[str, str]
    thought: str
    execution_time: float
    messages: list[dict[str, Any]]
    extra_info: dict[str, Any]
    turn_statistics: TurnStatistics | None


# required fields go here
class _HistoryItem(TypedDict):
    role: str
    content: str | list[dict[str, Any]]
    message_type: Literal["thought", "action", "observation"]


# see _HistoryItem for required fields
class HistoryItem(_HistoryItem, total=False):
    agent: str
    is_demo: bool
    thought: str
    action: str | None
    tool_calls: list[dict[str, str]] | None
    tool_call_ids: list[str] | None
    tags: list[str]
    cache_control: dict[str, Any] | None
    """HistoryProcessors can add these tags to enable special processing"""


History = list[HistoryItem]
Turns = list[list[HistoryItem]]
Trajectory = list[TrajectoryStep]


class GlobalStats(PydanticBaseModel):
    """This class tracks usage numbers (costs etc.) across all instances."""

    total_cost: float = 0
    """Cumulative cost for all instances so far"""

    last_query_timestamp: float = 0
    """Timestamp of the last query. Currently only used with API models."""

class TokenStats(PydanticBaseModel):
    """Token usage statistics."""
    
    raw_input: int = 0
    cached_input: int = 0
    output: int = 0
    internal_reasoning: Optional[int] = None

class TurnStatistics(PydanticBaseModel):
    """Statistics about a single turn."""

    cost: float
    """Cost of the turn in USD."""
    tokens: TokenStats | None = None
    internal_reasoning: Optional[str] = None
    inference_time: float
    """Inference time in milliseconds from model completion call start to completion."""

class SummaryMetadata(PydanticBaseModel):
    """Metadata about a LLM-generated trajectory summary."""
    
    summary: str
    context: list[dict[str, Any]]
    statistics: TurnStatistics | None = None
    

class InstanceStats(PydanticBaseModel):
    """This object tracks usage numbers (costs etc.) for a single instance."""

    instance_cost: float = 0
    tokens: TokenStats = TokenStats()
    api_calls: int = 0

    # Legacy properties for backward compatibility
    @property
    def tokens_sent(self) -> int:
        """Legacy property: total input tokens sent"""
        return self.tokens.cached_input + self.tokens.raw_input
    
    @property
    def tokens_received(self) -> int:
        """Legacy property: total output tokens received"""
        return self.tokens.output

    def __add__(self, other: InstanceStats) -> InstanceStats:
        # Handle regular fields
        result = {
            'instance_cost': self.instance_cost + other.instance_cost,
            'api_calls': self.api_calls + other.api_calls,
            'tokens': TokenStats(
                raw_input=self.tokens.raw_input + other.tokens.raw_input,
                cached_input=self.tokens.cached_input + other.tokens.cached_input,
                output=self.tokens.output + other.tokens.output
            )
        }
        return InstanceStats(**result)

    def __sub__(self, other: InstanceStats) -> InstanceStats:
        # Handle regular fields
        result = {
            'instance_cost': self.instance_cost - other.instance_cost,
            'api_calls': self.api_calls - other.api_calls,
            'tokens': TokenStats(
                raw_input=self.tokens.raw_input - other.tokens.raw_input,
                cached_input=self.tokens.cached_input - other.tokens.cached_input,
                output=self.tokens.output - other.tokens.output
            )
        }
        return InstanceStats(**result)

# todo: Make this actually have the dataclasses instead of dict versions
class AgentInfo(TypedDict, total=False):
    # same as `APIStats` from models.py
    # TODO better to move the stats data classes into the types aswell, mixing them into the model file is weird.
    model_stats: InstanceStats
    agent_model_stats: InstanceStats
    summary_model_stats: InstanceStats | None
    exit_status: str | None
    submission: str | None
    # same as `ReviewerResult`
    review: dict[str, Any]
    edited_files30: str
    edited_files50: str
    edited_files70: str
    # only if summarizer is used
    summarizer: dict
    swe_agent_hash: str
    swe_agent_version: str
    swe_rex_version: str
    swe_rex_hash: str

class AgentRunResult(PydanticBaseModel):
    info: AgentInfo
    trajectory: Trajectory

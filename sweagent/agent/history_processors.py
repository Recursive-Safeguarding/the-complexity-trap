from __future__ import annotations

import copy
import re
from abc import abstractmethod
from typing import Annotated, List, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator, PrivateAttr

from sweagent.agent.models import AbstractModel
from sweagent.types import History, HistoryItem, Turns, SummaryMetadata
from sweagent.utils.log import get_logger


class AbstractHistoryProcessor(Protocol):
    @abstractmethod
    def __call__(self, history: History) -> History:
        raise NotImplementedError


# Utility functions
# -----------------


def _get_content_text(entry: HistoryItem) -> str:
    if isinstance(entry["content"], str):
        return entry["content"]
    assert len(entry["content"]) == 1, "Expected single message in content"
    return entry["content"][0]["text"]


def _set_content_text(entry: HistoryItem, text: str) -> None:
    if isinstance(entry["content"], str):
        entry["content"] = text
    else:
        assert len(entry["content"]) == 1, "Expected single message in content"
        entry["content"][0]["text"] = text


def _clear_cache_control(entry: HistoryItem) -> None:
    if isinstance(entry["content"], list):
        assert len(entry["content"]) == 1, "Expected single message in content"
        entry["content"][0].pop("cache_control", None)
    entry.pop("cache_control", None)


def _set_cache_control(entry: HistoryItem) -> None:
    if not isinstance(entry["content"], list):
        entry["content"] = [  # type: ignore
            {
                "type": "text",
                "text": _get_content_text(entry),
                "cache_control": {"type": "ephemeral"},
            }
        ]
    else:
        entry["content"][0]["cache_control"] = {"type": "ephemeral"}
    if entry["role"] == "tool":
        # Workaround for weird bug
        entry["content"][0].pop("cache_control", None)
        entry["cache_control"] = {"type": "ephemeral"}

def _should_trigger_limit_aware(
    *,
    history: History,
    context_model: AbstractModel | None,
    fraction: float,
    min_tokens: int,
    logger,
    label: str,
) -> bool:
    """Return True when limit-aware compaction should trigger.

    If we cannot confidently estimate tokens/context window, we fall back to
    the existing behavior (trigger=True).
    """
    if context_model is None:
        return True

    context_window = getattr(context_model, "model_context_window", None)
    if context_window is None or context_window <= 0:
        return True

    try:
        tokens = context_model.count_tokens(history)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug(f"{label}: token estimation failed; falling back to default trigger: {exc}")
        return True

    threshold = max(min_tokens, int(context_window * fraction))
    if tokens < threshold:
        logger.debug(
            f"{label}: skipping compaction (tokens={tokens:,} < threshold={threshold:,}, "
            f"context_window={context_window:,}, fraction={fraction:.2f})"
        )
        return False

    logger.debug(
        f"{label}: triggering compaction (tokens={tokens:,} >= threshold={threshold:,}, "
        f"context_window={context_window:,}, fraction={fraction:.2f})"
    )
    return True


# History processors
# ------------------


class DefaultHistoryProcessor(BaseModel):
    type: Literal["default"] = "default"
    """Do not change. Used for (de)serialization."""

    # pydantic config
    model_config = ConfigDict(extra="forbid")

    def __call__(self, history: History) -> History:
        return history

class LastNObservations(BaseModel):
    """Keep the last n observations or remove tagged observations."""

    n: int
    """Number of observations to keep."""

    polling: int = 1
    """How many steps to keep between updating the number of observations to keep.
    This is useful for caching, as we want to remove more and more messages, but every
    time we change the history, we need to cache everything again.
    Effectively, we will now keep between `n` and `n+polling` observations.
    """

    always_remove_output_for_tags: set[str] = {"remove_output"}
    """Any observation with a `tags` field containing one of these strings will be elided,
    even if it is one of the last n observations.
    """

    always_keep_output_for_tags: set[str] = {"keep_output"}
    """Any observation with a `tags` field containing one of these strings will be kept,
    even if it is not one of the last n observations.
    """

    enable_limit_aware_trigger: bool = False
    """Only apply masking when near the model context limit."""

    limit_trigger_fraction: float = 0.9
    """Trigger when estimated tokens exceed this fraction of the context window."""

    limit_trigger_min_tokens: int = 0
    """Absolute minimum token threshold for triggering (takes precedence)."""

    type: Literal["last_n_observations"] = "last_n_observations"
    """Do not change. Used for (de)serialization."""

    _context_model: AbstractModel | None = PrivateAttr(default=None)
    _logger = get_logger("swea-hp", emoji="🧠")

    # pydantic config
    model_config = ConfigDict(extra="forbid")

    @field_validator("n")
    def validate_n(cls, n: int) -> int:
        if n <= 0:
            msg = "n must be a positive integer"
            raise ValueError(msg)
        return n

    @field_validator("limit_trigger_fraction")
    def validate_limit_trigger_fraction(cls, fraction: float) -> float:
        if not (0 < fraction <= 1):
            msg = "limit_trigger_fraction must be in (0, 1]"
            raise ValueError(msg)
        return fraction

    @field_validator("limit_trigger_min_tokens")
    def validate_limit_trigger_min_tokens(cls, min_tokens: int) -> int:
        if min_tokens < 0:
            msg = "limit_trigger_min_tokens must be non-negative"
            raise ValueError(msg)
        return min_tokens

    def set_context_model(self, model: AbstractModel) -> None:
        self._context_model = model

    def _should_mask(self, history: History) -> bool:
        if not self.enable_limit_aware_trigger:
            return True
        return _should_trigger_limit_aware(
            history=history,
            context_model=self._context_model,
            fraction=self.limit_trigger_fraction,
            min_tokens=self.limit_trigger_min_tokens,
            logger=self._logger,
            label="LastNObservations",
        )

    def _get_omit_indices(self, history: History) -> list[int]:
        observation_indices = [
            idx
            for idx, entry in enumerate(history)
            if entry["message_type"] == "observation" and not entry.get("is_demo", False)
        ]
        last_removed_idx = max(0, (len(observation_indices) // self.polling) * self.polling - self.n)
        # Note: We never remove the first observation, as it is the instance template
        return observation_indices[1:last_removed_idx]

    def __call__(self, history: History) -> History:
        if not self._should_mask(history):
            return history
        new_history = []
        omit_content_idxs = self._get_omit_indices(history)
        for idx, entry in enumerate(history):
            # Normalize tags to a comparable set of strings. Some processors
            # store dict tags like {"type": "summary"} which are unhashable.
            raw_tags = entry.get("tags", [])
            tags: set[str] = set()
            for tag in raw_tags:
                if isinstance(tag, dict):
                    tag_type = tag.get("type")
                    if isinstance(tag_type, str):
                        tags.add(tag_type)
                elif isinstance(tag, str):
                    tags.add(tag)

            if ((idx not in omit_content_idxs) or (tags & self.always_keep_output_for_tags)) and not (
                tags & self.always_remove_output_for_tags
            ):
                new_history.append(entry)
            else:
                data = entry.copy()
                assert data["message_type"] == "observation", (
                    f"Expected observation for dropped entry, got: {data['message_type']}"
                )
                text = _get_content_text(data)
                _set_content_text(data, f"Old environment output: ({len(text.splitlines())} lines omitted)")
                new_history.append(data)
        return new_history


class TagToolCallObservations(BaseModel):
    """Adds tags to history items for specific tool calls."""

    type: Literal["tag_tool_call_observations"] = "tag_tool_call_observations"
    """Do not change. Used for (de)serialization."""

    tags: set[str] = {"keep_output"}
    """Add the following tag to all observations matching the search criteria."""

    function_names: set[str] = set()
    """Only consider observations made by tools with these names."""

    # pydantic config
    model_config = ConfigDict(extra="forbid")

    def _add_tags(self, entry: HistoryItem) -> None:
        tags = set(entry.get("tags", []))
        tags.update(self.tags)
        entry["tags"] = list(tags)

    def _should_add_tags(self, entry: HistoryItem) -> bool:
        if entry["message_type"] != "action":
            return False
        function_calls = entry.get("tool_calls", [])
        if not function_calls:
            return False
        function_names = {call["function"]["name"] for call in function_calls}
        return bool(self.function_names & function_names)

    def __call__(self, history: History) -> History:
        for entry in history:
            if self._should_add_tags(entry):
                self._add_tags(entry)
        return history


class ClosedWindowHistoryProcessor(BaseModel):
    """Elide outdated windows, replacing them with line-count summaries."""

    type: Literal["closed_window"] = "closed_window"
    """Do not change. Used for (de)serialization."""

    _pattern = re.compile(r"^(\d+)\:.*?(\n|$)", re.MULTILINE)
    _file_pattern = re.compile(r"\[File:\s+(.*)\s+\(\d+\s+lines\ total\)\]")

    # pydantic config
    model_config = ConfigDict(extra="forbid")

    def __call__(self, history: History) -> History:
        new_history = list()
        windows = set()
        for entry in reversed(history):
            data = entry.copy()
            if data["role"] != "user":
                new_history.append(entry)
                continue
            if data.get("is_demo", False):
                new_history.append(entry)
                continue
            matches = list(self._pattern.finditer(entry["content"]))
            if len(matches) >= 1:
                file_match = self._file_pattern.search(entry["content"])
                if file_match:
                    file = file_match.group(1)
                else:
                    continue
                if file in windows:
                    start = matches[0].start()
                    end = matches[-1].end()
                    data["content"] = (
                        entry["content"][:start]
                        + f"Outdated window with {len(matches)} lines omitted...\n"
                        + entry["content"][end:]
                    )
                windows.add(file)
            new_history.append(data)
        return list(reversed(new_history))


class CacheControlHistoryProcessor(BaseModel):
    """This history processor adds manual cache control marks to the history.
    Use this when running with anthropic claude.
    """

    type: Literal["cache_control"] = "cache_control"
    """Do not change. Used for (de)serialization."""

    last_n_messages: int = 2
    """Add cache control to the last n user messages (and clear it for anything else).
    In most cases this should be set to 2 (caching for multi-turn conversations).
    When resampling and running concurrent instances, you want to set it to 1.
    If set to <= 0, any set cache control will be removed from all messages.
    """

    last_n_messages_offset: int = 0
    """E.g., set to 1 to start cache control after the second to last user message.
    This can be useful in rare cases, when you want to modify the last message after
    we've got the completion and you want to avoid cache mismatch.
    """

    tagged_roles: list[str] = ["user", "tool"]
    """Only add cache control to messages with these roles."""

    # pydantic config
    model_config = ConfigDict(extra="forbid")

    def __call__(self, history: History) -> History:
        new_history = []
        n_tagged = 0
        for i_entry, entry in enumerate(reversed(history)):
            # Clear cache control from previous messages
            _clear_cache_control(entry)
            if (
                n_tagged < self.last_n_messages
                and entry["role"] in self.tagged_roles
                and i_entry >= self.last_n_messages_offset
            ):
                _set_cache_control(entry)
                n_tagged += 1
            new_history.append(entry)
        return list(reversed(new_history))


class RemoveRegex(BaseModel):
    """This history processor can remove arbitrary content from history items"""

    remove: list[str] = ["<diff>.*</diff>"]
    """Regex patterns to remove from history items"""

    keep_last: int = 0
    """Keep the last n history items unchanged"""

    type: Literal["remove_regex"] = "remove_regex"
    """Do not change. Used for (de)serialization."""

    # pydantic config
    model_config = ConfigDict(extra="forbid")

    def __call__(self, history: History) -> History:
        new_history = []
        for i_entry, entry in enumerate(reversed(history)):
            entry = copy.deepcopy(entry)
            if i_entry < self.keep_last:
                new_history.append(entry)
            else:
                text = _get_content_text(entry)
                for pattern in self.remove:
                    text = re.sub(pattern, "", text, flags=re.DOTALL)
                    _set_content_text(entry, text)
                new_history.append(entry)
        return list(reversed(new_history))

class SummarizeEveryNTurns(BaseModel):
    """Summarize every N turns. Skips first two (system/user prompts)."""

    n: int = 2
    """Number of turns to summarize."""

    keep_last_m_turns: int = 0
    """Recent turns to keep unsummarized. Summarization triggers when
    unprocessed turns >= n + keep_last_m_turns."""

    enable_static_checkpointing: bool = False
    """Keep only the most recent summary, updating it every n turns.
    Otherwise summaries are appended to the history."""

    extract_action_from_turns: bool = False
    """Extract actions from turns and include them in the summary."""

    max_kept_action_length: int = -1
    max_kept_reasoning_length: int = -1
    omit_turns: bool = False

    enable_summary_synthesis: bool = False
    """Generate an updated checkpoint incorporating prior checkpoint and new turns,
    rather than summarizing only the new turns.
    Requires enable_static_checkpointing=True."""

    compact_before_summary: bool = False
    """Truncate long observations before sending to the summarizer.
    Only affects summarizer input, not the agent's actual context."""

    max_observation_length_for_summary: int = 5000
    """Max chars for observations sent to summarizer (when compact_before_summary=True)."""

    enable_limit_aware_trigger: bool = False
    """Only summarize when near the model context limit."""

    limit_trigger_fraction: float = 0.9
    """Trigger when estimated tokens exceed this fraction of the context window."""

    limit_trigger_min_tokens: int = 0
    """Absolute minimum token threshold for triggering (takes precedence)."""

    pure_on_demand: bool = False
    """With limit-aware enabled, skip periodic N-turn checks entirely.
    Summarize ALL unprocessed turns (except keep_last_m_turns) when
    context threshold is exceeded."""

    summaries: list[SummaryMetadata] = Field(default_factory=list)
    """All summaries generated, with metadata about token usage."""

    type: Literal["summarize_every_n_turns"] = "summarize_every_n_turns"
    """Do not change. Used for (de)serialization."""

    _processed_history_cache: History = PrivateAttr(default_factory=list)
    """Preprocessed history prefix (system + user prompt + summaries)."""

    _n_summarized_turns: int = PrivateAttr(default=0)
    """Turn count for static checkpointing where cache doesn't grow."""

    _model: AbstractModel | None = PrivateAttr(default=None)
    _context_model: AbstractModel | None = PrivateAttr(default=None)

    def _count_summaries_in_cache(self) -> int:
        """Count summary items in cache (backward compat for manual cache copying)."""
        return sum(1 for item in self._processed_history_cache
                   if item.get("tags") == [{'type': 'summary'}])

    _logger = get_logger("swea-lm", emoji="🤖")

    # pydantic config
    model_config = ConfigDict(extra="forbid")

    def set_model(self, model: AbstractModel):
        self._model = model

    def set_context_model(self, model: AbstractModel) -> None:
        self._context_model = model

    @field_validator("n")
    def validate_n(cls, n: int) -> int:
        if n <= 0:
            msg = "n must be a positive integer"
            raise ValueError(msg)
        return n

    @field_validator("keep_last_m_turns")
    def validate_keep_last_m_turns(cls, m: int) -> int:
        if m < 0:
            msg = "keep_last_m_turns must be non-negative"
            raise ValueError(msg)
        return m

    @field_validator("limit_trigger_fraction")
    def validate_limit_trigger_fraction(cls, fraction: float) -> float:
        if not (0 < fraction <= 1):
            msg = "limit_trigger_fraction must be in (0, 1]"
            raise ValueError(msg)
        return fraction

    @field_validator("limit_trigger_min_tokens")
    def validate_limit_trigger_min_tokens(cls, min_tokens: int) -> int:
        if min_tokens < 0:
            msg = "limit_trigger_min_tokens must be non-negative"
            raise ValueError(msg)
        return min_tokens

    @model_validator(mode="after")
    def _validate_synthesis_requires_static(self) -> "SummarizeEveryNTurns":
        if self.enable_summary_synthesis and not self.enable_static_checkpointing:
            raise ValueError("enable_summary_synthesis requires enable_static_checkpointing=true")
        return self

    def _extract_turns_from(self, history: History) -> Turns:
        """Split history into ReAct-style turns: (reasoning + action, observation) pairs.

        Handles both function-calling (role="tool") and non-function-calling
        (role="user" with message_type="observation") flows.
        """
        turns = []
        for i, step in enumerate(history):
            is_observation = (
                step["role"] == "tool" or
                (step["role"] == "user" and step.get("message_type") == "observation")
            )
            if is_observation:
                if i == 0:
                    continue
                if history[i-1]["role"] == "assistant":
                    assistant_message = history[i-1]
                    if any(isinstance(tag, dict) and tag.get('type') == 'summary' for tag in assistant_message.get('tags', [])):
                        continue
                    turns.append([assistant_message, step])
        return turns

    def _convert_turns_to_history_items(self, turns: Turns) -> History:
        """Flatten turns back to a list of history items."""
        history_items = []
        for turn in turns:
            history_items.extend(turn)
        return history_items

    def _omit_turns(self, turns: Turns) -> History:
        omitted_count = len(turns)
        return [{
            "role": "assistant",
            "content": f"Previous {omitted_count} turns omitted for brevity.",
            "message_type": "thought",
            # TODO: tags are dicts here but SWE-agent uses mixed tag formats elsewhere.
            # Only matters if chaining with a tag-dependent processor.
            "tags": [{'type': 'summary'}]
        }]

    def _compact_turns_for_summary(self, turns: Turns) -> Turns:
        """Truncate long observations for summarizer input (does not modify originals)."""
        compacted = []
        for turn in turns:
            compacted_turn = []
            for item in turn:
                item_copy = copy.deepcopy(item)
                is_observation = (
                    item_copy.get("role") == "tool" or
                    (item_copy.get("role") == "user" and item_copy.get("message_type") == "observation")
                )
                if is_observation:
                    content = _get_content_text(item_copy)
                    if len(content) > self.max_observation_length_for_summary:
                        truncated = content[:self.max_observation_length_for_summary - 50]
                        truncated += f"\n...[truncated {len(content) - self.max_observation_length_for_summary + 50} chars for summary]..."
                        _set_content_text(item_copy, truncated)
                compacted_turn.append(item_copy)
            compacted.append(compacted_turn)
        return compacted

    def _summarize_turns(self, turns: Turns, summary_context: str) -> tuple[History, bool]:
        """Generate a summary for the provided turns.

        Returns (summary_history, did_summarize). If API call limit is reached,
        returns ([], False) so the caller can preserve turns instead of dropping them.
        """
        if 0 < self._model.config.per_instance_call_limit <= self._model.stats.api_calls:
            self._logger.info(f"Skipping summary: API limit reached ({self._model.stats.api_calls} calls)")
            return [], False

        has_previous = bool(self._processed_history_cache)
        is_synthesis = self.enable_summary_synthesis and has_previous

        turns_for_summary = self._compact_turns_for_summary(turns) if self.compact_before_summary else turns

        summary_metadata = self._model.query_for_summary(
            summary_context,
            turns_for_summary,
            self.extract_action_from_turns,
            self.max_kept_action_length,
            self.max_kept_reasoning_length,
            synthesize=is_synthesis,
        )
        self.summaries.append(summary_metadata)

        if is_synthesis:
            wrapper = f"State checkpoint (cumulative):\n{summary_metadata.summary}"
        else:
            wrapper = f"Checkpoint for the last {len(turns)} turns:\n{summary_metadata.summary}"

        return [{
            "role": "assistant",
            "content": wrapper,
            "message_type": "thought",
            "tags": [{'type': 'summary'}]
        }], True

    def __call__(self, history: History) -> History:
        turns = self._extract_turns_from(history)

        # Fallback to counting summaries in cache for tests that manually copy it
        if self._n_summarized_turns > 0:
            n_processed_turns = self._n_summarized_turns
        else:
            n_processed_turns = self._count_summaries_in_cache() * self.n
        unprocessed_turns = turns[n_processed_turns:]

        if self.enable_limit_aware_trigger and not _should_trigger_limit_aware(
            history=history,
            context_model=self._context_model,
            fraction=self.limit_trigger_fraction,
            min_tokens=self.limit_trigger_min_tokens,
            logger=self._logger,
            label="SummarizeEveryNTurns",
        ):
            if self._processed_history_cache:
                return self._processed_history_cache + self._convert_turns_to_history_items(unprocessed_turns)
            return history

        # Pure on-demand: summarize ALL unprocessed turns when above threshold
        if self.pure_on_demand and self.enable_limit_aware_trigger:
            if len(unprocessed_turns) <= self.keep_last_m_turns:
                if self._processed_history_cache:
                    return self._processed_history_cache + self._convert_turns_to_history_items(unprocessed_turns)
                return history

            if self.keep_last_m_turns > 0:
                turns_to_summarize = unprocessed_turns[:-self.keep_last_m_turns]
                turns_to_keep = unprocessed_turns[-self.keep_last_m_turns:]
            else:
                turns_to_summarize = unprocessed_turns
                turns_to_keep = []
            self._logger.info(
                f"On-demand: summarizing {len(turns_to_summarize)} turns, keeping {len(turns_to_keep)}"
            )
        elif len(unprocessed_turns) < self.n + self.keep_last_m_turns:
            if self._processed_history_cache:
                return self._processed_history_cache + self._convert_turns_to_history_items(unprocessed_turns)
            else:
                return history
        else:
            turns_to_summarize = unprocessed_turns[:self.n]
            turns_to_keep = unprocessed_turns[self.n:]

        if self._model is None or self.omit_turns:
            if self._model is None:
                self._logger.warning("No model set for SummarizeEveryNTurns. "
                                    "Set a model via set_model() to enable summarization. "
                                    "Omitting turns instead.")
            else:
                self._logger.warning("Omitting turns instead of summarizing.")
            summary = self._omit_turns(turns_to_summarize)
            did_summarize = True
        else:
            summary_context = f'<PREVIOUS_CHECKPOINT>\n{self._processed_history_cache[-1]["content"]}\n</PREVIOUS_CHECKPOINT>\n' if self._processed_history_cache else \
                f'<PROBLEM_STATEMENT>\n{history[1]["content"]}\n</PROBLEM_STATEMENT>\n'
            summary, did_summarize = self._summarize_turns(turns_to_summarize, summary_context)

        if not did_summarize:
            self._logger.warning("Summarization skipped (API limit) - preserving turns")
            if self._processed_history_cache:
                return self._processed_history_cache + self._convert_turns_to_history_items(unprocessed_turns)
            else:
                return history

        self._n_summarized_turns = n_processed_turns + len(turns_to_summarize)

        if self.enable_static_checkpointing:
            self._processed_history_cache = history[:2] + summary
            return history[:2] + summary + self._convert_turns_to_history_items(turns_to_keep)
        else:
            if not self._processed_history_cache:
                self._processed_history_cache = history[:2] + summary
            else:
                self._processed_history_cache += summary
            return self._processed_history_cache + self._convert_turns_to_history_items(turns_to_keep)


class DeduplicateToolOutputs(BaseModel):
    """Replace duplicate tool outputs with self-contained placeholders.

    Hashes recomputed each call (stateless). Placeholders include preview lines
    so the agent doesn't lose context. Chain before other processors.
    """

    type: Literal["deduplicate_tool_outputs"] = "deduplicate_tool_outputs"
    """Do not change. Used for (de)serialization."""

    min_length: int = 200
    """Only deduplicate outputs >= this character length. Short outputs are kept as-is."""

    keep_first_lines: int = 3
    """Number of preview lines to include in placeholder to avoid 'reference rot'."""

    scope: Literal["global", "per_tool"] = "per_tool"
    """Hash scope: 'per_tool' only dedupes identical outputs from same tool (safer),
    'global' dedupes across all tools (more aggressive)."""

    model_config = ConfigDict(extra="forbid")

    def __call__(self, history: History) -> History:
        import hashlib

        seen: dict[str, tuple[int, str]] = {}  # key -> (first_step, preview)
        new_history = []
        step_num = 0

        for entry in history:
            # Skip non-observations and demo entries
            if entry.get("role") != "tool" or entry.get("is_demo"):
                new_history.append(entry)
                continue

            content = _get_content_text(entry)

            # Skip short outputs
            if len(content) < self.min_length:
                new_history.append(entry)
                step_num += 1
                continue

            # Compute hash (optionally scoped by tool name)
            tool_name = entry.get("name", entry.get("tool_name", "unknown"))
            content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
            key = f"{tool_name}:{content_hash}" if self.scope == "per_tool" else content_hash

            if key in seen:
                first_step, preview = seen[key]
                lines = content.count('\n') + 1
                chars = len(content)
                entry = entry.copy()
                placeholder = (
                    f"[Duplicate output - {lines} lines, {chars} chars, hash={content_hash[:8]}]\n"
                    f"Preview:\n{preview}\n"
                    f"[First seen at step {first_step}]"
                )
                _set_content_text(entry, placeholder)
            else:
                # Store first occurrence with preview
                preview_lines = '\n'.join(content.split('\n')[:self.keep_first_lines])
                seen[key] = (step_num, preview_lines)

            new_history.append(entry)
            step_num += 1

        return new_history


HistoryProcessor = Annotated[
    DefaultHistoryProcessor
    | LastNObservations
    | ClosedWindowHistoryProcessor
    | TagToolCallObservations
    | CacheControlHistoryProcessor
    | RemoveRegex
    | DeduplicateToolOutputs
    | SummarizeEveryNTurns,
    Field(discriminator="type"),
]

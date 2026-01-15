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

    type: Literal["last_n_observations"] = "last_n_observations"
    """Do not change. Used for (de)serialization."""

    # pydantic config
    model_config = ConfigDict(extra="forbid")

    @field_validator("n")
    def validate_n(cls, n: int) -> int:
        if n <= 0:
            msg = "n must be a positive integer"
            raise ValueError(msg)
        return n

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
        new_history = []
        omit_content_idxs = self._get_omit_indices(history)
        for idx, entry in enumerate(history):
            tags = entry.get("tags", [])

            # Hacky workaround. I set the tags to be a dict because there was some internal LiteLLM issue otherwise.
            tags = set([tags[0]['type']]) if (len(tags) > 0 and isinstance(tags[0], dict) and tags[0].get("type") == "summary") else set(tags)

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
    """Number of turns to summarized."""

    keep_last_m_turns: int = 0
    """Number of recent turns to keep unsummarized. When there are >= n + keep_last_m_turns
      unprocessed turns, only the first n turns will be summarized, 
      while the last keep_last_m_turns will be kept in the history."""
    
    enable_static_checkpointing: bool = False
    """If enabled we only keep the most recent summary and dynamically update it every n turns to the
    state we observed in these turns. Otherwise summaries are appended to the history."""
    
    extract_action_from_turns: bool = False
    """Whether to extract the action from the turns. If set, the actions will be extracted from the turns
    and added to the summary."""

    max_kept_action_length: int = -1
    max_kept_reasoning_length: int = -1
    omit_turns: bool = False

    enable_summary_synthesis: bool = False
    """When True, generate an UPDATED checkpoint that incorporates the prior
    checkpoint and new turns, rather than summarizing only the new turns.
    Requires enable_static_checkpointing=True to prevent duplicate accumulated context."""

    compact_before_summary: bool = False
    """When True, preprocess turns to remove redundancy before sending to the summarizer.
    This truncates long observations to reduce summarizer input tokens without affecting
    the agent's actual context. Low risk since it only affects what the summarizer sees."""

    max_observation_length_for_summary: int = 5000
    """Maximum character length for observations sent to the summarizer.
    Only used when compact_before_summary=True. Longer outputs are truncated."""

    summaries: list[SummaryMetadata] = Field(default_factory=list)
    """A list of dictionaries of all summaries generated by this history processor. Includes the summary,
    the input it was conditioned on and metadata about token usage."""
 
    type: Literal["summarize_every_n_turns"] = "summarize_every_n_turns"
    """Do not change. Used for (de)serialization."""

    _processed_history_cache: History = PrivateAttr(default_factory=list)
    """The history as preprocessed the last time this history processor took action.
        Contains the system and user prompt. Is the prefix for the unprocessed turns."""

    _n_summarized_turns: int = PrivateAttr(default=0)
    """Summarized turn count. Used for static checkpointing where cache doesn't grow."""

    _model: AbstractModel | None = None

    def _count_summaries_in_cache(self) -> int:
        """Count summary items in cache for backward compatibility with manual cache copying."""
        return sum(1 for item in self._processed_history_cache
                   if item.get("tags") == [{'type': 'summary'}])

    _logger = get_logger("swea-lm", emoji="🤖")

    # pydantic config
    model_config = ConfigDict(extra="forbid")

    def set_model(self, model: AbstractModel):
        self._model = model

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

    @model_validator(mode="after")
    def _validate_synthesis_requires_static(self) -> "SummarizeEveryNTurns":
        if self.enable_summary_synthesis and not self.enable_static_checkpointing:
            raise ValueError("enable_summary_synthesis requires enable_static_checkpointing=true")
        return self

    def _extract_turns_from(self, history: History) -> Turns:
        """Splits the history into turns. We define turns in the ReAct style:
        A turn is a (reasoning + action, observation) pair.

        Handles both function-calling flows (role="tool") and non-function-calling
        flows (role="user" with message_type="observation").
        """
        turns = []
        for i, step in enumerate(history):
            # Check for observations: either tool role or user with observation type
            is_observation = (
                step["role"] == "tool" or
                (step["role"] == "user" and step.get("message_type") == "observation")
            )
            if is_observation:
                if i == 0:
                    # Skip if this is somehow the first message (shouldn't happen)
                    continue
                if history[i-1]["role"] == "assistant":
                    assistant_message = history[i-1]
                    if any(isinstance(tag, dict) and tag.get('type') == 'summary' for tag in assistant_message.get('tags', [])):
                        continue
                    turns.append([assistant_message, step])
                # For non-function-calling, we may not always have an assistant before observation
                # In that case, just skip (e.g., initial problem statement)
        return turns

    def _convert_turns_to_history_items(self, turns: Turns) -> History:
        """Convert turns back to a flat list of history items."""
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
            # LiteLLM expects a list of dicts, but SWE-agent somehow a list of tags.
            # Not sure how exactly to amend this. For now this is however only a problem if we want to use
            # this history processor with another history processor that uses the tags field.
            "tags": [{'type': 'summary'}]
        }]

    def _compact_turns_for_summary(self, turns: Turns) -> Turns:
        """Preprocess turns to remove redundancy before sending to the summarizer.

        This truncates long observations to reduce token usage in the summarizer input.
        The original turns are NOT modified - only the summarizer sees the compacted version.
        """
        compacted = []
        for turn in turns:
            compacted_turn = []
            for item in turn:
                item_copy = copy.deepcopy(item)
                # Truncate observations (tool outputs or user observations in non-FC mode)
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
        """
        Generate a summary HistoryItem for the provided turns using the configured model.

        If the agent has used up all available API calls, we skip summary generation and signal
        that summarization was skipped so the caller can handle turn preservation.

        Args:
            turns (Turns): The turns to summarize, where each turn is a (assistant, tool) pair.
            summary_context (str): The context to prepend to the turns for summarization.
                This is initially the user prompt and if available, the previous summary.

        Returns:
            tuple[History, bool]: A tuple of (summary_history, did_summarize) where:
                - summary_history: Single-item list containing the summary
                - did_summarize: True if actual summarization happened, False if skipped
        """
        if 0 < self._model.config.per_instance_call_limit <= self._model.stats.api_calls:
            self._logger.info(f"Skipping summary generation because the agent has used up all available API calls: {self._model.stats.api_calls} >= {self._model.config.per_instance_call_limit}")
            # Signal that summarization was skipped - caller must NOT drop turns
            return [], False

        # Only synthesize when there's actually a previous checkpoint to synthesize FROM
        has_previous = bool(self._processed_history_cache)
        is_synthesis = self.enable_summary_synthesis and has_previous

        # Compact turns before summarization if enabled (reduces summarizer input tokens)
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

        # Use appropriate wrapper label based on synthesis mode
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

        # counter indexes into full history; fallback to counting summaries in cache
        # for tests that manually copy the cache
        if self._n_summarized_turns > 0:
            n_processed_turns = self._n_summarized_turns
        else:
            n_processed_turns = self._count_summaries_in_cache() * self.n
        unprocessed_turns = turns[n_processed_turns:]

        if len(unprocessed_turns) < self.n + self.keep_last_m_turns:
            if self._processed_history_cache:
                return self._processed_history_cache + [step for turn in unprocessed_turns for step in turn]
            else:
                return history
        else:
            turns_to_summarize = unprocessed_turns[:self.n]
            turns_to_keep = unprocessed_turns[self.n:]

            if self._model is None or self.omit_turns:
                if self._model is None:
                    self._logger.warning("No model set for SummarizeEveryNTurns history processor. "
                                        "If this is not intentional, please set a model using the set_model method. "
                                        "Omitting turns instead of summarizing!")
                else:
                    self._logger.warning("Omitting turns instead of summarizing!")
                summary = self._omit_turns(turns_to_summarize)
                did_summarize = True  # Omitting is intentional, treat as "handled"
            else:
                summary_context = f'<PREVIOUS_CHECKPOINT>\n{self._processed_history_cache[-1]["content"]}\n</PREVIOUS_CHECKPOINT>\n' if self._processed_history_cache else \
                    f'<PROBLEM_STATEMENT>\n{history[1]["content"]}\n</PROBLEM_STATEMENT>\n'
                summary, did_summarize = self._summarize_turns(turns_to_summarize, summary_context)

            # if summarization failed (API limit), preserve turns instead of dropping them
            if not did_summarize:
                self._logger.warning("Summarization skipped due to API limits - preserving turns in context")
                if self._processed_history_cache:
                    return self._processed_history_cache + self._convert_turns_to_history_items(unprocessed_turns)
                else:
                    return history

            self._n_summarized_turns = n_processed_turns + len(turns_to_summarize)

            if self.enable_static_checkpointing:
                # Replace cache (static)
                self._processed_history_cache = history[:2] + summary
                return history[:2] + summary + self._convert_turns_to_history_items(turns_to_keep)
            else:
                # Append (dynamic)
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

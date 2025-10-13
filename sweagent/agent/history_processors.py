from __future__ import annotations

import copy
import re
from abc import abstractmethod
from typing import Annotated, List, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, PrivateAttr

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
    """For each value in history, keep track of which windows have been shown.
    We want to mark windows that should stay open (they're the last window for a particular file)
    Then we'll replace all other windows with a simple summary of the window (i.e. number of lines)
    """

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
    """This history processor summarizes the every n turns of the conversation.
    A turn consists of a (reasoning/action, observation) pair. The first two 
    turns consisting of system and user prompt are not summarized."""

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

    summaries: list[SummaryMetadata] = []
    """A list of dictionaries of all summaries generated by this history processor. Includes the summary, 
    the input it was conditioned on and metadata about token usage."""
 
    type: Literal["summarize_every_n_turns"] = "summarize_every_n_turns"
    """Do not change. Used for (de)serialization."""

    _processed_history_cache: History = PrivateAttr(default_factory=list)
    """The history as preprocessed the last time this history processor took action.
        Contains the system and user prompt. Is the prefix for the unprocessed turns."""
    
    _model: AbstractModel | None = None

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

    def _extract_turns_from(self, history: History) -> Turns:
        """Splits the history into turns. We define turns in the ReAct style: 
        A turn is a (reasoning + action, observation) pair."""
        turns = []
        for i, step in enumerate(history):
            if step["role"] == "tool":
                if i == 0:
                    raise AssertionError("Unexpected history format. Tool call cannot be the first message in the history. "
                                         "No system and user prompt found!")
                if history[i-1]["role"] == "assistant":
                    assistant_message = history[i-1]
                    if any(isinstance(tag, dict) and tag.get('type') == 'summary' for tag in assistant_message.get('tags', [])):
                        continue
                    turns.append([assistant_message, step])
                else:
                    raise AssertionError("Unexpected history format. If a step in the history is a tool call, "
                                         "it must be preceded by an assistant message specifying the "
                                         "tool to call in the next step.")
            else:
                continue
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
    
    def _summarize_turns(self, turns: Turns, summary_context: str) -> History:
        """
        Generate a summary HistoryItem for the provided turns using the configured model.

        If the agent has used up all available API calls, we skip summary generation and return
        a placeholder summary instead of the original turns to maintain consistent cache counting.

        Args:
            turns (Turns): The turns to summarize, where each turn is a (assistant, tool) pair.
            summary_context (str): The context to prepend to the turns for summarization.
                This is initially the user prompt and if available, the previous summary.

        Returns:
            History: A single-item list containing either a real summary or a placeholder
            summary when API limits are exceeded.
        """
        summarized_turns = len(turns)
        
        if 0 < self._model.config.per_instance_call_limit <= self._model.stats.api_calls:
            self._logger.info(f"Skipping summary generation because the agent has used up all available API calls: {self._model.stats.api_calls} >= {self._model.config.per_instance_call_limit}")
            return [{
                "role": "assistant",
                "content": f"Checkpoint for the last {summarized_turns} turns: (Summary skipped due to API call limit)",
                "message_type": "thought", 
                "tags": [{'type': 'summary'}]
            }]
         
        
        summary_metadata = self._model.query_for_summary(summary_context, turns, self.extract_action_from_turns, self.max_kept_action_length, self.max_kept_reasoning_length)
        self.summaries.append(summary_metadata)
    
        return [{
            "role": "assistant",
            "content": f"Checkpoint for the last {len(turns)} turns:\n{summary_metadata.summary}",
            "message_type": "thought", 
            "tags": [{'type': 'summary'}]
        }]

    def __call__(self, history: History) -> History:
        turns = self._extract_turns_from(history)
                
        # System and user prompt are complete turns that are never summarized.
        # A summary turn corresponds to self.n raw turns.
        n_processed_turns = (len(self._processed_history_cache) - 2) * self.n if self._processed_history_cache else 0 
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
            else:
                summary_context = f'<PREVIOUS_CHECKPOINT>\n{self._processed_history_cache[-1]["content"]}\n</PREVIOUS_CHECKPOINT>\n' if self._processed_history_cache else \
                    f'<PROBLEM_STATEMENT>\n{history[1]["content"]}\n</PROBLEM_STATEMENT>\n'
                summary = self._summarize_turns(turns_to_summarize, summary_context)

            if not self._processed_history_cache:
                self._processed_history_cache = history[:2] + summary
            else:
                self._processed_history_cache += summary

            if self.enable_static_checkpointing:
                return history[:2] + summary + self._convert_turns_to_history_items(turns_to_keep)
            else:
                return self._processed_history_cache + self._convert_turns_to_history_items(turns_to_keep)

HistoryProcessor = Annotated[
    DefaultHistoryProcessor
    | LastNObservations
    | ClosedWindowHistoryProcessor
    | TagToolCallObservations
    | CacheControlHistoryProcessor
    | RemoveRegex
    | SummarizeEveryNTurns,
    Field(discriminator="type"),
]

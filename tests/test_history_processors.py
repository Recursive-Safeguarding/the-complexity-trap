import json
from pathlib import Path

import pytest

from sweagent.agent.history_processors import LastNObservations, TagToolCallObservations, SummarizeEveryNTurns
from sweagent.types import History


def get_history(traj_path: Path):
    return json.loads((traj_path).read_text())["history"]


def count_elided_observations(history: History):
    return len([entry for entry in history if "Old environment output" in entry["content"]])


@pytest.fixture
def test_history(test_trajectories_path: Path):
    return get_history(
        test_trajectories_path
        / "gpt4__swe-agent-test-repo__default_from_url__t-0.00__p-0.95__c-3.00__install-1/6e44b9__sweagenttestrepo-1c2844.traj"
    )


def test_last_n_observations(test_history: History):
    processor = LastNObservations(n=3)
    new_history = processor(test_history)
    total_observations = len([entry for entry in test_history if entry["message_type"] == "observation"])
    anchor_observations = 1  # keep the first observation (instance template)
    expected_elided_observations = total_observations - 3 - anchor_observations
    assert count_elided_observations(new_history) == expected_elided_observations


def test_add_tag_to_edits(test_history: History):
    processor = TagToolCallObservations(tags={"test"}, function_names={"edit"})
    new_history = processor(test_history)
    for entry in new_history:
        if entry.get("action", "").startswith("edit "):  # type: ignore
            assert entry.get("tags") == ["test"], entry


@pytest.fixture
def simple_history():
    """A simple history with system prompt, user prompt, and two complete turns."""
    return [
        {"role": "system", "content": "You are an AI assistant", "message_type": "system"},
        {"role": "user", "content": "Hello", "message_type": "user"},
        {"role": "assistant", "content": "I'll check something", "message_type": "thought", "action": "check"},
        {"role": "tool", "content": "Result of check", "message_type": "observation"},
        {"role": "assistant", "content": "I'll look up info", "message_type": "thought", "action": "lookup"},
        {"role": "tool", "content": "Information found", "message_type": "observation"}
    ]


@pytest.fixture
def extended_history():
    """A history with system prompt, user prompt, and 7 complete turns."""
    return [
        {"role": "system", "content": "You are an AI assistant", "message_type": "system"},
        {"role": "user", "content": "Hello", "message_type": "user"},
        {"role": "assistant", "content": "I'll check something", "message_type": "thought", "action": "check"},
        {"role": "tool", "content": "Result of check", "message_type": "observation"},
        {"role": "assistant", "content": "I'll look up info", "message_type": "thought", "action": "lookup"},
        {"role": "tool", "content": "Information found", "message_type": "observation"},
        {"role": "assistant", "content": "I'll analyze the data", "message_type": "thought", "action": "analyze"},
        {"role": "tool", "content": "Analysis complete", "message_type": "observation"},
        {"role": "assistant", "content": "I'll verify the results", "message_type": "thought", "action": "verify"},
        {"role": "tool", "content": "Verification successful", "message_type": "observation"},
        {"role": "assistant", "content": "I'll process the findings", "message_type": "thought", "action": "process"},
        {"role": "tool", "content": "Processing complete", "message_type": "observation"},
        {"role": "assistant", "content": "I'll summarize the data", "message_type": "thought", "action": "summarize"},
        {"role": "tool", "content": "Summary generated", "message_type": "observation"},
        {"role": "assistant", "content": "I'll finalize the report", "message_type": "thought", "action": "finalize"},
        {"role": "tool", "content": "Report finalized", "message_type": "observation"}
    ]

def test_extract_turns_from_simple_history(simple_history):
    """Test that _extract_turns_from correctly extracts turns from a simple history."""
    processor = SummarizeEveryNTurns(n=2)
    turns = processor._extract_turns_from(simple_history)
    
    assert len(turns) == 2
    assert turns[0] == [simple_history[2], simple_history[3]]
    assert turns[1] == [simple_history[4], simple_history[5]]


def test_extract_turns_with_summary(simple_history):
    """Test that _extract_turns_from correctly skips summary messages."""
    processor = SummarizeEveryNTurns(n=2)
    
    # Add a summary tag to one of the assistant messages
    simple_history[2]["tags"] = [{"type": "summary"}]

    # Should not raise an error, should skip the summary message
    turns = processor._extract_turns_from(simple_history)
    
    # Should only extract 1 turn (the second one), skipping the first which has a summary tag
    assert len(turns) == 1
    assert turns[0] == [simple_history[4], simple_history[5]]


def test_extract_turns_from_simple_history_with_n_is_3(simple_history):
    """Test that _extract_turns_from correctly extracts turns another value of n."""
    processor = SummarizeEveryNTurns(n=3)
    turns = processor._extract_turns_from(simple_history)
    
    assert len(turns) == 2
    # Check each turn has correct structure
    assert turns[0] == [simple_history[2], simple_history[3]]
    assert turns[1] == [simple_history[4], simple_history[5]]



def test_summarize_fewer_than_n_turns(simple_history):
    """Test when there are fewer than N turns."""
    processor = SummarizeEveryNTurns(n=3)
    new_history = processor(simple_history)
    
    # With N=3 and only 2 turns, should return original history
    assert new_history == simple_history
    # Cache should be empty
    assert not processor._processed_history_cache


def test_summarize_exactly_n_turns(simple_history):
    """Test when there are exactly N turns."""
    processor = SummarizeEveryNTurns(n=2)
    new_history = processor(simple_history)
    
    # Should have system, user prompt, and then a summary message
    assert len(new_history) == 3
    assert new_history[0] == simple_history[0]
    assert new_history[1] == simple_history[1]
    assert new_history[2]["role"] == "assistant"
    assert "turns omitted" in new_history[2]["content"]
    assert new_history[2]["tags"] == [{"type": "summary"}]
    
    # Cache should be updated
    assert processor._processed_history_cache == new_history


def test_summarize_with_partial_unprocessed_turns(extended_history):
    """Test when there are partial unprocessed turns after summarization."""
    processor = SummarizeEveryNTurns(n=2)
    
    # Should not yet summarize
    new_history = processor(extended_history[:4])
    assert new_history == extended_history[:4]
    
    # This should trigger a new summarization
    new_history = processor(extended_history[:6])
    
    # Should have system, user prompt, and first summary
    assert len(new_history) == 3
    assert new_history[0] == extended_history[0]
    assert new_history[1] == extended_history[1]
    assert "turns omitted" in new_history[2]["content"]
    assert new_history[2]["tags"] == [{"type": "summary"}]

    # Should return previous history plus new turn
    new_history = processor(extended_history[:8])
    assert len(new_history) == 5




def test_extract_turns_invalid_format():
    """Test that _extract_turns_from gracefully handles edge cases.

    The function now skips invalid sequences instead of raising, to support
    non-function-calling flows where observations may not follow the strict
    (assistant, observation) pattern.
    """
    processor = SummarizeEveryNTurns(n=2)

    # Tool as first message - should skip, return empty turns
    invalid_history_1 = [
        {"role": "tool", "content": "Invalid start", "message_type": "observation"}
    ]
    turns = processor._extract_turns_from(invalid_history_1)
    assert turns == [], "Observation at index 0 should be skipped"

    # Tool not preceded by assistant - should skip that observation
    invalid_history_2 = [
        {"role": "system", "content": "System prompt", "message_type": "system"},
        {"role": "user", "content": "User message", "message_type": "user"},
        {"role": "tool", "content": "Invalid sequence", "message_type": "observation"}
    ]
    turns = processor._extract_turns_from(invalid_history_2)
    assert turns == [], "Observation not preceded by assistant should be skipped"


def test_extract_turns_with_user_role_observations():
    """Test that _extract_turns_from handles non-function-calling mode.

    In non-function-calling flows (e.g., Bedrock models), observations come
    as role='user' with message_type='observation' instead of role='tool'.
    The processor should extract turns from both formats.
    """
    processor = SummarizeEveryNTurns(n=2)

    # History with role='user' observations (non-FC mode)
    non_fc_history = [
        {"role": "system", "content": "You are an AI assistant", "message_type": "system"},
        {"role": "user", "content": "Hello", "message_type": "user"},
        {"role": "assistant", "content": "I'll check something", "message_type": "thought", "action": "check"},
        {"role": "user", "content": "Result of check", "message_type": "observation"},  # Non-FC observation
        {"role": "assistant", "content": "I'll look up info", "message_type": "thought", "action": "lookup"},
        {"role": "user", "content": "Information found", "message_type": "observation"},  # Non-FC observation
    ]

    turns = processor._extract_turns_from(non_fc_history)

    # Should extract 2 turns, same as function-calling mode
    assert len(turns) == 2
    assert turns[0] == [non_fc_history[2], non_fc_history[3]]
    assert turns[1] == [non_fc_history[4], non_fc_history[5]]


def test_compact_turns_for_summary_with_user_role_observations():
    """Test that _compact_turns_for_summary handles role='user' observations.

    The compaction should truncate long observations regardless of whether
    they come as role='tool' or role='user' with message_type='observation'.
    """
    processor = SummarizeEveryNTurns(n=2, max_observation_length_for_summary=50)

    # Turns with role='user' observations (non-FC mode)
    turns = [
        [
            {"role": "assistant", "content": "I'll check", "message_type": "thought"},
            {"role": "user", "content": "A" * 100, "message_type": "observation"},  # Long observation
        ],
        [
            {"role": "assistant", "content": "I'll look", "message_type": "thought"},
            {"role": "tool", "content": "B" * 100, "message_type": "observation"},  # Long tool observation
        ],
    ]

    compacted = processor._compact_turns_for_summary(turns)

    # Both observations should be truncated
    assert len(compacted[0][1]["content"]) < 100, "User observation should be truncated"
    assert len(compacted[1][1]["content"]) < 100, "Tool observation should be truncated"
    assert "..." in compacted[0][1]["content"], "User observation should have ellipsis"
    assert "..." in compacted[1][1]["content"], "Tool observation should have ellipsis"


def test_with_incrementally_added_turns(extended_history):
    """Test SummarizeEveryNTurns when turns are added incrementally using extended_history."""
    processor = SummarizeEveryNTurns(n=2)

    # Start with just system and user prompts
    history = extended_history[:2]
    summarized_history = processor(history)
    assert summarized_history == history

    # Add a first turn (assistant + tool)
    history = extended_history[:4]
    summarized_history = processor(history)
    assert summarized_history == history

    # Add a second turn (assistant + tool)
    history = extended_history[:6]
    summarized_history = processor(history)
    assert len(summarized_history) == 3  # system, user, and summary
    assert summarized_history[2]["tags"] == [{"type": "summary"}]

    # Add a third turn (assistant + tool)
    history = extended_history[:8]
    summarized_history = processor(history)
    assert len(summarized_history) == 5  # system, user, summary, and the new turn (2 messages)
    assert summarized_history[2]["tags"] == [{"type": "summary"}]

    # Add a fourth turn (assistant + tool)
    history = extended_history[:10]
    summarized_history = processor(history)
    assert len(summarized_history) == 4  # system, user, first summary, second summary
    assert summarized_history[2]["tags"] == [{"type": "summary"}]
    assert summarized_history[3]["tags"] == [{"type": "summary"}]


def test_successive_processing(extended_history):
    """Test that successive processing works correctly with the cached state, explicitly checking the cache."""
    processor = SummarizeEveryNTurns(n=2)

    # Start with just system and user prompts
    history = extended_history[:2]
    result1 = processor(history)
    assert result1 == history
    assert processor._processed_history_cache == []  # Cache should be empty as no summarization happened

    # Add 2 turns (first 2 turns = 4 more entries)
    history = extended_history[:6]
    result2 = processor(history)
    assert len(result2) == 3  # system, user, summary
    assert result2[2]["tags"] == [{"type": "summary"}]
    # Cache should only contain system, user, and the summary
    assert processor._processed_history_cache == result2

    # Create a new processor to simulate restarting the process
    new_processor = SummarizeEveryNTurns(n=2)
    # Manually set the cache to simulate persistent state
    new_processor._processed_history_cache = processor._processed_history_cache.copy()

    # Add 1 more turn (next 2 entries, not enough for a new summary)
    history = extended_history[:8]
    result3 = new_processor(history)
    assert len(result3) == 5  # system, user, summary, turn 3 (assistant + tool)
    assert result3[2]["tags"] == [{"type": "summary"}]
    # Cache should still only contain system, user, and the summary
    assert new_processor._processed_history_cache == result2

    # Add 1 more turn (now enough for a new summary)
    history = extended_history[:10]
    result4 = new_processor(history)
    assert len(result4) == 4  # system, user, summary1, summary2
    assert result4[2]["tags"] == [{"type": "summary"}] 
    assert result4[3]["tags"] == [{"type": "summary"}] 
    # Cache should now contain system, user, and both summaries
    assert new_processor._processed_history_cache == result4

    # Process one more time without adding new content - should be the same
    result5 = new_processor(history)
    assert result5 == result4
    assert new_processor._processed_history_cache == result4


def test_summarize_with_api_limit_exceeded():
    """Test that SummarizeEveryNTurns continues to work correctly when summary model hits API call limit."""
    from sweagent.types import InstanceStats
    from sweagent.agent.models import AbstractModel
    
    class MockModelWithAPILimit(AbstractModel):
        def __init__(self, api_call_limit=1):
            self.config = type('Config', (), {
                'per_instance_call_limit': api_call_limit
            })()
            self.stats = InstanceStats()
            self.stats.api_calls = 0
            self.summary_system_prompt = "Test system prompt"
        
        def query(self, history, action_prompt="> "):
            # Not used in this test
            return {"message": "test"}
        
        def query_for_summary(self, context, turns, extract_action_from_turns=False,
                               max_kept_action_length=-1, max_kept_reasoning_length=-1,
                               synthesize=False):
            from sweagent.types import SummaryMetadata
            self.stats.api_calls += 1
            return SummaryMetadata(
                summary=f"Mock summary of {len(turns)} turns",
                context=[{"role": "system", "content": context}],
            )
    
    # Create a processor with n=2
    processor = SummarizeEveryNTurns(n=2)
    
    # Create a mock model that will exceed API limit after 1 call
    mock_model = MockModelWithAPILimit(api_call_limit=1)
    processor.set_model(mock_model)
    
    # Create extended history with 6 turns (12 entries after system + user)
    extended_history = [
        {"role": "system", "content": "System prompt", "message_type": "system"},
        {"role": "user", "content": "User message", "message_type": "user"},
    ]
    
    # Add 6 turns (12 entries)
    for i in range(6):
        extended_history.extend([
            {"role": "assistant", "content": f"Assistant message {i+1}", "message_type": "action"},
            {"role": "tool", "content": f"Tool response {i+1}", "message_type": "observation"}
        ])
    
    # First call: should work normally (API limit not exceeded)
    history1 = extended_history[:6]  # system + user + 2 turns
    result1 = processor(history1)
    assert len(result1) == 3  # system, user, summary
    assert "Mock summary of 2 turns" in result1[2]["content"]
    assert mock_model.stats.api_calls == 1

    # second call: hits API limit, turns are preserved (not dropped)
    history2 = extended_history[:10]  # system + user + 4 turns total
    result2 = processor(history2)
    # Should have: cache (sys, user, summary1) + unprocessed turns (2 turns = 4 items) = 7 items
    assert len(result2) == 7, f"Expected 7 items but got {len(result2)}: turns preserved when API limit hit"
    assert mock_model.stats.api_calls == 1  # Should not increment (summarization skipped)
    # Verify the preserved turns are there
    assert result2[3]["role"] == "assistant"  # Turn 3 assistant
    assert result2[4]["role"] == "tool"  # Turn 3 tool

    # third call: API limit still in effect, turns still preserved
    history3 = extended_history[:14]  # system + user + 6 turns total
    result3 = processor(history3)
    # Should have: cache (3) + 4 unprocessed turns (8 items) = 11 items
    assert len(result3) == 11, f"Expected 11 items but got {len(result3)}"
    assert mock_model.stats.api_calls == 1  # Still no increment

    print("✓ API call limit handling preserves turns correctly")


def test_extract_turns_with_existing_summaries():
    """Test that _extract_turns_from correctly handles history with existing summary messages."""
    processor = SummarizeEveryNTurns(n=2)
    
    # Create history with existing summary messages
    history_with_summaries = [
        {"role": "system", "content": "System prompt", "message_type": "system"},
        {"role": "user", "content": "User message", "message_type": "user"},
        {"role": "assistant", "content": "Summary of previous turns", "message_type": "thought", "tags": [{"type": "summary"}]},
        {"role": "assistant", "content": "Assistant message 1", "message_type": "action"},
        {"role": "tool", "content": "Tool response 1", "message_type": "observation"},
        {"role": "assistant", "content": "Assistant message 2", "message_type": "action"},
        {"role": "tool", "content": "Tool response 2", "message_type": "observation"},
    ]
    
    # Should not raise an error and should extract 2 turns (ignoring the summary)
    turns = processor._extract_turns_from(history_with_summaries)
    assert len(turns) == 2
    assert turns[0] == [history_with_summaries[3], history_with_summaries[4]]
    assert turns[1] == [history_with_summaries[5], history_with_summaries[6]]
    
    print("✓ Existing summaries are handled correctly")


def test_keep_last_m_turns_validation():
    """Test that keep_last_m_turns validation works correctly."""
    # Negative value should raise ValueError
    with pytest.raises(ValueError, match="keep_last_m_turns must be non-negative"):
        SummarizeEveryNTurns(n=2, keep_last_m_turns=-1)
    
    # Zero should be valid
    processor = SummarizeEveryNTurns(n=2, keep_last_m_turns=0)
    assert processor.keep_last_m_turns == 0
    
    # Positive values should be valid
    processor = SummarizeEveryNTurns(n=2, keep_last_m_turns=3)
    assert processor.keep_last_m_turns == 3


def test_keep_last_m_turns_zero(extended_history):
    """Test that keep_last_m_turns=0 behaves the same as the original implementation."""
    processor_with_zero = SummarizeEveryNTurns(n=2, keep_last_m_turns=0)
    processor_default = SummarizeEveryNTurns(n=2)
    
    # Both should behave identically
    for i in range(2, len(extended_history), 2):
        history_slice = extended_history[:i]
        result_zero = processor_with_zero(history_slice)
        result_default = processor_default(history_slice)
        assert result_zero == result_default


def test_keep_last_m_turns_smaller_than_n(extended_history):
    """Test keep_last_m_turns with a value smaller than n (keep_last_m_turns=1, n=3)."""
    processor = SummarizeEveryNTurns(n=3, keep_last_m_turns=1)
    
    # Need at least n + keep_last_m_turns = 4 turns before summarization
    # Start with 3 turns (6 entries after system + user) - should not summarize
    history = extended_history[:8]  # system + user + 3 turns
    result = processor(history)
    assert result == history  # No summarization should occur
    
    # Add 1 more turn (4 total) - should trigger summarization
    history = extended_history[:10]  # system + user + 4 turns
    result = processor(history)
    # Should have: system + user + summary + 1 remaining turn
    assert len(result) == 5
    assert result[0] == extended_history[0]  # system
    assert result[1] == extended_history[1]  # user
    assert result[2]["tags"] == [{"type": "summary"}]  # summary
    # Last turn should be kept
    assert result[3] == extended_history[8]  # last turn's assistant message
    assert result[4] == extended_history[9]  # last turn's tool message


def test_keep_last_m_turns_larger_than_n(extended_history):
    """Test keep_last_m_turns with a value larger than n (keep_last_m_turns=5, n=2)."""
    processor = SummarizeEveryNTurns(n=2, keep_last_m_turns=5)
    
    # Start with 6 turns (12 entries after system + user) - should not summarize
    history = extended_history[:14]
    result = processor(history)
    assert result == history
    
    result = processor(extended_history)
    
    # Should have: system + user + summary + 5 remaining turns (10 entries) = 13
    assert len(result) == 13
    assert result[0] == extended_history[0]  # system
    assert result[1] == extended_history[1]  # user
    assert result[2]["tags"] == [{"type": "summary"}]
    # Remaining 5 turns (10 entries) == keep_last_m_turns should be kept
    assert len(result[3:]) == 10


def test_keep_last_m_turns_equal_to_n(extended_history):
    """Test keep_last_m_turns when it equals n (both set to 2)."""
    processor = SummarizeEveryNTurns(n=2, keep_last_m_turns=2)
    
    # Need at least n + keep_last_m_turns = 4 turns before summarization
    result = processor(extended_history[:10])
    
    # Should have: system + user + summary + 2 remaining turns (4 entries) = 7
    assert len(result) == 7
    assert result[0] == extended_history[0]  # system
    assert result[1] == extended_history[1]  # user
    assert result[2]["tags"] == [{"type": "summary"}]  # summary
    # Last 2 turns (4 entries) should be kept
    assert len(result[3:]) == 4


def test_keep_last_m_turns_incremental_processing(extended_history):
    """Test that keep_last_m_turns works correctly with incremental history processing."""
    processor = SummarizeEveryNTurns(n=2, keep_last_m_turns=1)
    
    # With n=2, keep_last_m_turns=1, need at least 3 turns before summarization
    
    # Start with 2 turns - should not summarize (need 3 total)
    history = extended_history[:6]  # system + user + 2 turns
    result1 = processor(history)
    assert result1 == history  # No summarization
    
    # Add third turn - should trigger summarization
    history = extended_history[:8]  # system + user + 3 turns
    result2 = processor(history)
    assert len(result2) == 5  # system + user + summary + 1 remaining turn (2 entries)
    assert result2[2]["tags"] == [{"type": "summary"}]
    # Last turn should be kept
    assert result2[3] == extended_history[6]  # last turn's assistant message
    assert result2[4] == extended_history[7]  # last turn's tool message
    
    # Add fourth turn - should not trigger new summarization yet (need 3 unprocessed again)
    history = extended_history[:10]  # system + user + 4 turns
    result3 = processor(history)
    assert len(result3) == 7  # previous summary result + 1 new turn (2 entries)
    
    # Add fifth turn - should trigger second summarization 
    history = extended_history[:12]  # system + user + 5 turns
    result4 = processor(history)
    assert len(result4) == 6  # system + user + summary1 + summary2 + 1 remaining turn (2 entries)
    # Should have two summary messages
    summary_count = len([item for item in result4 if item.get("tags") == [{"type": "summary"}]])
    assert summary_count == 2


class DummyContextModel:
    def __init__(self, *, context_window: int, tokens: int):
        self.model_context_window = context_window
        self._tokens = tokens

    def count_tokens(self, history: History) -> int:  # pragma: no cover - trivial
        return self._tokens

    def set_tokens(self, tokens: int) -> None:
        self._tokens = tokens


def _has_summary(history: History) -> bool:
    for entry in history:
        for tag in entry.get("tags", []):
            if isinstance(tag, dict) and tag.get("type") == "summary":
                return True
    return False


def test_limit_aware_last_n_observations_skips_below_threshold(extended_history: History):
    processor = LastNObservations(
        n=3,
        enable_limit_aware_trigger=True,
        limit_trigger_fraction=0.9,
    )
    processor.set_context_model(DummyContextModel(context_window=1000, tokens=100))

    new_history = processor(extended_history)

    assert new_history == extended_history
    assert count_elided_observations(new_history) == 0


def test_limit_aware_last_n_observations_triggers_near_limit(extended_history: History):
    processor = LastNObservations(
        n=3,
        enable_limit_aware_trigger=True,
        limit_trigger_fraction=0.9,
    )
    processor.set_context_model(DummyContextModel(context_window=1000, tokens=950))

    new_history = processor(extended_history)

    total_observations = len([entry for entry in extended_history if entry["message_type"] == "observation"])
    anchor_observations = 1  # keep the first observation (instance template)
    expected_elided_observations = total_observations - 3 - anchor_observations
    assert count_elided_observations(new_history) == expected_elided_observations


def test_limit_aware_summary_skips_below_threshold(extended_history: History):
    processor = SummarizeEveryNTurns(
        n=2,
        keep_last_m_turns=0,
        omit_turns=True,
        enable_limit_aware_trigger=True,
        limit_trigger_fraction=0.9,
    )
    processor.set_context_model(DummyContextModel(context_window=1000, tokens=100))

    new_history = processor(extended_history)

    assert new_history == extended_history
    assert not _has_summary(new_history)


def test_limit_aware_summary_triggers_near_limit(extended_history: History):
    processor = SummarizeEveryNTurns(
        n=2,
        keep_last_m_turns=0,
        omit_turns=True,
        enable_limit_aware_trigger=True,
        limit_trigger_fraction=0.9,
    )
    processor.set_context_model(DummyContextModel(context_window=1000, tokens=950))

    new_history = processor(extended_history)

    assert new_history != extended_history
    assert _has_summary(new_history)


def test_limit_aware_summary_preserves_compacted_view_when_below_threshold(extended_history: History):
    context_model = DummyContextModel(context_window=1000, tokens=950)
    processor = SummarizeEveryNTurns(
        n=2,
        keep_last_m_turns=0,
        omit_turns=True,
        enable_limit_aware_trigger=True,
        limit_trigger_fraction=0.9,
    )
    processor.set_context_model(context_model)

    # First call compacts (near limit)
    compacted = processor(extended_history[:6])
    assert _has_summary(compacted)
    assert len(compacted) == 3

    # Subsequent call should keep the compacted view even if we're now below the threshold
    context_model.set_tokens(100)
    result = processor(extended_history[:8])
    assert _has_summary(result)
    assert len(result) < len(extended_history[:8])


def test_messages_cache_invalidates_on_history_growth():
    from sweagent.agent.agents import DefaultAgent

    calls = 0

    class CountingProcessor:
        def __call__(self, history: History) -> History:
            nonlocal calls
            calls += 1
            return history

    agent = object.__new__(DefaultAgent)
    agent.name = "agent"
    agent.history_processors = [CountingProcessor()]
    agent.history = [
        {
            "agent": "agent",
            "role": "system",
            "content": "s",
            "message_type": "system_prompt",
        }
    ]

    _ = agent.messages
    assert calls == 1

    # Grow history without manually clearing cache; messages should recompute.
    agent.history.append(
        {
            "agent": "agent",
            "role": "user",
            "content": "u",
            "message_type": "observation",
        }
    )

    _ = agent.messages
    assert calls == 2

    # No history change -> cache should hold.
    _ = agent.messages
    assert calls == 2

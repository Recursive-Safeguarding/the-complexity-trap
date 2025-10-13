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
    # extra -1 because instance template is kept
    expected_elided_observations = total_observations - 3 - 1
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
    """Test that _extract_turns_from raises an error for invalid history formats."""
    processor = SummarizeEveryNTurns(n=2)

    # Tool as first message
    invalid_history_1 = [
        {"role": "tool", "content": "Invalid start", "message_type": "observation"}
    ]
    with pytest.raises(AssertionError) as exc1:
        processor._extract_turns_from(invalid_history_1)
    assert "unexpected history format" in str(exc1.value).lower()
    assert "tool call cannot be the first message" in str(exc1.value).lower()

    # Tool not preceded by assistant
    invalid_history_2 = [
        {"role": "system", "content": "System prompt", "message_type": "system"},
        {"role": "user", "content": "User message", "message_type": "user"},
        {"role": "tool", "content": "Invalid sequence", "message_type": "observation"}
    ]
    with pytest.raises(AssertionError) as exc2:
        processor._extract_turns_from(invalid_history_2)
    assert "unexpected history format" in str(exc2.value).lower()
    assert "preceded by an assistant message" in str(exc2.value).lower()


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
        
        def query_for_summary(self, context, turns):
            self.stats.api_calls += 1
            return f"Mock summary of {len(turns)} turns"
    
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
    
    # Second call: should hit API limit but still return a summary placeholder
    history2 = extended_history[:10]  # system + user + 4 turns total
    result2 = processor(history2)
    assert len(result2) == 4  # system, user, first summary, second summary/placeholder
    assert "Summary skipped due to API call limit" in result2[3]["content"]
    assert mock_model.stats.api_calls == 1  # Should not increment
    
    # Third call: should still work (even though API limit was hit before)
    # This tests that the counting logic wasn't broken by the API limit
    history3 = extended_history[:14]  # system + user + 6 turns total
    result3 = processor(history3)
    # Should have 5 items: system, user, summary1, placeholder2, summary3/placeholder3
    assert len(result3) == 5
    assert "summary" in result3[4]["content"].lower()
    
    print("✓ API call limit handling works correctly")


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
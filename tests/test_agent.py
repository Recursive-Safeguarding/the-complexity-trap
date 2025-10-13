import pytest
import yaml
from swerex.exceptions import SwerexException
from swerex.runtime.abstract import Action, BashObservation, Observation
from swerex.runtime.dummy import DummyRuntime

from sweagent import CONFIG_DIR
from sweagent.agent.agents import DefaultAgent, DefaultAgentConfig
from sweagent.agent.models import InstantEmptySubmitModelConfig, PredeterminedTestModel
from sweagent.agent.problem_statement import EmptyProblemStatement, TextProblemStatement
from sweagent.environment.swe_env import SWEEnv
from sweagent.tools.parsing import FunctionCallingParser, Identity, ThoughtActionParser
from sweagent.tools.tools import ToolConfig
from sweagent.agent.models import GenericAPIModelConfig
from sweagent.types import InstanceStats
from sweagent.agent.history_processors import SummarizeEveryNTurns
from sweagent.agent.models import GrazieModel
from litellm.utils import token_counter

def test_dummy_env(dummy_env):
    pass


@pytest.fixture
def identity_agent_config():
    return DefaultAgentConfig(
        model=InstantEmptySubmitModelConfig(),
        tools=ToolConfig(
            parse_function=Identity(),
        ),
    )


@pytest.fixture
def thought_action_agent_config():
    return DefaultAgentConfig(
        model=InstantEmptySubmitModelConfig(),
        tools=ToolConfig(
            parse_function=ThoughtActionParser(),
        ),
    )


@pytest.fixture
def function_calling_agent_config():
    return DefaultAgentConfig(
        model=InstantEmptySubmitModelConfig(),
        tools=ToolConfig(
            parse_function=FunctionCallingParser(),
        ),
    )


@pytest.fixture
def default_agent_config():
    config = yaml.safe_load((CONFIG_DIR / "default_no_fcalls.yaml").read_text())
    config["agent"]["model"] = {"name": "instant_empty_submit"}
    print(yaml.dump(config))
    return DefaultAgentConfig.model_validate(config["agent"])


@pytest.fixture
def default_agent(default_agent_config: DefaultAgentConfig) -> DefaultAgent:
    a = DefaultAgent.from_config(default_agent_config)
    a.tools.mock_state = {"open_file": "asdf123", "working_dir": "/root"}
    return a


@pytest.fixture
def test_agent(identity_agent_config: DefaultAgentConfig) -> DefaultAgent:
    return DefaultAgent.from_config(identity_agent_config)


@pytest.fixture
def thought_action_agent(thought_action_agent_config: DefaultAgentConfig) -> DefaultAgent:
    return DefaultAgent.from_config(thought_action_agent_config)


@pytest.fixture
def function_calling_agent(function_calling_agent_config: DefaultAgentConfig) -> DefaultAgent:
    return DefaultAgent.from_config(function_calling_agent_config)


def test_exit_cost(dummy_env: SWEEnv, test_agent: DefaultAgent, tmp_path):
    test_agent.model = PredeterminedTestModel(["raise_cost"])  # type: ignore
    r = test_agent.run(
        problem_statement=EmptyProblemStatement(),
        env=dummy_env,
        output_dir=tmp_path,
    )
    assert r.info["exit_status"] == "exit_cost"  # type: ignore


def test_exit_context(dummy_env: SWEEnv, test_agent: DefaultAgent, tmp_path):
    test_agent.model = PredeterminedTestModel(["raise_context"])  # type: ignore
    r = test_agent.run(
        problem_statement=EmptyProblemStatement(),
        env=dummy_env,
        output_dir=tmp_path,
    )
    assert r.info["exit_status"] == "exit_context"  # type: ignore


def test_exit_model_error(dummy_env: SWEEnv, test_agent: DefaultAgent, tmp_path):
    test_agent.model = PredeterminedTestModel(["raise_runtime"])  # type: ignore
    r = test_agent.run(
        problem_statement=EmptyProblemStatement(),
        env=dummy_env,
        output_dir=tmp_path,
    )
    assert r.info["exit_status"] == "exit_environment_error"  # type: ignore


def test_exit_format(dummy_env: SWEEnv, thought_action_agent: DefaultAgent, tmp_path):
    thought_action_agent.model = PredeterminedTestModel(["a", "b", "c", "d"])  # type: ignore
    r = thought_action_agent.run(
        problem_statement=EmptyProblemStatement(),
        env=dummy_env,
        output_dir=tmp_path,
    )
    assert r.info["exit_status"] == "exit_format"  # type: ignore


def test_exit_blocklist(dummy_env: SWEEnv, test_agent: DefaultAgent, tmp_path):
    test_agent.model = PredeterminedTestModel(["vim", "python", "su", "nano"])  # type: ignore
    r = test_agent.run(
        problem_statement=EmptyProblemStatement(),
        env=dummy_env,
        output_dir=tmp_path,
    )
    assert r.info["exit_status"] == "exit_format"  # type: ignore


class RuntimeRaisesFirst(DummyRuntime):
    async def run_in_session(self, action: Action) -> Observation:
        if action.action_type == "bash" and action.command == "raise":
            raise SwerexException()
        return await super().run_in_session(action)


def test_early_exit(dummy_env: SWEEnv, test_agent: DefaultAgent, tmp_path):
    test_agent.model = PredeterminedTestModel(["raise"])  # type: ignore
    test_agent._catch_errors = True
    dummy_env.deployment.runtime = RuntimeRaisesFirst()  # type: ignore
    r = test_agent.run(
        problem_statement=EmptyProblemStatement(),
        env=dummy_env,
        output_dir=tmp_path,
    )
    assert r.info["exit_status"] == "exit_environment_error"  # type: ignore


def test_run_step_by_step_checking_history(dummy_env: SWEEnv, default_agent: DefaultAgent, tmp_path):
    a = default_agent
    a.model = PredeterminedTestModel(["asdf", "```\nls\n```", "```\necho 'asdf'\n```", "raise_cost"])  # type: ignore
    a.setup(dummy_env, TextProblemStatement(text="asdf123"))
    dummy_env.deployment.runtime.run_in_session_outputs = [  # type: ignore
        BashObservation(output="file_a file_b"),
        BashObservation(output=""),  # set last action
        BashObservation(output="asdf"),
        BashObservation(output=""),
    ]
    assert "asdf123" in a._problem_statement.get_problem_statement()  # type: ignore
    # system template and demo and instance template
    assert len(a.messages) == 3
    system_prompt = a.messages[0]["content"]
    assert "You are an autonomous programmer" in system_prompt
    demo = a.messages[1]["content"]
    # print(demo)
    assert "demonstration" in demo  # demo
    assert "marshmallow" in demo  # demo
    instance_template = a.messages[2]["content"]
    assert "the following issue within our repository" in instance_template
    assert "asdf123" in instance_template
    assert len(a.trajectory) == 0
    print(a.step())
    assert len(a.trajectory) == 2  # we requery once because format error
    assert len(a.messages) == 5  # first action performed + observation
    print(yaml.dump(a.messages, indent=2))
    assert a.messages[3]["content"].strip() == "```\nls\n```"
    assert "file_a file_b" in a.messages[4]["content"]
    assert "Open file: asdf123" in a.messages[4]["content"]
    assert "Current directory: /root" in a.messages[4]["content"]
    print(a.step())
    print(yaml.dump(a.messages, indent=2))
    assert len(a.trajectory) == 3
    assert len(a.messages) == 7
    print(a.step())
    assert len(a.trajectory) == 4
    assert a.info["exit_status"] == "exit_cost"  # type: ignore


# todo: fixme; Needs real environment or mocking of read_file
@pytest.mark.xfail
def test_run_autosubmit(dummy_env: SWEEnv, default_agent: DefaultAgent, tmp_path):
    a = default_agent
    a.model = PredeterminedTestModel(["raise_cost"])  # type: ignore
    a.setup(dummy_env, EmptyProblemStatement())
    dummy_env.write_file("/root/model.patch", "mysubmission")
    dummy_env.deployment.runtime.run_in_session_outputs = [  # type: ignore
        BashObservation(output=""),
        BashObservation(output=r"<<SWE_AGENT_SUBMISSION>>\nmysubmission\n<<SWE_AGENT_SUBMISSION>>"),
    ]
    r = a.step()
    assert a.info is not None
    assert a.info["exit_status"] == "submitted (exit_cost)"  # type: ignore
    assert a.info["submission"] == "mysubmission"  # type: ignore
    assert r.done
    assert r.submission == "mysubmission"
    assert r.exit_status == "submitted (exit_cost)"
    assert not r.action
    assert "cost limit" in r.thought


def test_show_no_output_template(dummy_env: SWEEnv, default_agent: DefaultAgent, tmp_path):
    a = default_agent
    a.templates.next_step_no_output_template = "no output template"
    a.setup(dummy_env, EmptyProblemStatement())
    a.model = PredeterminedTestModel(["```\nls\n```", "```\ntest\n```"])  # type: ignore
    dummy_env.deployment.runtime.run_in_session_outputs = [BashObservation(output="")]  # type: ignore
    a.step()
    a.step()
    # todo: actually test that the template is used


# todo: fixme; Needs real environment or mocking of read_file
@pytest.mark.xfail
def test_successful_submission(dummy_env: SWEEnv, default_agent: DefaultAgent, tmp_path):
    a = default_agent
    a.model = PredeterminedTestModel(["```\nsubmit\n```"])  # type: ignore
    a.setup(dummy_env, EmptyProblemStatement())
    dummy_env.write_file("/root/model.patch", "test")
    dummy_env.deployment.runtime.run_in_session_outputs = BashObservation(output=r"<<SWE_AGENT_SUBMISSION>>")  # type: ignore
    a.step()
    assert a.info["exit_status"] == "submitted"  # type: ignore
    assert a.info["submission"] == "test"  # type: ignore
    assert a.trajectory[-1]["observation"] == "test"


def test_human_exit(dummy_env: SWEEnv, default_agent: DefaultAgent, tmp_path):
    a = default_agent
    a.model = PredeterminedTestModel(["```\nexit\n```"])  # type: ignore
    a.setup(dummy_env, EmptyProblemStatement())
    r = a.step()
    assert r.done
    assert r.exit_status == "exit_command"
    assert r.action.strip() == "exit"


def test_function_calling(dummy_env: SWEEnv, function_calling_agent: DefaultAgent, tmp_path):
    a = function_calling_agent
    # Simulate a valid function call response from the model
    valid_response = {
        "message": "I'll list the contents of the directory",
        "tool_calls": [{"function": {"name": "bash", "arguments": '{"command": "ls"}'}, "id": "abc123"}],
    }
    a.model = PredeterminedTestModel([valid_response])  # type: ignore
    a.setup(dummy_env, EmptyProblemStatement())
    dummy_env.deployment.runtime.run_in_session_outputs = [  # type: ignore
        BashObservation(output="file1 file2"),
        BashObservation(output="file1 file2"),  # TODO, there's actually a bug in swe-rex, requiring two observations
    ]  # type: ignore
    r = a.step()
    assert not r.done, "Expected not done, because we haven't submitted yet"
    assert r.action.strip() == "ls", "Expected the tool call to be executed"
    assert "file1 file2" in r.observation, "Expected the tool call to return the output of the command"

def test_stats_injection():
    """Test that stats are properly injected and shared between models."""
    
    # Create a config with both main model and summary model
    config = DefaultAgentConfig(
        model=GenericAPIModelConfig(name="gpt-4o-mini"),
        summary_model=GenericAPIModelConfig(name="gpt-3.5-turbo"),
        history_processors=[SummarizeEveryNTurns(n=2)]
    )
    
    # Create agent from config
    agent = DefaultAgent.from_config(config)
    
    # Verify that the agent has stats
    assert hasattr(agent, 'stats'), "Agent should have stats attribute"
    assert isinstance(agent.stats, InstanceStats), "Agent stats should be InstanceStats instance"
    
    # Verify that the main model has the same stats object as the agent
    assert agent.model.stats is agent.stats, "Main model should share stats with agent"
    
    # Check if there's a summary model (different from main model)
    summary_processor = None
    for processor in agent.history_processors:
        if isinstance(processor, SummarizeEveryNTurns):
            summary_processor = processor
            break
    
    if summary_processor and summary_processor._model:
        assert summary_processor._model.stats is agent.stats, "Summary model should share stats with agent"
        print("✓ Summary model shares stats with agent")
    
    # Test that stats are properly updated when models are used
    initial_cost = agent.stats.instance_cost
    initial_calls = agent.stats.api_calls
    
    # Simulate some cost and API calls
    agent.stats.instance_cost += 0.05
    agent.stats.api_calls += 1
    
    # Verify that all models see the same updated stats
    assert agent.model.stats.instance_cost == initial_cost + 0.05, "Main model should see updated cost"
    assert agent.model.stats.api_calls == initial_calls + 1, "Main model should see updated API calls"
    
    if summary_processor and summary_processor._model:
        assert summary_processor._model.stats.instance_cost == initial_cost + 0.05, "Summary model should see updated cost"
        assert summary_processor._model.stats.api_calls == initial_calls + 1, "Summary model should see updated API calls"
    
    print("✓ All models share the same stats object")
    print("✓ Stats updates are visible across all models")
    print("✓ Stats injection implementation is working correctly!")


def test_stats_injection_with_same_model():
    """Test stats injection when main model and summary model are the same."""
    
    config = DefaultAgentConfig(
        model=GenericAPIModelConfig(name="gpt-4o-mini"),
        history_processors=[SummarizeEveryNTurns(n=2)]
    )
    
    agent = DefaultAgent.from_config(config)
    
    summary_processor = None
    for processor in agent.history_processors:
        if isinstance(processor, SummarizeEveryNTurns):
            summary_processor = processor
            break
    
    # Verify that the summary model is the same object as the main model
    if summary_processor:
        assert summary_processor._model is agent.model, "Summary model should be the same as main model when not specified"
        print("✓ Summary model correctly defaults to main model")
    
    # Verify stats are still shared
    assert agent.model.stats is agent.stats, "Main model should share stats with agent"
    
    print("✓ Stats injection works correctly when using same model for both main and summary")


def test_grazie_normalize_prompt_to_history():
    """Test the _normalize_prompt_to_history method for GrazieModel"""
    # Create a mock GrazieModel without initializing the client
    model = object.__new__(GrazieModel)
    model._cached_context = None
    
    # Test the normalization method
    system_prompt = "You are a helpful assistant."
    user_prompt = "Please help me with this task."
    
    result = model._normalize_prompt_to_history(system_prompt, user_prompt)
    
    expected = [
        {
            "role": "system",
            "content": system_prompt,
            "message_type": "system_prompt"
        },
        {
            "role": "user", 
            "content": user_prompt,
            "message_type": "observation"
        }
    ]
    
    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ _normalize_prompt_to_history works correctly")


def test_grazie_calculate_cached_tokens():
    """Test the _calculate_cached_tokens method for GrazieModel"""
    # Create a mock GrazieModel without initializing the client
    model = object.__new__(GrazieModel)
    
    # Test with no cached context
    model._cached_context = None
    current_history = [
        {"role": "system", "content": "System message", "message_type": "system_prompt"},
        {"role": "user", "content": "User message", "message_type": "observation"}
    ]
    
    cached_tokens = model._calculate_cached_tokens(current_history)
    assert cached_tokens == 0, f"Expected 0 cached tokens with no cache, got {cached_tokens}"
    
    # Test with identical cached context
    model._cached_context = current_history.copy()
    cached_tokens = model._calculate_cached_tokens(current_history)
    assert cached_tokens == token_counter(messages=model._cached_context)
    
    # Test with partial match
    model._cached_context = [
        {"role": "system", "content": "System message", "message_type": "system_prompt"},
        {"role": "user", "content": "Different user message", "message_type": "observation"}
    ]
    cached_tokens = model._calculate_cached_tokens(current_history)
    assert cached_tokens == token_counter(messages=[model._cached_context[0]])
    
    # Test with no match
    model._cached_context = [
        {"role": "system", "content": "Different system message", "message_type": "system_prompt"},
        {"role": "user", "content": "Different user message", "message_type": "observation"}
    ]
    cached_tokens = model._calculate_cached_tokens(current_history)
    assert cached_tokens == 0, f"Expected 0 cached tokens with no match, got {cached_tokens}"


def test_litellm_normalize_prompt_to_history():
    """Test the _normalize_prompt_to_history method for LiteLLMModel"""
    from sweagent.agent.models import LiteLLMModel
    
    # Create a mock LiteLLMModel without full initialization
    model = object.__new__(LiteLLMModel)
    model._cached_context = None
    
    # Test the normalization method
    system_prompt = "You are a helpful assistant."
    user_prompt = "Please help me with this task."
    
    result = model._normalize_prompt_to_history(system_prompt, user_prompt)
    
    expected = [
        {
            "role": "system",
            "content": system_prompt,
            "message_type": "system_prompt"
        },
        {
            "role": "user", 
            "content": user_prompt,
            "message_type": "observation"
        }
    ]
    
    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ LiteLLMModel _normalize_prompt_to_history works correctly")


def test_litellm_calculate_cached_tokens():
    """Test the _calculate_cached_tokens method for LiteLLMModel"""
    from sweagent.agent.models import LiteLLMModel
    
    # Create a mock LiteLLMModel without full initialization
    model = object.__new__(LiteLLMModel)
    
    # Test with no cached context
    model._cached_context = None
    current_history = [
        {"role": "system", "content": "System message", "message_type": "system_prompt"},
        {"role": "user", "content": "User message", "message_type": "observation"}
    ]
    
    cached_tokens = model._calculate_cached_tokens(current_history)
    assert cached_tokens == 0, f"Expected 0 cached tokens with no cache, got {cached_tokens}"
    
    # Test with identical cached context
    model._cached_context = current_history.copy()
    cached_tokens = model._calculate_cached_tokens(current_history)
    assert cached_tokens == token_counter(messages=model._cached_context)
    
    # Test with partial match
    model._cached_context = [
        {"role": "system", "content": "System message", "message_type": "system_prompt"},
        {"role": "user", "content": "Different user message", "message_type": "observation"}
    ]
    cached_tokens = model._calculate_cached_tokens(current_history)
    assert cached_tokens == token_counter(messages=[model._cached_context[0]])
    
    # Test with no match
    model._cached_context = [
        {"role": "system", "content": "Different system message", "message_type": "system_prompt"},
        {"role": "user", "content": "Different user message", "message_type": "observation"}
    ]
    cached_tokens = model._calculate_cached_tokens(current_history)
    assert cached_tokens == 0, f"Expected 0 cached tokens with no match, got {cached_tokens}"
    
    print("✓ LiteLLMModel _calculate_cached_tokens works correctly")
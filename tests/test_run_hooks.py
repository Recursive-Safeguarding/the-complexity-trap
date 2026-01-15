import os
import re
from types import SimpleNamespace

import pytest

from sweagent.agent.problem_statement import GithubIssue
from sweagent.run.hooks.open_pr import OpenPRConfig, OpenPRHook
from sweagent.run.hooks import open_pr as open_pr_module
from sweagent.utils.github import InvalidGithubURL
from sweagent.types import AgentRunResult


@pytest.fixture(autouse=True)
def _stub_github_api(monkeypatch):
    pattern = re.compile(r"github\\.com/[^/]+/[^/]+/issues/(\\d+)")

    def _fake_issue(url: str, *, token: str = ""):
        match = pattern.search(url)
        if not match:
            raise InvalidGithubURL(f"Invalid GitHub issue URL: {url}")
        issue_number = match.group(1)
        if issue_number == "16":
            return SimpleNamespace(state="closed", assignee=None, locked=False)
        if issue_number == "17":
            return SimpleNamespace(state="open", assignee="someone", locked=False)
        if issue_number == "18":
            return SimpleNamespace(state="open", assignee=None, locked=True)
        return SimpleNamespace(state="open", assignee=None, locked=False)

    def _fake_commits(org: str, repo: str, issue_number: str, *, token: str = "") -> list[str]:
        if str(issue_number) == "19":
            return ["https://github.com/swe-agent/test-repo/commit/abc123"]
        return []

    monkeypatch.setattr(open_pr_module, "_get_gh_issue_data", _fake_issue)
    monkeypatch.setattr(open_pr_module, "_get_associated_commit_urls", _fake_commits)


@pytest.fixture
def open_pr_hook_init_for_sop():
    hook = OpenPRHook(config=OpenPRConfig(skip_if_commits_reference_issue=True))
    hook._token = os.environ.get("GITHUB_TOKEN", "")
    hook._problem_statement = GithubIssue(github_url="https://github.com/swe-agent/test-repo/issues/1")
    return hook


@pytest.fixture
def agent_run_result():
    return AgentRunResult(
        info={
            "submission": "asdf",
            "exit_status": "submitted",
        },
        trajectory=[],
    )


def test_should_open_pr_fail_submission(open_pr_hook_init_for_sop, agent_run_result):
    hook = open_pr_hook_init_for_sop
    agent_run_result.info["submission"] = None
    assert not hook.should_open_pr(agent_run_result)


def test_should_open_pr_fail_exit(open_pr_hook_init_for_sop, agent_run_result):
    hook = open_pr_hook_init_for_sop
    agent_run_result.info["exit_status"] = "fail"
    assert not hook.should_open_pr(agent_run_result)


def test_should_open_pr_fail_invalid_url(open_pr_hook_init_for_sop, agent_run_result):
    hook = open_pr_hook_init_for_sop
    hook._problem_statement = SimpleNamespace(github_url="asdf")
    assert not hook.should_open_pr(agent_run_result)


def test_should_open_pr_fail_closed(open_pr_hook_init_for_sop, agent_run_result):
    hook = open_pr_hook_init_for_sop
    hook._problem_statement = GithubIssue(github_url="https://github.com/swe-agent/test-repo/issues/16")
    assert not hook.should_open_pr(agent_run_result)


def test_should_open_pr_fail_assigned(open_pr_hook_init_for_sop, agent_run_result):
    hook = open_pr_hook_init_for_sop
    hook._problem_statement = GithubIssue(github_url="https://github.com/swe-agent/test-repo/issues/17")
    assert not hook.should_open_pr(agent_run_result)


def test_should_open_pr_fail_locked(open_pr_hook_init_for_sop, agent_run_result):
    hook = open_pr_hook_init_for_sop
    hook._problem_statement = GithubIssue(github_url="https://github.com/swe-agent/test-repo/issues/18")
    assert not hook.should_open_pr(agent_run_result)


def test_should_open_pr_fail_has_pr(open_pr_hook_init_for_sop, agent_run_result):
    hook = open_pr_hook_init_for_sop
    hook._problem_statement = GithubIssue(github_url="https://github.com/swe-agent/test-repo/issues/19")
    assert not hook.should_open_pr(agent_run_result)


def test_should_open_pr_success_has_pr_override(open_pr_hook_init_for_sop, agent_run_result, monkeypatch):
    hook = open_pr_hook_init_for_sop
    hook._problem_statement = GithubIssue(github_url="https://github.com/swe-agent/test-repo/issues/19")
    hook._config = OpenPRConfig(skip_if_commits_reference_issue=False)
    # Ensure deterministic behavior regardless of external GitHub state.
    monkeypatch.setattr(
        open_pr_module,
        "_get_gh_issue_data",
        lambda url, *, token="": SimpleNamespace(state="open", assignee=None, locked=False),
    )
    monkeypatch.setattr(
        open_pr_module,
        "_get_associated_commit_urls",
        lambda org, repo, issue_number, *, token="": ["https://github.com/swe-agent/test-repo/commit/abc123"],
    )
    assert hook.should_open_pr(agent_run_result)

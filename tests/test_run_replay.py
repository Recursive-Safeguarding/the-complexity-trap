from __future__ import annotations

import os
import shutil
import subprocess

import pytest
from swerex.deployment.config import DockerDeploymentConfig

from sweagent.run.run_replay import RunReplay, RunReplayConfig


@pytest.fixture
def rr_config(swe_agent_test_repo_traj, tmp_path, swe_agent_test_repo_clone):
    return RunReplayConfig(
        traj_path=swe_agent_test_repo_traj,
        deployment=DockerDeploymentConfig(image="python:3.11"),
        output_dir=tmp_path,
    )


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(
            ["docker", "info"], capture_output=True, text=True, timeout=5
        )
    except Exception:
        return False
    return result.returncode == 0


def test_replay(rr_config):
    if not _docker_available():
        pytest.skip("Docker not available")
    if not os.environ.get("GITHUB_TOKEN"):
        pytest.skip("Requires GITHUB_TOKEN (avoids flaky unauthenticated GitHub API rate limiting)")
    rr = RunReplay.from_config(rr_config, _catch_errors=False, _require_zero_exit_code=True)
    rr.main()


def test_run_cli_help():
    args = [
        "sweagent",
        "run-replay",
        "--help",
    ]
    output = subprocess.run(args, capture_output=True)
    assert output.returncode == 0
    assert "Replay a trajectory file" in output.stdout.decode()

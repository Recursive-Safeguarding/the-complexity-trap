from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import pytest
from swerex.deployment.config import DockerDeploymentConfig, DummyDeploymentConfig

from sweagent.environment.repo import LocalRepoConfig
from sweagent.environment.swe_env import EnvironmentConfig, SWEEnv

# this is a hack and should be removed when we have a better solution
_this_dir = Path(__file__).resolve().parent
root_dir = _this_dir.parent
package_dir = root_dir / "sweagent"
sys.path.insert(0, str(root_dir))
sys.path.insert(1, str(package_dir))

# Ensure repo-local resources are used when running tests from an installed package.
os.environ.setdefault("SWE_AGENT_CONFIG_DIR", str(root_dir / "config"))
os.environ.setdefault("SWE_AGENT_TOOLS_DIR", str(root_dir / "tools"))
os.environ.setdefault("SWE_AGENT_TRAJECTORY_DIR", str(root_dir / "trajectories"))

# Ensure the venv entrypoints (e.g., sweagent) are discoverable in subprocess tests.
venv_bin = root_dir / ".venv" / "bin"
if venv_bin.exists():
    os.environ["PATH"] = f"{venv_bin}:{os.environ.get('PATH', '')}"

_DOCKER_AVAILABLE: tuple[bool, str] | None = None


def _docker_available_cached() -> tuple[bool, str]:
    """Return (ok, reason) for whether Docker is usable.

    Many integration tests require a running Docker daemon. When Docker is not
    available (common on fresh laptops/CI runners), we skip those tests rather
    than failing with opaque socket errors.
    """
    global _DOCKER_AVAILABLE
    if _DOCKER_AVAILABLE is not None:
        return _DOCKER_AVAILABLE
    try:
        subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=True,
        )
        # Many of our Docker integration tests also need network egress from
        # inside containers (e.g., apt-get during image builds). A running daemon
        # is not sufficient if Docker can't pull images or reach package mirrors.
        try:
            subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "debian:bookworm-slim",
                    "bash",
                    "-lc",
                    # Fast-ish, representative probe: package index fetch + one small install.
                    "apt-get update -qq && apt-get install -y -qq wget >/dev/null",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=90,
                check=True,
            )
            _DOCKER_AVAILABLE = (True, "")
        except subprocess.TimeoutExpired:
            _DOCKER_AVAILABLE = (False, "docker ok but in-container network check timed out")
        except subprocess.CalledProcessError as e:
            tail = ""
            try:
                tail = (e.stderr or b"").decode("utf-8", "ignore").strip().splitlines()[-1]
            except Exception:
                tail = ""
            msg = "docker ok but in-container network check failed"
            if tail:
                msg = f"{msg}: {tail}"
            _DOCKER_AVAILABLE = (False, msg)
    except FileNotFoundError:
        _DOCKER_AVAILABLE = (False, "docker not installed")
    except subprocess.TimeoutExpired:
        _DOCKER_AVAILABLE = (False, "docker info timed out")
    except subprocess.CalledProcessError:
        _DOCKER_AVAILABLE = (False, "docker daemon not running")
    except Exception as e:
        _DOCKER_AVAILABLE = (False, str(e))
    return _DOCKER_AVAILABLE


def _docker_is_required() -> bool:
    """Whether Docker must be available for this test run.

    In CI we prefer to fail fast rather than silently skipping integration tests.
    """
    explicit = os.environ.get("SWE_AGENT_REQUIRE_DOCKER")
    if explicit is not None:
        return explicit.strip().lower() in ("1", "true", "yes", "y", "t")
    return os.environ.get("CI", "").strip().lower() == "true"


def pytest_runtest_setup(item) -> None:
    # Our slow tests are integration-heavy and generally require Docker.
    if "slow" in item.keywords:
        ok, reason = _docker_available_cached()
        if not ok:
            if _docker_is_required():
                pytest.fail(f"Docker required for slow tests but unavailable: {reason}")
            pytest.skip(f"Skipping slow test (Docker unavailable: {reason})")


@pytest.fixture
def test_data_path() -> Path:
    p = _this_dir / "test_data"
    assert p.is_dir()
    return p


@pytest.fixture
def test_trajectories_path(test_data_path) -> Path:
    p = test_data_path / "trajectories"
    assert p.is_dir()
    return p


@pytest.fixture
def test_ctf_trajectories_path(test_data_path) -> Path:
    p = test_data_path / "trajectories" / "ctf"
    assert p.is_dir()
    return p


@pytest.fixture
def ctf_data_path(test_data_sources_path) -> Path:
    p = test_data_sources_path / "ctf"
    assert p.is_dir()
    return p


@pytest.fixture
def test_data_sources_path(test_data_path) -> Path:
    p = test_data_path / "data_sources"
    assert p.is_dir()
    return p


@pytest.fixture
def test_trajectory_path(test_trajectories_path) -> Path:
    traj = (
        test_trajectories_path
        / "gpt4__swe-agent__test-repo__default_from_url__t-0.00__p-0.95__c-3.00__install-1"
        / "swe-agent__test-repo-i1.traj"
    )
    assert traj.exists()
    return traj


@pytest.fixture
def test_trajectory(test_trajectory_path):
    return json.loads(test_trajectory_path.read_text())


@pytest.fixture(scope="module")
def test_env_args(
    tmpdir_factory,
) -> Generator[EnvironmentConfig]:
    """This will use a persistent container"""
    local_repo_path = tmpdir_factory.getbasetemp() / "test-repo"
    clone_cmd = ["git", "clone", "https://github.com/swe-agent/test-repo", str(local_repo_path)]
    subprocess.run(clone_cmd, check=True)
    test_env_args = EnvironmentConfig(
        deployment=DockerDeploymentConfig(image="python:3.11"),
        repo=LocalRepoConfig(path=Path(local_repo_path)),
    )
    yield test_env_args
    shutil.rmtree(local_repo_path)


@pytest.fixture
def dummy_env_args() -> EnvironmentConfig:
    return EnvironmentConfig(
        deployment=DummyDeploymentConfig(),
        repo=None,
    )


@pytest.fixture
def dummy_env(dummy_env_args) -> Generator[SWEEnv, None, None]:
    env = SWEEnv.from_config(dummy_env_args)
    env.start()
    yield env
    env.close()


@contextmanager
def swe_env_context(env_args):
    """Context manager to make sure we close the shell on the container
    so that we can reuse it.
    """

    env = SWEEnv.from_config(env_args)
    env.start()
    try:
        yield env
    finally:
        env.close()


@pytest.fixture
def swe_agent_test_repo_clone(tmp_path):
    local_repo_path = tmp_path / "test-repo"
    clone_cmd = ["git", "clone", "https://github.com/swe-agent/test-repo", local_repo_path]
    subprocess.run(clone_cmd, check=True)
    return local_repo_path


@pytest.fixture
def swe_agent_test_repo_traj(test_trajectories_path) -> Path:
    p = (
        test_trajectories_path
        / "gpt4__swe-agent-test-repo__default_from_url__t-0.00__p-0.95__c-3.00__install-1"
        / "6e44b9__sweagenttestrepo-1c2844.traj"
    )
    assert p.is_file()
    return p

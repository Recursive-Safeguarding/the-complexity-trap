from __future__ import annotations

import subprocess

from scripts import run_sweep


def test_run_evaluation_timeout_terminates_and_kills(tmp_path, monkeypatch):
    preds_path = tmp_path / "preds.json"
    preds_path.write_text("{}")

    monkeypatch.setattr(run_sweep, "_check_docker_available", lambda: (True, ""))

    import sweagent.utils.sbcli as sbcli

    monkeypatch.setattr(sbcli, "get_dataset_name", lambda subset, backend: "dummy-dataset")

    class FakeStdout:
        def __init__(self, lines: list[str] | None = None):
            self._lines = list(lines or [])
            self._index = 0
            self.closed = False

        def readline(self):
            if self._index < len(self._lines):
                line = self._lines[self._index]
                self._index += 1
                return line
            return ""

        def close(self):
            self.closed = True

    class FakeProcess:
        def __init__(self):
            self.stdout = FakeStdout([])
            self.terminated = False
            self.killed = False
            self.returncode = None
            self._wait_calls = 0

        def wait(self, timeout=None):
            self._wait_calls += 1
            if self._wait_calls in (1, 2):
                raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout)
            self.returncode = -9 if self.killed else -15 if self.terminated else 0
            return self.returncode

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

    fake_process = FakeProcess()
    monkeypatch.setattr(run_sweep.subprocess, "Popen", lambda *args, **kwargs: fake_process)

    result = run_sweep.run_evaluation(tmp_path, subset="verified", max_workers=1, eval_backend="docker")

    assert result is None
    assert fake_process.terminated is True
    assert fake_process.killed is True
    assert (tmp_path / "evaluation.log").exists()

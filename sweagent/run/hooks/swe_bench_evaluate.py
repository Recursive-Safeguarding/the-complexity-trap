"""SweBench evaluation hook.

Will be automatically added to `run_batch` if `SWEBenchInstances.evaluate` is set to true
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path
from threading import Lock
from time import time

from sweagent.run.hooks.abstract import RunHook
from sweagent.run.merge_predictions import merge_predictions
from sweagent.types import AgentRunResult
from sweagent.utils.log import get_logger


class SweBenchEvaluate(RunHook):
    _SUBSET_MAP = {
        "lite": "swe-bench_lite",
        "verified": "swe-bench_verified",
        # sb-cli doesn't have a dedicated mini subset; predictions constrain the evaluated IDs.
        "verified-mini": "swe-bench_verified",
        "multimodal": "swe-bench_multimodal",
    }

    def __init__(self, output_dir: Path, subset: str, split: str, continuous_submission_every: int = 0) -> None:
        super().__init__()
        self.output_dir = output_dir
        self.subset = subset
        self.split = split
        self.continuous_submission_every = continuous_submission_every
        self.logger = get_logger("SB-evaluate", emoji="😬")
        self.merge_lock = Lock()
        self.last_evaluation_time = time()
        self.evaluation_interval = continuous_submission_every
        self._running_calls = []
        # We need to add a suffix to the run_id to avoid collisions when you reuse the name of your run
        self._time_suffix = datetime.now().strftime("%Y%m%d%H%M%S%f")

    @property
    def run_id(self) -> str:
        return f"{self.output_dir.name}_{self._time_suffix}"

    def _get_sb_call(
        self,
        preds_path: Path,
        *,
        submit_only: bool = False,
        instance_ids: list[str] | None = None,
    ) -> list[str]:
        args = [
            "sb-cli",
            "submit",
            self._SUBSET_MAP[self.subset],
            self.split,
            "--predictions_path",
            str(preds_path),
            "--run_id",
            self.run_id,
            "--output_dir",
            str(self.output_dir / "sb-cli-reports"),
        ]
        if instance_ids:
            args.extend(["--instance_ids", ",".join(instance_ids)])
        if submit_only:
            args.extend(["--wait_for_evaluation", "0", "--gen_report", "0", "--verify_submission", "0"])
        return args

    def check_running_calls(self) -> None:
        """Warn if one of the running calls failed."""
        for call in self._running_calls:
            if call.poll() is not None:
                if call.returncode != 0:
                    self.logger.error("Failed to submit results to SweBench eval: %s", call.stderr.read())
                self._running_calls.remove(call)

    def on_instance_completed(self, *, result: AgentRunResult):
        if self.evaluation_interval == 0:
            return

        current_time = time()
        if current_time - self.last_evaluation_time < self.evaluation_interval:
            return

        with self.merge_lock:
            merge_predictions([self.output_dir], self.output_dir / "tmppreds.json")
            self.last_evaluation_time = current_time

        instance_ids = None
        if self.subset == "verified-mini":
            try:
                from sweagent.utils.sbcli import MAX_MINI_INSTANCES, _extract_instance_ids_from_preds

                instance_ids = _extract_instance_ids_from_preds(self.output_dir / "tmppreds.json")
                if not instance_ids:
                    self.logger.warning(
                        "Skipping continuous sb-cli submission for verified-mini: 0 extracted instance IDs"
                    )
                    return
                if len(instance_ids) > MAX_MINI_INSTANCES:
                    self.logger.warning(
                        "Skipping continuous sb-cli submission for verified-mini: %d extracted IDs (cap=%d)",
                        len(instance_ids),
                        MAX_MINI_INSTANCES,
                    )
                    return
            except Exception as e:
                self.logger.warning(
                    "Skipping continuous sb-cli submission for verified-mini: failed to extract instance IDs (%s)",
                    e,
                )
                return

        self._running_calls.append(
            subprocess.Popen(
                self._get_sb_call(
                    preds_path=self.output_dir / "tmppreds.json",
                    submit_only=True,
                    instance_ids=instance_ids,
                ),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        )

    def move_sb_cli_report(self) -> None:
        """Move report from `sb-cli-reports` to `results.json`."""
        output_dir = self.output_dir / "sb-cli-reports"
        if not output_dir.exists():
            self.logger.warning("No SweBench report found at %s", output_dir)
            return
        (self.output_dir / "results.json").unlink(missing_ok=True)
        reports = [p for p in output_dir.glob("*.json") if not p.name.endswith(".response.json")]
        if not reports:
            self.logger.warning("No SweBench report JSON found at %s", output_dir)
            return
        if len(reports) > 1:
            # sb-cli can write multiple JSON files; pick the newest non-response report.
            reports.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            self.logger.warning(
                "Found %d SweBench report JSON files at %s; using newest: %s",
                len(reports),
                output_dir,
                reports[0].name,
            )
        reports[0].rename(self.output_dir / "results.json")

    def on_end(self) -> None:
        self.logger.info("Submitting results to SWE-Bench")
        try:
            from sweagent.utils.sbcli import run_sbcli_evaluation
        except Exception as e:
            self.logger.error("Failed to import sb-cli evaluation wrapper: %s", e)
            return

        success, error, _results_path = run_sbcli_evaluation(
            preds_path=self.output_dir / "preds.json",
            subset=self.subset,
            run_id=self.run_id,
            output_dir=self.output_dir,
        )
        if not success:
            self.logger.error("Failed to submit results to SweBench eval: %s", error)
            return

        # remove temporary predictions if they exist
        if (self.output_dir / "tmppreds.json").exists():
            (self.output_dir / "tmppreds.json").unlink()

"""How RoboDojo runs are summarised."""

from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from unittest.mock import Mock


REPO_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "vlm_runner_under_test", REPO_ROOT / "scripts/robodojo/run_vlm_experiment.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

OVERRIDES = {
    "max_decisions": 240, "max_zero_progress": 5, "max_consecutive_errors": 2, "max_tokens": 8000,
    "max_attempts": 6, "reasoning_effort": "high", "request_timeout_s": 600.0,
    "decision_budget_s": 660.0, "episode_budget_s": 9000.0,
    "cameras": ["cam_head", "cam_left_wrist", "cam_right_wrist"],
    "visual_aids": {"markers": True, "grid": True, "enhance": True}, "measure_gripper": True,
    "icl": {"demo_bank": str(REPO_ROOT / "demos/robodojo"), "demo_count": 1},
}


class ExperimentFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="vlm-experiments-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        robodojo_patch = patch.object(runner, "ROBODOJO_ROOT", self.root / "RoboDojo")
        robodojo_patch.start()
        self.addCleanup(robodojo_patch.stop)
        for target in ("subprocess.Popen", "urllib.request.urlopen"):
            guard = patch(target, side_effect=AssertionError("tests must not launch processes or contact APIs"))
            guard.start()
            self.addCleanup(guard.stop)
        self.condition = {"model": "offline-model", "task": "general_pickup"}

    def make_run(self, run_id="example", actual_ids=(0, 1), successes=(False, False)):
        run_dir = self.root / "runs" / run_id
        run_dir.mkdir(parents=True)
        manifest = {
            "run_id": run_id, **self.condition, "profile": self.condition["model"],
            "seed": 0, "eval_num": 2, "hostname": "test-host", "state": "finished", "exit_code": 0,
            "overrides": {**OVERRIDES, "log_dir": str(run_dir / "decisions")},
            "robot": "arx_x5", "action_type": "joint",
            "policy_env": "vlm_policy", "eval_env": "RoboDojo",
            "wall_timeout_s": 4000,
        }
        runner.write_json(run_dir / "manifest.json", manifest)
        result_path = (runner.ROBODOJO_ROOT / "eval_result/RoboDojo/general_pickup/vlm_agent/arx_x5"
                       / f"0_ckpt_name={run_id},action_type=joint" / run_id / "_result.json")
        result_path.parent.mkdir(parents=True)
        runner.write_json(result_path, {
            "eval_time": 2, "success_rate": sum(successes) / 2, "score": sum(successes) * 50,
            "details": {str(index): {"layout_id": layout_id, "success": successes[index % len(successes)]}
                        for index, layout_id in enumerate(actual_ids)},
        })
        return run_dir, manifest, result_path

    def decision(self, run_dir, index, **record):
        episode_dir = run_dir / "decisions" / run_dir.name / "episode_001"
        episode_dir.mkdir(parents=True, exist_ok=True)
        runner.write_json(episode_dir / f"decision_{index:04d}.json", {"episode": 1, "decision": index, **record})
        return episode_dir


class ExperimentSummaryTest(ExperimentFixture):
    def test_main_text_plan_and_recovered_retry_summary(self):
        run_dir, _, _ = self.make_run()
        self.decision(run_dir, 1, plan="align then grasp", api_calls=2,
                      requests=[{"attempts": 2, "retry_history": [{"status_code": 429}]}])
        summary = runner.summarize_run(run_dir)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["demo_bank"], OVERRIDES["icl"]["demo_bank"])
        self.assertEqual(summary["rate_limit_count"], 1)
        self.assertEqual(summary["retry_count"], 1)

    def test_failed_latency_is_retained_separately_from_response_latency(self):
        run_dir, _, _ = self.make_run()
        self.decision(run_dir, 1, latency_s=2, attempts=1, usage={"total_tokens": 10})
        self.decision(run_dir, 2, latency_s=60, attempts=1, error="VLM request failed: request timeout")
        summary = runner.summarize_run(run_dir)
        self.assertEqual(summary["mean_latency_s"], 2)
        self.assertEqual(summary["mean_request_latency_s"], 31)
        self.assertEqual(summary["failed_request_latency_count"], 1)
        self.assertEqual(summary["api_call_count"], 2)

    def test_renderer_crash_stops_only_this_child_group_without_waiting_for_timeout(self):
        log = self.root / "launcher.log"
        log.write_text("[Error] GPU crash is detected\n")
        process = Mock()
        process.poll.return_value = None
        with patch.object(runner, "stop_process_group") as stop:
            self.assertEqual(runner.wait_for_run(process, log, 4000), "renderer_crash")
        stop.assert_called_once_with(process)
        process.wait.assert_not_called()

    def test_completed_policy_failures_are_not_reported_as_successes(self):
        run_dir, _, _ = self.make_run()
        self.decision(run_dir, 1, latency_s=3, usage={"total_tokens": 42},
                      execution={"requested": 4, "executed": 3, "ik_failed": 1})
        summary = runner.summarize_run(run_dir)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["successes"], 0)
        self.assertEqual(summary["success_rate"], 0)
        self.assertEqual(summary["usage"]["total_tokens"], 42)
        self.assertEqual(summary["execution"]["ik_failed"], 1)

    def test_actual_episode_summary_marks_model_error_abort(self):
        run_dir, _, _ = self.make_run()
        self.decision(run_dir, 1, error="VLM request failed after 1 attempt(s): TimeoutError: timed out")
        episode_dir = self.decision(run_dir, 2, error="no JSON object found in the reply")
        runner.write_json(episode_dir / "episode_summary.json", {
            "phase": "episode_end", "policy_stop_reason": "consecutive_model_errors",
        })
        summary = runner.summarize_run(run_dir)
        self.assertEqual(summary["status"], "model_error_abort")
        self.assertEqual(summary["model_error_abort_episode_count"], 1)
        self.assertEqual(summary["decision_api_error_count"], 1)
        self.assertEqual(summary["decision_parse_error_count"], 1)

    def test_http_timeouts_and_action_parse_errors_are_distinct(self):
        examples = {
            "VLM request failed after 1 attempt(s): HTTP 503": "api",
            "TimeoutError: timed out": "api",
            "empty completion": "api",
            "no JSON object found in the reply": "parse",
            "reply has no action object": "parse",
            "unexpected local error": "other",
        }
        for error, kind in examples.items():
            with self.subTest(error=error):
                self.assertEqual(runner.model_error_kind({"error": error}), kind)
        self.assertIsNone(runner.model_error_kind({}))

    def test_missing_result_is_incomplete_not_zero_success_measurement(self):
        run_dir, _, result_path = self.make_run()
        result_path.unlink()
        summary = runner.summarize_run(run_dir)
        self.assertEqual(summary["status"], "incomplete")
        self.assertEqual(summary["completed_episodes"], 0)
        self.assertIsNone(summary["success_rate"])

    def test_summary_lists_every_run_with_its_layouts(self):
        self.make_run("first", successes=(False, True))
        self.make_run("second", actual_ids=(0, 2), successes=(True, True))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.aggregate(self.root), 0)
        report = runner.read_json(self.root / "summary.json")
        self.assertEqual([item["run_id"] for item in report["runs"]], ["first", "second"])
        with (self.root / "results.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([(row["run_id"], row["layout_ids"], row["successes"]) for row in rows],
                         [("first", "0,1", "1"), ("second", "0,2", "2")])


if __name__ == "__main__":
    unittest.main()

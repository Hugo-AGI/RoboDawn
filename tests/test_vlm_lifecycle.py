"""Interrupted runs finish their manifest and leave no child processes behind."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "scripts/robodojo/run_vlm_experiment.py"
spec = importlib.util.spec_from_file_location("vlm_lifecycle_runner", RUNNER_PATH)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def running(pid: int) -> bool:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[0] not in ("Z", "X")
    except (FileNotFoundError, ProcessLookupError):
        return False


@unittest.skipUnless(sys.platform.startswith("linux"), "requires Linux process identity and signals")
class LifecycleProcessTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="vlm-lifecycle-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.processes = []
        self.groups = set()
        self.addCleanup(self.stop_created_processes)

    def stop_created_processes(self):
        for manifest_path in self.root.glob("experiment/runs/*/manifest.json"):
            group = json.loads(manifest_path.read_text()).get("process_group")
            if group is not None:
                self.groups.add(group)
        markers = list(self.root.glob("tree-2.json"))
        for marker in markers:
            try:
                self.groups.add(os.getpgid(json.loads(marker.read_text())["pid"]))
            except (OSError, ValueError):
                pass
        for process in self.processes:
            if process.poll() is None:
                process.kill()
        for group in self.groups:
            try:
                os.killpg(group, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for process in self.processes:
            process.wait(timeout=10)

    def write_script(self, filename, content):
        path = self.root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(content), encoding="utf-8")
        return path

    def launch(self, path):
        log_path = path.with_suffix(".log")
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, "-B", str(path)], cwd=self.root,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
        self.processes.append(process)
        self.groups.add(process.pid)
        return process, log_path

    def wait_until(self, predicate, process=None, log_path=None, timeout=15):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            if process is not None and process.poll() is not None:
                self.fail(f"process exited {process.returncode}: {log_path.read_text()}")
            time.sleep(0.02)
        self.fail(f"condition not reached: {log_path.read_text() if log_path else ''}")

    def make_run_process(self, hold_after_wait=False):
        worker = self.write_script("tree.py", """
            import json
            import os
            from pathlib import Path
            import signal
            import subprocess
            import sys
            import time

            root, depth = Path(sys.argv[1]), int(sys.argv[2])
            child = None

            def terminate(signum, frame):
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
                signal.signal(signal.SIGHUP, signal.SIG_IGN)
                if child is not None:
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
                raise SystemExit(128 + signum)

            signal.signal(signal.SIGTERM, terminate)
            signal.signal(signal.SIGHUP, terminate)
            if depth:
                child = subprocess.Popen([sys.executable, __file__, str(root), str(depth - 1)])
                while not (root / f"tree-{depth - 1}.json").exists():
                    time.sleep(0.01)
            (root / f"tree-{depth}.json").write_text(json.dumps({"pid": os.getpid()}))
            while True:
                time.sleep(60)
        """)
        self.write_script("scripts/robodojo/run_vlm_eval.sh",
                          f"exec {shlex.quote(sys.executable)} {shlex.quote(str(worker))} "
                          f"{shlex.quote(str(self.root))} 2\n")
        conda = self.write_script("conda.sh", ":\n")
        layout = self.root / "RoboDojo/Assets/Eval_Layout/RoboDojo/arx_x5/0"
        layout.mkdir(parents=True)
        (layout / "general_pickup_0.json").write_text("{}\n")
        script = self.write_script("run_wrapper.py", f"""
            import argparse
            import importlib.util
            from pathlib import Path
            import time

            spec = importlib.util.spec_from_file_location("lifecycle_runner", {str(RUNNER_PATH)!r})
            runner = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(runner)
            runner.REPO_ROOT = Path({str(self.root)!r})
            runner.ROBODOJO_ROOT = runner.REPO_ROOT / "RoboDojo"
            runner.gpu_info = lambda gpu: {{"requested": gpu, "query": None}}
            runner.load_vlm_profile = lambda name: {{"name": name, "model": name}}
            if {hold_after_wait!r}:
                original_wait = runner.wait_for_run

                def held_wait(*args, **kwargs):
                    result = original_wait(*args, **kwargs)
                    (runner.REPO_ROOT / "wait_returned").touch()
                    while not (runner.REPO_ROOT / "release_wait").exists():
                        time.sleep(0.01)
                    return result

                runner.wait_for_run = held_wait
            args = argparse.Namespace(
                run_id="interrupted", task="general_pickup", eval_num=1, seed=0,
                max_decisions=1, max_tokens=1500, wall_timeout=120, episode_budget=120, model="offline-model",
                gpu="0", demo_count=1,
                experiment_dir=runner.REPO_ROOT / "experiment",
                policy_env="offline", eval_env="offline", conda_sh=Path({str(conda)!r}),
            )
            raise SystemExit(runner.run(args))
        """)
        process, log = self.launch(script)
        try:
            self.wait_until(lambda: (self.root / "tree-2.json").exists(), process, log)
        finally:
            manifest_path = self.root / "experiment/runs/interrupted/manifest.json"
            if manifest_path.exists():
                group = json.loads(manifest_path.read_text()).get("process_group")
                if group is not None:
                    self.groups.add(group)
        pids = [json.loads((self.root / f"tree-{depth}.json").read_text())["pid"] for depth in range(3)]
        self.groups.add(pids[-1])
        return process, log, manifest_path, pids

    def check_run_signal(self, signum, leader_exits=False):
        process, log, manifest_path, pids = self.make_run_process(hold_after_wait=leader_exits)
        self.assertTrue(all(running(pid) for pid in pids))
        self.assertNotEqual(os.getpgid(pids[-1]), os.getpgid(process.pid))
        if leader_exits:
            os.kill(pids[-1], signal.SIGKILL)
            self.wait_until((self.root / "wait_returned").exists, process, log)
        process.send_signal(signum)
        if leader_exits:
            (self.root / "release_wait").touch()
        process.wait(timeout=20)
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest["state"], "finished", log.read_text())
        self.assertTrue(manifest["interrupted"])
        self.assertNotEqual(process.returncode, 0)
        self.wait_until(lambda: not any(running(pid) for pid in pids), log_path=log)
        summary = json.loads(manifest_path.with_name("summary.json").read_text())
        self.assertNotEqual(summary["status"], "completed")

    def test_run_sigterm_finishes_manifest_and_cleans_child_group(self):
        self.check_run_signal(signal.SIGTERM)

    def test_run_sighup_finishes_manifest_and_cleans_child_group(self):
        self.check_run_signal(signal.SIGHUP)

    def test_run_sigint_finishes_manifest_and_cleans_child_group(self):
        self.check_run_signal(signal.SIGINT)

    def test_run_termination_cleans_group_after_leader_exits(self):
        self.check_run_signal(signal.SIGTERM, leader_exits=True)


if __name__ == "__main__":
    unittest.main()

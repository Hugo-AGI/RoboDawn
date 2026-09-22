"""Demonstration bank loading and prompt placement, without execution."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation/policies"))
from vlm_agent import icl_demos, main_prompts  # noqa: E402
from vlm_agent.agent import AgentConfig, VLMAgent  # noqa: E402

BANK = Path(__file__).resolve().parents[1] / "demos/robodojo"
TASK = "stack_bowls"


class RecordingClient:
    """Capture requests; a test reply is never submitted to a simulation."""

    model = "offline-test-client"

    def __init__(self):
        self.requests = []

    def complete(self, messages, **kwargs):
        self.requests.append(messages)
        return {"text": '{"commands": ["wait"]}', "usage": {}, "finish_reason": "stop",
                "latency_s": 0.0, "attempts": 1}


class ICLStorageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="vlm-icl-storage-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for target in ("urllib.request.urlopen", "subprocess.Popen"):
            guard = patch(target, side_effect=AssertionError("tests must not contact APIs or launch processes"))
            guard.start()
            self.addCleanup(guard.stop)

    def test_shipped_bank_has_one_entry_per_manifest_task_with_the_recorded_md5(self):
        import hashlib
        manifest = json.loads((BANK / "MANIFEST.json").read_text())
        for task, entry in manifest["tasks"].items():
            with self.subTest(task=task):
                packed = BANK / task / "demo.json"
                self.assertTrue(packed.is_file())
                self.assertEqual(hashlib.md5(packed.read_bytes()).hexdigest(), entry["demo_md5"])
                self.assertEqual(entry["episodes"], 5)

    def test_bank_entry_round_trips_through_save_demo(self):
        demo = icl_demos.build_demo(BANK / TASK)
        self.assertEqual(demo.task, TASK)
        self.assertTrue(demo.frames)
        self.assertGreater(demo.n_images(), 0)
        packed_dir = self.root / "packed" / TASK
        icl_demos.save_demo(demo, packed_dir)
        reloaded = icl_demos.build_demo(packed_dir)
        self.assertEqual(reloaded.instruction, demo.instruction)
        # the image files are renamed on save; everything the prompt is built from is unchanged
        strip = lambda frames: [(f.turn, f.label, [blob for _, blob in f.images], f.state_line, f.plan, f.commands,
                                 f.failed, f.effect, f.source_index) for f in frames]
        self.assertEqual(strip(reloaded.frames), strip(demo.frames))
        self.assertEqual(reloaded.source, {**demo.source, "images_scaled": True})

    def test_messages_carry_every_frame_and_its_source_index(self):
        demo = icl_demos.build_demo(BANK / TASK)
        messages = icl_demos.demo_messages(demo)
        self.assertEqual([message["role"] for message in messages], ["user", "assistant"])
        parts = messages[0]["content"]
        text = "\n".join(part["text"] for part in parts if part["type"] == "text")
        self.assertIn(f'Its instruction was: "{demo.instruction}"', text)
        self.assertEqual(sum(part["type"] == "image_url" for part in parts), demo.n_images())
        for frame in demo.frames:
            self.assertIn(f"--- DEMO {frame.label}", text)
            if frame.source_index is not None:
                self.assertIn(f"source: expert trace[{frame.source_index}]", text)
            if frame.commands:
                self.assertIn("commands: " + json.dumps(frame.commands), text)
        self.assertIn("END OF THE DEMONSTRATION.", text)

    def test_bank_resolution_and_missing_entries(self):
        self.assertEqual(icl_demos.resolve_demo_dirs(str(BANK), TASK), [BANK / TASK])
        self.assertEqual(icl_demos.resolve_demo_dirs(str(BANK), "no_such_task"), [])
        with self.assertRaises(FileNotFoundError):
            icl_demos.load_demos({"demo_bank": str(BANK), "demo_count": 1}, "no_such_task")
        with self.assertRaises(FileNotFoundError):
            icl_demos.resolve_demo_dirs(str(self.root / "missing"), TASK)
        for icl in ({}, {"demo_bank": ""}, {"demo_bank": str(self.root / "missing")}, {"demo_path": str(BANK / TASK)}):
            with self.subTest(icl=icl), self.assertRaises(ValueError):
                AgentConfig({"icl": icl})

    def test_system_prompt_tells_the_model_how_to_read_the_demonstration(self):
        for memory in (False, True):
            with self.subTest(memory=memory):
                prompt = main_prompts.system_prompt(None, 4, memory, context={"table_z": 76.5})
                self.assertIn(main_prompts.DEMO_NOTE, prompt)

    def test_agent_places_demo_before_current_turn_on_each_request(self):
        state = {"left_ee_pose": [-0.25, -0.22, 0.98, 0, 1, 0, 0], "right_ee_pose": [0.25, -0.22, 0.98, 0, 1, 0, 0]}
        observation = {"state": state, "vision": {}, "task_name": TASK, "instruction": "Stack the bowls."}
        with contextlib.redirect_stdout(io.StringIO()):
            cfg = AgentConfig({"log_dir": str(self.root / "logs"), "log_images": False, "icl": {"demo_bank": str(BANK)}})
            client = RecordingClient()
            agent = VLMAgent(cfg, client, task_name=TASK)
            for _ in range(2):
                agent.observe(observation)
                agent.act()
        self.assertEqual(len(client.requests), 2)
        for turn, messages in enumerate(client.requests, start=1):
            self.assertEqual([message["role"] for message in messages], ["system", "user", "assistant", "user"])
            current = messages[-1]["content"][-1]["text"]
            self.assertIn(f"TURN {turn}.", current)
            self.assertIn("CURRENT STATE:", current)
            self.assertEqual(messages[1:3], icl_demos.demo_messages(agent.icl_demos))
        self.assertEqual(client.requests[0][:3], client.requests[1][:3])

    def test_a_task_without_a_bank_entry_cannot_start(self):
        state = {"left_ee_pose": [-0.25, -0.22, 0.98, 0, 1, 0, 0], "right_ee_pose": [0.25, -0.22, 0.98, 0, 1, 0, 0]}
        observation = {"state": state, "vision": {}, "task_name": "no_such_task", "instruction": "x"}
        with contextlib.redirect_stdout(io.StringIO()):
            cfg = AgentConfig({"log_dir": str(self.root / "logs"), "log_images": False, "icl": {"demo_bank": str(BANK)}})
            agent = VLMAgent(cfg, RecordingClient(), task_name="no_such_task")
            agent.observe(observation)
            with self.assertRaises(FileNotFoundError):
                agent.act()


if __name__ == "__main__":
    unittest.main()

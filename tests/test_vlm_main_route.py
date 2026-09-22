"""Discrete commands on RoboDojo, without a model or a simulator."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation/policies"))
from vlm_agent import commands, deploy, gripper_settle, main_prompts, main_route, motion  # noqa: E402
from vlm_agent.action_parser import parse_action_response  # noqa: E402
from vlm_agent.agent import AgentConfig, VLMAgent  # noqa: E402
from vlm_agent.vlm_client import VLMError  # noqa: E402

TABLE_Z = 0.765
# Fingers pointing forward (+Y): a +90 degree turn about world Z of the identity (fingers +X).
FORWARD = (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))
OFFSET = main_route.TCP_OFFSET_M


def cmd(line: str) -> commands.Command:
    return commands.parse_command(line)


class GrammarTest(unittest.TestCase):
    def test_every_command_kind_parses_and_survives_the_wire(self):
        for line, kind, text in (
            ("left move z -5", "move", "left move z -5.0"),
            ("RIGHT rotate yaw 30 deg", "rotate", "right rotate yaw +30.0"),
            ("left point down45", "point", "left point down45"),
            ("left point down30", "point", "left point down30"),
            ("right gripper close", "gripper", "right gripper 0.00"),
            ("left gripper 0.3", "gripper", "left gripper 0.30"),
            ("`left home`", "home", "left home"),
            ("wait # settle", "wait", "wait"),
            ("Done.", "done", "done"),
        ):
            with self.subTest(line=line):
                parsed = cmd(line)
                self.assertEqual((parsed.kind, parsed.text()), (kind, text))
                self.assertEqual(commands.Command.from_dict(parsed.to_dict()), parsed)

    def test_magnitudes_are_clipped_and_nonsense_is_an_error(self):
        clipped = cmd("left move x 50")
        self.assertEqual((clipped.value, clipped.clipped), (20.0, True))
        self.assertEqual(cmd("right rotate roll -120").value, -90.0)
        self.assertTrue(cmd("right rotate roll -120").clipped)
        self.assertFalse(cmd("right rotate roll -120").rounded)
        parsed, errors = commands.parse_command_list(["left move q 3", "both home", 3, "left move x 4"])
        self.assertEqual([c.text() for c in parsed], ["left move x +4.0"])
        self.assertEqual(len(errors), 3)
        with self.assertRaises(commands.CommandError):
            commands.Command.from_dict({"kind": "move", "arm": "left", "axis": "x", "value": 500})
        with self.assertRaises(commands.CommandError):
            commands.Command.from_dict({"kind": "teleport", "arm": "left"})


class GeometryTest(unittest.TestCase):
    """Commands act on the fingertip centre, 0.145 m ahead of the wrist RoboDojo reports."""

    def setUp(self):
        self.pos = np.array([-0.3, -0.2, 1.05])
        self.quat = np.array(FORWARD)
        self.tcp = self.pos + OFFSET * np.array([0.0, 1.0, 0.0])

    def test_move_translates_the_fingertips_along_a_world_axis(self):
        target = main_route.resolve(cmd("left move x 10"), self.pos, self.quat)
        np.testing.assert_allclose(target["tcp"], self.tcp + [0.1, 0, 0], atol=1e-9)
        np.testing.assert_allclose(target["pos"], self.pos + [0.1, 0, 0], atol=1e-9)
        np.testing.assert_allclose(target["quat"], self.quat, atol=1e-9)

    def test_rotate_turns_about_the_fingertips_not_the_wrist(self):
        target = main_route.resolve(cmd("left rotate yaw 90"), self.pos, self.quat)
        np.testing.assert_allclose(target["tcp"], self.tcp, atol=1e-9)
        np.testing.assert_allclose(main_route.forward_axis(target["quat"]), [-1.0, 0.0, 0.0], atol=1e-9)
        # the wrist swings behind the (fixed) fingertips
        np.testing.assert_allclose(target["pos"], self.tcp + [OFFSET, 0.0, 0.0], atol=1e-9)

    def test_rotations_are_rounded_to_15_degree_steps(self):
        # the grammar offers 15 deg levels and the reply parser rounds anything else to the nearest one,
        # never to zero; parse_command itself stays exact (the wire form carries whatever angle it was given)
        for line, value in (("left rotate yaw 20", 15.0), ("left rotate yaw 25", 30.0), ("left rotate yaw 5", 15.0),
                            ("left rotate yaw -50", -45.0), ("left rotate yaw 60", 60.0), ("left rotate yaw -100", -90.0)):
            with self.subTest(line=line):
                parsed, errors = commands.parse_command_list([line])
                self.assertEqual(errors, [])
                self.assertEqual(parsed[0].value, value)
                self.assertEqual(parsed[0].rounded, value != max(-90.0, min(90.0, float(line.split()[-1]))))
                self.assertEqual(commands.Command.from_dict(parsed[0].to_dict()), parsed[0])
        self.assertEqual(cmd("left rotate yaw 20").value, 20.0)
        self.assertEqual(commands.Command.from_dict({"kind": "rotate", "arm": "left", "axis": "yaw", "value": 40}).value, 40.0)

    def test_point_presets_keep_the_fingertips_in_place(self):
        for preset, direction in (("down", [0, 0, -1]), ("forward", [0, 1, 0]), ("down45", [0, 0.7071, -0.7071]),
                                  ("down15", [0, 0.2588, -0.9659]), ("down30", [0, 0.5, -0.8660]),
                                  ("down60", [0, 0.8660, -0.5]), ("down75", [0, 0.9659, -0.2588])):
            with self.subTest(preset=preset):
                target = main_route.resolve(cmd(f"left point {preset}"), self.pos, self.quat)
                np.testing.assert_allclose(target["tcp"], self.tcp, atol=1e-9)
                np.testing.assert_allclose(main_route.forward_axis(target["quat"]), direction, atol=1e-4)
                # zero roll: the fingers close along world -X, the preset convention
                np.testing.assert_allclose(main_route.finger_axis(target["quat"]), [-1, 0, 0], atol=1e-4)

    def test_home_returns_the_wrist_to_the_recorded_pose(self):
        home = [0.1, 0.2, 1.1, *FORWARD]
        target = main_route.resolve(cmd("left home"), self.pos, self.quat, home_pose=home)
        np.testing.assert_allclose(target["pos"], home[:3])
        np.testing.assert_allclose(target["tcp"], np.array(home[:3]) + OFFSET * np.array([0.0, 1.0, 0.0]), atol=1e-9)

    def test_state_is_reported_in_centimetres_at_the_fingertips(self):
        state = main_route.arm_state("left", self.pos, self.quat, opening=0.31, closed_floor=0.185)
        self.assertEqual(state["position_cm"], [-30.0, -5.5, 105.0])
        self.assertEqual(state["approach"], [0.0, 1.0, 0.0])
        self.assertTrue(state["held"])
        self.assertIsNone(main_route.arm_state("left", self.pos, self.quat)["held"])

    def test_rpc_timeout_covers_all_retries(self):
        config = SimpleNamespace(request_timeout_s=120.0)
        client = SimpleNamespace(_client=SimpleNamespace(config=config))
        deploy._configure_rpc_timeout(client, {"decision_budget_s": 660})
        self.assertEqual(config.request_timeout_s, 690)


# ---------------------------------------------------------------- a tracking environment
class FakeRobot:
    def __init__(self, side):
        self.type = "target"
        self.arm_name = f"{side}_arm"
        self.gripper_name = f"{side}_ee"
        self.robot_name = f"x5_{side}"
        self.entity_origin_pose = [0.0, 0.0, TABLE_Z, 1.0, 0.0, 0.0, 0.0]


class FakeObsManager:
    collect_interval = 10
    dt = 0.004


class TrackingEnv:
    """Perfect tracking: an end-effector action sets the wrist pose, a joint hold keeps it.

    The measured finger opening follows the command but never below the physical
    closed floor, and never below an ``obstruction`` standing in for an object.
    """

    task_name = "test"
    step_lim = 200

    def __init__(self, obstruction=None, deploy_cfg=None):
        self.robots = [FakeRobot("left"), FakeRobot("right")]
        self.poses = {"left": [-0.3, -0.2, 1.05, *FORWARD], "right": [0.3, -0.2, 1.05, *FORWARD]}
        self.commanded = {"left": 1.0, "right": 1.0}
        self.floor = {"left": 0.185, "right": 0.185}
        self.obstruction = dict(obstruction or {})
        self.take_action_cnt = [0]
        self.success = [True]
        self.ended = False
        self.executed = []
        self.obs_manager = FakeObsManager()
        self.deploy_cfg = {"vlm_agent": dict(deploy_cfg or {})}
        self.robot_manager = SimpleNamespace(
            robot_list=self.robots,
            get_real_endpose=lambda robot, env_idx_list=None: {0: list(self.poses[robot.arm_name.split("_")[0]])},
            process_name=lambda name: name if name.endswith("_state") else name + "_joint_state",
            get_joint=lambda robot, env_idx_list=None: {0: [0.0] * 6},
        )

    def measured(self):
        return {side: max(self.commanded[side], self.floor[side], self.obstruction.get(side, 0.0))
                for side in ("left", "right")}

    def floors(self):
        return dict(self.floor)

    def is_episode_end(self):
        return self.ended

    def get_obs(self):
        return {"state": {f"{side}_ee_pose": list(self.poses[side]) for side in ("left", "right")}
                | {f"{side}_ee_joint_state": [self.commanded[side]] for side in ("left", "right")},
                "vision": {}, "instruction": "pick up the block"}

    def take_action(self, action):
        self.take_action_cnt[0] += 1
        self.executed.append(action)
        for side in ("left", "right"):
            if action.get(f"{side}_ee_pose") is not None:
                self.poses[side] = [float(v) for v in action[f"{side}_ee_pose"]]
            if action.get(f"{side}_ee_joint_state") is not None:
                self.commanded[side] = float(action[f"{side}_ee_joint_state"][0])
        if self.take_action_cnt[0] >= self.step_lim:
            self.ended = True


class FakeJointModule:
    @staticmethod
    def hold_actions(env, grippers, count):
        return [{f"{side}_arm_joint_state": [0.0] * 6, f"{side}_ee_joint_state": [grippers.get(side, 1.0)]}
                for _ in range(count) for side in ("left",)] and [
            {"left_arm_joint_state": [0.0] * 6, "left_ee_joint_state": [grippers.get("left", 1.0)],
             "right_arm_joint_state": [0.0] * 6, "right_ee_joint_state": [grippers.get("right", 1.0)]}
            for _ in range(count)]


def tracking_plan(env_, request):
    """A planner stand-in: four steps that arrive at the requested pose, the other arm held."""
    arm = request["arm"]
    other = "right" if arm == "left" else "left"
    arrive = {f"{arm}_ee_pose": list(request["target_pose"]),
              f"{arm}_ee_joint_state": [request["grippers"].get(arm, 1.0)],
              f"{other}_ee_pose": list(env_.poses[other]),
              f"{other}_ee_joint_state": [request["grippers"].get(other, 1.0)]}
    return [arrive] * 4, {"plan_status": "Success", "plan_points": 40, "truncated": False}


def failing_plan(env_, request):
    return [], {"plan_status": "Fail", "planner_error": None}


def make_hooks(env, plan=None):
    def execute_control(env_, action, report, settle, settle_module, *, added=False):
        targets = settle_module.action_targets(env_, action)
        measure = settle.needs_measurement(targets)
        before = env_.measured() if measure else {}
        floors = env_.floors() if measure else {}
        previous = env_.take_action_cnt[0]
        env_.take_action(action)
        executed = env_.take_action_cnt[0] - previous
        if executed <= 0:
            settle.interrupt("native")
            return
        report["executed"] += executed
        report["settle_steps" if added else "motion_executed"] += executed
        settle.advance(targets, before, env_.measured() if measure else {}, floors, added=added)

    def stop_reason(env_, deadline):
        return "native" if env_.is_episode_end() else None

    return SimpleNamespace(
        execute_control=execute_control,
        plan_keypose=plan or tracking_plan,
        gripper_real=lambda env_: env_.measured(),
        gripper_closed_floor=lambda env_: env_.floors(),
        stop_reason=stop_reason,
        ik_ok=lambda env_, arm, pose: None,
        settle_module=gripper_settle,
        joint_module=FakeJointModule(),
    )


CONFIG = {"controller": {},
          "gripper_settle": {"enabled": True, "min_s": 0.04, "stable_s": 0.04, "max_s": 0.4, "tolerance": 0.002}}


def sequence(*lines):
    return motion.sequence([cmd(line).to_dict() for line in lines], {"turn": 1},
                           {"hold_steps": 3, "gripper_steps": 6})


class ExecutorTest(unittest.TestCase):
    def run_sequence(self, env, envelope, hooks=None, session=None):
        hooks = hooks or make_hooks(env)
        session = session if session is not None else {"home_poses": None}
        return main_route.run_sequence(env, envelope, CONFIG, hooks, session, deadline=1e12,
                                       control_dt=0.04, commanded_grippers=dict(env.commanded))

    def test_a_turn_runs_whole_motions_and_judges_them_by_the_reached_fingertips(self):
        env = TrackingEnv()
        report, stopped = self.run_sequence(env, sequence("left move x 10", "left point down", "wait"))
        self.assertIsNone(stopped)
        results = report["command_results"]
        self.assertEqual([r["command"] for r in results], ["left move x +10.0", "left point down", "wait"])
        self.assertTrue(all(r["ok"] for r in results), results)
        self.assertLess(results[0]["position_error_cm"], 0.1)
        self.assertGreater(results[0]["steps_cost"], 1)
        self.assertEqual(results[2]["steps_cost"], 3)
        self.assertEqual(report["executed"], sum(r["steps_cost"] for r in results))
        # the second command was resolved from the pose reached by the first
        np.testing.assert_allclose(results[1]["after"]["left"]["position_cm"][:2], [-20.0, -5.5], atol=0.05)
        self.assertEqual(results[1]["after"]["left"]["approach"], [0.0, 0.0, -1.0])
        self.assertEqual(results[0]["after"]["left"]["wrist_cm"][0], -20.0)

    def test_gripper_commands_report_what_stopped_the_fingers(self):
        env = TrackingEnv(obstruction={"left": 0.31})
        report, _ = self.run_sequence(env, sequence("left gripper close", "right gripper close", "left gripper open"))
        left_close, right_close, left_open = report["command_results"]
        self.assertTrue(left_close["ok"] and left_close["held"])
        self.assertIn("something is between them", left_close["note"])
        self.assertEqual(left_close["value"], 0.0)
        self.assertFalse(right_close["held"])
        self.assertIn("nothing between them", right_close["note"])
        self.assertEqual(left_open["note"], "gripper opened")
        self.assertFalse(left_open["held"])
        self.assertEqual(report["gripper_settle"]["left"]["status"], "settled")
        self.assertGreaterEqual(left_close["steps_cost"], 6)

    def test_planner_failure_leaves_the_arm_where_it_was(self):
        env = TrackingEnv()
        report, _ = self.run_sequence(env, sequence("left move z -5"), make_hooks(env, plan=failing_plan))
        result = report["command_results"][0]
        self.assertFalse(result["ok"])
        self.assertIn("could not find a path", result["note"])
        self.assertEqual(result["blocked_reason"], "planner")
        self.assertEqual(report["executed"], 0)

    def test_a_planned_motion_is_executed_and_checked(self):
        env = TrackingEnv()

        def plan(env_, request):
            arrive = {"left_ee_pose": list(request["target_pose"]), "left_ee_joint_state": [request["grippers"]["left"]],
                      "right_ee_pose": list(env_.poses["right"]), "right_ee_joint_state": [1.0]}
            return [arrive] * 4, {"plan_status": "Success", "plan_points": 40, "truncated": False}

        report, _ = self.run_sequence(env, sequence("left move z -5"), make_hooks(env, plan=plan))
        result = report["command_results"][0]
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["plan_status"], "Success")
        self.assertEqual(result["steps_cost"], 14)  # four motion steps plus ten settling steps
        self.assertAlmostEqual(env.poses["left"][2], 1.0)

    def test_home_uses_the_pose_recorded_on_the_first_turn(self):
        env = TrackingEnv()
        session = {"home_poses": None}
        self.run_sequence(env, sequence("left move x 10", "left point down"), session=session)
        report, _ = self.run_sequence(env, sequence("left home"), session=session)
        self.assertTrue(report["command_results"][0]["ok"])
        np.testing.assert_allclose(env.poses["left"], [-0.3, -0.2, 1.05, *FORWARD], atol=1e-6)

    def test_home_joint_return_bypasses_planner_and_settles(self):
        env = TrackingEnv()
        hooks = make_hooks(env)
        hooks.plan_keypose = mock.Mock(side_effect=AssertionError("home should bypass planner"))
        home = np.array(env.poses["left"])
        env.poses["left"][0] += 0.1
        action = {"left_ee_pose": home.tolist()}
        with mock.patch.object(main_route, "_joint_home_actions", return_value=[action]):
            report, _ = main_route.run_sequence(env, sequence("left home"), CONFIG,
                                               hooks, {"home_poses": {"left": home}}, math.inf, 0.04, {})
        hooks.plan_keypose.assert_not_called()
        self.assertTrue(report["command_results"][0]["ok"])
        self.assertEqual(report["command_results"][0]["motion_steps"], 11)

    def test_main_long_move_is_not_truncated(self):
        env = TrackingEnv()
        def plan(env, request):
            self.assertGreater(request["max_plan_steps"], 60)
            return [{"left_ee_pose": request["target_pose"]}] * 75, {"plan_status": "Success"}
        report, _ = main_route.run_sequence(env, sequence("left move x 10"), CONFIG,
                                           make_hooks(env, plan=plan), {}, math.inf, 0.04, {})
        self.assertTrue(report["command_results"][0]["ok"])
        self.assertEqual(report["command_results"][0]["motion_steps"], 85)

    def test_done_on_the_wire_is_a_wait_and_the_native_end_stops_the_turn(self):
        env = TrackingEnv()
        env.step_lim = 5
        report, stopped = self.run_sequence(env, sequence("done", "left move x 10"))
        self.assertEqual(report["command_results"][0]["kind"], "wait")
        self.assertEqual(stopped, "native")
        self.assertEqual(len(report["command_results"]), 2)
        self.assertEqual(env.take_action_cnt[0], 5)


class ParserTest(unittest.TestCase):
    def test_main_replies_carry_command_strings_not_an_action_object(self):
        parsed, meta = parse_action_response('{"scene": "s", "commands": ["left move x 5", "done"], "memory": "m"}')
        self.assertEqual(parsed["commands"], ["left move x 5", "done"])
        self.assertIsNone(meta["error"])
        parsed, _ = parse_action_response('{"commands": "left move x 5"}')
        self.assertEqual(parsed["commands"], ["left move x 5"])
        parsed, _ = parse_action_response('{"commands": ["wait"], "memory": 4}')
        self.assertEqual(parsed["memory"], "4")
        for bad in ('{"commands": []}', '{"action": {"arm": "left"}}', '{"commands": [3]}'):
            with self.subTest(bad=bad):
                self.assertIsNone(parse_action_response(bad)[0])


# ---------------------------------------------------------------- the policy-server side
class Replies:
    model = "test-model"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0

    def complete(self, *args, **kwargs):
        self.calls += 1
        reply = self.replies[min(self.calls - 1, len(self.replies) - 1)]
        if isinstance(reply, Exception):
            raise reply
        return {"text": reply if isinstance(reply, str) else json.dumps(reply),
                "usage": {"total_tokens": 10}, "finish_reason": "stop", "latency_s": 0.2, "attempts": 1}


BANK = Path(__file__).resolve().parents[1] / "demos/robodojo"
TASK = "stack_bowls"


def observation(gripper=1.0, measured=None):
    state = {}
    for side, x in (("left", -0.3), ("right", 0.3)):
        state[f"{side}_ee_pose"] = [x, -0.2, 1.05, *FORWARD]
        state[f"{side}_ee_joint_state"] = [gripper]
    obs = {"state": state, "vision": {}, "instruction": "Pick up the red block",
           "gripper_closed_floor": {"left": 0.185, "right": 0.185}}
    if measured is not None:
        obs["gripper_real"] = dict(measured)
    return obs


def after_states(env_like=None, left_z=1.05, left_opening=1.0):
    return {"left": main_route.arm_state("left", [-0.3, -0.2, left_z], FORWARD, left_opening, 0.185),
            "right": main_route.arm_state("right", [0.3, -0.2, 1.05], FORWARD, 1.0, 0.185)}


class AgentTest(unittest.TestCase):
    def make_agent(self, *replies, **overrides):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        raw = {"log_dir": temp.name, "log_images": False, "max_consecutive_errors": 2, "icl": {"demo_bank": str(BANK)}}
        raw.update(overrides)
        client = Replies(*replies)
        agent = VLMAgent(AgentConfig(raw), client, task_name=TASK)
        agent.note_execution({"phase": "episode_start", "steps_used": 0, "control_dt_s": 0.04})
        agent.observe(observation())
        return agent, client

    def record(self, agent, index):
        return json.loads((agent.log_dir / f"decision_{index:04d}.json").read_text())

    def test_a_turn_becomes_a_sequence_truncated_at_done_with_per_command_feedback(self):
        agent, _ = self.make_agent({"scene": "block at (10, 0)", "progress": "", "memory": "block right of centre",
                                    "plan": "grasp", "commands": ["left point down", "left move z -5", "done", "left home"]})
        envelope = agent.act()
        self.assertEqual(envelope["kind"], "sequence")
        # the turn stops at "done" (the model's own "left home" after it is dropped) and the harness then
        # sends both arms home for the checker
        self.assertEqual([c["text"] for c in envelope["commands"]],
                         ["left point down", "left move z -5.0", "left home", "right home"])
        self.assertEqual((envelope["options"]["hold_steps"], envelope["options"]["gripper_steps"]), (3, 6))
        record = self.record(agent, 1)
        self.assertIn("ROBOT: a dual-arm ARX X5", record["system_prompt"])
        self.assertIn("z = 76.5", record["system_prompt"])
        self.assertNotIn("{table_z", record["system_prompt"])
        self.assertIn("TASK: Pick up the red block", record["prompt"])
        self.assertIn("gripper position (fingertip centre) = (-30.0, -5.5, 105.0)", record["prompt"])
        self.assertIn("(empty - this is the first turn)", record["prompt"])
        self.assertEqual(agent.main_memory.scratchpad, "block right of centre")
        self.assertEqual(record["pending_results"][-1]["command"], "done")

        results = [
            {"command": "left point down", "kind": "point", "arm": "left", "ok": True, "note": "", "steps_cost": 12,
             "after": after_states()},
            {"command": "left move z -5.0", "kind": "move", "arm": "left", "ok": False, "steps_cost": 0,
             "note": "the target was not reached: the arm did NOT move; remaining error 5.0 cm / 0 deg", "after": after_states()},
        ]
        agent.note_execution({"command_results": results, "executed": 12, "steps_used": 12})
        agent.observe(observation())
        agent.act()
        prompt = self.record(agent, 2)["prompt"]
        self.assertIn("RESULT OF YOUR LAST COMMANDS:\n- left point down: ok\n- left move z -5.0: FAILED (the target was not reached", prompt)
        self.assertIn("- done: FAILED (NOT finished", prompt)
        self.assertIn("steps used: 12\n", prompt + "\n")
        self.assertIn("YOUR NOTES FROM PREVIOUS TURNS", prompt)
        self.assertIn("block right of centre", prompt)
        self.assertIn("turn 1: left point down -> ok; left move z -5.0 -> FAILED", prompt)

    def test_grasp_facts_follow_the_measured_close(self):
        agent, _ = self.make_agent({"commands": ["left gripper close"]}, {"commands": ["left move z 10"]},
                                   {"commands": ["left gripper open"]}, {"commands": ["wait"]})
        agent.act()
        close = {"command": "left gripper 0.00", "kind": "gripper", "arm": "left", "ok": True, "value": 0.0, "held": True,
                 "opening": 0.31, "note": "fingers stopped at opening 0.31: something is between them (probably grasped)",
                 "steps_cost": 8, "after": after_states(left_z=0.95, left_opening=0.31)}
        agent.note_execution({"command_results": [close], "executed": 8, "steps_used": 8})
        agent.observe(observation(measured={"left": 0.31, "right": 1.0}))
        agent.act()
        prompt = self.record(agent, 2)["prompt"]
        # fingers point forward, so the fingertip centre shares the wrist height of 0.95 m
        self.assertIn("LEFT arm closed on an object with the fingertips at z = 95.0 (still holding it)", prompt)
        self.assertIn("lower the fingertips to z = 95.0 + (h - 76.5)", prompt)
        agent.note_execution({"command_results": [{"command": "left move z +10.0", "kind": "move", "arm": "left",
                                                   "ok": True, "note": "", "steps_cost": 10, "after": after_states(left_z=1.05, left_opening=0.185)}],
                              "executed": 10, "steps_used": 18})
        agent.observe(observation(measured={"left": 0.185, "right": 1.0}))
        agent.act()
        self.assertIn("the object was LOST", self.record(agent, 3)["prompt"])
        agent.note_execution({"command_results": [{"command": "left gripper 1.00", "kind": "gripper", "arm": "left",
                                                   "ok": True, "value": 1.0, "held": False, "note": "gripper opened",
                                                   "steps_cost": 6, "after": after_states()}], "executed": 6, "steps_used": 24})
        agent.observe(observation(measured={"left": 1.0, "right": 1.0}))
        agent.act()
        self.assertNotIn("closed on an object", self.record(agent, 4)["prompt"])

    def test_the_instruction_loses_its_reset_clause(self):
        # the harness homes the arms on done, so the sentence is dropped from the instruction
        for raw, clean in (("Align the blocks, then reset the robot arm.", "Align the blocks."),
                           ("Push the T to the pad and reset the robot arms", "Push the T to the pad."),
                           ("Stack the bowls.", "Stack the bowls."), ("", "")):
            with self.subTest(raw=raw):
                self.assertEqual(main_prompts.strip_reset_clause(raw), clean)
        agent, _ = self.make_agent({"commands": ["wait"]})
        agent.observe(dict(observation(), instruction="Sort the cards, then reset the robot arm."))
        agent.act()
        self.assertIn("TASK: Sort the cards.", self.record(agent, 1)["prompt"])
        self.assertNotIn("reset the robot", self.record(agent, 1)["prompt"])

    def test_three_consecutive_done_turns_end_the_episode(self):
        agent, client = self.make_agent({"commands": ["done"]})
        for turn in (1, 2):
            envelope = agent.act()
            self.assertEqual(envelope["kind"], "sequence")
            # the first done sends both arms home; a repeated done has nothing left to execute and holds still
            expected = [("home", "left"), ("home", "right")] if turn == 1 else [("wait", None)]
            self.assertEqual([(c["kind"], c["arm"]) for c in envelope["commands"]], expected)
            self.assertEqual(self.record(agent, turn).get("auto_home"), True if turn == 1 else None)
            agent.note_execution({"command_results": [{"command": "wait", "kind": "wait", "arm": None, "ok": True,
                                                       "note": "waited", "steps_cost": 3, "after": after_states()}],
                                  "executed": 3, "steps_used": 3 * turn})
            agent.observe(observation())
        self.assertEqual(agent.act(), {"kind": "stop", "reason": "agent_done"})
        self.assertEqual(client.calls, 3)

    def test_a_successful_turn_resets_the_done_count(self):
        agent, _ = self.make_agent({"commands": ["done"]}, {"commands": ["left move x 5"]}, {"commands": ["done"]})
        for _ in range(3):
            envelope = agent.act()
            self.assertEqual(envelope["kind"], "sequence")
            agent.note_execution({"command_results": [], "executed": 3, "steps_used": 3})
            agent.observe(observation())
        self.assertEqual(agent.main_done_count, 1)

    def test_bad_replies_hold_still_and_are_reported_back(self):
        agent, _ = self.make_agent("I cannot see the block", VLMError("boom", attempts=1, latency_s=0.1, status_code=500, retryable=True))
        envelope = agent.act()
        self.assertEqual([c["kind"] for c in envelope["commands"]], ["wait"])
        self.assertIn("not a valid JSON object", agent.main_pending_results[0]["note"])
        agent.note_execution({"command_results": [{"command": "wait", "kind": "wait", "arm": None, "ok": True,
                                                   "note": "waited", "steps_cost": 3, "after": after_states()}],
                              "executed": 3, "steps_used": 3})
        agent.observe(observation())
        envelope = agent.act()
        self.assertEqual([c["kind"] for c in envelope["commands"]], ["wait"])
        self.assertIn("- (unparseable reply): FAILED", self.record(agent, 2)["prompt"])
        self.assertEqual(agent.stop_reason, "consecutive_model_errors")
        agent.observe(observation())
        self.assertEqual(agent.act(), {"kind": "stop", "reason": "consecutive_model_errors"})

    def test_config_defaults_cameras_and_rejects_unknown_main_keys(self):
        cfg = AgentConfig({"icl": {"demo_bank": str(BANK)}})
        self.assertEqual(cfg.cameras, ["cam_head", "cam_left_wrist", "cam_right_wrist"])
        self.assertEqual(cfg.main["max_commands_per_turn"], 4)
        self.assertTrue(cfg.main["profile"].endswith("robodojo_x5_main.yaml"))
        for raw in ({"main": {"max_commands": 4}}, {"main": {"done_limit": 0}}, {"main": {"history_turns": -1}},
                    {"controller": {"unknown_option": 0.1}}, {"icl": {}}):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                AgentConfig({"icl": {"demo_bank": str(BANK)}, **raw})


class DispatchTest(unittest.TestCase):
    """The simulation client routes a sequence envelope through main_route and reports per command."""

    def setUp(self):
        planner = mock.patch.object(deploy, "_plan_keypose", side_effect=lambda env, request, config: tracking_plan(env, request))
        planner.start()
        self.addCleanup(planner.stop)

    def client(self, envelope):
        calls = {"reports": []}

        def call(func_name, obs=None):
            if func_name == "get_action":
                return envelope
            if func_name == "report_execution":
                calls["reports"].append(obs)

        client = mock.Mock()
        client.call.side_effect = call
        return client, calls

    @mock.patch.dict(os.environ, {"VLM_AGENT_OVERRIDES": '{"max_decisions": 1}'})
    def test_a_sequence_is_executed_with_per_command_results(self):
        env = TrackingEnv()
        client, calls = self.client(sequence("left move x 10", "wait"))
        deploy.eval_one_episode(env, client)
        turn = [report for report in calls["reports"] if "command_results" in report][-1]
        self.assertEqual([r["ok"] for r in turn["command_results"]], [True, True])
        self.assertEqual(turn["executed"], env.take_action_cnt[0])
        self.assertGreater(turn["executed"], 3)
        self.assertEqual(calls["reports"][-1]["termination_reason"], "decision_budget")
        self.assertAlmostEqual(env.poses["left"][0], -0.2)

    @mock.patch.dict(os.environ, {"VLM_AGENT_OVERRIDES": '{"max_decisions": 1}'})
    def test_the_control_step_limit_is_lifted_for_every_episode(self):
        # RoboDojo's per-frame step limit is not the budget of this policy: the decision count is.
        env = TrackingEnv()
        for _ in range(2):
            env.ended = False
            env.take_action_cnt = [0]
            client, calls = self.client(sequence("wait"))
            deploy.eval_one_episode(env, client)
            self.assertEqual(env.step_lim, deploy.UNBOUNDED_STEPS)
            self.assertEqual(calls["reports"][-1]["termination_reason"], "decision_budget")


if __name__ == "__main__":
    unittest.main()

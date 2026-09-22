"""The motion envelope and the planned joint trajectory.

The cuRobo backend is exercised against a stand-in for RoboDojo's
``robot_manager``, so the parts a simulator cannot be asked about in a unit test
- key names, joint ordering, how many environment steps a plan costs, what
happens when planning fails - are still pinned.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import unittest
from unittest import mock

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is required by the policy
    np = None

POLICY_DIR = Path(__file__).resolve().parents[1] / "evaluation/policies/vlm_agent"


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"vlm_{name}", POLICY_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


motion = load("motion")
if np is not None:
    controller = load("controller")
    curobo_motion = load("curobo_motion")
    deploy = load("deploy")


class FakeRobot:
    def __init__(self, side, joints):
        self.type = "target"
        self.arm_name = f"{side}_arm"
        self.gripper_name = f"{side}_ee"
        self.robot_name = f"x5_{side}"
        self.entity_origin_pose = [0.0, 0.0, 0.765, 1.0, 0.0, 0.0, 0.0]
        self.joints = list(joints)


class FakePlanner:
    """Stands in for one RoboDojo ``CuroboPlanner``."""

    def __init__(self, result=None, dt=0.004, error=None):
        self.result = result
        self.dt = dt
        self.error = error
        self.calls = []

    def plan_path(self, curr_joint_pos, target_ee_pose, real_robot_pose, constraint_pose=None):
        self.calls.append({"start": list(curr_joint_pos), "target": list(target_ee_pose),
                           "origin": list(real_robot_pose), "constraint": constraint_pose})
        if self.error is not None:
            raise self.error
        return self.result


class FakeRobotManager:
    def __init__(self, robots, planners):
        self.robot_list = robots
        self.planner = planners
        self.ik_solver = planners

    def process_name(self, name):
        return name if name.endswith("_state") else name + "_joint_state"

    def get_joint(self, robot, env_idx_list=None):
        return {0: list(robot.joints)}

    def get_real_endpose(self, robot, env_idx_list=None, is_relative=True):
        return {0: [0.1, -0.2, 1.0, 1.0, 0.0, 0.0, 0.0]}


class FakeObsManager:
    collect_interval = 10
    dt = 0.004


class FakeEnv:
    """Enough of ``EvalEnv`` for the motion path, including the episode loop."""

    task_name = "test"
    step_lim = 100

    def __init__(self, result=None, error=None, planner_dt=0.004, deploy_cfg=None):
        self.left = FakeRobot("left", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        self.right = FakeRobot("right", [1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
        self.planners = {robot.robot_name: FakePlanner(result, planner_dt, error)
                         for robot in (self.left, self.right)}
        self.robot_manager = FakeRobotManager([self.left, self.right], self.planners)
        self.obs_manager = FakeObsManager()
        self.success = [True]
        self.take_action_cnt = [0]
        self.executed = []
        self.deploy_cfg = {"vlm_agent": dict(deploy_cfg or {})}

    def is_episode_end(self):
        return not self.success[0]

    def get_obs(self):
        return {}

    def take_action(self, action):
        self.take_action_cnt[0] += 1
        self.executed.append(action)


def keypose_request(**overrides):
    request = {
        "arm": "left",
        "target_pose": [0.1, -0.2, 1.0, 1.0, 0.0, 0.0, 0.0],
        "grippers": {"left": 0.0, "right": 1.0},
        "executed_motion": True,
        "settle_steps": 3,
    }
    request.update(overrides)
    return request


def trajectory(points, joints=6):
    return {"status": "Success",
            "position": np.arange(points * joints, dtype=float).reshape(points, joints),
            "velocity": np.zeros((points, joints))}


class EnvelopeTest(unittest.TestCase):
    def test_a_sequence_envelope_is_normalised(self):
        envelope = motion.normalize({"kind": "sequence", "commands": [{"kind": "wait"}]})
        self.assertEqual(envelope["kind"], motion.KIND_SEQUENCE)
        self.assertEqual((envelope["commands"], envelope["plan"], envelope["options"]), ([{"kind": "wait"}], {}, {}))

    def test_unknown_or_malformed_envelopes_stop_the_loop(self):
        for result in ({"kind": "teleport"}, "sequence", None, {}, [{"left_ee_pose": [0] * 7}], {"kind": "sequence"}):
            self.assertEqual(motion.normalize(result)["kind"], motion.KIND_STOP, result)

    def test_stop_carries_a_reason(self):
        self.assertEqual(motion.stop("wall_budget")["reason"], "wall_budget")
        self.assertEqual(motion.stop()["reason"], "policy_stop")


@unittest.skipIf(np is None, "numpy is required by the policy controller")
class CuroboBackendTest(unittest.TestCase):
    def test_plan_becomes_joint_actions_that_command_both_arms(self):
        env = FakeEnv(trajectory(21))
        actions, info = curobo_motion.plan_actions(env, keypose_request())
        self.assertEqual(info["plan_status"], "Success")
        self.assertEqual(info["stride"], 10)
        self.assertEqual(info["plan_points"], 21)
        self.assertEqual(len(actions), 2)
        for action in actions:
            # Exactly the keys EvalEnv.get_action_type reads as a joint action.
            self.assertEqual(sorted(action), ["left_arm_joint_state", "left_ee_joint_state",
                                              "right_arm_joint_state", "right_ee_joint_state"])
            self.assertEqual(action["right_arm_joint_state"], env.right.joints)
            self.assertEqual(action["left_ee_joint_state"], [0.0])
            self.assertEqual(action["right_ee_joint_state"], [1.0])
        positions = trajectory(21)["position"]
        self.assertEqual(actions[0]["left_arm_joint_state"], positions[10].tolist())
        self.assertEqual(actions[-1]["left_arm_joint_state"], positions[20].tolist())

    def test_the_final_planned_pose_is_always_commanded(self):
        env = FakeEnv(trajectory(15))
        actions, _ = curobo_motion.plan_actions(env, keypose_request())
        self.assertEqual(actions[-1]["left_arm_joint_state"], trajectory(15)["position"][14].tolist())

    def test_the_planner_is_asked_for_the_requested_pose_in_the_arm_frame(self):
        env = FakeEnv(trajectory(11))
        curobo_motion.plan_actions(env, keypose_request())
        call = env.planners["x5_left"].calls[0]
        self.assertEqual(call["start"], env.left.joints)
        self.assertEqual(call["target"], [0.1, -0.2, 1.0, 1.0, 0.0, 0.0, 0.0])
        self.assertEqual(call["origin"], env.left.entity_origin_pose)
        self.assertEqual(env.planners["x5_right"].calls, [])

    def test_a_long_path_is_truncated_to_the_step_budget(self):
        env = FakeEnv(trajectory(400))
        actions, info = curobo_motion.plan_actions(env, keypose_request(), {"max_plan_steps": 12})
        self.assertEqual(len(actions), 12)
        self.assertTrue(info["truncated"])

    def test_planning_failure_executes_nothing_and_is_reported(self):
        env = FakeEnv({"status": "Fail"})
        actions, info = curobo_motion.plan_actions(env, keypose_request())
        self.assertEqual(actions, [])
        self.assertEqual(info["plan_status"], "Fail")
        self.assertIsNone(info["planner_error"])

    def test_a_broken_planner_is_reported_not_raised(self):
        env = FakeEnv(error=RuntimeError("cuda graph capture failed"))
        actions, info = curobo_motion.plan_actions(env, keypose_request())
        self.assertEqual(actions, [])
        self.assertEqual(info["plan_status"], "Error")
        self.assertIn("RuntimeError", info["planner_error"])

    def test_an_unknown_arm_is_reported_not_raised(self):
        env = FakeEnv(trajectory(11))
        actions, info = curobo_motion.plan_actions(env, keypose_request(arm="middle"))
        self.assertEqual(actions, [])
        self.assertIn("middle", info["planner_error"])

    def test_a_gripper_only_decision_holds_both_arms_without_planning(self):
        env = FakeEnv(trajectory(11))
        request = keypose_request(executed_motion=False, settle_steps=4)
        actions, info = curobo_motion.plan_actions(env, request)
        self.assertEqual(info["plan_status"], "skipped")
        self.assertEqual(env.planners["x5_left"].calls, [])
        self.assertEqual(len(actions), 4)
        for action in actions:
            self.assertEqual(action["left_arm_joint_state"], env.left.joints)
            self.assertEqual(action["right_arm_joint_state"], env.right.joints)
            self.assertEqual(action["left_ee_joint_state"], [0.0])

    def test_settle_steps_cannot_exceed_the_step_budget(self):
        env = FakeEnv(trajectory(11))
        request = keypose_request(executed_motion=False, settle_steps=99)
        actions, _ = curobo_motion.plan_actions(env, request, {"max_plan_steps": 5})
        self.assertEqual(len(actions), 5)

    def test_stride_follows_the_environment_control_period(self):
        env = FakeEnv(trajectory(11))
        self.assertEqual(curobo_motion.stride_for(env, env.planners["x5_left"]), 10)
        self.assertEqual(curobo_motion.stride_for(env, FakePlanner(dt=0.04)), 1)
        self.assertEqual(curobo_motion.stride_for(env, FakePlanner(dt=0.004), override=3), 3)
        env.obs_manager = None
        self.assertEqual(curobo_motion.stride_for(env, FakePlanner(dt=0.004)),
                         curobo_motion.DEFAULT_STRIDE)

    def test_gripper_commands_are_clipped_into_the_normalized_range(self):
        env = FakeEnv(trajectory(11))
        actions, _ = curobo_motion.plan_actions(env, keypose_request(grippers={"left": 2.0, "right": -1.0}))
        self.assertEqual(actions[0]["left_ee_joint_state"], [1.0])
        self.assertEqual(actions[0]["right_ee_joint_state"], [0.0])

    def test_warmup_reports_every_arm_and_survives_a_failure(self):
        env = FakeEnv(trajectory(11))
        self.assertEqual(sorted(curobo_motion.warmup(env)), ["left", "right"])
        broken = FakeEnv(error=RuntimeError("no planner"))
        self.assertIn("error", curobo_motion.warmup(broken)["left"])


@unittest.skipIf(np is None, "numpy is required by the policy controller")
class EpisodeDispatchTest(unittest.TestCase):
    """The episode loop has to route by the envelope's kind, not by its config."""

    def client(self, envelope, limit=1):
        calls = {"actions": 0, "reports": []}

        def call(func_name, obs=None):
            if func_name == "get_action":
                calls["actions"] += 1
                return envelope
            if func_name == "report_execution":
                calls["reports"].append(obs)

        return call, calls

    @mock.patch.dict(os.environ, {"VLM_AGENT_OVERRIDES": "{}"})
    def test_a_stop_envelope_ends_the_episode_without_success(self):
        env = FakeEnv(trajectory(21))
        call, calls = self.client(motion.stop("permanent_api_error"))
        client = mock.Mock()
        client.call.side_effect = call
        deploy.eval_one_episode(env, client)
        self.assertEqual(env.take_action_cnt[0], 0)
        self.assertEqual(calls["reports"][-1]["termination_reason"], "policy_stop")
        self.assertFalse(env.success[0])

    @mock.patch.dict(os.environ, {"VLM_AGENT_OVERRIDES": '{"max_decisions": 1}'})
    def test_the_planner_is_warmed_up_before_the_first_decision(self):
        env = FakeEnv(trajectory(21))
        call, calls = self.client(motion.stop("offline_check"))
        client = mock.Mock()
        client.call.side_effect = call
        deploy.eval_one_episode(env, client)
        warmups = [report for report in calls["reports"] if "planner_warmup" in report]
        self.assertEqual(len(warmups), 1)
        self.assertEqual(sorted(warmups[0]["planner_warmup"]), ["left", "right"])


if __name__ == "__main__":
    unittest.main()

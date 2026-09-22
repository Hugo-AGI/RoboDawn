"""Agent state across turns and episodes, without a model or simulator."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation/policies"))
from vlm_agent.agent import AgentConfig, VLMAgent
from vlm_agent.controller import ControllerError
from vlm_agent import deploy
from vlm_agent.vlm_client import VLMError


BANK = Path(__file__).resolve().parents[1] / "demos/robodojo"
TASK = "stack_bowls"


def observation(gripper=1.0, closed_floor=None):
    state = {}
    for side, x in (("left", -0.3), ("right", 0.3)):
        state[f"{side}_ee_pose"] = [x, -0.2, 1.05, 1, 0, 0, 0]
        state[f"{side}_ee_joint_state"] = [gripper]
    obs = {"state": state, "vision": {}, "instruction": "Pick up a block"}
    if closed_floor is not None:
        obs["gripper_closed_floor"] = dict.fromkeys(("left", "right"), closed_floor)
    return obs


class Replies:
    model = "test-model"

    def __init__(self, *replies):
        self.replies = iter(replies)
        self.calls = 0

    def complete(self, *args, **kwargs):
        self.calls += 1
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return {"text": reply if isinstance(reply, str) else json.dumps(reply),
                "usage": {"total_tokens": 10}, "finish_reason": "stop", "latency_s": 0.2, "attempts": 1}


class AgentStateTest(unittest.TestCase):
    def make_agent(self, *responses):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        client = Replies(*responses)
        agent = VLMAgent(AgentConfig({"log_dir": temp.name, "log_images": False, "icl": {"demo_bank": str(BANK)}}),
                         client, task_name=TASK)
        agent.observe(observation())
        return agent, client

    def test_timeout_keeps_latency_and_permanent_error_survives_episode_reset(self):
        failure = VLMError("VLM request failed: HTTP 401", attempts=1, latency_s=0.4, status_code=401)
        agent, client = self.make_agent(failure)
        envelope = agent.act()
        self.assertEqual([c["kind"] for c in envelope["commands"]], ["wait"])
        record = json.loads((agent.log_dir / "decision_0001.json").read_text())
        self.assertEqual(record["latency_s"], 0.4)
        self.assertEqual(record["api_calls"], 1)
        agent.reset()
        agent.observe(observation())
        self.assertEqual(agent.act(), {"kind": "stop", "reason": "permanent_api_error"})
        self.assertEqual(client.calls, 1)

    def test_expired_episode_budget_does_not_call_model(self):
        agent, client = self.make_agent({"commands": ["wait"]})
        obs = observation()
        obs["policy_budget_s"] = 0
        agent.observe(obs)
        self.assertEqual(agent.act(), {"kind": "stop", "reason": "wall_budget"})
        self.assertEqual(client.calls, 0)

    def test_invalid_robot_quaternion_does_not_become_an_identity_pose(self):
        agent, client = self.make_agent({"commands": ["wait"]})
        obs = observation()
        obs["state"]["right_ee_pose"][3:] = [0, 0, 0, 0]
        with self.assertRaises(ControllerError):
            agent.observe(obs)
        self.assertIsNone(agent.states)
        self.assertEqual(client.calls, 0)

    def test_two_consecutive_model_errors_end_the_episode(self):
        agent, client = self.make_agent("not json", "still not json", {"commands": ["wait"]})
        first = agent.act()
        self.assertEqual([c["kind"] for c in first["commands"]], ["wait"])
        self.assertIsNone(agent.stop_reason)
        second = agent.act()
        self.assertEqual([c["kind"] for c in second["commands"]], ["wait"])
        self.assertEqual(agent.stop_reason, "consecutive_model_errors")
        self.assertEqual(agent.act(), {"kind": "stop", "reason": "consecutive_model_errors"})
        self.assertEqual(client.calls, 2)


class GripperMeasurementTest(unittest.TestCase):
    def test_closed_floor_uses_primary_joint_physical_bound_and_drive_sign(self):
        for sign, physical_limits, expected in ((1, [0.0, 0.044], 0.01 / 0.054),
                                                 (-1, [-0.01, 0.034], 0.01 / 0.054),
                                                 (-1, [0.0, 0.044], 0.0),
                                                 (1, [-0.02, 0.044], 0.0)):
            with self.subTest(sign=sign, limits=physical_limits):
                robot = SimpleNamespace(type="target", ee_type="gripper", arm_name="right_arm",
                                        gripper_scale=[-0.01, 0.044], gripper_move={"sign": sign},
                                        gripper_joint_indices=[1, 2])
                key = SimpleNamespace(root_physx_view=SimpleNamespace(
                    get_dof_limits=lambda: np.array([[[-3.0, 3.0], physical_limits, [-9.0, 9.0]]])))
                env = SimpleNamespace(robot_manager=SimpleNamespace(robot_list=[robot], robot_key=[key]))
                deploy._gripper_floor_disabled = False
                self.assertAlmostEqual(deploy._gripper_closed_floor(env)["right"], expected)

    def test_unavailable_physical_limits_remain_unknown_without_disabling_real_measurement(self):
        robot = SimpleNamespace(type="target", ee_type="gripper", arm_name="right_arm",
                                gripper_scale=[-0.01, 0.044], gripper_move={"sign": 1},
                                gripper_joint_indices=[0, 1])
        env = SimpleNamespace(robot_manager=SimpleNamespace(robot_list=[robot],
                              get_end_effector_real_val=lambda *args, **kwargs: {0: [0.0, 0.0]}))
        deploy._gripper_floor_disabled = False
        deploy._gripper_probe_disabled = False
        self.assertEqual(deploy._gripper_closed_floor(env), {})
        self.assertEqual(deploy._gripper_real(env), {"right": 0.1852})

    def test_real_finger_measurement_matches_command_normalization_in_both_directions(self):
        for sign in (1, -1):
            for value, expected in ((-0.01, 0.0), (0.017, 0.5), (0.044, 1.0)):
                with self.subTest(sign=sign, value=value):
                    robot = SimpleNamespace(type="target", ee_type="gripper", arm_name="right_arm",
                                            gripper_scale=[-0.01, 0.044], gripper_move={"sign": sign})
                    manager = SimpleNamespace(robot_list=[robot],
                                              get_end_effector_real_val=lambda *args, **kwargs: {0: [value, value]})
                    deploy._gripper_probe_disabled = False
                    measured = deploy._gripper_real(SimpleNamespace(robot_manager=manager))
                    self.assertAlmostEqual(measured["right"], expected if sign == 1 else 1 - expected)

    def make_env(self, step_limit=200):
        robots = [SimpleNamespace(type="target", ee_type="gripper", arm_name=f"{side}_arm",
                                  gripper_name=f"{side}_ee", gripper_scale=[-0.01, 0.044],
                                  gripper_move={"sign": 1}, gripper_joint_indices=[6, 7])
                  for side in ("left", "right")]
        keys = [SimpleNamespace(joint_names=[f"joint{i}" for i in range(1, 9)],
                                root_physx_view=SimpleNamespace(get_dof_limits=lambda: np.array(
                                    [[[-3.0, 3.0]] * 6 + [[0.0, 0.044], [0.0, 0.044]]])))
                for _ in robots]
        raw = {robot.arm_name: [0.044, 0.044] for robot in robots}
        manager = SimpleNamespace(robot_list=robots, robot_key=keys,
                                  process_name=lambda name: name + "_joint_state",
                                  get_joint=lambda robot, **kwargs: {0: [0.1] * 6},
                                  get_real_endpose=lambda robot, **kwargs: {0: [0.3, -0.2, 1.0, 1, 0, 0, 0]},
                                  get_end_effector_real_val=lambda robot, **kwargs: {0: raw[robot.arm_name]})
        env = SimpleNamespace(task_name="test", step_lim=step_limit, take_action_cnt=[0], success=[True],
                              robot_manager=manager, executed=[], env_origins=np.zeros((1, 3)),
                              deploy_cfg={"vlm_agent": {}})
        env.is_episode_end = lambda: env.take_action_cnt[0] >= step_limit or not env.success[0]
        env.get_obs = lambda: {"vision": {"cam_head": {"color": np.zeros((8, 8, 3), dtype=np.uint8)}}}
        env.camera_manager = SimpleNamespace(camera_names=[["cam_head"]],
                                             cameras=[[SimpleNamespace(get_intrinsics_matrix=lambda: np.eye(3))]],
                                             get_camera_extrinsics=lambda index, env_idx: np.eye(4))
        return env

    def test_regular_observation_includes_closed_floor_without_rescaling_measurement(self):
        env = self.make_env()
        env.deploy_cfg = {"vlm_agent": {"measure_gripper": True,
                         "visual_aids": {"markers": False, "grid": False, "enhance": False}}}
        observations = []

        def call(func_name, obs=None):
            if func_name == "update_obs":
                observations.append(obs)
            if func_name == "get_action":
                return {"kind": "stop", "reason": "offline_check"}

        deploy.eval_one_episode(env, SimpleNamespace(call=call))
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0]["gripper_real"], {"left": 1.0, "right": 1.0})
        self.assertAlmostEqual(observations[0]["gripper_closed_floor"]["right"], 0.01 / 0.054)
        self.assertEqual(env.executed, [])


if __name__ == "__main__":
    unittest.main()

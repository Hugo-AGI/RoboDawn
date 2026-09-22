"""Skeleton for driving a REAL robot with the same VLM controller used on RoboTwin 2.0.

Implement the four methods marked TODO, write a profile YAML from
``harness/configs/real_robot_profile_template.yaml`` and run::

    LLM_API_KEY=... LLM_API_BASE=https://<endpoint>/v1 python harness/examples/real_robot_skeleton.py \
        --instruction "put the red cup on the plate" --model gemini-3.8-flash \
        --profile my_robot_profile.yaml --output results/real/cup_001

Design rules that made the simulator version work and that carry over:

* the controlled point is the TOOL CENTRE POINT (between the fingertips), not the wrist;
* one command = one *complete* planned motion of the requested magnitude, executed
  until the arm settles, so that the next observation is consistent;
* report planner/IK failures honestly ("the arm did NOT move") and wait for the
  gripper to finish closing before reading its opening;
* draw the fingertip marker and a metric grid on the overview image (world->pixel
  projection needs the camera intrinsics/extrinsics), and provide a wrist camera.

``--mock`` runs the whole loop against a fake robot (blank images) to check the
API key, the prompt and the parsing without touching hardware.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import yaml  # noqa: E402

from harness.agent.llm_client import ChatClient  # noqa: E402
from harness.agent.mllm_agent import AgentConfig, MLLMDiscreteAgent  # noqa: E402
from harness.core import watchdog  # noqa: E402
from harness.core.commands import Command  # noqa: E402
from harness.core.env import DiscreteEnvBase, EpisodeInfo, View  # noqa: E402

TCP_OFFSET_M = 0.0  # distance from the pose your controller reports to the fingertip centre, along the approach axis


class RealRobotEnv(DiscreteEnvBase):
    """Adapter between the VLM controller and a real arm. Replace the TODOs."""

    def __init__(self, instruction: str, step_limit: int = 300, arms=("right",)) -> None:
        self.instruction = instruction
        self._step_limit = step_limit
        self.arms = tuple(arms)
        self._steps = 0
        self._success = False

    # ---------------------------------------------------------------- TODO 1: hardware I/O
    def _read_pose(self, arm: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (fingertip position in metres, 3x3 rotation) in the world frame described in the profile.

        Rotation convention used by the whole harness: column 0 = approach direction (fingers point
        along it), column 1 = finger open/close axis, column 2 = the remaining axis.
        """
        raise NotImplementedError

    def _read_gripper(self, arm: str) -> float:
        """Measured opening, 0 = fully closed .. 1 = fully open (from the joint, not the command)."""
        raise NotImplementedError

    def _capture(self) -> dict[str, np.ndarray]:
        """RGB uint8 arrays, e.g. {"overview": ..., "right_camera": ...}."""
        raise NotImplementedError

    def _move_to(self, arm: str, tcp_m: np.ndarray, rotation: np.ndarray) -> bool:
        """Plan and execute a motion to the given fingertip pose; block until settled. Return False if
        the planner/IK failed (the arm must then stay where it was)."""
        raise NotImplementedError

    def _set_gripper(self, arm: str, opening: float) -> None:
        """Command the gripper and block until the fingers stop moving."""
        raise NotImplementedError

    # -------------------------------------------------------------- interface
    def reset_episode(self, episode_index: Optional[int] = None) -> EpisodeInfo:
        self._steps = 0
        self._success = False
        return EpisodeInfo(task_name="real", task_config="real", episode_index=episode_index or 0, seed=0,
                           instruction=self.instruction, step_limit=self._step_limit)

    @property
    def success(self) -> bool:
        return self._success  # TODO 2: set from a checker, a keyboard confirmation or a VLM judge

    @property
    def steps_used(self) -> int:
        return self._steps

    @property
    def step_limit(self) -> int:
        return self._step_limit

    def _arm_state(self, arm: str) -> dict:
        p, R = self._read_pose(arm)
        return {"position_cm": [round(float(v) * 100, 2) for v in p],
                "approach": [round(float(v), 2) for v in R[:, 0]],
                "finger_axis": [round(float(v), 2) for v in R[:, 1]],
                "gripper_real": round(float(self._read_gripper(arm)), 3)}

    def state(self) -> dict:
        s = {"steps_used": self._steps, "step_limit": self._step_limit, "success": self._success}
        for arm in self.arms:
            s[arm] = self._arm_state(arm)
        return s

    def observe(self) -> dict:
        return {"images": self._capture(), "state": self.state()}

    def views(self, obs: dict, cameras: list[str]) -> list[View]:
        # TODO 3: annotate the overview like harness/robotwin/env.py::annotate_head_image
        # (fingertip marker + metric grid on the table plane) once you have camera calibration.
        return super().views(obs, cameras)

    def execute(self, cmd: Command) -> dict:
        watchdog.beat(cmd.text())
        res = {"command": cmd.text(), "kind": cmd.kind, "arm": cmd.arm, "ok": True, "note": ""}
        if cmd.kind in ("done", "wait"):
            res["after"] = {a: self._arm_state(a) for a in self.arms}
            return res
        if self.budget_exhausted:
            res.update(ok=False, note="step budget exhausted")
            return res
        arm = cmd.arm if cmd.arm in self.arms else self.arms[0]
        p, R = self._read_pose(arm)
        self._steps += 1
        if cmd.kind == "gripper":
            self._set_gripper(arm, float(cmd.value))
            opening = self._read_gripper(arm)
            res["note"] = ("fingers stopped at opening %.2f: something is between them (probably grasped)" % opening
                           if cmd.value < 0.5 and opening > 0.08 else
                           "fingers closed fully: nothing between them" if cmd.value < 0.5 else "gripper opened")
        else:
            target_p, target_R = p.copy(), R.copy()
            if cmd.kind == "move":
                target_p[{"x": 0, "y": 1, "z": 2}[cmd.axis]] += cmd.value / 100.0
            elif cmd.kind == "rotate":
                axis = {"roll": [1, 0, 0], "pitch": [0, 1, 0], "yaw": [0, 0, 1]}[cmd.axis]
                target_R = _axis_angle(axis, math.radians(cmd.value)) @ R
            elif cmd.kind == "point":
                target_R = ORIENTATION_PRESETS[cmd.axis]        # TODO 4: presets for your robot
            elif cmd.kind == "home":
                target_p, target_R = HOME_POSE[arm]
            ok = self._move_to(arm, target_p, target_R)
            if not ok:
                res.update(ok=False, note="motion planner could not reach the target; the arm stayed near its "
                                          "previous pose. Try a smaller step or a different direction.")
        res["after"] = {a: self._arm_state(a) for a in self.arms}
        res["success"] = self.success
        return res


def _axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


R_FORWARD = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)          # approach +y
ORIENTATION_PRESETS = {
    "forward": R_FORWARD,
    "down": _axis_angle([1, 0, 0], -math.pi / 2) @ R_FORWARD,                  # approach -z
    "down45": _axis_angle([1, 0, 0], -math.pi / 4) @ R_FORWARD,
}
HOME_POSE = {"right": (np.array([0.30, -0.20, 0.95]), R_FORWARD), "left": (np.array([-0.30, -0.20, 0.95]), R_FORWARD)}


class MockRobotEnv(RealRobotEnv):
    """Fake robot for dry runs: kinematics are exact, images are blank, success is never reached."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._pose = {arm: (HOME_POSE[arm][0].copy(), HOME_POSE[arm][1].copy()) for arm in self.arms}
        self._grip = {arm: 1.0 for arm in self.arms}

    def _read_pose(self, arm):
        return self._pose[arm][0].copy(), self._pose[arm][1].copy()

    def _read_gripper(self, arm):
        return self._grip[arm]

    def _capture(self):
        img = np.full((480, 640, 3), 235, dtype=np.uint8)
        return {"overview": img, "right_camera": img[:240, :320].copy()}

    def _move_to(self, arm, p, R):
        if p[2] < 0.74 or abs(p[0]) > 0.5:
            return False
        self._pose[arm] = (p, R)
        return True

    def _set_gripper(self, arm, opening):
        self._grip[arm] = opening


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--profile", default=str(REPO / "harness/configs/real_robot_profile_template.yaml"))
    ap.add_argument("--output", default="results/real/episode")
    ap.add_argument("--max_turns", type=int, default=30)
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()
    watchdog.start(stall_seconds=900)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    profile = yaml.safe_load(Path(args.profile).read_text()) or {}
    cfg = AgentConfig(model=args.model, max_turns=args.max_turns, profile=profile, cameras=["overview", "wrist"])
    client = ChatClient(model=cfg.model, timeout_s=cfg.timeout_s, max_tokens=cfg.max_tokens,
                        log_path=out / "llm_calls.jsonl")
    env = (MockRobotEnv if args.mock else RealRobotEnv)(args.instruction)
    ep = env.reset_episode(0)
    res = MLLMDiscreteAgent(cfg, client, out).run_episode(env, ep.instruction, out)
    (out / "result.json").write_text(json.dumps(res.__dict__, indent=1))
    print(json.dumps(res.__dict__, indent=1))


if __name__ == "__main__":
    main()

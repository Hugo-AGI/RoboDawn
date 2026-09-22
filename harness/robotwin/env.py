"""RoboTwin 2.0 discrete "button" environment.

Wraps one RoboTwin task environment behind a small discrete command
interface (see :mod:`harness.robotwin.commands`).  Every command is executed
as ONE RoboTwin ``take_action(..., action_type='ee')`` call, i.e. one curobo
plan to the new end-effector pose followed by execution until the arm has
settled.  Hence one discrete command costs exactly one step of the official
per-task step budget (``task_config/_eval_step_limit.yml``).

The episode protocol (seed selection through the expert check, instruction
sampling, step limit, success test) mirrors ``script/eval_policy.py`` of the
official RoboTwin 2.0 release so that success rates are comparable with the
public leaderboard.

Usage::

    env = RoboTwinDiscreteEnv("place_empty_cup", task_config="demo_clean", seed=0)
    ep = env.reset_episode(episode_index=0)      # -> EpisodeInfo (instruction, seed, ...)
    obs = env.observe()                          # images + proprioception (+ privileged state)
    res = env.execute(parse_command("left move z -5"))
    env.close()
"""

from __future__ import annotations

import importlib
import io
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

from ..core import watchdog
from ..core.commands import Command
from ..core.env import DiscreteEnvBase, EpisodeInfo, View

logger = logging.getLogger(__name__)

def _find_robotwin_root() -> Path:
    """$ROBOTWIN_ROOT, else the RoboTwin submodule of this repository, else ~/RoboTwin.

    Any of them must be an official RoboTwin 2.0 checkout with its assets downloaded."""
    if os.environ.get("ROBOTWIN_ROOT"):
        return Path(os.environ["ROBOTWIN_ROOT"]).resolve()
    for cand in (Path(__file__).resolve().parents[2] / "RoboTwin", Path.home() / "RoboTwin"):
        if (cand / "envs").is_dir():
            return cand.resolve()
    raise FileNotFoundError("RoboTwin not found: initialise the RoboTwin submodule or set ROBOTWIN_ROOT "
                            "to an official RoboTwin 2.0 checkout")


ROBOTWIN_ROOT = _find_robotwin_root()


def _bootstrap_robotwin(root: Path) -> None:
    """Make the RoboTwin package importable and its relative asset paths valid."""
    root = Path(root).resolve()
    if not (root / "envs").is_dir():
        raise FileNotFoundError(f"RoboTwin root {root} does not contain envs/")
    for p in (str(root), str(root / "description" / "utils")):
        if p not in sys.path:
            sys.path.insert(0, p)
    # RoboTwin loads assets with paths relative to the CWD ("./assets/objects/...").
    os.chdir(root)


_bootstrap_robotwin(ROBOTWIN_ROOT)
logging.getLogger("curobo").setLevel(logging.WARNING)

import transforms3d as t3d  # noqa: E402
from envs import CONFIGS_PATH  # noqa: E402
from envs.utils.create_actor import UnStableError  # noqa: E402
from envs.utils.actor_utils import Actor  # noqa: E402
from generate_episode_instructions import generate_episode_descriptions  # noqa: E402


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------

def quat_to_rpy_deg(q) -> list[float]:
    """wxyz quaternion -> [roll, pitch, yaw] in degrees (static xyz convention)."""
    r, p, y = t3d.euler.quat2euler(q, axes="sxyz")
    return [math.degrees(r), math.degrees(p), math.degrees(y)]


def rotate_quat_world(q, axis: str, deg: float):
    """Rotate orientation q (wxyz) about a world axis by deg degrees."""
    axis_vec = {"roll": [1, 0, 0], "pitch": [0, 1, 0], "yaw": [0, 0, 1]}[axis]
    r_delta = t3d.axangles.axangle2mat(axis_vec, math.radians(deg))
    r_new = r_delta @ t3d.quaternions.quat2mat(q)
    return t3d.quaternions.mat2quat(r_new)


def quat_angle_deg(q1, q2) -> float:
    d = abs(float(np.dot(np.asarray(q1) / np.linalg.norm(q1), np.asarray(q2) / np.linalg.norm(q2))))
    return math.degrees(2.0 * math.acos(min(1.0, d)))


def _to_py(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, dict):
        return {k: _to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_py(v) for v in o]
    return o


# ----------------------------------------------------------------------------
# data classes
# ----------------------------------------------------------------------------



DEFAULT_SEED_CACHE_DIR = Path(__file__).resolve().parents[1] / "valid_seeds"
VIDEO_FPS = 5

# Two harness-owned cameras. "agent_camera" is a wide perspective overview placed in front of the
# robot's central post (so the post does not occlude the table centre); "top_camera" looks straight
# down so that the metric grid is undistorted and x/y positions can be read off directly.
DEFAULT_AGENT_CAMERA = {
    "enabled": True,
    "name": "agent_camera",
    "width": 640,
    "height": 480,
    "fovy_deg": 60.0,
    "position": [0.0, -0.36, 1.50],
    "forward": [0.0, 0.42, -0.91],
    "left": [-1.0, 0.0, 0.0],
}
DEFAULT_TOP_CAMERA = {
    "enabled": True,
    "name": "top_camera",
    "width": 640,
    "height": 480,
    "fovy_deg": 50.0,
    "position": [0.0, -0.10, 1.65],
    "forward": [0.0, 0.0, -1.0],
    "left": [-1.0, 0.0, 0.0],
}

TCP_OFFSET_M = 0.12  # tool centre point (between the fingertips) lies 12 cm ahead of the wrist link

# canonical orientations (world frame): approach = R[:,0], finger axis = R[:,1], up = R[:,2]
R_FORWARD = t3d.euler.euler2mat(0.0, 0.0, math.pi / 2.0, axes="sxyz")           # approach +y
ORIENTATION_PRESETS = {
    "forward": R_FORWARD,
    "down": t3d.axangles.axangle2mat([1, 0, 0], -math.pi / 2.0) @ R_FORWARD,     # approach -z
    "down45": t3d.axangles.axangle2mat([1, 0, 0], -math.pi / 4.0) @ R_FORWARD,   # approach (0, .71, -.71)
}


@dataclass
class ArmState:
    position_cm: list[float]          # TOOL CENTRE POINT (between the fingertips), world frame, centimetres
    wrist_cm: list[float]             # wrist link position (RoboTwin "endpose"), centimetres
    approach: list[float]             # unit vector the fingers point along (world frame)
    finger_axis: list[float]          # unit vector along which the fingers open/close (world frame)
    rpy_deg: list[float]              # roll/pitch/yaw about world axes, degrees (for logging only)
    quat_wxyz: list[float]
    gripper: float                    # 0 = closed .. 1 = open (commanded value)
    gripper_real: float               # measured normalised opening


def wrist_to_tcp(pose7: np.ndarray) -> np.ndarray:
    R = t3d.quaternions.quat2mat(pose7[3:])
    return np.asarray(pose7[:3], dtype=float) + R[:, 0] * TCP_OFFSET_M


def tcp_to_wrist(tcp_m: np.ndarray, quat_wxyz) -> np.ndarray:
    R = t3d.quaternions.quat2mat(quat_wxyz)
    return np.asarray(tcp_m, dtype=float) - R[:, 0] * TCP_OFFSET_M


# ----------------------------------------------------------------------------
# the environment
# ----------------------------------------------------------------------------

class RoboTwinDiscreteEnv(DiscreteEnvBase):
    """One RoboTwin 2.0 task behind the discrete command interface (see harness.core.env)."""

    def __init__(
        self,
        task_name: str,
        task_config: str = "demo_randomized",
        seed: int = 0,
        instruction_type: str = "unseen",
        record_video: bool = True,
    ) -> None:
        self.task_name = task_name
        self.task_config = task_config
        self.seed = seed
        self.instruction_type = instruction_type
        self.record_video = record_video
        # extra cameras owned by the harness (not part of the official observation)
        self.agent_camera_cfg = dict(DEFAULT_AGENT_CAMERA)
        self.top_camera_cfg = dict(DEFAULT_TOP_CAMERA)
        self._extra_cameras: dict = {}
        self._initial_objects: dict = {}

        self.args = self._load_task_args(task_name, task_config)
        self.env = self._make_task_env(task_name)

        self.current_seed = 100000 * (1 + seed)
        self.episode_index = -1
        self.episode: Optional[EpisodeInfo] = None
        # cache of expert-validated seeds: {"seeds": [...], "infos": [...]} indexed by valid-episode number
        self.seed_cache_dir = DEFAULT_SEED_CACHE_DIR
        self.seed_cache = self._load_seed_cache()
        self.frames: list[np.ndarray] = []
        self.command_log: list[dict] = []
        self._episode_open = False

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _load_task_args(task_name, task_config) -> dict:
        with open(Path(CONFIGS_PATH) / f"{task_config}.yml", "r", encoding="utf-8") as f:
            args = yaml.safe_load(f)
        args["task_name"] = task_name
        args["task_config"] = task_config
        args["ckpt_setting"] = "discrete_harness"
        args["policy_name"] = "discrete_harness"

        with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
            embodiment_types = yaml.safe_load(f)
        with open(os.path.join(CONFIGS_PATH, "_camera_config.yml"), "r", encoding="utf-8") as f:
            camera_config = yaml.safe_load(f)

        def embodiment_file(kind):
            path = embodiment_types[kind]["file_path"]
            if path is None:
                raise ValueError(f"no embodiment file for {kind}")
            return path

        def embodiment_config(robot_file):
            with open(os.path.join(robot_file, "config.yml"), "r", encoding="utf-8") as f:
                return yaml.safe_load(f)

        embodiment = args["embodiment"]
        head_type = args["camera"]["head_camera_type"]
        args["head_camera_h"] = camera_config[head_type]["h"]
        args["head_camera_w"] = camera_config[head_type]["w"]
        if len(embodiment) == 1:
            args["left_robot_file"] = embodiment_file(embodiment[0])
            args["right_robot_file"] = embodiment_file(embodiment[0])
            args["dual_arm_embodied"] = True
        elif len(embodiment) == 3:
            args["left_robot_file"] = embodiment_file(embodiment[0])
            args["right_robot_file"] = embodiment_file(embodiment[1])
            args["embodiment_dis"] = embodiment[2]
            args["dual_arm_embodied"] = False
        else:
            raise ValueError("embodiment items should be 1 or 3")
        args["left_embodiment_config"] = embodiment_config(args["left_robot_file"])
        args["right_embodiment_config"] = embodiment_config(args["right_robot_file"])

        args["eval_mode"] = True
        args["render_freq"] = 0
        args["eval_video_save_dir"] = None   # we record our own video (no ffmpeg dependency)
        args["eval_video_log"] = False
        return args

    @staticmethod
    def _make_task_env(task_name):
        module = importlib.import_module(f"envs.{task_name}")
        return getattr(module, task_name)()

    # --------------------------------------------------------------- episodes
    def _seed_cache_path(self) -> Path:
        return self.seed_cache_dir / f"{self.task_name}__{self.task_config}__seed{self.seed}.json"

    def _load_seed_cache(self) -> dict:
        p = self._seed_cache_path()
        if p.is_file():
            try:
                d = json.loads(p.read_text())
                if d.get("seed_base") == 100000 * (1 + self.seed):
                    return d
            except Exception:  # noqa: BLE001
                pass
        return {"seed_base": 100000 * (1 + self.seed), "seeds": [], "infos": [], "next_seed": 100000 * (1 + self.seed)}

    def save_seed_cache(self) -> None:
        p = self._seed_cache_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(self.seed_cache))
        os.replace(tmp, p)

    def validate_seeds(self, count: int, save_every: int = 1) -> dict:
        """Extend the valid-seed cache until it holds `count` expert-solvable seeds."""
        while len(self.seed_cache["seeds"]) < count:
            seed = int(self.seed_cache["next_seed"])
            ok, info, reason = self._expert_check(seed)
            self.seed_cache["next_seed"] = seed + 1
            if ok:
                self.seed_cache["seeds"].append(seed)
                self.seed_cache["infos"].append(_to_py(info))
                if len(self.seed_cache["seeds"]) % save_every == 0:
                    self.save_seed_cache()
            else:
                logger.info("seed %d rejected: %s", seed, reason)
        self.save_seed_cache()
        return self.seed_cache

    def _expert_check(self, seed: int):
        """Run the built-in expert on a seed. Returns (ok, info_dict, reason)."""
        watchdog.beat(f"expert check seed {seed}")
        try:
            self.env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **self.args)
            expert_info = self.env.play_once()
            ok = bool(self.env.plan_success and self.env.check_success())
            self.env.close_env()
        except UnStableError as exc:
            self.env.close_env()
            return False, {}, f"unstable: {exc}"
        except Exception as exc:  # noqa: BLE001
            logger.warning("expert check raised for seed %s: %s", seed, exc)
            try:
                self.env.close_env()
            except Exception:  # noqa: BLE001
                pass
            return False, {}, f"exception: {exc}"
        info = expert_info.get("info", {}) if isinstance(expert_info, dict) else {}
        return ok, info, "" if ok else "expert failed"

    def reset_episode(self, episode_index: Optional[int] = None) -> EpisodeInfo:
        """Set up valid episode number `episode_index` following the official protocol.

        A seed is accepted only when the built-in expert solves it (expert
        check); the same seed is then re-initialised for the policy.  Valid
        seeds are cached so that shards of one task agree on the episode list.
        """
        if self._episode_open:
            self.close_episode()
        self.episode_index = self.episode_index + 1 if episode_index is None else episode_index

        if len(self.seed_cache["seeds"]) <= self.episode_index:
            self.validate_seeds(self.episode_index + 1)
        seed = int(self.seed_cache["seeds"][self.episode_index])
        expert_info = {"info": self.seed_cache["infos"][self.episode_index]}
        # Official protocol: the expert runs on the SAME task object right before the policy
        # episode. Several tasks (open_laptop, place_object_scale, put_object_cabinet, ...) set
        # attributes in play_once() that check_success() later reads, so the expert must be
        # replayed in-process even when the seed comes from the cache.
        ok, info, reason = self._expert_check(seed)
        if not ok:
            logger.warning("cached seed %d failed the expert replay (%s); continuing anyway", seed, reason)
        if info:
            expert_info = {"info": info}
        return self._start_episode(seed, expert_info)

    def _start_episode(self, seed: int, expert_info: dict) -> EpisodeInfo:
        watchdog.beat(f"start episode seed {seed}")
        # real episode for the policy
        self.env.setup_demo(now_ep_num=self.episode_index, seed=seed, is_test=True, **self.args)
        self._episode_open = True
        self._create_agent_camera()
        info_dict = expert_info.get("info", {}) if isinstance(expert_info, dict) else {}
        try:
            results = generate_episode_descriptions(self.task_name, [info_dict], 1)
            candidates = results[0][self.instruction_type]
            rng = np.random.RandomState(seed)
            instruction = str(rng.choice(candidates))
        except Exception as exc:  # noqa: BLE001
            logger.warning("instruction generation failed (%s); using task name", exc)
            instruction = self.task_name.replace("_", " ")
        self.env.set_instruction(instruction=instruction)

        self.frames = []
        self.command_log = []
        self.episode = EpisodeInfo(
            task_name=self.task_name,
            task_config=self.task_config,
            episode_index=self.episode_index,
            seed=seed,
            instruction=instruction,
            step_limit=int(self.env.step_lim),
            expert_info=_to_py(info_dict),
        )
        self.env.get_obs()
        self._initial_objects = {k: v.get("position_cm") for k, v in self.object_states().items()}
        self._record_frame()
        return self.episode

    def _create_agent_camera(self) -> None:
        self._extra_cameras = {}
        import sapien

        scene = self.env.scene
        for cfg in (self.agent_camera_cfg, self.top_camera_cfg):
            if not cfg.get("enabled", True):
                continue
            fwd = np.asarray(cfg["forward"], dtype=float)
            fwd /= np.linalg.norm(fwd)
            left = np.asarray(cfg["left"], dtype=float)
            left /= np.linalg.norm(left)
            up = np.cross(fwd, left)
            mat = np.eye(4)
            mat[:3, :3] = np.stack([fwd, left, up], axis=1)
            mat[:3, 3] = np.asarray(cfg["position"], dtype=float) + np.array([0.0, 0.0, float(self.env.table_z_bias)])
            cam = scene.add_camera(
                name=cfg["name"], width=int(cfg["width"]), height=int(cfg["height"]),
                fovy=math.radians(float(cfg["fovy_deg"])), near=0.1, far=100.0,
            )
            cam.entity.set_pose(sapien.Pose(mat))
            self._extra_cameras[cfg["name"]] = cam

    def _agent_camera_capture(self) -> dict:
        out = {}
        for name, cam in self._extra_cameras.items():
            cam.take_picture()
            rgba = cam.get_picture("Color")
            rgb = (np.clip(rgba[:, :, :3], 0.0, 1.0) * 255.0).astype(np.uint8)
            params = {"intrinsic_cv": np.asarray(cam.get_intrinsic_matrix()), "extrinsic_cv": np.asarray(cam.get_extrinsic_matrix())}
            out[name] = (np.ascontiguousarray(rgb), params)
        return out

    def close_episode(self) -> None:
        if self._episode_open:
            try:
                self.env.close_env(clear_cache=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning("close_env failed: %s", exc)
            self._episode_open = False

    def close(self) -> None:
        self.close_episode()

    # ------------------------------------------------------------------ state
    @property
    def steps_used(self) -> int:
        return int(self.env.take_action_cnt)

    @property
    def step_limit(self) -> int:
        return int(self.env.step_lim)

    @property
    def success(self) -> bool:
        return bool(self.env.eval_success)

    @property
    def budget_exhausted(self) -> bool:
        return self.steps_used >= self.step_limit

    def measured_gripper_opening(self, arm: str) -> float:
        """Normalised finger opening measured from the joint position (0 closed .. 1 open)."""
        robot = self.env.robot
        entity = robot.left_entity if arm == "left" else robot.right_entity
        joints = robot.left_gripper if arm == "left" else robot.right_gripper
        try:
            joint = joints[0][0]
            active = entity.get_active_joints()
            qpos = float(entity.get_qpos()[active.index(joint)])
            lo, hi = [float(v) for v in np.asarray(joint.get_limits()).reshape(-1)[:2]]
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = 0.0, 0.04765
            return float(np.clip((qpos - lo) / (hi - lo), 0.0, 1.0))
        except Exception:  # noqa: BLE001
            return float(robot.get_normal_real_gripper_val()[0 if arm == "left" else 1])

    def arm_state(self, arm: str) -> ArmState:
        pose = np.asarray(self.env.get_arm_pose(arm), dtype=float)
        R = t3d.quaternions.quat2mat(pose[3:])
        tcp = pose[:3] + R[:, 0] * TCP_OFFSET_M
        gripper = float(self.env.robot.get_left_gripper_val() if arm == "left" else self.env.robot.get_right_gripper_val())
        real = self.measured_gripper_opening(arm)
        return ArmState(
            position_cm=[round(float(v) * 100.0, 2) for v in tcp],
            wrist_cm=[round(float(v) * 100.0, 2) for v in pose[:3]],
            approach=[round(float(v), 2) for v in R[:, 0]],
            finger_axis=[round(float(v), 2) for v in R[:, 1]],
            rpy_deg=[round(v, 1) for v in quat_to_rpy_deg(pose[3:])],
            quat_wxyz=[float(v) for v in pose[3:]],
            gripper=round(gripper, 3),
            gripper_real=round(float(real), 3),
        )

    def object_states(self) -> dict[str, dict]:
        """Privileged object poses (only used for the oracle / state-assisted ablations)."""
        objs = {}
        for name, value in vars(self.env).items():
            if isinstance(value, Actor):
                try:
                    pose = value.get_pose()
                    entry = {
                        "position_cm": [round(float(v) * 100.0, 2) for v in pose.p],
                        "quat_wxyz": [float(v) for v in pose.q],
                        "model": value.get_name(),
                    }
                    fps = []
                    for i in range(8):
                        try:
                            fp = value.get_functional_point(i, "pose")
                            fps.append([round(float(v) * 100.0, 2) for v in fp.p])
                        except Exception:  # noqa: BLE001
                            break
                    if fps:
                        entry["functional_points_cm"] = fps
                    objs[name] = entry
                except Exception:  # noqa: BLE001
                    continue
        return objs

    @property
    def table_z_cm(self) -> float:
        """Height of the table top in cm (74 in demo_clean; lowered by up to 3 cm under domain randomisation)."""
        return round(74.0 + 100.0 * float(getattr(self.env, "table_z_bias", 0.0)), 1)

    def prompt_context(self) -> dict:
        return {"table_z": int(round(self.table_z_cm))}

    def state(self) -> dict:
        s = {
            "steps_used": self.steps_used,
            "step_limit": self.step_limit,
            "success": self.success,
            "table_z_cm": self.table_z_cm,
            "left": asdict(self.arm_state("left")),
            "right": asdict(self.arm_state("right")),
        }
        s["objects"] = self.object_states()
        return s

    # ----------------------------------------------------------- observation
    def observe(self) -> dict:
        """Return raw RGB images (uint8 HxWx3, RGB) and camera parameters."""
        obs = self.env.get_obs()
        cams = obs["observation"]
        images = {name: np.ascontiguousarray(cams[name]["rgb"]) for name in cams if "rgb" in cams[name]}
        params = {
            name: {
                "intrinsic_cv": np.asarray(cams[name]["intrinsic_cv"]),
                "extrinsic_cv": np.asarray(cams[name]["extrinsic_cv"]),
            }
            for name in cams
            if "intrinsic_cv" in cams[name]
        }
        for name, (rgb, prm) in self._agent_camera_capture().items():
            images[name] = rgb
            params[name] = prm
        return {"images": images, "camera_params": params, "state": self.state()}

    def project_points(self, camera: str, points_m: np.ndarray, camera_params: dict) -> np.ndarray:
        """Project Nx3 world points (metres) to pixel coordinates of a camera."""
        K = camera_params[camera]["intrinsic_cv"]
        E = camera_params[camera]["extrinsic_cv"]  # 3x4 world->camera (OpenCV)
        pts = np.asarray(points_m, dtype=float).reshape(-1, 3)
        cam = (E[:, :3] @ pts.T + E[:, 3:4]).T
        z = np.clip(cam[:, 2:3], 1e-6, None)
        uv = (K @ (cam / z).T).T
        return uv[:, :2]

    # ---------------------------------------------------------------- video
    def _record_frame(self, caption: str = "") -> None:
        """Append one video frame (wide agent camera if available, else the official head camera)."""
        if not self.record_video:
            return
        try:
            self.env._update_render()
            cam = self._extra_cameras.get(self.agent_camera_cfg["name"])
            if cam is not None:
                cam.take_picture()
                rgba = cam.get_picture("Color")
                rgb = (np.clip(rgba[:, :, :3], 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                self.env.cameras.update_picture()
                rgb = self.env.cameras.get_rgb()["head_camera"]["rgb"]
            frame = np.ascontiguousarray(rgb)
            if caption:
                from PIL import Image, ImageDraw

                img = Image.fromarray(frame)
                draw = ImageDraw.Draw(img)
                draw.rectangle([0, 0, img.width, 14], fill=(0, 0, 0))
                draw.text((4, 2), caption[:110], fill=(255, 255, 255))
                frame = np.asarray(img)
            self.frames.append(frame)
        except Exception as exc:  # noqa: BLE001
            logger.debug("frame capture failed: %s", exc)

    def save_video(self, path: Path) -> Optional[str]:
        if not self.frames:
            return None
        import imageio

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(str(path), self.frames, fps=VIDEO_FPS, codec="libx264", quality=6)
        return str(path)

    def _settle_gripper(self, arm: str, target: float, max_steps: int = 400, stable_needed: int = 20) -> int:
        """Step physics while holding all drive targets until the finger opening stops changing."""
        robot = self.env.robot
        other = "right" if arm == "left" else "left"
        other_val = float(robot.get_left_gripper_val() if other == "left" else robot.get_right_gripper_val())
        prev = self.measured_gripper_opening(arm)
        stable = 0
        steps = 0
        for steps in range(1, max_steps + 1):
            robot.set_gripper(target, arm)
            robot.set_gripper(other_val, other)
            self.env.scene.step()
            cur = self.measured_gripper_opening(arm)
            stable = stable + 1 if abs(cur - prev) < 5e-4 else 0
            prev = cur
            if stable >= stable_needed:
                break
        try:
            if self.env.check_success():
                self.env.eval_success = True
        except Exception:  # noqa: BLE001
            pass
        return steps

    # -------------------------------------------------------------- commands
    def _pose_of(self, arm: str) -> np.ndarray:
        return np.asarray(self.env.get_arm_pose(arm), dtype=float)

    def command_target(self, cmd: Command, pose: np.ndarray) -> np.ndarray:
        """Wrist pose (xyz + wxyz) that a move/rotate/point/home command asks for when the arm is at ``pose``."""
        tcp = wrist_to_tcp(pose)
        quat = np.asarray(pose[3:], dtype=float).copy()
        if cmd.kind == "move":
            tcp[{"x": 0, "y": 1, "z": 2}[cmd.axis]] += cmd.value / 100.0
        elif cmd.kind == "rotate":
            quat = rotate_quat_world(quat, cmd.axis, cmd.value)
        elif cmd.kind == "point":
            quat = t3d.quaternions.mat2quat(ORIENTATION_PRESETS[cmd.axis])
        elif cmd.kind == "home":
            robot = self.env.robot
            origin = np.asarray(robot.left_original_pose if cmd.arm == "left" else robot.right_original_pose, dtype=float)
            quat, tcp = origin[3:].copy(), wrist_to_tcp(origin)
        else:
            raise ValueError(f"{cmd.kind} has no target pose")
        return np.concatenate([tcp_to_wrist(tcp, quat), quat])

    def plan_only(self, arm: str, target_pose: np.ndarray, qpos: Optional[np.ndarray] = None):
        """Ask the motion planner whether ``arm`` can reach ``target_pose`` from joint state ``qpos`` (default:
        the current one) without moving anything.  Returns (ok, joint state at the end of the plan) so that
        a sequence of commands can be checked before any of them is executed."""
        robot = self.env.robot
        entity = robot.left_entity if arm == "left" else robot.right_entity
        names = robot.left_arm_joints_name if arm == "left" else robot.right_arm_joints_name
        active = [j.get_name() for j in entity.get_active_joints()]
        idx = [active.index(n) for n in names]
        qpos = np.array(entity.get_qpos() if qpos is None else qpos, dtype=np.float32)   # curobo wants float32
        plan = (robot.left_plan_path if arm == "left" else robot.right_plan_path)(
            np.asarray(target_pose, dtype=float), last_qpos=qpos)
        if plan.get("status") != "Success":
            return False, None
        q = qpos.copy()
        q[idx] = np.asarray(plan["position"], dtype=np.float32)[-1]
        return True, q

    def _physics_steps(self) -> int:
        """Physics steps taken so far on the current SAPIEN scene (scene.step is wrapped on first use;
        RoboTwin calls it once per control tick, so this is the robot's motion time in ticks)."""
        scene = self.env.scene
        counter = getattr(scene, "_harness_step_counter", None)
        if counter is None:
            orig = scene.step
            counter = [0]

            def counted():
                counter[0] += 1
                return orig()

            scene.step = counted
            scene._harness_step_counter = counter
        return counter[0]

    def _physics_dt(self) -> float:
        try:
            return float(self.env.scene.get_timestep())
        except Exception:  # noqa: BLE001
            return 1.0 / 250.0  # RoboTwin's default (envs/_base_task.py: set_timestep(1/250))

    def execute(self, cmd: Command) -> dict:
        """Execute one discrete command; returns a result dict."""
        t0 = time.time()
        watchdog.beat(f"execute {cmd.text()}")
        result = {
            "command": cmd.text(),
            "kind": cmd.kind,
            "arm": cmd.arm,
            "ok": True,
            "note": "",
            "steps_used_before": self.steps_used,
        }
        early = None
        if cmd.kind == "done":
            early = dict(ok=True, note="agent declared done")
        elif self.success:
            early = dict(ok=False, note="task already succeeded")
        elif self.budget_exhausted:
            early = dict(ok=False, note="step budget exhausted")
        if early is not None:
            snap = {"left": asdict(self.arm_state("left")), "right": asdict(self.arm_state("right"))}
            result.update(early, steps_used_after=self.steps_used, success=self.success, before=snap, after=snap, seconds=0.0)
            self.command_log.append(result)
            return result

        left_pose = self._pose_of("left")
        right_pose = self._pose_of("right")
        lg = float(self.env.robot.get_left_gripper_val())
        rg = float(self.env.robot.get_right_gripper_val())
        target_pose = None          # wrist-frame target handed to RoboTwin
        target_tcp = None           # tool-centre-point target used for error checking

        if cmd.kind == "gripper":
            if cmd.arm == "left":
                lg = float(cmd.value)
            else:
                rg = float(cmd.value)
        elif cmd.kind != "wait":
            target_pose = self.command_target(cmd, left_pose if cmd.arm == "left" else right_pose)
            target_tcp = wrist_to_tcp(target_pose)
            if cmd.arm == "left":
                left_pose = target_pose
            else:
                right_pose = target_pose

        action = np.concatenate([left_pose, [lg], right_pose, [rg]]).astype(np.float64)
        before = {"left": self.arm_state("left"), "right": self.arm_state("right")}
        phys0 = self._physics_steps()
        try:
            self.env.take_action(action, action_type="ee")
        except Exception as exc:  # noqa: BLE001
            logger.exception("take_action failed")
            result.update(ok=False, note=f"simulator error: {exc}")
        # how long the robot was moving for this command, in simulated time: one scene.step per control tick
        result["physics_steps"] = self._physics_steps() - phys0
        result["robot_seconds"] = round(result["physics_steps"] * self._physics_dt(), 3)
        if cmd.kind == "gripper" and result["ok"] and not self.success:
            # the fingers move slowly; let physics run (without consuming budget) until the opening settles,
            # otherwise a half-closed gripper looks like a successful grasp
            result["settle_steps"] = self._settle_gripper(cmd.arm, float(cmd.value))
        after = {"left": self.arm_state("left"), "right": self.arm_state("right")}
        self._record_frame(f"step {self.steps_used}/{self.step_limit}  {cmd.text()}" + ("  SUCCESS" if self.success else ""))

        if target_pose is not None and result["ok"]:
            reached = after[cmd.arm]
            pos_err_cm = float(np.linalg.norm(np.asarray(reached.position_cm) - target_tcp * 100.0))
            ang_err = quat_angle_deg(reached.quat_wxyz, target_pose[3:])
            result["position_error_cm"] = round(pos_err_cm, 2)
            result["orientation_error_deg"] = round(ang_err, 1)
            if pos_err_cm > 1.5 or ang_err > 8.0:
                result["ok"] = False
                # say what actually happened: the planner often gets most of the way before the
                # tolerance check fails, and "the arm did not move" would then mislead the model
                moved = np.asarray(reached.position_cm) - np.asarray(before[cmd.arm].position_cm)
                moved_cm = float(np.linalg.norm(moved))
                turned_deg = quat_angle_deg(reached.quat_wxyz, before[cmd.arm].quat_wxyz)
                result["moved_cm"] = round(moved_cm, 2)
                result["turned_deg"] = round(turned_deg, 1)
                if moved_cm < 0.5 and turned_deg < 2.0:
                    what = "the arm did NOT move"
                elif cmd.kind == "rotate":
                    what = f"the gripper turned only {turned_deg:.0f} deg of the requested {abs(cmd.value):.0f}"
                else:
                    what = (f"the fingertips moved ({moved[0]:+.1f}, {moved[1]:+.1f}, {moved[2]:+.1f}) cm, i.e. only part "
                            f"of the way")
                result["note"] = (
                    f"motion planner could not reach the target: {what}; remaining error {pos_err_cm:.1f} cm / "
                    f"{ang_err:.0f} deg (see CURRENT STATE for the actual pose). Try a smaller step, a different "
                    "direction, or move away from the table/robot body first."
                )
        if cmd.kind == "gripper" and result["ok"]:
            real = after[cmd.arm].gripper_real
            if cmd.value < 0.5 and real > 0.08:
                result["note"] = f"fingers stopped at opening {real:.2f}: something is between them (probably grasped)"
            elif cmd.value < 0.5:
                result["note"] = "fingers closed fully: nothing between them"
            else:
                result["note"] = "gripper opened"
        if cmd.clipped:
            result["note"] = (result["note"] + " " if result["note"] else "") + "(magnitude was clipped to the limit)"

        result.update(
            steps_used_after=self.steps_used,
            success=self.success,
            before={k: asdict(v) for k, v in before.items()},
            after={k: asdict(v) for k, v in after.items()},
            seconds=round(time.time() - t0, 2),
        )
        self.command_log.append(result)
        return result


    # ------------------------------------------------------------- viewing
    def views(self, obs: dict, cameras: list[str]) -> list[View]:
        """The annotated overview/top-down views and the wrist cameras named in ``cameras``."""
        from PIL import Image

        out: list[View] = []
        grid_note = (f"; the thin grey grid is drawn ON THE TABLE TOP (z = {int(round(self.table_z_cm))}) every 10 cm, "
                     "labelled x... along the top/bottom edges and y... along the side edges - read object x/y "
                     "positions from it")
        for cam in cameras:
            if cam == "wrist":
                for arm in ("left", "right"):
                    name = f"{arm}_camera"
                    if name not in obs["images"]:
                        continue
                    img = enhance_image(Image.fromarray(obs["images"][name]))
                    out.append(View(name, img, f"{name} (wrist camera of the {arm} arm, looking along its fingers)"))
                continue
            if cam not in obs["images"]:
                continue
            img = annotate_head_image(self, obs, camera=cam)
            marker_note = ("; markers: cyan circle L = left fingertips, orange circle R = right fingertips, line = wrist "
                           "to fingertips")
            axes_note = "; xyz legend"
            if cam == "agent_camera":
                cap = ("agent_camera (perspective overview from above the robot looking forward/down; the robot body "
                       "is at the bottom, +y is up the image" + marker_note + axes_note + grid_note + ")")
            elif cam == "top_camera":
                cap = ("top_camera (straight-down map view: +x to the right, +y up" + (marker_note and "; same markers")
                       + grid_note + "; use it to read x/y, use the perspective view for heights)")
            elif cam == "head_camera":
                cap = ("head_camera (the robot's own head camera, mounted above the torso looking down at the table; "
                       "+y is up the image, the left arm appears on the left" + marker_note + axes_note + grid_note + ")")
            else:
                cap = cam
            out.append(View(cam, img, cap))
        return out

    def should_stop(self) -> Optional[str]:
        """Stop early once an object that started on the table has fallen far below it."""
        if not self._initial_objects:
            return None
        for name, o in (self.object_states() or {}).items():
            p, p0 = o.get("position_cm"), self._initial_objects.get(name)
            if p and p0 and p0[2] >= 70.0 and p[2] < 55.0 and (p0[2] - p[2]) > 30.0:
                return "object_fell"
        return None


# ----------------------------------------------------------------------------
# image annotation helpers (pure PIL)
# ----------------------------------------------------------------------------

def enhance_image(img, cutoff: float = 1.0):
    """Auto-contrast (percentile stretch) so that the washed-out clean scene keeps object detail."""
    from PIL import ImageOps

    try:
        return ImageOps.autocontrast(img, cutoff=cutoff, preserve_tone=True)
    except TypeError:  # older PIL
        return ImageOps.autocontrast(img, cutoff=cutoff)


def annotate_head_image(env: RoboTwinDiscreteEnv, obs: dict, camera: str = "agent_camera"):
    """Return a PIL image of the camera view, contrast-enhanced, with the table grid, gripper markers and world axes."""
    from PIL import Image, ImageDraw

    if camera not in obs["images"]:
        camera = "head_camera"
    img = enhance_image(Image.fromarray(obs["images"][camera]))
    draw = ImageDraw.Draw(img)
    params = obs["camera_params"]
    if camera not in params:
        return img

    def px(points_m):
        uv = env.project_points(camera, np.asarray(points_m), params)
        return [(float(u), float(v)) for u, v in uv]

    _draw_table_grid(draw, px, img.size, table_z=0.74 + float(env.env.table_z_bias))

    # gripper tool points (circle = fingertip centre, line = wrist -> fingertips)
    for arm, color in (("left", (0, 200, 255)), ("right", (255, 120, 0))):
        st = obs["state"][arm]
        p = np.asarray(st["position_cm"]) / 100.0
        w = np.asarray(st["wrist_cm"]) / 100.0
        (u, v), (uw, vw) = px([p, w])
        r = 5
        draw.line([uw, vw, u, v], fill=color, width=2)
        draw.ellipse([u - r, v - r, u + r, v + r], outline=color, width=2)
        draw.text((u + r + 2, v - r - 2), arm[0].upper(), fill=color)

    # world axes legend near the front edge of the table
    origin = np.array([-0.45, -0.38, 0.745 + float(env.env.table_z_bias)])
    axes = {"+x": ([0.1, 0, 0], (255, 60, 60)), "+y": ([0, 0.1, 0], (60, 220, 60)), "+z": ([0, 0, 0.1], (80, 80, 255))}
    o = px([origin])[0]
    for name, (vec, color) in axes.items():
        e = px([origin + np.asarray(vec)])[0]
        draw.line([o, e], fill=color, width=2)
        draw.text(e, name, fill=color)
    return img


def _draw_table_grid(draw, px, size, table_z: float, x_range=(-0.5, 0.5), y_range=(-0.45, 0.25), step=0.1):
    """Draw a metric grid on the table plane: x labels along the near edge, y labels along the left edge."""
    W, H = size
    grey = (120, 120, 120)
    xs = np.arange(x_range[0], x_range[1] + 1e-6, step)
    ys = np.arange(y_range[0], y_range[1] + 1e-6, step)
    for x in xs:
        pts = px([[x, y, table_z] for y in ys])
        draw.line(pts, fill=grey, width=1)
    for y in ys:
        pts = px([[x, y, table_z] for x in xs])
        draw.line(pts, fill=grey, width=1)
    # labels: x along the near (bottom) and far (top) edges, y along both side edges
    for x in xs:
        for y_edge in (y_range[0], y_range[1]):
            (u, v), = px([[x, y_edge, table_z]])
            if 0 <= u < W and 0 <= v < H:
                draw.text((u - 8, min(max(v + 2, 0), H - 12)), f"x{int(round(x * 100)):+d}", fill=(60, 60, 60))
    for y in ys:
        for x_edge in (x_range[0], x_range[1]):
            (u, v), = px([[x_edge, y, table_z]])
            if -40 <= u < W and 0 <= v < H:
                draw.text((min(max(u + 2, 0), W - 34), v - 6), f"y{int(round(y * 100)):+d}", fill=(60, 60, 60))


def encode_png(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

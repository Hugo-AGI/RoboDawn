"""Environment interface for the discrete-command VLM harness.

A *discrete environment* hides one robot (simulated or real) behind three
calls: :meth:`DiscreteEnvBase.observe` (images + proprioceptive state),
:meth:`DiscreteEnvBase.execute` (one :class:`~harness.core.commands.Command`)
and :meth:`DiscreteEnvBase.reset_episode`.  The VLM agent in
``harness/agent`` only talks to this interface, so porting the controller to
a new simulator or a real arm means implementing this class (see
``harness/examples/real_robot_skeleton.py``) and writing a profile YAML.

State schema expected by the prompt builder (``harness/agent/prompts.py``)::

    {
      "steps_used": int, "step_limit": int, "success": bool,
      "left":  {"position_cm": [x, y, z],      # tool-centre point, world frame, centimetres
                "approach": [ax, ay, az],       # unit vector the fingers point along
                "finger_axis": [fx, fy, fz],    # unit vector the fingers open/close along
                "gripper_real": float},         # measured opening, 0 closed .. 1 open
      "right": {...same...},                   # omit for single-arm robots
      "objects": {...}                          # simulator object poses: never shown to the model; used to pick the
                                                #   demonstration entry (grasp side) and for the object-fell early stop
    }

Result dict returned by :meth:`execute` (keys used by the agent)::

    {"command": str, "kind": str, "arm": str|None, "ok": bool, "note": str,
     "after": {"left": {...arm state...}, "right": {...}}}   # "after" needed for grasp tracking
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .commands import Command


@dataclass
class EpisodeInfo:
    task_name: str
    task_config: str
    episode_index: int
    seed: int
    instruction: str
    step_limit: int
    expert_info: dict = field(default_factory=dict)


@dataclass
class View:
    """One image handed to the VLM this turn."""

    name: str
    image: Any            # PIL.Image.Image
    caption: str


class DiscreteEnvBase(ABC):
    """Contract between a robot (sim or real) and the VLM controller."""

    # ------------------------------------------------------------ episodes
    @abstractmethod
    def reset_episode(self, episode_index: Optional[int] = None) -> EpisodeInfo:
        """Prepare a new episode and return its description (instruction, step limit, ...)."""

    def close_episode(self) -> None:
        """Release per-episode resources (default: nothing)."""

    def close(self) -> None:
        """Release everything (default: close the episode)."""
        self.close_episode()

    # ---------------------------------------------------------------- state
    @abstractmethod
    def observe(self) -> dict:
        """Return ``{"images": {name: uint8 HxWx3 RGB array}, "state": self.state(), ...}``."""

    @abstractmethod
    def state(self) -> dict:
        """Proprioceptive state following the schema in the module docstring."""

    @property
    @abstractmethod
    def success(self) -> bool:
        """Whether the task checker has registered success."""

    @property
    @abstractmethod
    def steps_used(self) -> int:
        ...

    @property
    @abstractmethod
    def step_limit(self) -> int:
        ...

    @property
    def budget_exhausted(self) -> bool:
        return self.steps_used >= self.step_limit

    # -------------------------------------------------------------- actions
    @abstractmethod
    def execute(self, cmd: Command) -> dict:
        """Execute one discrete command and return the result dict (see module docstring)."""

    # ------------------------------------------------------------- viewing
    def views(self, obs: dict, cameras: list[str]) -> list[View]:
        """Images to show the VLM for this turn (default: every raw image, no annotation)."""
        from PIL import Image

        return [View(name, Image.fromarray(img), name) for name, img in obs["images"].items()]

    def should_stop(self) -> Optional[str]:
        """Harness-side early-stop reason (e.g. an object fell off the table), or None."""
        return None

    def prompt_context(self) -> dict:
        """Per-episode numbers substituted into the profile text (``{table_z}``, ``{table_z+5}`` ...).

        Return e.g. ``{"table_z": 71}`` when the scene is randomised; the default profile assumes the
        values written literally in the YAML.
        """
        return {}

    def save_video(self, path: Path) -> Optional[str]:
        """Write the episode video if the environment records one."""
        return None

"""Discrete command vocabulary for the RoboTwin 2.0 button harness.

The vocabulary is deliberately tiny so that a multimodal LLM can drive the
robot by emitting one short line per action, e.g.::

    left move z -5        # translate the left gripper 5 cm down (world frame)
    right rotate yaw 30   # rotate the right gripper 30 deg about world Z
    left gripper close
    right home
    done

Translation magnitudes are in centimetres and rotation magnitudes in degrees.
Both are clipped to the limits in :data:`LIMITS` so that a single command can
never fling the arm across the workspace.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

ARMS = ("left", "right")
MOVE_AXES = ("x", "y", "z")
ROT_AXES = ("roll", "pitch", "yaw")
GRIPPER_STATES = {"open": 1.0, "close": 0.0, "closed": 0.0, "half": 0.5}

# Safety limits applied to a single command.
LIMITS = {
    "move_cm": 20.0,
    "rotate_deg": 90.0,
}


@dataclass
class Command:
    """One discrete command.

    kind is one of ``move``, ``rotate``, ``point``, ``gripper``, ``home``, ``wait``, ``done``.
    """

    kind: str
    arm: Optional[str] = None
    axis: Optional[str] = None
    value: Optional[float] = None
    raw: str = ""
    clipped: bool = False
    extra: dict = field(default_factory=dict)

    def text(self) -> str:
        if self.kind == "move":
            return f"{self.arm} move {self.axis} {self.value:+.1f}"
        if self.kind == "rotate":
            return f"{self.arm} rotate {self.axis} {self.value:+.1f}"
        if self.kind == "gripper":
            return f"{self.arm} gripper {self.value:.2f}"
        if self.kind == "home":
            return f"{self.arm} home"
        if self.kind == "point":
            return f"{self.arm} point {self.axis}"
        return self.kind


class CommandError(ValueError):
    pass


_NUM = r"([-+]?\d+(?:\.\d+)?)"
_RE_MOVE = re.compile(rf"^(left|right)\s+move\s+([xyz])\s*{_NUM}\s*(cm)?$", re.I)
_RE_ROT = re.compile(rf"^(left|right)\s+rotate\s+(roll|pitch|yaw)\s*{_NUM}\s*(deg)?$", re.I)
_RE_GRIP = re.compile(rf"^(left|right)\s+gripper\s+(open|close|closed|half|{_NUM})$", re.I)
_RE_HOME = re.compile(r"^(left|right)\s+home$", re.I)
_RE_POINT = re.compile(r"^(left|right)\s+point\s+(down|forward|down45)$", re.I)
_RE_WAIT = re.compile(r"^wait$", re.I)
_RE_DONE = re.compile(r"^done$", re.I)


def parse_command(line: str) -> Command:
    """Parse one command line. Raises CommandError on unknown syntax."""
    text = line.strip().strip("`").strip()
    text = re.sub(r"\s+", " ", text)
    # tolerate trailing comments
    text = text.split("#", 1)[0].strip().rstrip(".").strip()
    if not text:
        raise CommandError("empty command")

    m = _RE_MOVE.match(text)
    if m:
        arm, axis, value = m.group(1).lower(), m.group(2).lower(), float(m.group(3))
        clipped = abs(value) > LIMITS["move_cm"]
        value = max(-LIMITS["move_cm"], min(LIMITS["move_cm"], value))
        return Command("move", arm, axis, value, raw=line, clipped=clipped)

    m = _RE_ROT.match(text)
    if m:
        arm, axis, value = m.group(1).lower(), m.group(2).lower(), float(m.group(3))
        clipped = abs(value) > LIMITS["rotate_deg"]
        value = max(-LIMITS["rotate_deg"], min(LIMITS["rotate_deg"], value))
        return Command("rotate", arm, axis, value, raw=line, clipped=clipped)

    m = _RE_GRIP.match(text)
    if m:
        arm, state = m.group(1).lower(), m.group(2).lower()
        if state in GRIPPER_STATES:
            value = GRIPPER_STATES[state]
        else:
            value = max(0.0, min(1.0, float(state)))
        return Command("gripper", arm, None, value, raw=line)

    m = _RE_HOME.match(text)
    if m:
        return Command("home", m.group(1).lower(), raw=line)

    m = _RE_POINT.match(text)
    if m:
        return Command("point", m.group(1).lower(), m.group(2).lower(), raw=line)

    if _RE_WAIT.match(text):
        return Command("wait", raw=line)
    if _RE_DONE.match(text):
        return Command("done", raw=line)

    raise CommandError(f"unrecognised command: {line!r}")


def parse_command_list(lines) -> tuple[list[Command], list[str]]:
    """Parse a list of lines; returns (commands, errors)."""
    commands, errors = [], []
    for line in lines:
        if not isinstance(line, str):
            errors.append(f"command is not a string: {line!r}")
            continue
        try:
            commands.append(parse_command(line))
        except CommandError as exc:
            errors.append(str(exc))
    return commands, errors


GRAMMAR_HELP = """\
Command grammar (one command per string, case-insensitive):
  <arm> move <axis> <cm>        translate that gripper along a WORLD axis, keeping its orientation.
                                <arm> is left|right, <axis> is x|y|z, <cm> is a signed number, |cm| <= 20.
  <arm> rotate <axis> <deg>     rotate that gripper about a WORLD axis through its own tool point.
                                <axis> is roll (about x) | pitch (about y) | yaw (about z), |deg| <= 90.
  <arm> point down|forward|down45
                                snap the gripper to a preset orientation: fingers pointing straight down,
                                straight forward (+y), or tilted 45 deg between the two. Use "rotate yaw"
                                afterwards to spin the finger-closing axis.
  <arm> gripper open|close      open or close the fingers of that arm (also accepts a number 0..1, 1 = open).
  <arm> home                    send that arm back to its initial pose.
  wait                          let physics settle for one step without moving.
  done                          declare the task complete (or impossible) and stop.
"""

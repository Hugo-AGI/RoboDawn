"""Prompt construction for the RoboTwin discrete-control agent.

The benchmark-specific knowledge lives in a *profile* (see
``harness/configs/robotwin2_profile.yaml``).  The profile is what the
"adaptation" step of the paper produces for a new benchmark: coordinate
frame conventions, workspace, camera descriptions, gripper facts and a
handful of tips distilled from a few calibration tasks.  Nothing in this
module is task specific.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from ..core.commands import GRAMMAR_HELP

DEFAULT_PROFILE = {
    # Robot-agnostic placeholders. Every deployment overrides these in its profile YAML
    # (see harness/configs/robotwin2_profile.yaml and real_robot_profile_template.yaml).
    "robot": "a robot manipulator with a parallel-jaw gripper (describe arms, reach and gripper opening here).",
    "frame": "World frame in centimetres: state +x/+y/+z directions as seen in the overview image and the table height.",
    "workspace": "Reachable region of the fingertips and any rules for choosing an arm.",
    "cameras": "Describe every image the model receives and any markers drawn on it.",
    "gripper": (
        "The 'gripper position' reported below is the tool centre point between the fingertips. 'approach' is the "
        "unit vector the fingers point along; 'finger axis' is the direction along which the fingers open and close. "
        "Gripper opening is reported 0 (closed) .. 1 (open); after a 'close' command an opening clearly above 0.1 "
        "means an object is held."
    ),
    "tips": [
        "Plan: orient the gripper, move above the target, align x/y using the overview and then the wrist camera, "
        "descend to grasp height, close, lift, transport at a safe height, descend, open.",
        "Each command is executed by a motion planner; if a command reports FAILED the arm did NOT move.",
        "Look at the images every turn and verify the effect of your last commands before continuing.",
    ],
}


_TEMPLATE_RE = re.compile(r"\{([a-z_]+)([+-]\d+)?\}")


def render_template(text: str, context: Optional[dict]) -> str:
    """Substitute ``{name}`` / ``{name+5}`` / ``{name-2}`` placeholders with numbers from ``context``.

    Placeholders whose name is not in the context are left untouched, so a profile written with
    literal numbers keeps working and a deployment can add its own keys.
    """
    if not context:
        return text

    def sub(m):
        name, delta = m.group(1), m.group(2)
        if name not in context:
            return m.group(0)
        val = context[name] + (int(delta) if delta else 0)
        return str(int(val)) if float(val).is_integer() else f"{val:g}"

    return _TEMPLATE_RE.sub(sub, text)


def _profile_text(profile: dict, context: Optional[dict] = None) -> str:
    p = {**DEFAULT_PROFILE, **(profile or {})}
    p = {k: (render_template(v, context) if isinstance(v, str) else [render_template(t, context) for t in v])
         for k, v in p.items()}
    tips = "\n".join(f"- {t}" for t in p.get("tips", []))
    return (
        f"ROBOT: {p['robot']}\n\n"
        f"COORDINATE FRAME: {p['frame']}\n\n"
        f"WORKSPACE: {p['workspace']}\n\n"
        f"CAMERAS: {p['cameras']}\n\n"
        f"GRIPPER: {p['gripper']}\n\n"
        f"TIPS:\n{tips}"
    )


DEMO_NOTE = (
    "One or more DEMONSTRATIONS are shown before your first turn: successful episodes of the same kind of task, recorded "
    "in different scenes (other object positions, colours, lighting and table height). Each shows what the controller saw "
    "at selected turns, its state and the commands it sent. Copy its strategy (order of sub-goals, how it aligned, "
    "grasp and place heights relative to the table, when it verified with the wrist camera), NOT its numbers: read "
    "the positions for YOUR scene from your own images and state."
)


def system_prompt(profile: Optional[dict], max_commands_per_turn: int, memory_enabled: bool,
                  context: Optional[dict] = None) -> str:
    memory_field = (
        '  "memory": "<rewrite your running notes: what you have achieved, what you learned (e.g. grasp height that '
        'worked, commands that failed), and what remains; keep it under 120 words>",\n'
        if memory_enabled
        else ""
    )
    return (
        "You are the controller of a robot in a physics simulator. You receive camera images and the robot state, "
        "and you reply with a few discrete commands that are executed in order. Then you get new images.\n\n"
        + _profile_text(profile, context)
        + "\n\n"
        + GRAMMAR_HELP
        + "\n" + DEMO_NOTE + "\n"
        + "\nRESPONSE FORMAT: reply with ONE JSON object and nothing else:\n"
        "{\n"
        '  "scene": "<one or two sentences: where the relevant objects and the grippers are, in cm>",\n'
        '  "progress": "<which sub-goal you are on and whether the last commands had the intended effect>",\n'
        + memory_field
        + '  "plan": "<the next few steps in words>",\n'
        f'  "commands": ["<command>", ...]   // 1 to {max_commands_per_turn} commands, executed in order\n'
        "}\n"
        "Keep every text field short (at most ~40 words each); the whole reply must stay well under 400 words. "
        "The episode ends automatically as soon as the benchmark checker registers success, so as long as you keep "
        "receiving turns the task is NOT complete yet. Only send \"done\" if you are sure nothing more can be done."
    )


def state_text(state: dict, grasp_facts: Optional[dict] = None) -> str:
    def arm(name: str) -> str:
        a = state[name]
        return (
            f"{name.upper()} gripper position (fingertip centre) = {tuple(a['position_cm'])} cm, "
            f"approach = {tuple(a['approach'])}, finger axis = {tuple(a['finger_axis'])}, "
            f"opening = {a['gripper_real']:.2f}"
        )

    arms = [name for name in ("left", "right") if isinstance(state.get(name), dict)]
    table_z = state.get("table_z_cm")
    table_line = [f"table top at z = {table_z:g}"] if table_z is not None else []
    lines = [arm(name) for name in arms] + table_line + [f"steps used: {state['steps_used']} / {state['step_limit']}"]
    for name, f in (grasp_facts or {}).items():
        cur = state[name]["gripper_real"]
        held = "still holding it" if cur > 0.05 else "but the fingers are now fully closed, so the object was LOST"
        tz = f"{table_z:g}" if table_z is not None else "74"
        lines.append(
            f"{name.upper()} arm closed on an object with the fingertips at z = {f['grasp_z']} ({held}). "
            f"To set it down on a surface at height h, lower the fingertips to z = {f['grasp_z']} + (h - {tz}) before opening."
        )
    return "\n".join(lines)


def turn_text(
    turn: int,
    instruction: str,
    state: dict,
    last_results: list[dict],
    memory_text: str,
    image_captions: list[str],
    grasp_facts: Optional[dict] = None,
) -> str:
    parts = [f"TASK: {instruction}", f"TURN {turn}."]
    if last_results:
        lines = []
        for r in last_results:
            tag = "ok" if r.get("ok") else "FAILED"
            note = r.get("note") or ""
            lines.append(f"- {r['command']}: {tag}{(' (' + note + ')') if note else ''}")
        parts.append("RESULT OF YOUR LAST COMMANDS:\n" + "\n".join(lines))
    parts.append("CURRENT STATE:\n" + state_text(state, grasp_facts))
    if memory_text:
        parts.append(memory_text)
    parts.append("IMAGES ATTACHED (in order): " + "; ".join(image_captions))
    parts.append("Reply with the JSON object.")
    return "\n\n".join(parts)


def parse_agent_json(text: str) -> dict:
    """Extract the first JSON object from a model reply (tolerates code fences and prose)."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in reply")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    # unterminated: try to salvage the commands list
    raise ValueError("unbalanced JSON object in reply")

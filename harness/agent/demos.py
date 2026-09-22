"""In-context demonstrations shown to the controller before its first turn.

A *task demonstration* is one successful episode of the task - the RoboTwin scripted expert replayed
through the discrete commands and captioned by a VLM - rendered to the form the policy reads
(``demo.json`` plus the frames it names): the key turns (first turn, every turn with a gripper /
orientation command, last turn, the final image), each with the overview image at reduced resolution,
the proprioceptive state, the plan text, the commands sent and - object-relative - what they achieved.
A *primer* is a task-independent recorded episode showing what each command does; it is rendered from
its ``trace.json`` and per-turn images at load time. Both live in ``demos/robotwin2/``.

Both are placed in one user message (followed by a short assistant acknowledgement) right after the
system prompt, so every turn sees them in the same position; the current turn keeps its own message.
"""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PIL import Image

from .llm_client import image_part, text_part


@dataclass
class DemoFrame:
    turn: int
    label: str                       # "turn 3" or "final state"
    images: list[tuple[str, bytes]]  # (view name, png bytes)
    state_line: str
    plan: str
    commands: list[str]
    failed: list[str]
    effect: str = ""                 # relative description of what the commands achieved (see annotate)
    scene: str = ""                  # where the objects were, as the demonstration's controller read them


@dataclass
class Demo:
    task: str
    instruction: str
    source: dict
    frames: list[DemoFrame] = field(default_factory=list)
    arm_note: str = ""               # why the demonstration used the arm(s) it used
    kind: str = "task"               # "task" (an episode of the task) or "primer" (what each command does)
    grasps: dict = field(default_factory=dict)   # arm -> {"object", "x"}: what each arm closed on first, and where it was

    def side_key(self) -> str:
        """'L', 'R', 'LR' ...: the table half of every grasped object, in arm order (compare demonstrations)."""
        return "".join("L" if g["x"] < 0 else "R" for _, g in sorted(self.grasps.items()))

    def n_images(self) -> int:
        return sum(len(f.images) for f in self.frames)


# --------------------------------------------------------------------------- building
def _state_line(state: dict, table_z_default: Optional[float] = None) -> str:
    parts = []
    for arm in ("left", "right"):
        a = state.get(arm)
        if isinstance(a, dict):
            p = a.get("position_cm", [0, 0, 0])
            parts.append(f"{arm.upper()} fingertips ({p[0]:.0f}, {p[1]:.0f}, {p[2]:.0f}) opening {a.get('gripper_real', 0):.2f}")
    table_z = state.get("table_z_cm", table_z_default)
    if table_z is not None:
        parts.append(f"table z = {float(table_z):g}")
    return "; ".join(parts)


def _is_key_turn(rec: dict) -> bool:
    cmds = " ".join(rec.get("commands") or [])
    return bool(re.search(r"\bgripper\b|\bpoint\b|\bhome\b", cmds))


def select_key_turns(trace: list[dict], max_frames: int) -> list[int]:
    """Indices into ``trace`` to show: first, last, gripper/orientation turns, then evenly filled."""
    n = len(trace)
    if n <= max_frames:
        return list(range(n))
    keep = {0, n - 1}
    keep.update(i for i, r in enumerate(trace) if _is_key_turn(r))
    if len(keep) > max_frames:
        # keep the first/last and thin the rest evenly
        mid = sorted(keep - {0, n - 1})
        step = len(mid) / (max_frames - 2)
        keep = {0, n - 1} | {mid[int(i * step)] for i in range(max_frames - 2)}
    while len(keep) < max_frames:
        # add the turn that is furthest from any kept one
        kept = sorted(keep)
        gaps = [(b - a, a, b) for a, b in zip(kept, kept[1:])]
        g, a, b = max(gaps)
        if g < 2:
            break
        keep.add((a + b) // 2)
    return sorted(keep)


_SKIP_OBJ = {"wall", "table", "cluttered_obj"}


def select_demo(demos: list, state: dict):
    """The demonstration whose grasped objects lie on the same table halves as the same-named objects in
    ``state`` (the episode's first state); ties and unknown objects fall back to the first entry."""
    objs = _objects(state)

    def score(d) -> int:
        n = 0
        for g in d.grasps.values():
            pos = objs.get(g.get("object"))
            if pos is not None and (pos[0] < 0) == (g["x"] < 0):
                n += 1
        return n
    return max(demos, key=score) if demos else None


def _objects(state: dict) -> dict:
    return {k: v.get("position_cm") for k, v in (state.get("objects") or {}).items()
            if k not in _SKIP_OBJ and v.get("position_cm")}


def _load_image(episode_dir: Path, stem: str, view: str, scale: float) -> Optional[bytes]:
    for ext in ("png", "jpg"):
        p = episode_dir / f"{stem}_{view}.{ext}"
        if p.is_file():
            img = Image.open(p).convert("RGB")
            if scale != 1.0:
                img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.BICUBIC)
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            return buf.getvalue()
    return None


def build_demo(episode_dir: Path, max_frames: int = 99, views: tuple[str, ...] = ("agent_camera",),
               scale: float = 0.5) -> Demo:
    """Render a recorded episode directory (``trace.json`` + per-turn images): the primer is shipped this way."""
    episode_dir = Path(episode_dir)
    trace = json.loads((episode_dir / "trace.json").read_text())
    trace = [r for r in trace if r.get("commands")]     # drop unparseable / errored turns
    if not trace:
        raise ValueError(f"{episode_dir}: no executed turns in trace.json")
    meta = {}
    src = episode_dir / "source.json"
    if src.is_file():
        meta = json.loads(src.read_text())
    instruction = meta.get("instruction")
    if not instruction:
        m = re.match(r"TASK: (.*)", trace[0].get("prompt", ""))
        instruction = m.group(1).strip() if m else meta.get("task", episode_dir.name)
    task = meta.get("task") or episode_dir.parent.parent.name
    frames = []
    for i in select_key_turns(trace, max_frames):
        rec = trace[i]
        turn = int(rec["turn"])
        images = [(v, png) for v in views if (png := _load_image(episode_dir, f"turn{turn:03d}", v, scale))]
        failed = [r["command"] for r in rec.get("results", []) if not r.get("ok") and r.get("kind") != "done"]
        state_line = _state_line(rec.get("state", {}), meta.get("table_z_cm"))
        frames.append(DemoFrame(turn=turn, label=f"turn {turn}", images=images, state_line=state_line,
                                plan=str(rec.get("plan") or "").strip(), commands=list(rec["commands"]), failed=failed))
    final = [(v, png) for v in views if (png := _load_image(episode_dir, "final", v, scale))]
    if final:
        frames.append(DemoFrame(turn=trace[-1]["turn"] + 1, label="final state (task checker registered success)",
                                images=final, state_line="", plan="", commands=[], failed=[]))
    source = {**meta, "episode_dir": str(episode_dir), "turns": len(trace)}
    kind = "primer" if meta.get("model") == "primer" else "task"
    if kind == "primer":
        frames = [f for f in frames if not f.label.startswith("final")]
    return Demo(task=task, instruction=instruction, source=source, frames=frames, kind=kind)


# --------------------------------------------------------------------------- rendering
def demo_messages(demos: "Demo | list[Demo]") -> list[dict]:
    """Chat messages (one user message + assistant acknowledgement) carrying one or more demonstrations."""
    demos = [demos] if isinstance(demos, Demo) else list(demos)
    content = []
    tasks = [d for d in demos if d.kind != "primer"]
    for demo in demos:
        content += _demo_content(demo, tasks.index(demo) + 1 if demo in tasks else 0, len(tasks))
    content.append(text_part("--- END OF THE DEMONSTRATION" + ("S" if len(demos) > 1 else "") + ". Your own episode starts "
                             "with the next message; its scene, object positions and table height differ, so measure "
                             "everything again from your own images."))
    return [
        {"role": "user", "content": content},
        {"role": "assistant", "content": "Understood. I will follow the demonstrated strategy and read all positions from my own images and state."},
    ]


def _demo_content(demo: Demo, index: int, total: int) -> list[dict]:
    n_img = demo.n_images()
    if demo.kind == "primer":
        header = (f"COMMAND PRIMER ({len(demo.frames)} steps, {n_img} images; not a task): what each command does, shown "
                  "in a clean scene from the start pose. For every step you see the image BEFORE the commands, the state, "
                  "an explanation and the commands; the next image shows their effect. Read it once - the task "
                  "demonstration follows.")
        tag = "PRIMER "
        content = [text_part(header)]
        for f in demo.frames:
            lines = [f"--- {tag}{f.label}"]
            for _, png in f.images:
                content.append(image_part(png))
            if f.state_line:
                lines.append(f"state: {f.state_line}")
            if f.plan:
                lines.append(f"explanation: {f.plan}")
            if f.commands:
                lines.append("commands: " + json.dumps(f.commands))
            content.append(text_part("\n".join(lines)))
        return content
    label = f"DEMONSTRATION {index} of {total}" if total > 1 else "DEMONSTRATION"
    header = (f"{label} (a successful episode of a similar task in a DIFFERENT scene; {len(demo.frames)} selected "
              f"turns, {n_img} images). Its instruction was: \"{demo.instruction}\".\n"
              "For each selected turn you see the image(s) the controller received, then its state, its plan and the "
              "commands it sent (which were executed before the next shown turn). Turns in between are omitted.")
    if demo.arm_note:
        header += "\n" + demo.arm_note
    if any(f.scene for f in demo.frames):
        header += ("\nEach turn also gives the SCENE as that controller read it off its own image (object positions in "
                   "cm); read your own scene the same way instead of copying those numbers.")
    if any(f.effect for f in demo.frames):
        header += ("\nEach turn also states the NET EFFECT of its commands relative to the objects (how far the fingertips "
                   "ended up from the object). Reproduce these object-relative effects in your scene; the command numbers "
                   "themselves are specific to the demonstration's object positions.")
    content = [text_part(header)]
    tag = f"DEMO {index} " if total > 1 else "DEMO "
    for f in demo.frames:
        lines = [f"--- {tag}{f.label}" + (f" [{', '.join(v for v, _ in f.images)}]" if f.images else "")]
        for _, png in f.images:
            content.append(image_part(png))
        if f.state_line:
            lines.append(f"state: {f.state_line}")
        if f.scene:
            lines.append(f"scene: {f.scene}")
        if f.plan:
            lines.append(f"plan: {f.plan}")
        if f.commands:
            lines.append("commands: " + json.dumps(f.commands))
        if f.effect:
            lines.append("net effect: " + f.effect)
        if f.failed:
            lines.append("(FAILED, arm did not move: " + ", ".join(f.failed) + ")")
        content.append(text_part("\n".join(lines)))
    return content


def save_demo(demo: Demo, out_dir: Path, image_format: str = "png", quality: int = 85) -> None:
    """Write the frames and a JSON description of the demonstration (for inspection / the gallery).
    ``image_format='jpeg'`` stores what the policy sees at a fraction of the size, which is what makes a
    rendered bank small enough to keep under version control."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = "jpg" if image_format == "jpeg" else "png"
    frames = []
    for f in demo.frames:
        files = []
        for view, png in f.images:
            name = f"frame{f.turn:03d}_{view}.{ext}"
            if ext == "jpg":
                buf = io.BytesIO()
                Image.open(io.BytesIO(png)).convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
                png = buf.getvalue()
            (out_dir / name).write_bytes(png)
            files.append(name)
        frames.append({"turn": f.turn, "label": f.label, "images": files, "state": f.state_line, "plan": f.plan,
                       "commands": f.commands, "failed": f.failed, "effect": f.effect, "scene": f.scene})
    (out_dir / "demo.json").write_text(json.dumps({"task": demo.task, "instruction": demo.instruction, "kind": demo.kind,
                                                   "source": demo.source, "arm_note": demo.arm_note, "grasps": demo.grasps,
                                                   "frames": frames}, indent=1))


def load_saved_demo(out_dir: Path) -> Demo:
    """The inverse of :func:`save_demo`: a demonstration rendered to frames + demo.json, as the bank ships them."""
    out_dir = Path(out_dir)
    meta = json.loads((out_dir / "demo.json").read_text())
    kept = meta["frames"]
    frames = [DemoFrame(turn=int(f["turn"]), label=f["label"],
                        images=[(name.split("_", 1)[1].rsplit(".", 1)[0], (out_dir / name).read_bytes()) for name in f["images"]],
                        state_line=f["state"], plan=f["plan"], commands=list(f["commands"]), failed=list(f.get("failed") or []),
                        effect=f.get("effect", ""), scene=f.get("scene", "")) for f in kept]
    return Demo(task=meta["task"], instruction=meta["instruction"], source=dict(meta.get("source") or {}, saved_from=str(out_dir)),
                frames=frames, arm_note=meta.get("arm_note", ""), kind=meta.get("kind", "task"),
                grasps=dict(meta.get("grasps") or {}))


MAX_BANK_ENTRIES = 9


def demo_dirs_for_task(bank: Optional[Path], task: str) -> list[Path]:
    """Every bank entry of a task: ``<bank>/<task>/`` plus ``<task>+2/``, ``<task>+3/`` ..."""
    if not bank:
        return []
    dirs = [Path(bank) / (task if k == 1 else f"{task}+{k}") for k in range(1, MAX_BANK_ENTRIES + 1)]
    return [d for d in dirs if (d / "demo.json").is_file()]


def load_demos(task: str, bank: Optional[Path] = None, primer: Optional[Path] = None) -> list[Demo]:
    """The primer followed by every bank entry of the task; the agent picks among the entries per episode."""
    demos = [build_demo(Path(primer))] if primer else []
    entries = demo_dirs_for_task(bank, task)
    if not entries:
        raise FileNotFoundError(f"no demonstration for task {task} in {bank}")
    return demos + [load_saved_demo(Path(d)) for d in entries]

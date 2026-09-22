"""Memory structures for the discrete-control agent.

Three complementary mechanisms, each switchable for ablations:

* ``history``    -- a compressed, machine-written log of previous turns
                    (commands issued, outcomes, gripper position afterwards).
* ``scratchpad`` -- free text the model itself rewrites every turn ("what I
                    have done, what I learned, what is next").  This is the
                    "give the model a scratch space" idea.
* ``episode facts`` -- harness-written facts that are expensive to
                    rediscover (e.g. which commands failed to plan).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class TurnRecord:
    turn: int
    commands: list[str]
    outcomes: list[str]
    steps_after: int
    arms: dict = field(default_factory=dict)   # arm name -> {"pos": [x, y, z], "grip": float}


@dataclass
class AgentMemory:
    use_history: bool = True
    use_scratchpad: bool = True
    history_turns: int = 12
    scratchpad: str = ""
    turns: list[TurnRecord] = field(default_factory=list)
    planner_failures: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    def record_turn(self, turn: int, commands: list[str], results: list[dict], state: dict) -> None:
        outcomes = []
        for r in results:
            tag = "ok" if r.get("ok") else "FAILED"
            note = (r.get("note") or "").strip()
            if note and len(note) > 90:
                note = note[:87] + "..."
            outcomes.append(f"{tag}{(' - ' + note) if note else ''}")
            if r.get("kind") in ("move", "rotate", "point", "home") and not r.get("ok"):
                self.planner_failures.append(r["command"])
        self.turns.append(
            TurnRecord(
                turn=turn,
                commands=list(commands),
                outcomes=outcomes,
                steps_after=int(state["steps_used"]),
                arms={a: {"pos": state[a]["position_cm"], "grip": state[a]["gripper_real"]}
                      for a in ("left", "right") if isinstance(state.get(a), dict)},
            )
        )

    def update_scratchpad(self, text: str | None) -> None:
        if text is None:
            return
        text = str(text).strip()
        if text:
            self.scratchpad = text[:2500]

    # ------------------------------------------------------------------
    def render(self) -> str:
        parts = []
        if self.use_scratchpad:
            parts.append("YOUR NOTES FROM PREVIOUS TURNS (you wrote these; update them in the `memory` field):\n"
                         + (self.scratchpad if self.scratchpad else "(empty - this is the first turn)"))
        if self.use_history:
            recent = self.turns[-self.history_turns:]
            if recent:
                lines = []
                if len(self.turns) > len(recent):
                    lines.append(f"... {len(self.turns) - len(recent)} earlier turns omitted ...")
                for t in recent:
                    cmds = "; ".join(f"{c} -> {o}" for c, o in zip(t.commands, t.outcomes))
                    after = ", ".join(f"{a[0].upper()}@{_fmt(v['pos'])} grip {v['grip']:.2f}" for a, v in t.arms.items())
                    lines.append(f"turn {t.turn}: {cmds} | after: {after}, steps used {t.steps_after}")
                parts.append("HISTORY OF YOUR COMMANDS AND THEIR OUTCOMES:\n" + "\n".join(lines))
            else:
                parts.append("HISTORY OF YOUR COMMANDS AND THEIR OUTCOMES:\n(none yet)")
        return "\n\n".join(parts)

    def to_json(self) -> dict:
        return {
            "scratchpad": self.scratchpad,
            "turns": [t.__dict__ for t in self.turns],
            "planner_failures": self.planner_failures,
        }


def _fmt(p) -> str:
    return "(" + ", ".join(f"{v:.0f}" for v in p) + ")"

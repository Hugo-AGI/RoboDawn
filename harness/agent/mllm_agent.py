"""Closed-loop multimodal-LLM controller over the discrete-command environment interface.

One *turn* = one model call: the model sees the views provided by the
environment (``DiscreteEnvBase.views``), the proprioceptive state, the
outcome of its previous commands and its memory, and answers with a JSON
object holding up to ``max_commands_per_turn`` commands which are executed
sequentially.  The controller is robot-agnostic: everything robot specific
comes from the environment implementation and the profile YAML.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from ..core.commands import parse_command_list
from ..core.env import DiscreteEnvBase
from .demos import Demo, demo_messages, load_demos, save_demo, select_demo
from .llm_client import ChatClient, image_part, text_part
from .memory import AgentMemory
from .prompts import parse_agent_json, system_prompt, turn_text

logger = logging.getLogger(__name__)
DEMO_ROOT = Path(__file__).resolve().parents[2] / "demos" / "robotwin2"


@dataclass
class AgentConfig:
    model: str = "gemini-3.8-flash"
    max_turns: int = 45                  # every reported run used 45
    max_commands_per_turn: int = 4
    memory_history: bool = True
    memory_scratchpad: bool = True
    history_turns: int = 12
    # views shown every turn: the annotated overview, the top-down map and both wrist cameras
    cameras: list = field(default_factory=lambda: ["agent_camera", "top_camera", "wrist"])
    temperature: float = 0.0
    max_tokens: int = 8000                # gateways count reasoning against it; 2000 truncates the JSON when the model thinks
    timeout_s: float = 300.0
    profile: dict = field(default_factory=dict)
    consecutive_parse_failures_allowed: int = 3
    done_limit: int = 3                   # stop after this many consecutive "done" replies without success
    # In-context demonstrations (harness/agent/demos.py): the command primer followed by the task's expert
    # demonstration - of the bank's entries for the task, the one whose grasped objects lie on the same
    # table halves as in the episode.
    demo_bank: str = str(DEMO_ROOT / "expert")      # <bank>/<task>[+k]/ per task
    demo_primer: str = str(DEMO_ROOT / "primer")    # task-independent "what each command does"


@dataclass
class EpisodeResult:
    success: bool
    turns: int
    steps_used: int
    step_limit: int
    finished_reason: str
    seconds: float
    model_seconds: float
    model_calls: int
    prompt_tokens: int
    completion_tokens: int


def _where(demo: Demo) -> Optional[str]:
    """The directory a demonstration was loaded from."""
    return demo.source.get("saved_from") or demo.source.get("episode_dir")


class MLLMDiscreteAgent:
    def __init__(self, cfg: AgentConfig, client: ChatClient, out_dir: Optional[Path] = None,
                 task: Optional[str] = None) -> None:
        self.cfg = cfg
        self.client = client
        self.out_dir = Path(out_dir) if out_dir else None
        self.demos: list[Demo] = load_demos(task or "", bank=cfg.demo_bank, primer=cfg.demo_primer)
        self.primer_demos = [d for d in self.demos if d.kind == "primer"]
        self.task_demos = [d for d in self.demos if d.kind != "primer"]     # candidates; one of them is shown
        self.demo_msgs = self._messages_for(self.task_demos[:1])
        n_task = 0
        for demo in self.demos:
            logger.info("in-context %s from %s: %d frames%s", demo.kind, _where(demo), len(demo.frames),
                        f" (grasps {demo.side_key()})" if demo.grasps else "")
            if demo.kind != "primer":
                n_task += 1
            if self.out_dir:
                save_demo(demo, self.out_dir / ("demo_primer" if demo.kind == "primer" else ("demo" if n_task == 1 else f"demo{n_task}")))

    def _messages_for(self, task_demos: list[Demo]) -> list[dict]:
        return demo_messages(self.primer_demos + list(task_demos))

    def _select_demos(self, state: dict) -> list[Demo]:
        """The task demonstration to show in this episode: the entry matching the episode's grasp sides."""
        if len(self.task_demos) <= 1:
            return self.task_demos[:1]
        return [select_demo(self.task_demos, state)]

    def demo_sources(self) -> list[dict]:
        return [{"kind": d.kind, "episode_dir": _where(d), "seed": d.source.get("seed"),
                 "task_config": d.source.get("task_config")} for d in self.demos]

    # ------------------------------------------------------------------
    def _images_for_turn(self, env: DiscreteEnvBase, obs: dict) -> tuple[list[dict], list[str], dict]:
        parts, captions, saved = [], [], {}
        for view in env.views(obs, list(self.cfg.cameras)):
            png = encode_png(view.image)
            parts.append(image_part(png))
            captions.append(view.caption)
            saved[view.name] = png
        return parts, captions, saved

    # ------------------------------------------------------------------
    def run_episode(self, env: DiscreteEnvBase, instruction: str, episode_dir: Optional[Path] = None) -> EpisodeResult:
        cfg = self.cfg
        memory = AgentMemory(use_history=cfg.memory_history, use_scratchpad=cfg.memory_scratchpad,
                             history_turns=cfg.history_turns)
        sys_prompt = system_prompt(cfg.profile, cfg.max_commands_per_turn, cfg.memory_scratchpad,
                                   context=env.prompt_context())
        episode_dir = Path(episode_dir) if episode_dir else None
        if episode_dir:
            episode_dir.mkdir(parents=True, exist_ok=True)
            (episode_dir / "system_prompt.txt").write_text(sys_prompt)
        trace = []
        last_results: list[dict] = []
        grasp_facts: dict[str, dict] = {}   # harness-tracked: fingertip z at which each arm closed on an object
        t_start = time.time()
        model_seconds = 0.0
        calls = 0
        prompt_tokens = completion_tokens = 0
        parse_failures = 0
        done_count = 0
        finished_reason = "max_turns"
        turn = 0

        demo_msgs = self.demo_msgs
        for turn in range(1, cfg.max_turns + 1):
            obs = env.observe()
            state = obs["state"]
            if turn == 1 and self.task_demos:
                shown = self._select_demos(state)
                demo_msgs = self._messages_for(shown)
                logger.info("demonstration(s) for this episode: %s", [_where(d) for d in shown])
            img_parts, captions, saved = self._images_for_turn(env, obs)
            if episode_dir:
                for name, png in saved.items():
                    (episode_dir / f"turn{turn:03d}_{name}.png").write_bytes(png)
            text = turn_text(turn, instruction, state, last_results, memory.render(), captions,
                             grasp_facts=grasp_facts)
            messages = [{"role": "system", "content": sys_prompt}] + demo_msgs + [
                {"role": "user", "content": img_parts + [text_part(text)]},
            ]
            reply = self.client.chat(messages, tag=f"turn{turn}")
            calls += 1
            model_seconds += reply.latency_s
            prompt_tokens += int(reply.usage.get("prompt_tokens") or 0)
            completion_tokens += int(reply.usage.get("completion_tokens") or 0)

            record = {"turn": turn, "state": state, "prompt": text, "reply": reply.text, "latency_s": round(reply.latency_s, 2),
                      "error": reply.error, "commands": [], "results": []}
            if turn == 1 and self.task_demos:
                record["demo"] = [_where(d) for d in shown]
            if reply.error:
                record["parse_error"] = f"model call failed: {reply.error}"
                trace.append(record)
                parse_failures += 1
                if parse_failures > cfg.consecutive_parse_failures_allowed:
                    finished_reason = "model_error"
                    break
                last_results = [{"command": "(no command)", "ok": False, "note": "your previous reply could not be obtained; reply with the JSON object"}]
                continue

            try:
                parsed = parse_agent_json(reply.text)
                raw_cmds = parsed.get("commands") or []
                if isinstance(raw_cmds, str):
                    raw_cmds = [raw_cmds]
                commands, errors = parse_command_list(raw_cmds)
                parse_failures = 0
            except Exception as exc:  # noqa: BLE001
                parse_failures += 1
                record["parse_error"] = str(exc)
                trace.append(record)
                logger.warning("turn %d: could not parse reply (%s)", turn, exc)
                if parse_failures > cfg.consecutive_parse_failures_allowed:
                    finished_reason = "parse_failure"
                    break
                last_results = [{"command": "(unparseable reply)", "ok": False,
                                 "note": f"your reply was not a valid JSON object ({exc}); reply with the JSON object only"}]
                continue

            memory.update_scratchpad(parsed.get("memory"))
            record.update(scene=parsed.get("scene"), progress=parsed.get("progress"), plan=parsed.get("plan"),
                          memory=parsed.get("memory"), command_errors=errors)
            commands = commands[: cfg.max_commands_per_turn]
            results = []
            done = False
            for cmd in commands:
                if cmd.kind == "done":
                    done = True
                    results.append({
                        "command": "done", "kind": "done", "arm": None, "ok": False,
                        "note": ("NOT finished: the benchmark checker has not registered success, so the episode continues. "
                                 "Re-read the instruction; typical reasons: the object is not where/how the task wants it "
                                 "(wrong spot, not high enough, not far enough to the side, wrong orientation), a gripper "
                                 "must be opened at the end, or a second object/arm is still required. Keep acting."),
                    })
                    break
                res = env.execute(cmd)
                results.append(res)
                _update_grasp_facts(grasp_facts, cmd, res)
                if env.success or env.budget_exhausted:
                    break
            for err in errors:
                results.append({"command": "(invalid)", "kind": "invalid", "arm": None, "ok": False, "note": err})
            record["commands"] = [c.text() for c in commands]
            record["results"] = [{k: v for k, v in r.items() if k not in ("before", "after")} for r in results]
            trace.append(record)
            memory.record_turn(turn, record["commands"], results, env.state())
            last_results = results

            if env.success:
                finished_reason = "success"
                break
            if done:
                done_count += 1
                if done_count >= cfg.done_limit:
                    finished_reason = "agent_done"
                    break
            else:
                done_count = 0
            if env.budget_exhausted:
                finished_reason = "step_budget"
                break
            stop = env.should_stop()
            if stop:
                # harness-side early stop (bookkeeping only; never shown to the model)
                finished_reason = stop
                break

        if episode_dir:
            # final observation (after the last command)
            try:
                for view in env.views(env.observe(), list(cfg.cameras)):
                    (episode_dir / f"final_{view.name}.png").write_bytes(encode_png(view.image))
            except Exception as exc:  # noqa: BLE001
                logger.warning("could not save the final observation: %s", exc)
        result = EpisodeResult(
            success=env.success, turns=turn, steps_used=env.steps_used, step_limit=env.step_limit,
            finished_reason=finished_reason, seconds=round(time.time() - t_start, 1), model_seconds=round(model_seconds, 1),
            model_calls=calls, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        )
        if episode_dir:
            (episode_dir / "trace.json").write_text(json.dumps(trace, indent=1, default=_json_default))
            (episode_dir / "memory.json").write_text(json.dumps(memory.to_json(), indent=1, default=_json_default))
        return result


def _update_grasp_facts(facts: dict, cmd, res: dict) -> None:
    """Remember at which fingertip height an arm closed on something (and forget it when it opens)."""
    if cmd.kind != "gripper" or not res.get("ok"):
        return
    arm = cmd.arm
    after = (res.get("after") or {}).get(arm) or {}
    opening = after.get("gripper_real", 0.0)
    if cmd.value < 0.5 and opening > 0.08:
        facts[arm] = {"grasp_z": round(float(after["position_cm"][2]), 1), "opening": round(float(opening), 2)}
    elif cmd.value >= 0.5:
        facts.pop(arm, None)


def encode_png(img) -> bytes:
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)

"""Evaluate the discrete-command MLLM agent on RoboTwin 2.0 tasks.

Example::

    CUDA_VISIBLE_DEVICES=0 python harness/run_robotwin_eval.py \
        --task place_empty_cup --episodes 10 --model gemini-3.8-flash \
        --api_base https://<your-endpoint>/v1 --api_key_file ~/.config/robodawn/key \
        --output results/rt2/gemini-3.8-flash/place_empty_cup/shard_0

The defaults are the configuration the reported results were produced with:
``demo_randomized`` scenes, 45 turns, the task's expert demonstration (chosen by
grasp side) preceded by the command primer, model reasoning on. Every episode
writes ``episode_<k>/`` with trace.json, memory.json, images per turn and an
mp4; the run directory gets results.json with the aggregate.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import logging
import os
import sys
import time

faulthandler.enable()
from dataclasses import asdict
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
# harness.robotwin.env chdirs into the RoboTwin root at import time (its assets use relative
# paths), so remember where the user launched us to resolve relative --output paths.
LAUNCH_CWD = Path.cwd()

from harness.core import watchdog  # noqa: E402
from harness.agent.llm_client import ChatClient  # noqa: E402
from harness.agent.mllm_agent import AgentConfig, MLLMDiscreteAgent  # noqa: E402
from harness.robotwin.env import RoboTwinDiscreteEnv  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True)
    p.add_argument("--task_config", default="demo_randomized", help="RoboTwin scene config; every reported run used demo_randomized")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--start_episode", type=int, default=0, help="skip this many valid episodes first (for sharding)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--instruction_type", default="unseen", choices=["seen", "unseen"])
    p.add_argument("--model", default="gemini-3.8-flash")
    p.add_argument("--agent_config", default=None, help="YAML with AgentConfig fields (overrides defaults)")
    p.add_argument("--profile", default=str(REPO / "harness" / "configs" / "robotwin2_profile.yaml"))
    p.add_argument("--output", required=True)
    p.add_argument("--max_turns", type=int, default=None)
    p.add_argument("--max_commands_per_turn", type=int, default=None)
    p.add_argument("--no_video", action="store_true")
    p.add_argument("--reasoning_effort", default=None,
                   help="override the effort this model family sends by default (llm_client.THINKING_EXTRA); "
                        "the endpoint decides which values it accepts")
    p.add_argument("--api_base", default=None, help="OpenAI-compatible base URL (or $LLM_API_BASE)")
    p.add_argument("--api_key_file", default=None,
                   help="file holding the API key for this run (default: $LLM_API_KEY or ~/.config/robodawn/key*)")
    p.add_argument("--inline_system", action="store_true",
                   help="send the system prompt as the first text of the first user message (some gateways switch "
                        "Gemini's thinking off when a request carries a system message)")
    p.add_argument("--rpm", type=float, default=None, help="client-side requests-per-minute limit for this process")
    p.add_argument("--max_tokens", type=int, default=None,
                   help="reply budget per turn (default 8000); gateways count reasoning tokens against it, so a heavy "
                        "thinker needs more")
    p.add_argument("--timeout_s", type=float, default=None, help="per-request timeout (default 300)")
    p.add_argument("--label", default="", help="free-form label stored in results.json")
    p.add_argument("--cameras", default=None,
                   help="comma-separated views, e.g. agent_camera,top_camera,wrist (default) or head_camera,wrist")
    # demonstrations: the defaults (AgentConfig) are the reported configuration,
    # the task's expert demonstration preceded by the command primer
    p.add_argument("--demo_bank", default=None, help="bank with <task>[+k]/ entries (default: demos/robotwin2/expert)")
    p.add_argument("--demo_primer", default=None, help="primer directory (default: demos/robotwin2/primer)")
    p.add_argument("--stall_timeout", type=float, default=900.0,
                   help="abort the process when nothing progresses for this many seconds (0 disables)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    watchdog.start(stall_seconds=args.stall_timeout)
    out = Path(args.output)
    if not out.is_absolute():
        out = LAUNCH_CWD / out
    out.mkdir(parents=True, exist_ok=True)

    cfg_dict = {}
    if args.agent_config:
        cfg_dict.update(yaml.safe_load(_resolve(args.agent_config).read_text()) or {})
    profile = {}
    if args.profile and _resolve(args.profile).is_file():
        profile = yaml.safe_load(_resolve(args.profile).read_text()) or {}
    cfg_dict["profile"] = profile
    cfg_dict["model"] = args.model
    if args.max_turns is not None:
        cfg_dict["max_turns"] = args.max_turns
    if args.max_commands_per_turn is not None:
        cfg_dict["max_commands_per_turn"] = args.max_commands_per_turn
    if args.max_tokens is not None:
        cfg_dict["max_tokens"] = args.max_tokens
    if args.timeout_s is not None:
        cfg_dict["timeout_s"] = args.timeout_s
    if args.cameras:
        cfg_dict["cameras"] = [c.strip() for c in args.cameras.split(",") if c.strip()]
    if args.demo_bank:
        cfg_dict["demo_bank"] = str(_resolve(args.demo_bank))
    if args.demo_primer:
        cfg_dict["demo_primer"] = str(_resolve(args.demo_primer))
    cfg = AgentConfig(**cfg_dict)
    (out / "agent_config.json").write_text(json.dumps(asdict(cfg), indent=1))

    client = ChatClient(
        model=cfg.model, timeout_s=cfg.timeout_s, temperature=cfg.temperature, max_tokens=cfg.max_tokens,
        log_path=out / "llm_calls.jsonl", base_url=args.api_base,
        extra_body={"reasoning_effort": args.reasoning_effort} if args.reasoning_effort else None,
        api_key=_pick_key(_resolve(args.api_key_file)) if args.api_key_file else None, rpm=args.rpm,
        inline_system=args.inline_system,
    )
    agent = MLLMDiscreteAgent(cfg, client, out, task=args.task)

    env = RoboTwinDiscreteEnv(args.task, task_config=args.task_config, seed=args.seed,
                              instruction_type=args.instruction_type, record_video=not args.no_video)

    results_path = out / "results.json"
    episodes = []
    if results_path.is_file():
        try:
            episodes = json.loads(results_path.read_text()).get("episodes", [])
        except Exception:  # noqa: BLE001
            episodes = []
    done_idx = {e["episode_index"] for e in episodes}

    t_run = time.time()
    for k in range(args.start_episode, args.start_episode + args.episodes):
        if k in done_idx:
            continue
        # advance through valid seeds without running the policy for skipped shards
        watchdog.beat(f"reset_episode {k}")
        ep = env.reset_episode(k)
        ep_dir = out / f"episode_{k:03d}"
        logging.info("episode %d seed %d: %s", k, ep.seed, ep.instruction)
        res = agent.run_episode(env, ep.instruction, ep_dir)
        video = env.save_video(ep_dir / f"episode_{k:03d}_{'success' if res.success else 'failure'}.mp4") if not args.no_video else None
        rec = {"episode_index": k, "seed": ep.seed, "instruction": ep.instruction, **asdict(res), "video": video}
        episodes.append(rec)
        env.close_episode()
        summary = _summary(args, cfg, episodes, time.time() - t_run, client.total_usage, agent.demo_sources())
        results_path.write_text(json.dumps(summary, indent=1))
        logging.info("episode %d -> success=%s turns=%d steps=%d reason=%s (%.0fs, model %.0fs)",
                     k, res.success, res.turns, res.steps_used, res.finished_reason, res.seconds, res.model_seconds)
    env.close()
    summary = _summary(args, cfg, episodes, time.time() - t_run, client.total_usage, agent.demo_sources())
    results_path.write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}, indent=1))


def _pick_key(path: Path) -> str:
    """One key from a key file; several keys (one per line) are spread over processes by pid."""
    keys = [l.strip() for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")]
    return keys[os.getpid() % len(keys)]


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else LAUNCH_CWD / p


def _summary(args, cfg, episodes, seconds, usage, demo_sources=()):
    n = len(episodes)
    succ = sum(1 for e in episodes if e["success"])
    return {
        "task": args.task,
        "task_config": args.task_config,
        "model": cfg.model,
        "label": args.label,
        "reasoning_effort": args.reasoning_effort,
        "episodes_done": n,
        "successes": succ,
        "success_rate": round(succ / n, 4) if n else None,
        "mean_turns": round(sum(e["turns"] for e in episodes) / n, 1) if n else None,
        "mean_steps": round(sum(e["steps_used"] for e in episodes) / n, 1) if n else None,
        "finished_reasons": {r: sum(1 for e in episodes if e["finished_reason"] == r) for r in sorted({e["finished_reason"] for e in episodes})},
        "wall_seconds": round(seconds, 1),
        "llm_usage": usage,
        "agent_config": asdict(cfg),
        "demos": demo_sources,
        "episodes": episodes,
    }


if __name__ == "__main__":
    main()

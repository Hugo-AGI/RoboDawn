#!/usr/bin/env python3
"""Run one RoboDojo task with the VLM policy, or summarize an experiment directory.

    python scripts/robodojo/run_vlm_experiment.py run --experiment-dir experiments/paper \
        --run-id gpt6_stack_bowls --model gpt-6-astra --task stack_bowls --gpu 0
    python scripts/robodojo/run_vlm_experiment.py summary --experiment-dir experiments/paper

Every run gets a directory with the launch manifest, the launcher log, the
per-turn decision logs and a summary; ``summary`` folds all runs of an
experiment into ``results.csv``. The defaults are the settings the reported
results were produced with.
"""

from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
ROBODOJO_ROOT = REPO_ROOT / "RoboDojo"
POLICY_ROOT = REPO_ROOT / "evaluation/policies/vlm_agent"
sys.path.insert(0, str(POLICY_ROOT.parent))
from vlm_agent.icl_demos import resolve_icl_path
from vlm_agent.vlm_client import load_vlm_profile

# The expert demonstrations the reported one-shot results were measured with.
DEFAULT_DEMO_BANK = REPO_ROOT / "demos/robodojo"


@contextmanager
def termination_signals():
    request = {"signum": None}

    def handle(signum, _frame):
        # Do not raise between Popen creating a child and assigning its handle.
        if request["signum"] is None:
            request["signum"] = signum

    previous = {sig: signal.signal(sig, handle) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        yield request
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def process_identity(pid: int) -> dict | None:
    pid = int(pid)
    if pid <= 0:
        raise ValueError("process ID must be positive")
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    fields = stat[stat.rindex(")") + 2:].split()
    if fields[0] in ("Z", "X"):
        return None
    return {"pid": pid, "start_ticks": int(fields[19]),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


def run_activity(manifest: dict) -> str:
    """Check local ownership; a remote or unidentifiable live group is unknown."""
    if manifest.get("state") not in ("running", "prepared"):
        return "inactive"
    if manifest.get("hostname") != socket.gethostname():
        return "unknown"
    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        owner = manifest.get("runner_identity")
        child = manifest.get("process_identity")
        if owner and owner["boot_id"] != boot_id:
            return "inactive"
        if owner and process_identity(owner["pid"]) == owner:
            return "active"
        if child:
            if child["boot_id"] != boot_id:
                return "inactive"
            actual = process_identity(child["pid"])
            if actual == child:
                return "active"
            if actual is not None:
                # The leader PID cannot be reused while its original group exists.
                return "inactive"
        group = manifest.get("process_group")
        if group is None or int(group) <= 0:
            return "unknown"
        try:
            os.killpg(int(group), 0)
        except ProcessLookupError:
            return "inactive"
        unreadable = False
        for path in Path("/proc").iterdir():
            if not path.name.isdecimal():
                continue
            try:
                stat = (path / "stat").read_text()
                fields = stat[stat.rindex(")") + 2:].split()
                if int(fields[2]) == int(group) and fields[0] not in ("Z", "X"):
                    # Never signal an orphaned or unidentifiable group from a scan.
                    return "unknown"
            except (FileNotFoundError, ProcessLookupError):
                continue
            except OSError:
                unreadable = True
        return "unknown" if unreadable else "inactive"
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return "unknown"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def gpu_info(gpu: str) -> dict:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--id=" + gpu,
             "--query-gpu=index,name,uuid,driver_version,memory.total", "--format=csv,noheader,nounits"],
            text=True, capture_output=True, timeout=10, check=True,
        )
        return {"requested": gpu, "query": result.stdout.strip()}
    except (OSError, subprocess.SubprocessError):
        return {"requested": gpu, "query": None}


def stop_process_group(process: subprocess.Popen) -> None:
    """Only the session created for this run is eligible for termination."""
    identity = getattr(process, "_vlm_identity", None)
    if isinstance(identity, dict):
        actual = process_identity(process.pid)
        if actual is not None and actual != identity:
            return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait(timeout=10)
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


def wait_for_run(process, log_path: Path, timeout_s: float, shutdown: dict | None = None) -> str | None:
    """End known native crashes immediately instead of waiting for crash dialogs."""
    deadline = time.monotonic() + timeout_s
    offset = 0
    tail = ""
    fatal_messages = ("GPU crash is detected", "[crash] Wrote dump file", "Segmentation fault (core dumped)")
    while process.poll() is None:
        if shutdown and shutdown["signum"] is not None:
            stop_process_group(process)
            return "interrupted"
        with log_path.open(encoding="utf-8", errors="replace") as log:
            log.seek(offset)
            tail = tail[-1024:] + log.read()
            offset = log.tell()
        if any(message in tail for message in fatal_messages):
            stop_process_group(process)
            return "renderer_crash"
        if time.monotonic() >= deadline:
            stop_process_group(process)
            return "wall_timeout"
        try:
            process.wait(timeout=min(1.0, max(0.001, deadline - time.monotonic())))
        except subprocess.TimeoutExpired:
            pass
    return None


def result_paths(manifest: dict) -> list[Path]:
    root = ROBODOJO_ROOT / "eval_result/RoboDojo" / manifest["task"] / "vlm_agent/arx_x5"
    prefix = f"{manifest['seed']}_ckpt_name={manifest['run_id']},"
    return sorted(
        path for path in root.glob("*/**/_result.json")
        if path.relative_to(root).parts[0].startswith(prefix)
        and path.parent.name == manifest["run_id"]
    )


def model_error_kind(record: dict) -> str | None:
    error = record.get("error")
    if not error:
        return None
    text = str(error).lower()
    if any(token in text for token in ("no json object", "no action object", "invalid action", "parse")):
        return "parse"
    if any(token in text for token in ("vlm request failed", "http ", "timeout", "timed out", "empty completion")):
        return "api"
    return "other"


def summarize_run(run_dir: Path) -> dict:
    manifest = read_json(run_dir / "manifest.json")
    found = result_paths(manifest)
    result = read_json(found[0]) if len(found) == 1 else {}
    details = result.get("details") or {}
    episodes = [detail for detail in details.values() if isinstance(detail, dict)]
    layout_ids = [int(detail["layout_id"]) for detail in episodes if "layout_id" in detail]
    records = []
    log_errors = 0
    reasons = []
    model_error_abort_episodes = 0
    for path in sorted((run_dir / "decisions").rglob("*.json")):
        try:
            record = read_json(path)
        except (ValueError, OSError):
            log_errors += 1
            continue
        reason = record.get("policy_stop_reason") or record.get("termination_reason") or record.get("stop_reason")
        if reason:
            reasons.append(str(reason))
        if path.name == "episode_summary.json" and reason == "consecutive_model_errors":
            model_error_abort_episodes += 1
        if path.name.startswith("decision_"):
            records.append(record)
    requests = [request for record in records for request in record.get("requests", [record])]
    timed_requests = [record for record in requests if isinstance(record.get("latency_s"), (int, float))]
    latencies = [float(record["latency_s"]) for record in timed_requests if model_error_kind(record) != "api"]
    all_latencies = [float(record["latency_s"]) for record in timed_requests]
    tokens = {key: sum(int((record.get("usage") or {}).get(key, 0) or 0) for record in records)
              for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
    execution = {key: sum(int((record.get("execution") or {}).get(key, 0) or 0) for record in records)
                 for key in ("requested", "executed", "ik_failed")}
    error_count = sum(bool(record.get("error")) for record in records)
    error_kinds = {kind: sum(model_error_kind(record) == kind for record in records)
                   for kind in ("api", "parse", "other")}
    model_error_aborted = "consecutive_model_errors" in reasons
    api_aborted = any(any(token in reason.lower() for token in ("api", "consecutive_errors"))
                      for reason in reasons)
    requested = int(manifest["eval_num"])
    completed = int(result.get("eval_time", len(episodes)) or 0)
    exit_code = manifest.get("exit_code")
    if manifest.get("timed_out"):
        status = "wall_timeout"
    elif len(found) > 1:
        status = "ambiguous_results"
    elif manifest.get("state") == "running":
        status = "running"
    elif model_error_aborted:
        status = "model_error_abort"
    elif api_aborted:
        status = "api_aborted"
    elif exit_code not in (None, 0):
        status = "infrastructure_failure"
    elif completed < requested:
        status = "incomplete"
    elif error_count:
        status = "completed_with_model_errors"
    else:
        status = "completed"
    summary = {
        "run_id": manifest["run_id"], "model": manifest["model"],
        "profile": manifest["profile"],
        "demo_bank": (manifest.get("overrides") or {}).get("icl", {}).get("demo_bank"),
        "task": manifest["task"], "seed": manifest["seed"], "hostname": manifest["hostname"],
        "status": status, "exit_code": exit_code, "requested_episodes": requested,
        "completed_episodes": completed, "successes": sum(bool(item.get("success")) for item in episodes),
        "success_rate": result.get("success_rate") if completed else None,
        "score": result.get("score") if completed else None,
        "wall_time_s": manifest.get("wall_time_s"), "decision_count": len(records),
        "api_call_count": sum(record.get("api_calls", record.get("attempts", 1)) for record in records),
        "retry_count": sum(max(0, request.get("attempts", 1) - 1) for request in requests),
        "transport_error_attempt_count": sum(len(request.get("retry_history", [])) for request in requests),
        "rate_limit_count": sum(failure.get("status_code") == 429 for request in requests
                                for failure in request.get("retry_history", [])),
        "recovered_response_count": sum(bool(record.get("parse", {}).get("recovered")) for record in records),
        "decision_error_count": error_count, "unreadable_log_count": log_errors,
        "decision_api_error_count": error_kinds["api"],
        "decision_parse_error_count": error_kinds["parse"],
        "decision_other_error_count": error_kinds["other"],
        "model_error_abort_episode_count": model_error_abort_episodes,
        "mean_latency_s": statistics.mean(latencies) if latencies else None,
        "p50_latency_s": statistics.median(latencies) if latencies else None,
        "mean_request_latency_s": statistics.mean(all_latencies) if all_latencies else None,
        "response_latency_count": len(latencies),
        "failed_request_latency_count": len(all_latencies) - len(latencies),
        "unknown_request_latency_count": len(requests) - len(timed_requests),
        "failure_reason": manifest.get("failure_reason"),
        "usage": tokens, "execution": execution, "termination_reasons": reasons,
        "result_paths": [str(path) for path in found], "details": details, "layout_ids": layout_ids,
    }
    if manifest.get("state") != "running":
        manifest["result_paths"] = summary["result_paths"]
        write_json(run_dir / "manifest.json", manifest)
    if result:
        write_json(run_dir / "result.json", result)
    write_json(run_dir / "summary.json", summary)
    return summary


def run(args: argparse.Namespace) -> int:
    with termination_signals() as shutdown:
        return _run(args, shutdown)


def _run(args: argparse.Namespace, shutdown: dict) -> int:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", args.run_id):
        raise ValueError("run-id must contain only letters, digits, dot, underscore, hyphen")
    if not re.fullmatch(r"[A-Za-z0-9_]+", args.task):
        raise ValueError("invalid task name")
    # A namespace built by hand (the lifecycle tests) may omit the timeout knobs.
    request_timeout = float(getattr(args, "request_timeout", 60))
    decision_budget = float(getattr(args, "decision_budget", 660))
    max_attempts = int(getattr(args, "max_attempts", 6))
    if min(args.eval_num, args.max_decisions, args.max_tokens, args.wall_timeout, args.episode_budget,
           request_timeout, decision_budget, max_attempts) <= 0 or args.seed < 0:
        raise ValueError("budgets and eval-num must be positive; seed must be nonnegative")
    profile = load_vlm_profile(args.model)
    run_dir = args.experiment_dir.resolve() / "runs" / args.run_id
    bank = resolve_icl_path(getattr(args, "demo_bank", None) or DEFAULT_DEMO_BANK)
    if not bank.is_dir():
        raise FileNotFoundError(f"demonstration bank not found: {bank}")
    overrides = {
        "log_dir": str(run_dir / "decisions"),
        "max_decisions": args.max_decisions, "max_zero_progress": 5,
        "max_consecutive_errors": 2, "max_tokens": args.max_tokens, "max_attempts": max_attempts,
        "reasoning_effort": getattr(args, "reasoning_effort", "high"),
        "request_timeout_s": request_timeout, "decision_budget_s": decision_budget,
        "episode_budget_s": args.episode_budget,
        "cameras": ["cam_head", "cam_left_wrist", "cam_right_wrist"],
        "visual_aids": {"markers": True, "grid": True, "enhance": True},
        "measure_gripper": True,
        "icl": {"demo_bank": str(bank), "demo_count": int(getattr(args, "demo_count", 1) or 1)},
    }
    manifest = {
        "run_id": args.run_id, "created_at": now(), "state": "prepared",
        "profile": profile["name"], "model": profile["model"],
        "transport": {"base_url": profile["base_url"]},
        "task": args.task, "seed": args.seed, "eval_num": args.eval_num,
        "hostname": socket.gethostname(), "gpu": gpu_info(args.gpu),
        "runner_identity": process_identity(os.getpid()),
        "robot": "arx_x5", "overrides": overrides,
        # The label RoboDojo names the result directory after: planned trajectories are joint targets.
        "action_type": "joint",
        "wall_timeout_s": args.wall_timeout,
        "policy_env": args.policy_env, "eval_env": args.eval_env,
    }
    if result_paths(manifest):
        raise FileExistsError("result files already exist for this run-id")
    layout_dir = ROBODOJO_ROOT / "Assets/Eval_Layout/RoboDojo/arx_x5" / str(args.seed)
    if not all((layout_dir / f"{args.task}_{index}.json").is_file() for index in range(args.eval_num)):
        raise FileNotFoundError("one or more requested fixed layouts are missing")
    if not args.conda_sh.is_file():
        raise FileNotFoundError("Conda initialization script does not exist")
    run_dir.mkdir(parents=True, exist_ok=False)
    command = ["bash", str(REPO_ROOT / "scripts/robodojo/run_vlm_eval.sh"),
               "--model", profile["name"], "--task", args.task, "--ckpt", args.run_id,
               "--seed", str(args.seed), "--eval-num", str(args.eval_num),
               "--env-gpu", args.gpu, "--policy-gpu", args.gpu,
               "--policy-env", args.policy_env, "--eval-env", args.eval_env]
    manifest["command"] = command
    environment = os.environ.copy()
    environment.update({
        "VLM_AGENT_OVERRIDES": json.dumps(overrides), "VLM_AGENT_RUN_TAG": args.run_id,
        "ROBODOJO_RUN_ID": args.run_id, "ROBODOJO_MAX_BASH_RETRIES": "1",
        "PYTHONUNBUFFERED": "1",
    })
    shell = f"source {shlex.quote(str(args.conda_sh))} && exec {shlex.join(command)}"
    started = time.monotonic()
    manifest.update(state="running", started_at=now())
    write_json(run_dir / "manifest.json", manifest)
    process = None
    try:
        with (run_dir / "launcher.log").open("x", encoding="utf-8") as log:
            if shutdown["signum"] is None:
                process = subprocess.Popen(
                    ["bash", "-lc", shell], cwd=REPO_ROOT, env=environment,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                )
                manifest["process_group"] = process.pid
                manifest["process_identity"] = process_identity(process.pid)
                process._vlm_identity = manifest["process_identity"]
                write_json(run_dir / "manifest.json", manifest)
                manifest["failure_reason"] = wait_for_run(process, run_dir / "launcher.log", args.wall_timeout, shutdown)
                manifest["timed_out"] = manifest["failure_reason"] == "wall_timeout"
                manifest["exit_code"] = process.returncode
    except (KeyboardInterrupt, OSError):
        if process is not None:
            stop_process_group(process)
        manifest["exit_code"] = process.returncode if process is not None else -1
        manifest["interrupted"] = True
    finally:
        if process is not None and (shutdown["signum"] is not None or manifest.get("interrupted") or process.poll() is None):
            stop_process_group(process)
        if shutdown["signum"] is not None:
            manifest.update(interrupted=True, failure_reason="interrupted",
                            termination_signal=signal.Signals(shutdown["signum"]).name,
                            exit_code=128 + shutdown["signum"])
        manifest.update(state="finished", finished_at=now(), wall_time_s=round(time.monotonic() - started, 3))
        write_json(run_dir / "manifest.json", manifest)
    summary = summarize_run(run_dir)
    print(json.dumps(summary, ensure_ascii=True), flush=True)
    if shutdown["signum"] is not None:
        return 128 + shutdown["signum"]
    return 0 if summary["status"] in ("completed", "completed_with_model_errors") else 1


def aggregate(experiment_dir: Path) -> int:
    summaries = [summarize_run(path.parent) for path in sorted((experiment_dir / "runs").glob("*/manifest.json"))]
    fields = ["run_id", "model", "profile", "demo_bank", "task", "seed", "hostname", "status",
              "requested_episodes", "completed_episodes", "successes", "success_rate", "score",
              "wall_time_s", "decision_count", "decision_error_count", "mean_latency_s", "p50_latency_s",
              "decision_api_error_count", "decision_parse_error_count", "decision_other_error_count",
              "model_error_abort_episode_count",
              "api_call_count", "retry_count", "transport_error_attempt_count", "rate_limit_count",
              "recovered_response_count",
              "mean_request_latency_s", "response_latency_count", "failed_request_latency_count",
              "unknown_request_latency_count", "failure_reason",
              "total_tokens", "ik_failed", "layout_ids"]
    rows = []
    for summary in summaries:
        row = {key: summary.get(key) for key in fields}
        row.update(total_tokens=summary["usage"]["total_tokens"], ik_failed=summary["execution"]["ik_failed"],
                   layout_ids=",".join(str(layout_id) for layout_id in summary["layout_ids"]))
        rows.append(row)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    write_json(experiment_dir / "summary.json", {"generated_at": now(), "runs": summaries})
    with (experiment_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"runs": len(summaries), "summary": str(experiment_dir / "summary.json")}), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    launch = commands.add_parser("run")
    launch.add_argument("--experiment-dir", type=Path, required=True)
    launch.add_argument("--run-id", required=True)
    launch.add_argument("--model", required=True, help="profile name or model ID from secrets.json")
    launch.add_argument("--demo-bank", type=Path, default=DEFAULT_DEMO_BANK,
                        help="demonstration bank with one <task>/ entry per task (default: demos/robodojo)")
    launch.add_argument("--demo-count", type=int, default=1, help="demonstrations per task from the bank")
    launch.add_argument("--task", required=True)
    launch.add_argument("--seed", type=int, default=0)
    launch.add_argument("--eval-num", type=int, default=5, help="fixed layouts 0..N-1 of the seed, one episode each")
    launch.add_argument("--gpu", default="0")
    launch.add_argument("--max-decisions", type=int, default=240, help="model turns per episode")
    launch.add_argument("--max-tokens", type=int, default=8000)
    launch.add_argument("--reasoning-effort", choices=("low", "medium", "high"), default="high")
    # Per-request timeout and the hard wall for one turn. The simulation
    # deploy extends the simulation RPC timeout to cover the retry budget.
    launch.add_argument("--request-timeout", type=float, default=600)
    launch.add_argument("--decision-budget", type=float, default=660)
    launch.add_argument("--max-attempts", type=int, default=6, help="initial request plus five retries by default")
    launch.add_argument("--wall-timeout", type=float, default=30000, help="seconds before the whole run is killed")
    launch.add_argument("--episode-budget", type=float, default=9000, help="seconds of wall clock per episode")
    launch.add_argument("--policy-env", default="vlm_policy")
    launch.add_argument("--eval-env", default="RoboDojo")
    launch.add_argument("--conda-sh", type=Path, default=Path.home() / "miniconda3/etc/profile.d/conda.sh")
    summary = commands.add_parser("summary", aliases=["aggregate"])
    summary.add_argument("--experiment-dir", type=Path, required=True)
    args = parser.parse_args()
    return run(args) if args.command == "run" else aggregate(args.experiment_dir.resolve())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        print(f"experiment error: {exc}", file=sys.stderr)
        raise SystemExit(2)

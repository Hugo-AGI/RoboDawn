"""Aggregate results.json files under a run directory into one table.

    python harness/scripts/aggregate_results.py results/rt2_v1 [--csv out.csv]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    rows = defaultdict(lambda: {"n": 0, "succ": 0, "turns": 0.0, "steps": 0.0, "model_s": 0.0, "prompt_tok": 0, "compl_tok": 0})
    for f in sorted(Path(args.run_dir).rglob("results.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        key = (d.get("model"), d.get("task"), d.get("task_config"))
        r = rows[key]
        for e in d.get("episodes", []):
            r["n"] += 1
            r["succ"] += int(bool(e.get("success")))
            r["turns"] += e.get("turns", 0)
            r["steps"] += e.get("steps_used", 0)
            r["model_s"] += e.get("model_seconds", 0)
            r["prompt_tok"] += e.get("prompt_tokens", 0)
            r["compl_tok"] += e.get("completion_tokens", 0)
    header = f"{'model':22s} {'task':28s} {'config':16s} {'n':>4s} {'succ':>5s} {'rate':>6s} {'turns':>6s} {'steps':>6s} {'llm_s/ep':>8s} {'ptok/ep':>8s}"
    print(header)
    lines = []
    per_model = defaultdict(lambda: [0, 0])
    for (model, task, cfg), r in sorted(rows.items()):
        n = r["n"] or 1
        rate = r["succ"] / n
        per_model[model][0] += r["succ"]
        per_model[model][1] += r["n"]
        print(f"{str(model):22s} {str(task):28s} {str(cfg):16s} {r['n']:4d} {r['succ']:5d} {rate:6.2f} {r['turns']/n:6.1f} {r['steps']/n:6.1f} {r['model_s']/n:8.0f} {r['prompt_tok']/n:8.0f}")
        lines.append([model, task, cfg, r["n"], r["succ"], round(rate, 4), round(r["turns"] / n, 1), round(r["steps"] / n, 1)])
    print("-" * len(header))
    for model, (s, n) in per_model.items():
        print(f"{str(model):22s} overall: {s}/{n} = {s / max(n, 1):.3f}")
    if args.csv:
        import csv

        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["model", "task", "task_config", "episodes", "successes", "success_rate", "mean_turns", "mean_steps"])
            w.writerows(lines)


if __name__ == "__main__":
    main()

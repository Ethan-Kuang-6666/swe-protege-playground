#!/usr/bin/env python3
"""Compare two mini-swe-agent batch runs instance by instance.

Reads the raw trajectories from both run directories, so it can clamp OpenAI's `-9999.0` sentinel
logprob (returned when a token's probability underflows -- a placeholder, not a measurement, that
otherwise dominates any mean it lands in).

    python compare_runs.py results/run1 results/run2-gpt4o
    python compare_runs.py results/run1 results/run2-gpt4o --instance astropy__astropy-13453
"""

import argparse
import json
from pathlib import Path

SENTINEL = -9000.0
"""Anything at or below this is a placeholder rather than a real log-probability."""


def get_steps(traj: Path, floor: float) -> list[dict]:
    """One record per model call: what it ran, and its sampled-token confidence."""
    messages = json.loads(traj.read_text())["messages"]
    steps = []
    for index, message in enumerate(messages):
        response = message.get("extra", {}).get("response")
        if not isinstance(response, dict):
            continue
        entries = ((response.get("choices") or [{}])[0].get("logprobs") or {}).get("content") or []
        if not entries:
            continue
        raw = [entry["logprob"] for entry in entries]
        logprobs = [max(x, floor) for x in raw]
        following = messages[index + 1].get("extra", {}) if index + 1 < len(messages) else {}
        steps.append(
            {
                "commands": [a["command"] for a in message.get("extra", {}).get("actions", [])],
                "returncode": following.get("returncode"),
                "n_tokens": len(logprobs),
                "avg_logp": sum(logprobs) / len(logprobs),
                "min_logp": min(logprobs),
                "n_clamped": sum(x <= SENTINEL for x in raw),
            }
        )
    return steps


def load_run(run_dir: Path, floor: float) -> dict[str, dict]:
    """Instance id -> summary, for every trajectory in a batch run directory."""
    runs = {}
    for traj in sorted(run_dir.glob("*/*.traj.json")):
        info = json.loads(traj.read_text())["info"]
        steps = get_steps(traj, floor)
        if not steps:
            continue
        runs[traj.parent.name] = {
            "steps": steps,
            "exit": info.get("exit_status", "?"),
            "patch": len(info.get("submission") or ""),
            "avg_logp": sum(s["avg_logp"] for s in steps) / len(steps),
            "min_logp": min(s["min_logp"] for s in steps),
            "n_clamped": sum(s["n_clamped"] for s in steps),
        }
    return runs


def summary(a: dict, b: dict, name_a: str, name_b: str) -> None:
    print(f"{'instance':<26} {'|':^3} {name_a[:26]:^26} {'|':^3} {name_b[:26]:^26}")
    print(f"{'':<26} {'|':^3} {'steps  avg_lp  min_lp exit':^26} {'|':^3} {'steps  avg_lp  min_lp exit':^26}")
    print("-" * 92)
    for inst in sorted(set(a) | set(b)):

        def cell(run: dict) -> str:
            if inst not in run:
                return f"{'-- missing --':^26}"
            r = run[inst]
            return f"{len(r['steps']):>5} {r['avg_logp']:>+7.4f} {r['min_logp']:>+7.2f} {r['exit'][:4]:<4}"

        print(f"{inst:<26} {'|':^3} {cell(a)} {'|':^3} {cell(b)}")
    print("-" * 92)
    for name, run in ((name_a, a), (name_b, b)):
        if not run:
            continue
        n = sum(len(r["steps"]) for r in run.values())
        mean = sum(r["avg_logp"] * len(r["steps"]) for r in run.values()) / n
        solved = sum(r["patch"] > 0 for r in run.values())
        clamped = sum(r["n_clamped"] for r in run.values())
        print(
            f"{name:<26} {len(run):>3} inst  {n:>5} steps  token-weighted avg_logp {mean:>+8.4f}  "
            f"non-empty patches {solved:>2}  clamped tokens {clamped}"
        )


def detail(a: dict, b: dict, inst: str, name_a: str, name_b: str) -> None:
    sa = a.get(inst, {}).get("steps", [])
    sb = b.get(inst, {}).get("steps", [])
    print(f"{inst}\n")
    print(f"{'#':>3}  {name_a[:42]:<42} {'|':^3} {name_b[:42]:<42}")
    print("-" * 96)
    for i in range(max(len(sa), len(sb))):

        def cell(steps: list[dict]) -> str:
            if i >= len(steps):
                return " " * 42
            s = steps[i]
            cmd = "; ".join(s["commands"]).replace("\n", " ")[:24]
            return f"{s['avg_logp']:>+7.4f} {s['min_logp']:>+6.2f} rc={str(s['returncode']):>4} {cmd:<24}"

        print(f"{i:>3}  {cell(sa)} {'|':^3} {cell(sb)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--instance", help="Show a step-by-step comparison for this instance instead")
    parser.add_argument("--floor", type=float, default=-20.0, help="Clamp sentinel logprobs to this value (0 disables)")
    args = parser.parse_args()

    floor = args.floor if args.floor else float("-inf")
    a, b = load_run(args.run_a, floor), load_run(args.run_b, floor)
    if args.instance:
        detail(a, b, args.instance, args.run_a.name, args.run_b.name)
    else:
        summary(a, b, args.run_a.name, args.run_b.name)


if __name__ == "__main__":
    main()

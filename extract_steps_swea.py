#!/usr/bin/env python3
"""Flatten a SWE-agent `.traj` into the same per-step records `extract_steps.py` produces for mini.

SWE-agent discards the raw model response, so `avglogp` and `expert` are read from the `extra_info`
that `UncertaintyModel` records on each step rather than recomputed from logprobs.

    python extract_steps_swea.py run_dir/*/ *.traj -o steps.json
"""

import argparse
import json
import re
from pathlib import Path


def extract(traj: dict) -> list[dict]:
    steps = []
    for i, step in enumerate(traj["trajectory"]):
        extra = step.get("extra_info") or {}
        action = (step.get("action") or "").strip()
        steps.append(
            {
                "step": i,
                "model": "expert" if extra.get("expert") else "student",
                "llm_output": step.get("thought") or "",
                "commands": [re.sub(r"\s+", " ", action)] if action else [],
                "format_error": not action,
                "returncode": None,
                "command_output": step.get("observation") or "",
                "n_tokens": None,
                "avg_logp": extra.get("avglogp"),
            }
        )
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trajectory", type=Path, help="SWE-agent .traj file")
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    steps = extract(json.loads(args.trajectory.read_text()))
    args.output.write_text(json.dumps(steps, indent=2))
    n_expert = sum(s["model"] == "expert" for s in steps)
    print(f"{args.trajectory.parent.name}: {len(steps)} steps, {n_expert} expert -> {args.output}")


if __name__ == "__main__":
    main()

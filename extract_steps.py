#!/usr/bin/env python3
"""Flatten a mini-swe-agent trajectory into one record per model call.

Each record holds what the model said, the command it ran, what that command printed, and the
sampled-token confidence metrics:

  avg_logp -- mean log-probability of the sampled tokens (AvgLogP, Eq. 1 of arXiv:2502.18581)
  min_logp -- smallest sampled-token log-probability

Requires the run to have been made with `logprobs: true`. Pure stdlib, so it runs anywhere the
trajectory file does -- no model, no GPU, no mini-swe-agent install.

    python extract_steps.py run1.traj.json -o steps.json
"""

import argparse
import json
import sys
from pathlib import Path

# Alternating per-step colours so adjacent steps stay visually distinct when scrolling.
STEP_COLORS = ("\033[36m", "\033[33m")  # cyan, yellow
DIM, RESET = "\033[2m", "\033[0m"


def get_logprobs(message: dict) -> list[float]:
    """Sampled-token logprobs of the model call that produced this message, if any."""
    response = message.get("extra", {}).get("response")
    if not isinstance(response, dict):  # absent, or a repr() fallback from a failed model_dump
        return []
    entries = ((response.get("choices") or [{}])[0].get("logprobs") or {}).get("content") or []
    return [entry["logprob"] for entry in entries]


def extract(messages: list[dict]) -> list[dict]:
    """One record per model call, in order.

    Steps whose output failed to parse are included: they are billed calls, and mini stores their
    response on the *user* message carrying the format error rather than on an assistant message.
    Skipping them would silently drop exactly the low-confidence steps.
    """
    steps = []
    for index, message in enumerate(messages):
        if not (logprobs := get_logprobs(message)):
            continue
        extra = message.get("extra", {})
        failed = extra.get("interrupt_type") == "FormatError"
        # On a format error the message content is the complaint sent back to the model, so the
        # model's own output lives under `model_response` instead.
        output = extra.get("model_response", "") if failed else (message.get("content") or "")
        # The observation always follows the message that triggered it, when there is one.
        following = messages[index + 1].get("extra", {}) if index + 1 < len(messages) else {}
        observed = "raw_output" in following
        steps.append(
            {
                "step": len(steps),
                "llm_output": output,
                "commands": [action["command"] for action in extra.get("actions", [])],
                "format_error": failed,
                "returncode": following.get("returncode") if observed else None,
                "command_output": following.get("raw_output") if observed else None,
                "n_tokens": len(logprobs),
                "avg_logp": sum(logprobs) / len(logprobs),
                "min_logp": min(logprobs),
            }
        )
    return steps


def truncate(steps: list[dict], limit: int) -> list[dict]:
    """Clip long command output, keeping head and tail, which is where the useful parts are."""
    for step in steps:
        text = step["command_output"]
        if text and len(text) > limit:
            half = limit // 2
            step["command_output"] = f"{text[:half]}\n... {len(text) - limit} chars elided ...\n{text[-half:]}"
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trajectory", type=Path, help="Trajectory .json file")
    parser.add_argument("-o", "--output", type=Path, help="Write records here as JSON (default: print table only)")
    parser.add_argument(
        "--max-output-chars", type=int, default=0, help="Clip command output to this many chars (0 = keep it all)"
    )
    parser.add_argument("--spacing", type=int, default=1, help="Blank lines between steps (0 = compact table)")
    parser.add_argument("--no-color", action="store_true", help="Disable colour even on a terminal")
    args = parser.parse_args()

    steps = extract(json.loads(args.trajectory.read_text())["messages"])
    if not steps:
        raise SystemExit(f"No logprobs in {args.trajectory}. The run needs `-c model.model_kwargs.logprobs=true`.")
    if args.max_output_chars:
        steps = truncate(steps, args.max_output_chars)

    # Escape codes would corrupt the text if this is piped into a file or a pager.
    color = not args.no_color and sys.stdout.isatty()
    paint = (lambda text, code: f"{code}{text}{RESET}") if color else (lambda text, code: text)
    gap = "\n" * args.spacing

    print(paint(f"{'#':>3} {'ntok':>5} {'avg_logp':>9} {'min_logp':>9} {'rc':>4}  command", DIM))
    print(paint("-" * 92, DIM))
    for step in steps:
        command = "!! format error" if step["format_error"] else "; ".join(step["commands"]).replace("\n", " ")
        returncode = "-" if step["returncode"] is None else step["returncode"]
        row = (
            f"{step['step']:>3} {step['n_tokens']:>5} {step['avg_logp']:>+9.4f} "
            f"{step['min_logp']:>+9.4f} {returncode:>4}  {command[:48]}"
        )
        print(paint(row, STEP_COLORS[step["step"] % len(STEP_COLORS)]), end=f"\n{gap}")
    n_failed = sum(step["format_error"] for step in steps)
    print(paint(f"{len(steps)} steps, {n_failed} format errors", DIM))

    if args.output:
        args.output.write_text(json.dumps(steps, indent=2))
        print(f"wrote {args.output} ({args.output.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()

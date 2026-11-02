#!/usr/bin/env python3
"""Flatten a mini-swe-agent trajectory into one record per model call.

Each record holds what the model said, the command it ran, what that command printed, and the
sampled-token confidence metrics:

  avg_logp -- mean log-probability of the sampled tokens (AvgLogP, Eq. 1 of arXiv:2502.18581)
  min_logp -- smallest sampled-token log-probability

The confidence fields are None for a provider that returns no logprobs (Anthropic, say), so a run
made without them still flattens -- you just get the commands and their output. Pure stdlib, so it
runs anywhere the trajectory file does -- no model, no GPU, no mini-swe-agent install.

    python extract_steps.py run1.traj.json -o steps.json
"""

import argparse
import json
import re
import sys
from bisect import bisect_right
from pathlib import Path

# Alternating per-step colours so adjacent steps stay visually distinct when scrolling.
STEP_COLORS = ("\033[36m", "\033[33m")  # cyan, yellow
DIM, RESET = "\033[2m", "\033[0m"

ACTION_REGEX = r"```mswea_bash_command\s*\n(.*?)\n```"
CERTAIN = -0.01  # a token at or above this is effectively deterministic
KS = (1, 3, 5, 10)
WINDOWS = (5, 10)


def get_logprob_entries(message: dict) -> list[dict]:
    """Per-token logprob records of the model call that produced this message, if any."""
    response = message.get("extra", {}).get("response")
    if not isinstance(response, dict):  # absent, or a repr() fallback from a failed model_dump
        return []
    return ((response.get("choices") or [{}])[0].get("logprobs") or {}).get("content") or []


def get_logprobs(message: dict) -> list[float]:
    """Sampled-token logprobs of the model call that produced this message, if any."""
    return [entry["logprob"] for entry in get_logprob_entries(message)]


def aggregate(logprobs: list[float]) -> dict | None:
    """Every single-pass token-probability signal we track, over one span of tokens.

    `mean` is LNS/avglogp; `topK` averages the K least probable tokens, the log-probability
    analogue of Top-K max entropy; `std` and `frac_certain` separate a uniformly predictable
    span from one with a few spikes of surprise, which mean and topK both flatten.
    """
    if not logprobs:
        return None
    values, mean = sorted(logprobs), sum(logprobs) / len(logprobs)
    return {
        "n": len(values),
        "mean": mean,
        "std": (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5,
        "frac_certain": sum(v >= CERTAIN for v in values) / len(values),
    } | {f"top{k}": sum(values[:k]) / min(k, len(values)) for k in KS}


def segments(entries: list[dict], action_regex: str = ACTION_REGEX) -> dict | None:
    """Signals over the THOUGHT prose, the command body, and the whole response.

    Tokens do not align to the code fence, so the response text is reconstructed from them and the
    regex match's character offsets are mapped back onto token indices.
    """
    if not entries:
        return None
    starts, position = [], 0
    for entry in entries:
        starts.append(position)
        position += len(entry["token"])
    values = [entry["logprob"] for entry in entries]
    spans = {"all": aggregate(values)}
    match = re.search(action_regex, "".join(entry["token"] for entry in entries), re.DOTALL)
    if match:
        lo = max(bisect_right(starts, match.start(1)) - 1, 0)
        spans["thought"] = aggregate(values[:lo])
        spans["command"] = aggregate(values[lo : bisect_right(starts, match.end(1) - 1)])
    return spans


def normalize(step: dict) -> str:
    """Command text as the plotting scripts compare it, so repeat flags and loop shading agree."""
    return re.sub(r"\s+", " ", "; ".join(step["commands"]).strip())


def add_repeat_fields(steps: list[dict]) -> list[dict]:
    """The string-matcher baselines, which every UE signal has to be scored against."""
    commands = [normalize(step) for step in steps]
    seen = 0
    for index, command in enumerate(commands):
        for window in WINDOWS:
            steps[index][f"repeat_{window}"] = bool(command) and command in commands[max(0, index - window) : index]
        steps[index]["n_repeats_so_far"] = seen  # strictly before this step, so it stays usable as a predictor
        seen += steps[index][f"repeat_{WINDOWS[-1]}"]
    return steps


def extract(messages: list[dict]) -> list[dict]:
    """One record per model call, in order.

    Steps whose output failed to parse are included: they are billed calls, and mini stores their
    response on the *user* message carrying the format error rather than on an assistant message.
    Skipping them would silently drop exactly the low-confidence steps.
    """
    steps = []
    for index, message in enumerate(messages):
        extra = message.get("extra", {})
        # A model call is what carries a response; the logprobs inside it are optional, since not
        # every provider returns them.
        if "response" not in extra:
            continue
        entries = get_logprob_entries(message)
        logprobs = [entry["logprob"] for entry in entries]
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
                "model": extra.get("response", {}).get("model", ""),
                "llm_output": output,
                "commands": [action["command"] for action in extra.get("actions", [])],
                "format_error": failed,
                "returncode": following.get("returncode") if observed else None,
                "command_output": following.get("raw_output") if observed else None,
                "n_tokens": len(logprobs) or None,
                "avg_logp": sum(logprobs) / len(logprobs) if logprobs else None,
                "min_logp": min(logprobs) if logprobs else None,
                "seg": segments(entries),
            }
        )
    return add_repeat_fields(steps)


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
        raise SystemExit(f"No model calls found in {args.trajectory}.")
    if args.max_output_chars:
        steps = truncate(steps, args.max_output_chars)

    # Escape codes would corrupt the text if this is piped into a file or a pager.
    color = not args.no_color and sys.stdout.isatty()
    paint = (lambda text, code: f"{code}{text}{RESET}") if color else (lambda text, code: text)
    gap = "\n" * args.spacing
    fmt = lambda value: "-" if value is None else f"{value:+.4f}"  # noqa: E731 -- providers without logprobs

    print(paint(f"{'#':>3} {'who':>6} {'ntok':>5} {'avg_logp':>9} {'min_logp':>9} {'rc':>4}  command", DIM))
    print(paint("-" * 92, DIM))
    for step in steps:
        command = "!! format error" if step["format_error"] else "; ".join(step["commands"]).replace("\n", " ")
        returncode = "-" if step["returncode"] is None else step["returncode"]
        who = "expert" if "gpt" in step["model"] or "claude" in step["model"] else "slm"
        row = (
            f"{step['step']:>3} {who:>6} {step['n_tokens'] or '-':>5} {fmt(step['avg_logp']):>9} "
            f"{fmt(step['min_logp']):>9} {returncode:>4}  {command[:48]}"
        )
        print(paint(row, STEP_COLORS[step["step"] % len(STEP_COLORS)]), end=f"\n{gap}")
    n_failed = sum(step["format_error"] for step in steps)
    print(paint(f"{len(steps)} steps, {n_failed} format errors", DIM))

    if args.output:
        args.output.write_text(json.dumps(steps, indent=2))
        print(f"wrote {args.output} ({args.output.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()

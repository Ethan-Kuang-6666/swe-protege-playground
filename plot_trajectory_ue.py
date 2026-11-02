#!/usr/bin/env python3
"""Plot every uncertainty signal for one trajectory, against the commands it actually issued.

Reads the steps.json produced by extract_steps.py, which must carry the `seg` field. Four stacked
panels share the step axis, so a signal can be read off at the exact step a loop begins:

  1. mean log-probability per segment -- THOUGHT prose vs command body vs the whole response
  2. aggregators over the whole response -- mean against the Top-K variants
  3. share of effectively deterministic tokens, per segment
  4. command identity -- one row per distinct command, so a repeat reads as a horizontal line

    python plot_trajectory_ue.py results_lite300/*/steps.json -o plots_ue/
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_trajectory import EXPERT, FORMAT_ERR, GRID, INK, INK2, LOOP, STUDENT, SURFACE, is_expert, loop_runs, norm

CUTOFF = -0.01
SEGMENT_COLORS = {"all": STUDENT, "thought": "#c97b1e", "command": "#1f8a7a"}
AGG_COLORS = {"mean": STUDENT, "top1": "#d14b8f", "top3": "#8a62d6", "top5": "#eda100", "top10": "#1baf7a"}


def series(steps: list[dict], segment: str, field: str) -> tuple[list[int], list[float]]:
    """Student steps that carry this signal. Expert steps are excluded: the logprobs are not theirs."""
    points = [
        (step["step"], (step.get("seg") or {}).get(segment, {}).get(field))
        for step in steps
        if not is_expert(step) and (step.get("seg") or {}).get(segment)
    ]
    return [p[0] for p in points if p[1] is not None], [p[1] for p in points if p[1] is not None]


def decorate(ax, steps: list[dict], runs: list[tuple[int, int]], experts: list[int]) -> None:
    for start, end in runs:
        ax.axvspan(start - 0.5, end + 0.5, color=LOOP, alpha=0.12, lw=0)
    for step in experts:
        ax.axvline(step, color=EXPERT, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.set_xlim(-0.8, len(steps) - 0.2)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def command_rows(steps: list[dict]) -> tuple[list[int | None], dict[str, int], dict[str, int]]:
    """Row index per step, plus the row and issue count of each distinct command."""
    rows, order, counts = [], {}, {}
    for step in steps:
        command = norm(step)
        if command and command not in order:
            order[command] = len(order)
        rows.append(order.get(command))
        counts[command] = counts.get(command, 0) + 1
    return rows, order, counts


def plot(steps_path: Path, out_dir: Path | None = None, cutoff: float = CUTOFF) -> Path:
    steps = json.loads(steps_path.read_text())
    if not any(step.get("seg") for step in steps):
        raise SystemExit(f"{steps_path} has no `seg` field -- re-run extract_steps.py over the trajectory first.")
    sibling = steps_path.parent / f"{steps_path.parent.name}.traj.json"
    swea = steps_path.parent / f"{steps_path.parent.name}.traj"
    info = json.loads((sibling if sibling.exists() else swea).read_text())["info"]
    runs, experts = loop_runs(steps), [s["step"] for s in steps if is_expert(s)]
    rows, order, counts = command_rows(steps)

    strip_height = max(1.4, 0.17 * len(order))
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(12, 7.2 + strip_height),
        dpi=150,
        sharex=True,
        gridspec_kw={"height_ratios": [2.4, 2.4, 1.3, strip_height], "hspace": 0.14},
    )
    fig.patch.set_facecolor(SURFACE)

    # 1. mean log-probability, one line per segment
    ax = axes[0]
    for segment, color in SEGMENT_COLORS.items():
        xs, ys = series(steps, segment, "mean")
        if xs:
            ax.plot(xs, ys, color=color, lw=1.9 if segment == "all" else 1.4, label=segment, zorder=2)
    ax.axhline(cutoff, color=INK2, lw=1, ls=(0, (2, 3)), zorder=1)
    ax.set_ylabel("mean logprob", color=INK2, fontsize=9)
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="lower left")
    decorate(ax, steps, runs, experts)

    # 2. aggregators over the whole response
    ax = axes[1]
    for field, color in AGG_COLORS.items():
        xs, ys = series(steps, "all", field)
        if xs:
            ax.plot(xs, ys, color=color, lw=1.9 if field == "mean" else 1.2, label=field, zorder=2)
    ax.set_ylabel("all-token aggregators", color=INK2, fontsize=9)
    ax.legend(fontsize=8, frameon=False, ncol=5, loc="lower left")
    decorate(ax, steps, runs, experts)

    # 3. share of effectively deterministic tokens
    ax = axes[2]
    for segment, color in SEGMENT_COLORS.items():
        xs, ys = series(steps, segment, "frac_certain")
        if xs:
            ax.plot(xs, ys, color=color, lw=1.6 if segment == "all" else 1.2, zorder=2)
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel(f"frac tokens\n>= {cutoff}", color=INK2, fontsize=9)
    decorate(ax, steps, runs, experts)

    # 4. command identity -- a repeat is a horizontal line
    ax = axes[3]
    for step, row in zip(steps, rows):
        if row is None:
            continue
        repeated = counts[norm(step)] > 1
        ax.scatter(
            step["step"],
            row,
            s=34,
            color=EXPERT if is_expert(step) else (LOOP if repeated else STUDENT),
            marker="x" if step["format_error"] else "o",
            edgecolor=SURFACE,
            lw=0.9,
            zorder=3,
        )
    for command, row in order.items():
        if counts[command] > 1:
            ax.axhline(row, color=LOOP, alpha=0.18, lw=0.9, zorder=1)
    ax.set_yticks([row for command, row in order.items() if counts[command] > 1])
    ax.set_yticklabels(
        [f"{command[:34]}  ×{counts[command]}" for command in order if counts[command] > 1],
        fontsize=7,
        color=INK2,
        family="monospace",
    )
    ax.set_ylim(-0.8, max(len(order) - 0.2, 1))
    ax.set_xlabel("step", color=INK2)
    ax.set_ylabel("command", color=INK2, fontsize=9)
    decorate(ax, steps, runs, experts)
    ax.grid(axis="y", lw=0)

    n_repeat = sum(step.get("repeat_10", False) for step in steps)
    axes[0].set_title(
        f"{steps_path.parent.name}   ·   {info.get('exit_status', '?')}   ·   {len(steps)} steps, "
        f"{len(experts)} expert, {n_repeat} repeat-10 steps, {len(order)} distinct commands",
        loc="left",
        fontsize=10,
        color=INK,
        pad=12,
    )
    fig.legend(
        handles=[
            Line2D([], [], color=STUDENT, lw=0, marker="o", ms=5, label="first issue of this command"),
            Line2D([], [], color=LOOP, lw=0, marker="o", ms=5, label="command issued more than once"),
            Line2D([], [], color=EXPERT, lw=0, marker="o", ms=5, label="expert step"),
            Line2D([], [], color=FORMAT_ERR, lw=0, marker="x", ms=6, label="format error"),
            Line2D([], [], color=LOOP, lw=6, alpha=0.12, label="consecutive-repeat run"),
        ],
        loc="upper center",
        fontsize=8,
        frameon=False,
        ncol=5,
        bbox_to_anchor=(0.5, 0.055),
    )

    out_dir = out_dir or steps_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{steps_path.parent.name}.ue.png"
    fig.savefig(out, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("steps", type=Path, nargs="+", help="steps.json file(s) from extract_steps.py")
    parser.add_argument("-o", "--out-dir", type=Path, help="Output directory (default: next to each steps.json)")
    parser.add_argument("--cutoff", type=float, default=CUTOFF, help="Reference cutoff to draw (default: %(default)s)")
    args = parser.parse_args()
    for path in args.steps:
        print(plot(path, args.out_dir, args.cutoff))


if __name__ == "__main__":
    main()

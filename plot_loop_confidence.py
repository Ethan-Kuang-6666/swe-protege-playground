#!/usr/bin/env python3
"""Does repetition drive the student's confidence up?

Reads the steps.json files produced by extract_steps.py across a whole run and draws three
panels:

  A  median avglogp against position within a run of identical consecutive commands
  B  the avglogp distribution, repeated steps against non-repeated ones
  C  per trajectory, how much it repeated against how confident it was

    python plot_loop_confidence.py results_lite300/*/steps.json -o loop_confidence.png
"""

import argparse
import json
import re
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
PLAIN, LOOP = "#2a78d6", "#eb6834"
MAX_POS = 5
"""Positions past this are pooled: repeat runs have a long tail and the bins would be empty."""


def norm(step: dict) -> str:
    return re.sub(r"\s+", " ", "; ".join(step["commands"]).strip())


def repeat_runs(steps: list[dict]):
    """(start, end) index ranges of >=2 consecutive steps issuing the same non-empty command."""
    runs, i = [], 0
    while i < len(steps):
        j = i
        while j + 1 < len(steps) and norm(steps[j + 1]) == norm(steps[i]) and norm(steps[i]):
            j += 1
        if j > i:
            runs.append((i, j))
        i = j + 1
    return runs


def collect(paths: list[Path]):
    by_position: dict[int, list[float]] = {}
    repeated: list[float] = []
    plain: list[float] = []
    per_traj: list[tuple[int, float]] = []
    for path in paths:
        steps = json.loads(path.read_text())
        runs = repeat_runs(steps)
        in_run = {i for a, b in runs for i in range(a, b + 1)}
        for a, b in runs:
            for offset, index in enumerate(range(a, b + 1)):
                if (value := steps[index]["avg_logp"]) is not None:
                    by_position.setdefault(min(offset, MAX_POS), []).append(value)
        for i, step in enumerate(steps):
            if step["avg_logp"] is None:
                continue
            (repeated if i in in_run else plain).append(step["avg_logp"])
        if values := [s["avg_logp"] for s in steps if s["avg_logp"] is not None]:
            per_traj.append((len(in_run), median(values)))
    return by_position, repeated, plain, per_traj


def style(ax, xlabel, ylabel, title):
    ax.set_facecolor(SURFACE)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.set_title(title, loc="left", fontsize=10, color=INK, pad=10)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors=INK2, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("steps", type=Path, nargs="+", help="steps.json files from extract_steps.py")
    parser.add_argument("-o", "--output", type=Path, default=Path("loop_confidence.png"))
    parser.add_argument("--cutoff", type=float, default=-0.001, help="Handoff cutoff to mark (default: %(default)s)")
    args = parser.parse_args()

    by_position, repeated, plain, per_traj = collect(args.steps)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)

    # A -- the gradient. Log scale, since the medians span more than an order of magnitude.
    ax = axes[0]
    positions = sorted(by_position)
    medians = [-median(by_position[p]) for p in positions]
    labels = [str(p) if p < MAX_POS else f"{MAX_POS}+" for p in positions]
    # A line, not bars: the y axis is logarithmic, so bars would have to hang from an arbitrary
    # baseline rather than grow from zero.
    ax.plot(range(len(positions)), medians, color=LOOP, lw=2, marker="o", ms=7, mec=SURFACE, mew=1.5)
    for i, (p, m) in enumerate(zip(positions, medians)):
        ax.text(i, m * 0.78, f"n={len(by_position[p])}", ha="center", fontsize=7, color=INK2)
    ax.axhline(-args.cutoff, color=INK2, lw=1, ls=(0, (2, 3)))
    ax.text(len(positions) - 0.4, -args.cutoff, f" cutoff {args.cutoff}", fontsize=8, color=INK2, va="center")
    ax.set_yscale("log")
    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels(labels)
    ax.invert_yaxis()
    style(
        ax,
        "position within a run of identical commands",
        "|median avglogp|  (log, nearer top = more certain)",
        "A  confidence climbs with every repetition",
    )

    # B -- where the two populations sit relative to the cutoff.
    ax = axes[1]
    # Half-decade bins from the deepest observed value up to zero: the mass sits within a few
    # thousandths of 0, so linear bins would put everything in one column.
    bins = [-(10 ** (-k / 2)) for k in range(0, 13)]
    if (floor := min(plain + repeated)) < bins[0]:
        bins.insert(0, floor)
    ax.hist([plain, repeated], bins=bins, color=[PLAIN, LOOP], label=["non-repeated", "repeated"], rwidth=0.88)
    ax.axvline(args.cutoff, color=INK2, lw=1, ls=(0, (2, 3)))
    ax.set_xscale("symlog", linthresh=1e-6)
    ax.set_xlim(bins[0], -1e-6)  # avglogp is never positive, so the symlog right half is empty
    ax.legend(fontsize=8, frameon=False)
    style(ax, "avglogp (symlog)", "steps", "B  repeated steps cluster above the cutoff")

    # C -- does a trajectory that repeats more read as more confident overall?
    ax = axes[2]
    ax.scatter(
        [n for n, _ in per_traj],
        [-v for _, v in per_traj],
        s=18,
        color=LOOP,
        alpha=0.5,
        edgecolor="none",
    )
    ax.set_yscale("log")
    ax.invert_yaxis()
    style(ax, "repeated-command steps in the trajectory", "|median avglogp| (log)", "C  per trajectory")

    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {args.output}")

    print(
        f"\nrepeated     n={len(repeated):6}  median {median(repeated):+.6f}  >= cutoff {sum(x >= args.cutoff for x in repeated) / len(repeated) * 100:.1f}%"
    )
    print(
        f"non-repeated n={len(plain):6}  median {median(plain):+.6f}  >= cutoff {sum(x >= args.cutoff for x in plain) / len(plain) * 100:.1f}%"
    )


if __name__ == "__main__":
    main()

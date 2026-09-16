#!/usr/bin/env python3
"""Plot avg_logp per step for a trajectory, marking repeated commands and expert steps.

Reads the steps.json produced by extract_steps.py. Expert steps are drawn as vertical
markers rather than points, since their logprobs from OpenAI contain -9999 sentinels.

    python plot_trajectory.py results_lite20/*/steps.json   # writes <instance>.png next to each steps.json
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CUTOFF = -0.005
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
STUDENT, LOOP, EXPERT, FORMAT_ERR = "#2a78d6", "#eb6834", "#4a3aa7", "#e34948"


def norm(step: dict) -> str:
    return re.sub(r"\s+", " ", "; ".join(step["commands"]).strip())


def is_expert(step: dict) -> bool:
    return any(k in step["model"] for k in ("gpt", "claude"))


def loop_runs(steps: list[dict]) -> list[tuple[int, int]]:
    """(start, end) index ranges of >=2 consecutive student steps issuing the same command."""
    runs, i = [], 0
    while i < len(steps):
        if is_expert(steps[i]):
            i += 1
            continue
        j = i
        while j + 1 < len(steps) and not is_expert(steps[j + 1]) and norm(steps[j + 1]) == norm(steps[i]):
            j += 1
        if j > i:
            runs.append((i, j))
        i = j + 1
    return runs


def plot(steps_path: Path, out_dir: Path | None = None) -> Path:
    steps = json.loads(steps_path.read_text())
    traj = json.loads((steps_path.parent / f"{steps_path.parent.name}.traj.json").read_text())["info"]
    xs = [s["step"] for s in steps if not is_expert(s)]
    ys = [s["avg_logp"] for s in steps if not is_expert(s)]
    experts = [s["step"] for s in steps if is_expert(s)]
    runs = loop_runs(steps)
    fmt_errs = [s["step"] for s in steps if s["format_error"] and not is_expert(s)]

    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for a, b in runs:
        ax.axvspan(a - 0.5, b + 0.5, color=LOOP, alpha=0.12, lw=0)
    for x in experts:
        ax.axvline(x, color=EXPERT, lw=1.5, ls=(0, (4, 3)), zorder=1)
    ax.axhline(CUTOFF, color=INK2, lw=1, ls=(0, (2, 3)), zorder=1)
    ax.plot(xs, ys, color=STUDENT, lw=2, zorder=2, solid_joinstyle="round")
    ax.scatter(xs, ys, s=28, color=STUDENT, edgecolor=SURFACE, lw=1.2, zorder=3)
    loop_idx = {i for a, b in runs for i in range(a, b + 1)}
    ax.scatter(
        [s["step"] for s in steps if s["step"] in loop_idx],
        [s["avg_logp"] for s in steps if s["step"] in loop_idx],
        s=40,
        color=LOOP,
        edgecolor=SURFACE,
        lw=1.2,
        zorder=4,
    )
    if fmt_errs:
        ax.scatter(
            fmt_errs,
            [s["avg_logp"] for s in steps if s["step"] in fmt_errs],
            s=60,
            marker="x",
            color=FORMAT_ERR,
            lw=1.5,
            zorder=5,
        )

    ymin = min(ys + [CUTOFF]) - 0.04
    ax.set_ylim(ymin, 0.02)
    ax.set_xlim(-0.8, len(steps) - 0.2)
    ax.text(
        len(steps) - 0.2, CUTOFF, f" cutoff {CUTOFF}", va="center", ha="left", fontsize=8, color=INK2, clip_on=False
    )
    for x in experts:
        ax.text(x, 0.02, "E", ha="center", va="bottom", fontsize=8, color=EXPERT, fontweight="bold")

    ax.set_xlabel("step", color=INK2)
    ax.set_ylabel("avg logprob (student)", color=INK2)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors=INK2, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)

    n_exp, n_loop = len(experts), sum(b - a + 1 for a, b in runs)
    ax.set_title(
        f"{steps_path.parent.name}   ·   {traj['exit_status']}   ·   {len(steps)} steps, "
        f"{n_exp} expert, {n_loop} repeated-command steps, ${traj['model_stats']['instance_cost']:.2f}",
        loc="left",
        fontsize=10,
        color=INK,
        pad=14,
    )
    handles = [
        Line2D([], [], color=STUDENT, lw=2, marker="o", ms=5, label="student avg logprob"),
        Line2D([], [], color=LOOP, lw=0, marker="o", ms=6, label="repeated-command run (same command ≥2× in a row)"),
        Line2D([], [], color=EXPERT, lw=1.5, ls=(0, (4, 3)), label="expert takes this step"),
        Line2D([], [], color=INK2, lw=1, ls=(0, (2, 3)), label="handoff cutoff"),
    ]
    if fmt_errs:
        handles.append(Line2D([], [], color=FORMAT_ERR, lw=0, marker="x", ms=7, label="format error"))
    ax.legend(
        handles=handles, loc="lower left", fontsize=8, frameon=False, ncol=len(handles), bbox_to_anchor=(0, -0.32)
    )

    out_dir = out_dir or steps_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{steps_path.parent.name}.png"
    fig.savefig(out, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("steps", type=Path, nargs="+", help="steps.json file(s) from extract_steps.py")
    parser.add_argument("-o", "--out-dir", type=Path, help="Output directory (default: next to each steps.json)")
    args = parser.parse_args()
    for p in args.steps:
        print(plot(p, args.out_dir))


if __name__ == "__main__":
    main()

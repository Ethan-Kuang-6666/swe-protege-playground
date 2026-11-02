"""Post-hoc uncertainty-estimation analysis over saved mini-swe-agent trajectories.

Computes token-probability signals per step, separately over the THOUGHT prose, the command
body, and the whole response, then scores each against step-level degeneration labels.
"""

import json
import re
from bisect import bisect_right
from pathlib import Path

import numpy as np
import typer

ACTION_REGEX = r"```mswea_bash_command\s*\n(.*?)\n```"
SEGMENTS = ("thought", "command", "all")
KS = (1, 3, 5, 10)


def segment_slices(tokens: list[dict], action_regex: str) -> dict[str, slice]:
    """Token index spans for the thought prose and the command body."""
    starts, pos = [], 0
    for t in tokens:
        starts.append(pos)
        pos += len(t["token"])
    match = re.search(action_regex, "".join(t["token"] for t in tokens), re.DOTALL)
    if not match:
        return {"all": slice(0, len(tokens))}
    lo = max(bisect_right(starts, match.start(1)) - 1, 0)
    hi = bisect_right(starts, match.end(1) - 1)
    return {"thought": slice(0, lo), "command": slice(lo, hi), "all": slice(0, len(tokens))}


def signals(logprobs: np.ndarray) -> dict[str, float]:
    """Mean logprob (LNS) plus Top-K aggregation over the K least probable tokens."""
    if not len(logprobs):
        return {}
    ascending = np.sort(logprobs)
    return {"mean": float(ascending.mean()), "n_tokens": len(ascending)} | {
        f"top{k}": float(ascending[:k].mean()) for k in KS
    }


def step_records(traj: Path, action_regex: str) -> list[dict]:
    """One record per assistant step: the command, and every signal on every segment."""
    messages = json.loads(traj.read_text())["messages"]
    records = []
    for message in (m for m in messages if m["role"] == "assistant"):
        tokens = (message.get("extra", {}).get("response", {}).get("choices", [{}])[0].get("logprobs") or {}).get(
            "content"
        ) or []
        actions = message.get("extra", {}).get("actions") or [{}]
        record = {"command": actions[0].get("command", ""), "n_tokens_total": len(tokens)}
        for name, span in segment_slices(tokens, action_regex).items():
            for signal, value in signals(np.array([t["logprob"] for t in tokens[span]])).items():
                record[f"{name}.{signal}"] = value
        records.append(record)
    return records


def label_steps(commands: list[str], window: int = 10) -> tuple[list[bool], list[bool]]:
    """Exact repeats (the string-matcher target) and prefix repeats (parametric drift)."""
    prefixes = [" ".join(c.split()[:2]) for c in commands]
    exact, near = [], []
    for i, command in enumerate(commands):
        recent = slice(max(0, i - window), i)
        is_exact = command in commands[recent]
        exact.append(is_exact)
        near.append(not is_exact and prefixes[i] in prefixes[recent])
    return exact, near


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), float)
    ranks[order] = np.arange(1, len(values) + 1)
    sorted_values, i = values[order], 0
    while i < len(sorted_values):
        j = i
        while j + 1 < len(sorted_values) and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j + 2) / 2
        i = j + 1
    return ranks


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """P(score of a positive > score of a negative). Higher score is predicted-degenerate."""
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if not n_pos or not n_neg:
        return float("nan")
    return float((_ranks(scores)[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _rejection_auc(scores: np.ndarray, good: np.ndarray) -> float:
    """Mean precision of the retained set as the highest-scoring steps are progressively rejected."""
    retained = good[np.argsort(scores, kind="mergesort")]
    return float((np.cumsum(retained) / np.arange(1, len(good) + 1)).mean())


def prr(scores: np.ndarray, labels: np.ndarray) -> float:
    """Prediction-Rejection Ratio: 0.0 matches random rejection, 1.0 matches the oracle."""
    good = ~labels
    random, oracle = good.mean(), _rejection_auc(-good.astype(float), good)
    if oracle == random:
        return float("nan")
    return (_rejection_auc(scores, good) - random) / (oracle - random)


def main(
    run_dir: Path = typer.Argument(..., help="Directory of per-instance trajectory folders"),
    out: Path = typer.Option(Path("ue_steps.json"), "--out", help="Where to write the per-step table"),
    action_regex: str = typer.Option(ACTION_REGEX, "--action-regex"),
    window: int = typer.Option(10, "--window", help="Lookback for the repeat detector"),
) -> None:
    rows = []
    for traj in sorted(run_dir.glob("*/*.traj.json")):
        records = step_records(traj, action_regex)
        exact, near = label_steps([r["command"] for r in records], window)
        for i, (record, is_exact, is_near) in enumerate(zip(records, exact, near)):
            rows.append(record | {"instance": traj.parent.name, "step": i, "exact": is_exact, "near": is_near})
    out.write_text(json.dumps(rows))

    exact = np.array([r["exact"] for r in rows])
    near = np.array([r["near"] for r in rows])
    print(f"{len(rows)} steps over {len({r['instance'] for r in rows})} instances")
    print(f"exact repeats {exact.mean():.1%} | prefix repeats among the rest {near[~exact].mean():.1%}\n")

    for label, name, subset in [(exact, "exact repeat", np.ones(len(rows), bool)), (near, "prefix repeat", ~exact)]:
        print(f"--- predicting {name} (n={subset.sum()}, base rate {label[subset].mean():.1%}) ---")
        print(f"{'signal':<18}{'AUROC':>8}{'PRR':>8}{'n':>9}")
        for segment in SEGMENTS:
            for signal in ("mean", *(f"top{k}" for k in KS)):
                key = f"{segment}.{signal}"
                have = np.array([key in r for r in rows]) & subset
                if have.sum() < 100:
                    continue
                scores = np.array([r[key] for r in rows if key in r])[have[np.array([key in r for r in rows])]]
                print(f"{key:<18}{auroc(scores, label[have]):>8.3f}{prr(scores, label[have]):>8.3f}{have.sum():>9}")
        print()


if __name__ == "__main__":
    typer.run(main)

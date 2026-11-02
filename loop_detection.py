"""Loop detection from token-level uncertainty over saved mini-swe-agent trajectories.

  extract   writes <instance>/steps.json next to each trajectory: readable steps plus certainty metrics
  label     an LLM judge marks the step where each trajectory's first loop starts, or no loop
  evaluate  first-alarm TP/FP/FN per metric x trigger x direction; threshold tuned on one split, scored on the other

Metrics are single-pass, from the sampled-token logprobs and the logged top-k alternatives only
(LNS, perplexity, CSP, mean/min probability, mean/max/top-K entropy, plus std, near-certain fraction and margin).
`evaluate` orients each so that larger means more certain, then tries both "fire when certain" (our hypothesis)
and "fire when uncertain" (the conventional cascade).
"""

import json
import random
import re
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
import numpy as np
import typer
from jinja2 import Template

app = typer.Typer(add_completion=False)
ACTION_REGEX = r"```mswea_bash_command\s*\n(.*?)\n```"
CERTAIN = -0.01
KS = (3, 5, 10)
REPEAT_WINDOW = 10
TRIGGERS = ("static", "streak2", "streak3", "mean3", "mean5", "delta5", "deltaall")
LESS_IS_CERTAIN = ("perplexity", "entropy", "std_logp")


def token_arrays(entries: list[dict]) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Sampled-token logprobs, plus entropy and top-1/top-2 margin when the alternatives were logged.

    Entropy is over the top-k alternatives renormalised to sum to one, the usual truncated estimate.
    Alternatives are floats in slimmed trajectories and {"token", "logprob", ...} dicts in raw ones.
    """
    logprobs = np.array([e["logprob"] for e in entries], float)
    alts = [
        sorted((a if isinstance(a, (int, float)) else a["logprob"] for a in e.get("top_logprobs") or []), reverse=True)
        for e in entries
    ]
    n = min(map(len, alts), default=0)
    if n < 2:
        return logprobs, None, None
    p = np.exp(np.array([a[:n] for a in alts]))
    q = p / p.sum(axis=1, keepdims=True)
    return logprobs, -(q * np.log(np.clip(q, 1e-30, None))).sum(axis=1), p[:, 0] - p[:, 1]


def metrics(logprobs: np.ndarray, entropy: np.ndarray | None, margin: np.ndarray | None) -> dict[str, float]:
    if not len(logprobs):
        return {}
    probs = np.exp(logprobs)
    out = {
        "lns": logprobs.mean(),
        "perplexity": np.exp(-logprobs.mean()),
        "csp": logprobs.sum(),
        "mean_prob": probs.mean(),
        "min_prob": probs.min(),
        "std_logp": logprobs.std(),
        "frac_certain": (logprobs >= CERTAIN).mean(),
    }
    if entropy is not None:
        high = np.sort(entropy)[::-1]
        out |= {"mean_entropy": high.mean(), "max_entropy": high[0]} | {f"top{k}_entropy": high[:k].mean() for k in KS}
        out |= {"mean_margin": margin.mean(), "min_margin": margin.min()}
    return {k: float(v) for k, v in out.items()}


def segment_slices(entries: list[dict]) -> dict[str, slice]:
    """Token spans of the THOUGHT prose and the command body, found by mapping regex offsets back onto tokens."""
    starts, position = [], 0
    for entry in entries:
        starts.append(position)
        position += len(entry["token"])
    match = re.search(ACTION_REGEX, "".join(e["token"] for e in entries), re.DOTALL)
    if not match:
        return {"all": slice(0, len(entries))}
    lo = max(bisect_right(starts, match.start(1)) - 1, 0)
    return {
        "thought": slice(0, lo),
        "command": slice(lo, bisect_right(starts, match.end(1) - 1)),
        "all": slice(0, len(entries)),
    }


def trajectory_steps(messages: list[dict], last_n_observations: int) -> list[dict]:
    """One readable record per model step, with its certainty metrics on every segment.

    `visible_outputs` mirrors UncertaintyAgent.get_context: the steps whose output the model still saw in full
    when it wrote this step; every older observation had been replaced by a placeholder.
    """
    steps, commands, owner = [], [], {}  # owner: observation message index -> step that produced it
    for i, message in enumerate(messages):
        if message["role"] == "user" and i > 1 and steps:
            owner[i] = steps[-1]["step"]
        response = message.get("extra", {}).get("response")
        if message["role"] != "assistant" or not isinstance(response, dict):
            continue
        observations = [j for j in range(2, i) if messages[j]["role"] == "user"]
        elided = set(observations[:-last_n_observations])
        entries = (response["choices"][0].get("logprobs") or {}).get("content") or []
        logprobs, entropy, margin = token_arrays(entries)
        command = (message["extra"].get("actions") or [{}])[0].get("command", "")
        normalized = " ".join(command.split())
        observation = messages[i + 1] if i + 1 < len(messages) else {}
        pick = lambda a, span: None if a is None else a[span]  # noqa: E731
        steps.append(
            {
                "step": len(steps),
                "thought": message["content"].split("```")[0].strip(),
                "command": command,
                "returncode": observation.get("extra", {}).get("returncode"),
                "output": observation.get("content", ""),
                "visible_outputs": sorted({owner[j] for j in observations if j not in elided and j in owner}),
                "exact_repeat": bool(normalized) and normalized in commands[-REPEAT_WINDOW:],
                "n_tokens": len(entries),
                "metrics": {
                    name: metrics(logprobs[span], pick(entropy, span), pick(margin, span))
                    for name, span in segment_slices(entries).items()
                },
            }
        )
        commands.append(normalized)
    return steps


def load_steps(run_dir: Path) -> dict[str, dict]:
    return {p.parent.name: json.loads(p.read_text()) for p in sorted(run_dir.glob("*/steps.json"))}


@app.command()
def extract(run_dir: Path = typer.Argument(..., help="Directory of per-instance trajectory folders")) -> None:
    for path in sorted(run_dir.glob("*/*.traj.json")):
        traj = json.loads(path.read_text())
        messages = traj["messages"]
        task = re.search(r"<pr_description>(.*?)</pr_description>", messages[1]["content"], re.DOTALL)
        steps = trajectory_steps(messages, traj["info"]["config"]["agent"]["last_n_observations"])
        record = {
            "instance": path.parent.name,
            "task": (task.group(1) if task else messages[1]["content"]).strip(),
            "steps": steps,
        }
        (path.parent / "steps.json").write_text(json.dumps(record, indent=1))
        print(f"{path.parent.name}: {len(steps)} steps")


JUDGE = Template("""\
You are labelling trajectories of a small software-engineering agent that tries to fix a GitHub issue by running shell
commands. We want to know where the agent STALLS: where it stops making real progress and keeps doing the same thing.

<definition>
A STALL is a stretch of at least 3 repetitions of a pattern in which the agent makes no real progress. It includes:
- exact_loop: the same command, run again and again.
- cycle: two or more commands repeated in turn, e.g. the same edit, then the same test, then the same error, again.
- near_repeat: similar commands with only surface variation -- another line range of the same file, a slightly
  different grep for the same thing, a reworded edit that fails the same way -- whose results do not materially change.
Repeating an action whose outcome stays the same (same error, no effect, empty output) is a stall even when the
commands differ a little. NOT a stall: exploration that reads different things, or iteration in which each attempt
changes something that matters and the results change in a way the agent learns from.
</definition>

Find the FIRST stall. Report the step where it begins -- the first step that belongs to the repeated pattern, not the
step where the repetition becomes obvious -- and the last step of that stall (the final step if the agent never
escapes it). The exact start is often ambiguous by a step or two; pick the earliest step from which the agent was no
longer making progress. If the agent never stalls, report null for both.

<issue>
{{ task }}
</issue>

<trajectory>
{% for s in steps -%}
### Step {{ s.step }}
THOUGHT: {{ s.thought }}
COMMAND: {{ s.command }}
OUTPUT: {{ s.output }}

{% endfor -%}
</trajectory>

Answer with exactly this JSON inside answer tags:
<answer>{"stall_start": <step or null>, "stall_end": <step or null>, "kind": "<exact_loop | cycle | near_repeat | null>", "pattern": "<one sentence describing what repeats>", "evidence_steps": [<step numbers>]}</answer>
""")


@app.command()
def label(
    run_dir: Path = typer.Argument(..., help="Run directory containing <instance>/steps.json"),
    out: Path = typer.Option(None, "--out", help="Default: <run_dir>/ue_labels.json; resumed if it exists"),
    model: str = typer.Option("claude-opus-5-5", "--model"),
    effort: str = typer.Option("high", "--effort"),
    workers: int = typer.Option(8, "--workers"),
    limit: int = typer.Option(0, "--limit", help="Only label this many new instances (0 = all)"),
) -> None:
    out = out or run_dir / "ue_labels.json"
    trajectories = load_steps(run_dir)
    labels = json.loads(out.read_text()) if out.exists() else {}
    todo = [i for i in trajectories if i not in labels][: limit or None]
    client = anthropic.Anthropic()

    def judge(instance: str) -> tuple[str, dict | None]:
        response = client.beta.messages.create(
            model=model,
            max_tokens=16000,
            output_config={"effort": effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": JUDGE.render(**trajectories[instance])}],
        )
        if response.stop_reason == "refusal":  # left unlabelled, so a rerun retries it
            print(f"{instance}: judge refused ({response.stop_details.category}), skipped")
            return instance, None
        text = "".join(b.text for b in response.content if b.type == "text")
        answer = json.loads(re.search(r"<answer>(.*?)</answer>", text, re.DOTALL).group(1))
        return instance, answer | {"judge": response.model, "raw": text}

    with ThreadPoolExecutor(workers) as pool:
        for instance, result in pool.map(judge, todo):
            if result is None:
                continue
            labels[instance] = result
            out.write_text(json.dumps(labels, indent=1))
            print(
                f"{instance}: stall {result['stall_start']}-{result['stall_end']} ({result['kind']})  {result['pattern']}"
            )


def fire_scores(certainty: np.ndarray, trigger: str) -> np.ndarray:
    """Per-step score whose first crossing of the threshold is the alarm."""
    if trigger == "static":
        return certainty
    if trigger.startswith("streak"):  # k consecutive steps at or above the threshold
        k = int(trigger.removeprefix("streak"))
        return np.array([certainty[s - k + 1 : s + 1].min() if s >= k - 1 else np.nan for s in range(len(certainty))])
    if trigger.startswith("mean"):  # moving average of the last w steps
        w = int(trigger.removeprefix("mean"))
        return np.array([certainty[s - w + 1 : s + 1].mean() if s >= w - 1 else np.nan for s in range(len(certainty))])
    if trigger == "deltaall":  # jump above the mean of all earlier steps in the trajectory
        return np.array([certainty[s] - certainty[:s].mean() if s else np.nan for s in range(len(certainty))])
    w = int(trigger.removeprefix("delta"))  # jump above the mean of the previous w steps
    return np.array(
        [certainty[s] - certainty[max(0, s - w) : s].mean() if s else np.nan for s in range(len(certainty))]
    )


def first_alarms(fires: list[np.ndarray], thresholds: np.ndarray) -> np.ndarray:
    """Alarm step for each (threshold, trajectory), -1 if the trigger never fires."""
    alarms = np.full((len(thresholds), len(fires)), -1)
    for j, f in enumerate(fires):
        running = np.fmax.accumulate(np.where(np.isnan(f), -np.inf, f)) if len(f) else f
        index = np.searchsorted(running, thresholds, side="left")
        alarms[:, j] = np.where(index < len(f), index, -1)
    return alarms


def score(alarms: np.ndarray, starts: np.ndarray, grace: int = 0) -> dict[str, np.ndarray]:
    """TP: first alarm at or after the stall start, or up to `grace` steps before it, since the labelled start is
    fuzzy by a step or two. FP: any earlier alarm, or any alarm in a stall-free trajectory."""
    fired, loop = alarms >= 0, starts >= 0
    tp = fired & loop & (alarms >= starts - grace)
    n_loop = max(loop.sum(), 1)
    return {
        "tpr": tp.sum(1) / n_loop,
        "fpr": (fired & ~tp).sum(1) / len(starts),
        "fnr": (loop & ~fired).sum(1) / n_loop,
        "tp": tp,
    }


def signal_values(steps: list[dict], signal: str) -> np.ndarray:
    if signal == "exact_repeat":
        return np.array([float(s["exact_repeat"]) for s in steps])
    segment, name = signal.split(".")
    return np.array([s["metrics"].get(segment, {}).get(name, np.nan) for s in steps])


@app.command()
def evaluate(
    run_dir: Path = typer.Argument(..., help="Run directory containing <instance>/steps.json"),
    labels_file: Path = typer.Option(None, "--labels", help="Default: <run_dir>/ue_labels.json"),
    n_tune: int = typer.Option(100, "--n-tune"),
    seed: int = typer.Option(0, "--seed"),
    max_fpr: float = typer.Option(0.1, "--max-fpr"),
    grace: int = typer.Option(2, "--grace", help="Alarms this many steps before the labelled start still count"),
    out: Path = typer.Option(None, "--out", help="Default: <run_dir>/ue_eval.json"),
    show: int = typer.Option(30, "--show"),
    holdout: Path = typer.Option(
        None, "--holdout", help="File of instance ids kept out of tune/test and scored as a third split"
    ),
    thresholds_from: Path = typer.Option(
        None, "--thresholds-from", help="ue_eval.json of another run; also score its thresholds unchanged"
    ),
) -> None:
    trajectories = load_steps(run_dir)
    labels = json.loads((labels_file or run_dir / "ue_labels.json").read_text())
    frozen = (
        {(r["signal"], r["fires_when"], r["trigger"]): r["threshold"] for r in json.loads(thresholds_from.read_text())}
        if thresholds_from
        else {}
    )
    held = set(holdout.read_text().split()) & labels.keys() if holdout else set()
    ids = sorted(i for i in trajectories if i in labels and i not in held)
    random.Random(seed).shuffle(ids)
    splits = {"tune": ids[:n_tune], "test": ids[n_tune:]} | ({"holdout": sorted(held)} if held else {})
    starts = {
        s: np.array([-1 if labels[i]["stall_start"] is None else labels[i]["stall_start"] for i in v])
        for s, v in splits.items()
    }
    print(" | ".join(f"{s}: {len(v)} trajectories, {(starts[s] >= 0).sum()} with a stall" for s, v in splits.items()))

    names = sorted(
        {
            f"{seg}.{m}"
            for t in trajectories.values()
            for s in t["steps"]
            for seg, ms in s["metrics"].items()
            for m in ms
        }
    )
    results = []
    for signal in ["exact_repeat", *names]:
        sign = -1 if any(k in signal for k in LESS_IS_CERTAIN) else 1
        for direction in (1,) if signal == "exact_repeat" else (1, -1):
            for trigger in TRIGGERS:
                fires = {
                    s: [
                        fire_scores(direction * sign * signal_values(trajectories[i]["steps"], signal), trigger)
                        for i in v
                    ]
                    for s, v in splits.items()
                }
                pooled = np.concatenate(fires["tune"])
                pooled = pooled[~np.isnan(pooled)]
                if not len(pooled):
                    continue
                thresholds = np.append(np.unique(np.quantile(pooled, np.linspace(0, 1, 2001))), np.inf)
                tune = score(first_alarms(fires["tune"], thresholds), starts["tune"], grace)
                ok = np.flatnonzero(tune["fpr"] <= max_fpr)
                if not len(ok):
                    continue
                best = ok[np.lexsort((tune["fpr"][ok], -tune["tpr"][ok]))[0]]
                key = (signal, "certain" if direction == 1 else "uncertain", trigger)
                row = {
                    "signal": signal,
                    "fires_when": key[1],
                    "trigger": trigger,
                    "threshold": float(direction * sign * thresholds[best]),
                    **{f"tune_{k}": float(tune[k][best]) for k in ("tpr", "fpr", "fnr")},
                }
                # stored thresholds are in the metric's own units; the sign flip back is its own inverse
                chosen = {"": thresholds[best]} | ({"_frozen": direction * sign * frozen[key]} if key in frozen else {})
                if key in frozen:
                    row["frozen_threshold"] = frozen[key]
                for split in [s for s in splits if s != "tune"]:
                    for tag, threshold in chosen.items():
                        alarms = first_alarms(fires[split], np.array([threshold]))
                        graced, strict = score(alarms, starts[split], grace), score(alarms, starts[split])
                        delays = (alarms[0] - starts[split])[graced["tp"][0]]
                        row |= {f"{split}{tag}_{k}": float(graced[k][0]) for k in ("tpr", "fpr", "fnr")}
                        row |= {f"{split}{tag}_strict_{k}": float(strict[k][0]) for k in ("tpr", "fpr")}
                        row[f"{split}{tag}_median_delay"] = float(np.median(delays)) if len(delays) else None
                results.append(row)
    results.sort(key=lambda r: (-r["tune_tpr"], r["tune_fpr"]))
    (out or run_dir / "ue_eval.json").write_text(json.dumps(results, indent=1))

    print(f"\nthreshold chosen on tune to maximise TPR subject to FPR <= {max_fpr}, grace {grace}; sorted by tune TPR")
    scored = [s + tag for s in splits if s != "tune" for tag in ("", "_frozen")[: 1 + bool(frozen)]]
    print(
        f"{'signal':<26}{'fires':<11}{'trigger':<9}{'thresh':>10}{'tuneTPR':>9}{'tuneFPR':>9}"
        + "".join(f"{s + 'TPR':>19}{s + 'FPR':>19}{'delay':>7}" for s in scored)
    )
    for r in results[:show]:
        cols = "".join(
            f"{'-':>19}{'-':>19}{'-':>7}"
            if s + "_tpr" not in r
            else f"{r[s + '_tpr']:>19.2f}{r[s + '_fpr']:>19.2f}{'-' if r[s + '_median_delay'] is None else format(r[s + '_median_delay'], '.0f'):>7}"
            for s in scored
        )
        print(
            f"{r['signal']:<26}{r['fires_when']:<11}{r['trigger']:<9}{r['threshold']:>10.4g}{r['tune_tpr']:>9.2f}{r['tune_fpr']:>9.2f}{cols}"
        )


if __name__ == "__main__":
    app()

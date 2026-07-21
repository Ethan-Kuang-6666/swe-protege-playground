"""Turn batch-generated protégé trajectories into an SFT dataset.

Reads resolved (test-passing) trajectories written by `swebench_protege.py` and verified by
`verify_protege_patches.py` (`resolved_instances.txt` in each batch dir is the only input signal used
to pick instances), strips each down to a clean OpenAI-style tool-calling message list
(system/user/assistant/tool, with `ask_expert_llm` calls intact), and writes train/eval JSONL +
parquet files. One repo is held out entirely for eval to check generalization to an unseen repo.

The parquet's `messages` column is a native nested list-of-dicts (matches verl's
`MultiTurnSFTDataset`, which reads it directly via pandas). `tool_calls[].function.arguments` is kept
as a JSON *string* (not a nested dict) inside it: `bash` and `ask_expert_llm` have differently-shaped
arguments, and Arrow requires one fixed struct schema across every element of a list column, so a dict
there gets silently null-padded with the other tool's fields once written to parquet.

There's no `tools` column at all: every trajectory was generated with the exact same 2-tool action
space (`ProtegeModel._query` always calls `litellm.completion(..., tools=[BASH_TOOL,
ASK_EXPERT_LLM_TOOL])`), so it'd be identical dead weight on every row -- and, being nested dicts with
differently-shaped `parameters.properties` between the two tools, unsafe to store as a real parquet
struct column for the same reason as `arguments` above. Training-side, `ProtegeSFTDataset` (see
protege_sft_data/protege_sft_dataset.py) sets `self.tools` directly to the static 2-tool list instead
of reading it from the dataframe.
"""

import glob
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import typer

from minisweagent.models.utils.actions_toolcall import ASK_EXPERT_LLM_TOOL, BASH_TOOL
from minisweagent.utils.log import logger

app = typer.Typer(rich_markup_mode="rich", add_completion=False)

TOOLS = [BASH_TOOL, ASK_EXPERT_LLM_TOOL]

# A handful of early oauthlib trajectories contain tool calls whose `arguments` is technically valid
# JSON but semantically garbage (e.g. degenerate model output with stray braces/commentary leaking in
# as extra keys) -- reject any tool call whose keys aren't a subset of what that tool actually accepts.
# `timeout` is allowed for `bash` even though it's not in BASH_TOOL's declared schema: real trajectories
# use it, it's just not currently documented there.
_ALLOWED_TOOL_CALL_ARGS = {"bash": {"command", "timeout"}, "ask_expert_llm": {"question", "budget_tokens"}}


def _is_well_formed(message: dict) -> bool:
    for tc in message["tool_calls"]:
        try:
            args = json.loads(tc["function"]["arguments"])
        except json.JSONDecodeError:
            return False
        if not isinstance(args, dict) or not set(args) <= _ALLOWED_TOOL_CALL_ARGS.get(tc["function"]["name"], set()):
            return False
    return True


@dataclass
class SFTExample:
    instance_id: str
    repo: str
    messages: list[dict]
    num_expert_calls: int


def _clean_message(message: dict) -> dict:
    tool_calls = [
        {
            "id": tc["id"],
            "type": "function",
            "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]},
        }
        for tc in message.get("tool_calls") or []
    ]
    return {
        "role": message["role"],
        "content": message.get("content") or "",
        "tool_calls": tool_calls,
        "tool_call_id": message.get("tool_call_id", ""),
    }


def load_example(traj_path: Path, instance_id: str) -> SFTExample | None:
    raw_messages = json.loads(traj_path.read_text())["messages"]
    messages = [_clean_message(m) for m in raw_messages if m["role"] != "exit"]
    if not all(_is_well_formed(m) for m in messages):
        return None
    num_expert_calls = sum(1 for m in messages for tc in m["tool_calls"] if tc["function"]["name"] == "ask_expert_llm")
    repo = instance_id.split(".")[0]
    return SFTExample(instance_id=instance_id, repo=repo, messages=messages, num_expert_calls=num_expert_calls)


def collect_examples(batch_dirs: list[Path]) -> list[SFTExample]:
    examples: list[SFTExample] = []
    seen_instance_ids: set[str] = set()
    n_malformed = 0
    for batch_dir in sorted(batch_dirs):
        resolved_file = batch_dir / "resolved_instances.txt"
        if not resolved_file.exists():
            logger.warning(f"No resolved_instances.txt in {batch_dir}, skipping")
            continue
        resolved_ids = [line.strip() for line in resolved_file.read_text().splitlines() if line.strip()]
        for instance_id in resolved_ids:
            if instance_id in seen_instance_ids:
                continue
            traj_path = batch_dir / instance_id / f"{instance_id}.traj.json"
            if not traj_path.exists():
                logger.warning(f"Missing trajectory for resolved instance {instance_id} in {batch_dir}")
                continue
            seen_instance_ids.add(instance_id)
            example = load_example(traj_path, instance_id)
            if example is None:
                n_malformed += 1
                logger.warning(f"Dropping {instance_id}: malformed tool-call arguments")
                continue
            examples.append(example)
    if n_malformed:
        logger.info(f"Dropped {n_malformed} trajectories with malformed tool-call arguments")
    return examples


def write_split(examples: list[SFTExample], output_dir: Path, name: str) -> None:
    import pandas as pd

    jsonl_rows = [
        {
            "instance_id": e.instance_id,
            "repo": e.repo,
            "num_expert_calls": e.num_expert_calls,
            "messages": e.messages,
            "tools": TOOLS,
        }
        for e in examples
    ]
    (output_dir / f"sft_{name}.jsonl").write_text("\n".join(json.dumps(r) for r in jsonl_rows) + "\n")

    # "messages" stays a native nested list-of-dicts (verl's MultiTurnSFTDataset reads it as such via
    # pandas). No "tools" column: it's identical on every row and unsafe to store as nested structs (see
    # module docstring), so ProtegeSFTDataset (protege_sft_dataset.py) sets it directly instead.
    parquet_rows = [
        {
            "instance_id": e.instance_id,
            "repo": e.repo,
            "num_expert_calls": e.num_expert_calls,
            "messages": e.messages,
        }
        for e in examples
    ]
    pd.DataFrame(parquet_rows).to_parquet(output_dir / f"sft_{name}.parquet")


@app.command()
def main(
    batches_glob: str = typer.Option(
        str(Path.home() / "projects" / "protege-batch-output*"),
        "--batches-glob",
        help="Glob pattern matching batch output directories",
    ),
    output_dir: Path = typer.Option(Path("sft_data"), "-o", "--output-dir", help="Where to write train/eval files"),
    holdout_repo: str = typer.Option(
        "scrapy__scrapy", "--holdout-repo", help="org__repo held out entirely for eval (empty string to disable)"
    ),
) -> None:
    """Build SFT train/eval sets from resolved protégé trajectories."""
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_dirs = [Path(p) for p in glob.glob(batches_glob) if Path(p).is_dir()]
    logger.info(f"Found {len(batch_dirs)} batch directories matching '{batches_glob}'")

    examples = collect_examples(batch_dirs)
    train = [e for e in examples if e.repo != holdout_repo]
    eval_ = [e for e in examples if e.repo == holdout_repo]

    write_split(train, output_dir, "train")
    write_split(eval_, output_dir, "eval")

    repo_counts = Counter(e.repo for e in examples)
    n_expert = sum(1 for e in examples if e.num_expert_calls > 0)
    logger.info(f"Wrote {len(train)} train / {len(eval_)} eval examples to {output_dir}")
    logger.info(f"Repo breakdown: {dict(repo_counts)}")
    logger.info(f"Expert called in {n_expert}/{len(examples)} ({n_expert / len(examples):.1%}) trajectories")


if __name__ == "__main__":
    app()

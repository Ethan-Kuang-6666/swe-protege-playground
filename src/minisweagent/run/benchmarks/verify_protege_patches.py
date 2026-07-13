"""Verify protege-generated patches against gold tests and report resolved instances.

Applies each non-empty patch in a batch's preds.json to a fresh copy of the instance's
buggy commit, restores the full test suite (removed from the working branch by SWE-smith),
and runs it. This is the authoritative check -- an instance's own `exit_status` (e.g.
"Submitted") only means a patch was produced, not that it's correct.
"""

import base64
import concurrent.futures
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import typer

from minisweagent.config import get_config_from_spec
from minisweagent.run.benchmarks.swebench_protege import DATASET_MAPPING, DEFAULT_CONFIG_FILE, get_sb_environment
from minisweagent.utils.log import logger
from minisweagent.utils.serialize import recursive_merge

app = typer.Typer(rich_markup_mode="rich", add_completion=False)


@dataclass
class VerificationResult:
    instance_id: str
    repo: str
    patch_nonempty: bool
    apply_ok: bool
    tests_passed: bool | None
    n_tests_failed: int | None
    expert_called: bool
    num_expert_calls: int
    exit_status: str
    cost: float
    num_steps: int


def _traj_metadata(output_path: Path, instance_id: str) -> dict:
    traj_path = output_path / instance_id / f"{instance_id}.traj.json"
    if not traj_path.exists():
        return {"exit_status": "", "cost": 0.0, "num_steps": 0, "num_expert_calls": 0}
    traj = json.loads(traj_path.read_text())
    messages = traj.get("messages", [])
    num_expert_calls = sum(
        1
        for m in messages
        for tc in (m.get("tool_calls") or [])
        if tc.get("function", {}).get("name") == "ask_expert_llm"
    )
    model_stats = traj.get("info", {}).get("model_stats", {})
    return {
        "exit_status": traj.get("info", {}).get("exit_status", ""),
        "cost": model_stats.get("instance_cost", 0.0),
        "num_steps": model_stats.get("api_calls", 0),
        "num_expert_calls": num_expert_calls,
    }


def _test_ids(instance: dict, field: str) -> list[str]:
    value = instance.get(field, [])
    return json.loads(value) if isinstance(value, str) else value


def verify_patch(instance: dict, patch: str, config: dict) -> tuple[bool, bool, int | None]:
    """Apply patch to a fresh environment, restore tests, run the instance's FAIL_TO_PASS + PASS_TO_PASS tests.

    Returns (apply_ok, tests_passed, n_tests_failed).
    """
    env = get_sb_environment(config, instance)
    try:
        b64 = base64.b64encode(patch.encode()).decode()
        apply_out = env.execute(
            {
                "command": f"cd /testbed && echo {b64} | base64 -d > /tmp/patch.diff && git apply /tmp/patch.diff && echo PATCH_OK"
            }
        )
        if "PATCH_OK" not in apply_out["output"]:
            return False, False, None
        env.execute(
            {"command": "cd /testbed && (git checkout main -- tests/ || git checkout master -- tests/ || true)"}
        )

        test_ids = _test_ids(instance, "FAIL_TO_PASS") + _test_ids(instance, "PASS_TO_PASS")
        ids_b64 = base64.b64encode("\n".join(test_ids).encode()).decode()
        test_out = env.execute(
            {
                "command": (
                    f"cd /testbed && echo {ids_b64} | base64 -d > /tmp/test_ids.txt && "
                    "source activate testbed && "
                    "python -m pytest $(cat /tmp/test_ids.txt) -q --tb=no -p no:cacheprovider"
                )
            },
            timeout=600,
        )
        n_failed = len(re.findall(r"^FAILED ", test_out["output"], re.MULTILINE))
        return True, test_out["returncode"] == 0, n_failed
    finally:
        env.cleanup()


def process_instance(instance: dict, patch: str, output_path: Path, config: dict) -> VerificationResult:
    instance_id = instance["instance_id"]
    meta = _traj_metadata(output_path, instance_id)
    apply_ok, tests_passed, n_failed = verify_patch(instance, patch, config)
    logger.info(f"{instance_id}: apply_ok={apply_ok} tests_passed={tests_passed} n_failed={n_failed}")
    return VerificationResult(
        instance_id=instance_id,
        repo=instance.get("repo", instance_id.split(".")[0]),
        patch_nonempty=True,
        apply_ok=apply_ok,
        tests_passed=tests_passed if apply_ok else False,
        n_tests_failed=n_failed,
        expert_called=meta["num_expert_calls"] > 0,
        num_expert_calls=meta["num_expert_calls"],
        exit_status=meta["exit_status"],
        cost=meta["cost"],
        num_steps=meta["num_steps"],
    )


@app.command()
def main(
    output: Path = typer.Option(..., "-o", "--output", help="Batch output directory containing preds.json"),
    subset: str = typer.Option("smith", "--subset", help="SWEBench subset to use or path to a dataset"),
    split: str = typer.Option("train", "--split", help="Dataset split"),
    workers: int = typer.Option(4, "-w", "--workers", help="Number of parallel verification workers"),
    config_spec: list[str] = typer.Option(
        [str(DEFAULT_CONFIG_FILE)], "-c", "--config", help="Config files used for generation"
    ),
) -> None:
    """Verify all non-empty patches in `output/preds.json` and write a resolved-instance report."""
    from datasets import load_dataset

    preds = json.loads((output / "preds.json").read_text())
    patches = {iid: d["model_patch"] for iid, d in preds.items() if d["model_patch"].strip()}
    logger.info(f"Verifying {len(patches)} non-empty patches out of {len(preds)} total instances")

    dataset_path = DATASET_MAPPING.get(subset, subset)
    instances = {
        inst["instance_id"]: inst for inst in load_dataset(dataset_path, split=split) if inst["instance_id"] in patches
    }

    config = recursive_merge(*(get_config_from_spec(spec) for spec in config_spec))

    results: list[VerificationResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_instance, instances[iid], patch, output, config): iid
            for iid, patch in patches.items()
        }
        for future in concurrent.futures.as_completed(futures):
            iid = futures[future]
            try:
                results.append(future.result())
            except Exception as e:
                logger.error(f"Error verifying {iid}: {e}", exc_info=True)
                meta = _traj_metadata(output, iid)
                results.append(
                    VerificationResult(
                        instance_id=iid,
                        repo=instances[iid].get("repo", iid.split(".")[0]),
                        patch_nonempty=True,
                        apply_ok=False,
                        tests_passed=False,
                        n_tests_failed=None,
                        expert_called=meta["num_expert_calls"] > 0,
                        num_expert_calls=meta["num_expert_calls"],
                        exit_status=meta["exit_status"],
                        cost=meta["cost"],
                        num_steps=meta["num_steps"],
                    )
                )

    csv_path = output / "verification_results.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(VerificationResult.__dataclass_fields__.keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(r.__dict__)

    resolved = [r.instance_id for r in results if r.tests_passed]
    (output / "resolved_instances.txt").write_text("\n".join(resolved) + "\n")

    n_expert = sum(1 for r in results if r.expert_called)
    logger.info(f"Resolved: {len(resolved)}/{len(results)} ({len(resolved) / len(results) * 100:.1f}%)")
    logger.info(f"Called expert: {n_expert}/{len(results)} ({n_expert / len(results) * 100:.1f}%)")
    logger.info(f"Wrote {csv_path} and {output / 'resolved_instances.txt'}")


if __name__ == "__main__":
    app()

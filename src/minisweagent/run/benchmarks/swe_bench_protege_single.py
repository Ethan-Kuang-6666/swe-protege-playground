"""Run swe-protege on a single swe-bench instance"""

from pathlib import Path

import typer
from datasets import load_dataset

from minisweagent import global_config_dir
from minisweagent.agents.protege import ProtegeAgent
from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.models.expert_model import ExpertModel
from minisweagent.models.protege_model import ProtegeModel
from minisweagent.run.benchmarks.swebench import (
    DATASET_MAPPING,
    get_sb_environment,
)
from minisweagent.utils.log import logger
from minisweagent.utils.serialize import UNSET, recursive_merge

DEFAULT_OUTPUT_FILE = global_config_dir / "last_swebench_protege_single_run.traj.json"
DEFAULT_CONFIG_FILE = builtin_config_dir / "benchmarks" / "protege_traj_generation.yaml"

app = typer.Typer(rich_markup_mode="rich", add_completion=False)

_CONFIG_SPEC_HELP_TEXT = """Path to config files, filenames, or key-value pairs.

[bold red]IMPORTANT:[/bold red] [red]If you set this option, the default config file will not be used.[/red]
So you need to explicitly set it e.g., with [bold green]-c swebench.yaml <other options>[/bold green]

Multiple configs will be recursively merged."""


# fmt: off
@app.command()
def main(
    subset: str = typer.Option("lite", "--subset", help="SWEBench subset to use or path to a dataset", rich_help_panel="Data selection"),
    split: str = typer.Option("dev", "--split", help="Dataset split", rich_help_panel="Data selection"),
    instance_spec: str = typer.Option(0, "-i", "--instance", help="SWE-Bench instance ID or index", rich_help_panel="Data selection"),
    model_name: str | None = typer.Option(None, "-m", "--model", help="Model to use", rich_help_panel="Basic"),
    expert_model_l1_name: str | None = typer.Option(None, "--l1", help="l1 (weaker) expert model", rich_help_panel="Basic"),
    expert_model_l2_name: str | None = typer.Option(None, "--l2", help="L2 (stronger) expert model", rich_help_panel="Basic"),
    environment_class: str | None = typer.Option(None, "--environment-class", help="Environment class to use (e.g., 'docker' or 'minisweagent.environments.docker.DockerEnvironment')", rich_help_panel="Advanced"),
    cost_limit: float | None = typer.Option(None, "-l", "--cost-limit", help="Cost limit. Set to 0 to disable."),
    config_spec: list[str] = typer.Option([str(DEFAULT_CONFIG_FILE)], "-c", "--config", help=_CONFIG_SPEC_HELP_TEXT, rich_help_panel="Basic"),
    exit_immediately: bool = typer.Option(False, "--exit-immediately", help="Exit immediately when the agent wants to finish instead of prompting.", rich_help_panel="Advanced"),
    output: Path | None = typer.Option(DEFAULT_OUTPUT_FILE, "-o", "--output", help="Output trajectory file", rich_help_panel="Basic"),
) -> None:
    # fmt: on
    """Run on a single SWE-Bench instance."""
    dataset_path = DATASET_MAPPING.get(subset, subset)
    logger.info(f"Loading dataset from {dataset_path}, split {split}...")
    instances = {
        inst["instance_id"]: inst  # type: ignore
        for inst in load_dataset(dataset_path, split=split)
    }
    if instance_spec.isnumeric():
        instance_spec = sorted(instances.keys())[int(instance_spec)]
    instance: dict = instances[instance_spec]  # type: ignore

    logger.info(f"Building agent config from specs: {config_spec}")
    configs = [get_config_from_spec(spec) for spec in config_spec]
    configs.append({
        "agent": {
            "cost_limit": cost_limit if cost_limit is not None else UNSET,
            "confirm_exit": False if exit_immediately else UNSET,
            "output_path": output or UNSET,
        },
        "model": {
            "model_name": model_name or UNSET,
        },
        "expert_model_l1": {
            "model_name": expert_model_l1_name or UNSET
        },
        "expert_model_l2": {
            "model_name": expert_model_l2_name or UNSET
        },
        "environment": {
            "environment_class": environment_class or UNSET,
        },
    })
    config = recursive_merge(*configs)

    env = get_sb_environment(config, instance)
    agent = ProtegeAgent(
        ProtegeModel(**config.get("model", {})),
        [
            ExpertModel(**config.get("expert_model_l1", {})),
            ExpertModel(**config.get("expert_model_l2", {})),
        ],
        env,
        **config["agent"],
    )
    agent.run(instance["problem_statement"], ground_truth_patch=instance.get("patch", ""))


if __name__ == "__main__":
    app()

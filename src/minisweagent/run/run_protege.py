"""
Use this file to run the swe-protege.
"""

import logging
import os
from pathlib import Path

import typer
import yaml
from minisweagent import package_dir
from minisweagent.agents.protege import ProtegeAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.expert_model import ExpertModel
from minisweagent.models.protege_model import ProtegeModel

app = typer.Typer()

@app.command()
def main(
    task: str = typer.Option(..., "-t", "--task", help="Task/problem statement", show_default=False, prompt=True),
    model_name: str = typer.Option(
        os.getenv("MSWEA_MODEL_NAME"),
        "-m",
        "--model",
        help="Model name (defaults to MSWEA_MODEL_NAME env var)",
        prompt="What model do you want to use?",
    ),
    expert_model_name: str = typer.Option(
        os.getenv("MSWEA_EXPERT_MODEL_NAME"),
        "-e",
        "--expert-model",
        help="Expert model name (defaults to MSWEA_EXPERT_MODEL_NAME env var)",
        prompt="What model do you want to use?",
    ),
) -> ProtegeAgent:
    logging.basicConfig(level=logging.DEBUG)
    config = yaml.safe_load(Path(package_dir / "config" / "protege.yaml").read_text())
    agent = ProtegeAgent(
        ProtegeModel(model_name=model_name, **config.get("model", {})),
        ExpertModel(model_name=expert_model_name, **config.get("expert_model", {})),
        LocalEnvironment(**config.get("environment", {})),
        **config["agent"],
    )
    
    agent.run(task)
    return agent


if __name__ == "__main__":
    app()


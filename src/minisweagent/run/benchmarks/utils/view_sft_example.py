"""Render one SFT example (from sft_train.jsonl / sft_eval.jsonl) as a readable markdown transcript."""

import json
from pathlib import Path

import typer

app = typer.Typer(rich_markup_mode="rich", add_completion=False)


def _format_tool_call(tc: dict) -> str:
    args = json.loads(tc["function"]["arguments"])
    return f"**→ `{tc['function']['name']}`**\n```json\n{json.dumps(args, indent=2)}\n```"


def render_example(example: dict, max_content_chars: int = 4000) -> str:
    lines = [
        f"# {example['instance_id']}",
        f"repo: `{example['repo']}`  |  expert calls: {example['num_expert_calls']}",
        "",
    ]
    for message in example["messages"]:
        lines.append(f"---\n**[{message['role'].upper()}]**")
        content = message["content"]
        if len(content) > max_content_chars:
            content = content[:max_content_chars] + f"\n...[{len(content) - max_content_chars} chars truncated]..."
        if content:
            lines.append(content)
        for tool_call in message["tool_calls"]:
            lines.append(_format_tool_call(tool_call))
        lines.append("")
    return "\n".join(lines)


def _load_example(jsonl_path: Path, index: int, instance_id: str) -> dict:
    for i, line in enumerate(jsonl_path.read_text().splitlines()):
        row = json.loads(line)
        if instance_id:
            if row["instance_id"] == instance_id:
                return row
        elif i == index:
            return row
    raise ValueError(f"No matching example found in {jsonl_path}")


@app.command()
def main(
    jsonl_path: Path = typer.Argument(..., help="Path to sft_train.jsonl or sft_eval.jsonl"),
    index: int = typer.Option(0, "--index", help="Row index to render (ignored if --instance-id is set)"),
    instance_id: str = typer.Option("", "--instance-id", help="Render this specific instance_id instead"),
    output: Path | None = typer.Option(None, "-o", "--output", help="Write markdown to this file instead of stdout"),
) -> None:
    """Render one SFT example as a readable markdown transcript."""
    example = _load_example(jsonl_path, index, instance_id)
    markdown = render_example(example)
    if output:
        output.write_text(markdown)
        print(f"Wrote {output}")
    else:
        print(markdown)


if __name__ == "__main__":
    app()

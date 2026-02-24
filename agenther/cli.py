"""Command-line interface for AgentHER."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from agenther.models import FailedTrajectory, OutputFormat
from agenther.pipeline import AgentHERPipeline, PipelineConfig

console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@click.group()
@click.version_option(package_name="agenther")
def main() -> None:
    """AgentHER — Hindsight Experience Replay for LLM Agents."""


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-o", "--output", "output_path", default=None, help="Output JSONL path")
@click.option("-m", "--model", default="gpt-4o", help="LLM model name")
@click.option("--base-url", default=None, help="OpenAI-compatible API base URL")
@click.option("--api-key", default=None, help="API key (or set OPENAI_API_KEY env var)")
@click.option(
    "-f", "--format",
    "output_format",
    type=click.Choice(["sft", "dpo", "sharegpt"]),
    default="sft",
    help="Output data format",
)
@click.option("--min-confidence", default=0.5, type=float, help="Minimum relabeling confidence")
@click.option("--llm-detector/--rule-detector", default=False, help="Use LLM for failure detection")
@click.option(
    "--llm-extractor/--rule-extractor", default=True, help="Use LLM for outcome extraction"
)
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging")
def run(
    input_file: str,
    output_path: str | None,
    model: str,
    base_url: str | None,
    api_key: str | None,
    output_format: str,
    min_confidence: float,
    llm_detector: bool,
    llm_extractor: bool,
    verbose: bool,
) -> None:
    """Process failed trajectories from INPUT_FILE and generate augmented training data."""
    _setup_logging(verbose)

    trajectories = _load_trajectories(input_file)
    if not trajectories:
        console.print("[red]No trajectories loaded. Check your input file.[/red]")
        sys.exit(1)

    console.print(f"[bold green]Loaded {len(trajectories)} trajectories[/bold green]")

    config = PipelineConfig(
        model=model,
        base_url=base_url,
        api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        use_llm_detector=llm_detector,
        use_llm_extractor=llm_extractor,
        output_format=OutputFormat(output_format),
        min_confidence=min_confidence,
    )

    pipeline = AgentHERPipeline(config)
    results, saved_path = pipeline.run_and_save(trajectories, output_path)

    # Summary table
    table = Table(title="AgentHER Results")
    table.add_column("Trajectory ID", style="cyan")
    table.add_column("Stage", style="magenta")
    table.add_column("Status", style="green")
    table.add_column("Error", style="red", max_width=50)

    for r in results:
        status = "[green]✓[/green]" if r.success else "[red]✗[/red]"
        table.add_row(r.trajectory_id, r.stage_reached, status, r.error or "")

    console.print(table)

    success_count = sum(1 for r in results if r.success)
    console.print(
        f"\n[bold]Summary:[/bold] {success_count}/{len(results)} trajectories relabeled"
    )
    if saved_path:
        console.print(f"[bold]Output:[/bold] {saved_path}")


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-v", "--verbose", is_flag=True)
def validate(input_file: str, verbose: bool) -> None:
    """Validate the format of an input trajectories file."""
    _setup_logging(verbose)

    trajectories = _load_trajectories(input_file)
    if trajectories:
        console.print(
            f"[green]✓ Valid: {len(trajectories)} trajectories parsed successfully[/green]"
        )
        for t in trajectories[:3]:
            preview = t.original_prompt[:80] + "..." if len(t.original_prompt) > 80 else t.original_prompt
            console.print(f"  • {t.trajectory_id}: {preview}")
    else:
        console.print("[red]✗ Invalid or empty input file[/red]")
        sys.exit(1)


def _load_trajectories(path: str) -> list[FailedTrajectory]:
    """Load trajectories from a JSON or JSONL file."""
    p = Path(path)
    content = p.read_text(encoding="utf-8").strip()

    trajectories: list[FailedTrajectory] = []

    if content.startswith("["):
        raw_list = json.loads(content)
        for item in raw_list:
            trajectories.append(FailedTrajectory.model_validate(item))
    else:
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            trajectories.append(FailedTrajectory.model_validate_json(line))

    return trajectories


if __name__ == "__main__":
    main()

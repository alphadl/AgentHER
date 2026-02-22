"""Quick-start example: run AgentHER on sample trajectories.

Usage:
    # Rule-based mode (no LLM cost, for testing the pipeline structure)
    python examples/run_example.py --rule-based

    # Full LLM mode (requires OPENAI_API_KEY)
    python examples/run_example.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from agenther.data_augmenter import DataAugmenter
from agenther.failure_detector import FailureDetector
from agenther.models import FailedTrajectory, OutputFormat
from agenther.outcome_extractor import OutcomeExtractor
from agenther.pipeline import AgentHERPipeline, PipelineConfig

console = Console()

EXAMPLE_FILE = Path(__file__).parent / "example_trajectories.json"


def run_step_by_step(rule_based: bool = True) -> None:
    """Demonstrate each pipeline stage individually."""
    console.rule("[bold blue]AgentHER Step-by-Step Demo")

    with open(EXAMPLE_FILE, encoding="utf-8") as f:
        raw = json.load(f)
    trajectories = [FailedTrajectory.model_validate(t) for t in raw]

    trajectory = trajectories[0]
    console.print(f"\n[cyan]Trajectory:[/cyan] {trajectory.trajectory_id}")
    console.print(f"[cyan]Original Prompt:[/cyan] {trajectory.original_prompt}")
    console.print(f"[cyan]Failure Reason:[/cyan] {trajectory.failure_reason}\n")

    # Stage 1: Failure Detection
    console.rule("Stage 1: Failure Detection")
    detector = FailureDetector()
    analysis = detector.detect(trajectory, use_llm=False)
    console.print(f"  Is failure: {analysis.is_failure}")
    console.print(f"  Type: {analysis.failure_type}")
    console.print(f"  Severity: {analysis.severity:.2f}")
    console.print(f"  Recoverable: {analysis.recoverable}")

    # Stage 2: Outcome Extraction
    console.rule("Stage 2: Outcome Extraction")
    extractor = OutcomeExtractor()
    outcome = extractor.extract(trajectory, use_llm=False)
    console.print("  Achievements:")
    for a in outcome.actual_achievements:
        console.print(f"    • {a}")
    console.print(f"  Limitations: {outcome.limitations}")

    if rule_based:
        console.print("\n[yellow]Skipping LLM stages (--rule-based mode)[/yellow]")
        console.print("Run without --rule-based to use the full LLM pipeline.\n")
        return

    # Stages 3-4 require LLM
    console.rule("Stages 3-4: Full Pipeline (LLM)")
    config = PipelineConfig(use_llm_detector=False, use_llm_extractor=True)
    pipeline = AgentHERPipeline(config)
    results, saved = pipeline.run_and_save(trajectories)

    for r in results:
        status = "✓" if r.success else "✗"
        console.print(f"  [{status}] {r.trajectory_id}: {r.stage_reached}")
        if r.relabeled:
            console.print(f"      Hindsight prompt: {r.relabeled.hindsight_prompt}")

    if saved:
        console.print(f"\n[green]Output saved to {saved}[/green]")


def main() -> None:
    parser = argparse.ArgumentParser(description="AgentHER example runner")
    parser.add_argument(
        "--rule-based",
        action="store_true",
        help="Run only rule-based stages (no LLM needed)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )

    run_step_by_step(rule_based=args.rule_based)


if __name__ == "__main__":
    main()

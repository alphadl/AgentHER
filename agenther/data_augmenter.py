"""Data Augmenter — Stage 4 of the AgentHER pipeline.

Packages relabeled trajectories into standard training data formats:
  - SFT (Supervised Fine-Tuning): single chosen conversation
  - DPO (Direct Preference Optimization): chosen + rejected pair
  - ShareGPT: multi-turn conversation format
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from agenther.models import (
    AugmentedSample,
    FailedTrajectory,
    OutputFormat,
    RelabeledData,
)

logger = logging.getLogger(__name__)


class DataAugmenter:
    """Converts relabeled trajectory data into training-ready formats."""

    def to_sft(
        self,
        trajectory: FailedTrajectory,
        relabeled: RelabeledData,
    ) -> AugmentedSample:
        """Create a supervised fine-tuning sample.

        The conversation uses the hindsight prompt as user input and
        reconstructs the agent's reasoning chain as the assistant response.
        """
        assistant_content = _build_assistant_response(trajectory)

        return AugmentedSample(
            source_trajectory_id=trajectory.trajectory_id,
            format=OutputFormat.SFT,
            hindsight_prompt=relabeled.hindsight_prompt,
            chosen=[
                {"role": "user", "content": relabeled.hindsight_prompt},
                {"role": "assistant", "content": assistant_content},
            ],
            metadata={
                "original_prompt": trajectory.original_prompt,
                "relabel_confidence": relabeled.confidence,
                "num_steps": len(trajectory.steps),
            },
        )

    def to_dpo(
        self,
        trajectory: FailedTrajectory,
        relabeled: RelabeledData,
    ) -> AugmentedSample:
        """Create a DPO training pair.

        - Chosen: hindsight prompt -> trajectory (now a "success")
        - Rejected: original prompt -> same trajectory (a known "failure")
        """
        assistant_content = _build_assistant_response(trajectory)

        return AugmentedSample(
            source_trajectory_id=trajectory.trajectory_id,
            format=OutputFormat.DPO,
            hindsight_prompt=relabeled.hindsight_prompt,
            chosen=[
                {"role": "user", "content": relabeled.hindsight_prompt},
                {"role": "assistant", "content": assistant_content},
            ],
            rejected=[
                {"role": "user", "content": trajectory.original_prompt},
                {"role": "assistant", "content": assistant_content},
            ],
            metadata={
                "original_prompt": trajectory.original_prompt,
                "relabel_confidence": relabeled.confidence,
                "rationale": relabeled.rationale,
            },
        )

    def to_sharegpt(
        self,
        trajectory: FailedTrajectory,
        relabeled: RelabeledData,
    ) -> AugmentedSample:
        """Create a ShareGPT-style multi-turn conversation."""
        turns: list[dict[str, str]] = [
            {"role": "user", "content": relabeled.hindsight_prompt},
        ]

        for step in trajectory.steps:
            reasoning = f"**Thought:** {step.thought}\n**Action:** {step.action_name}"
            if step.action_input:
                reasoning += f"\n**Input:** {json.dumps(step.action_input, ensure_ascii=False)}"
            turns.append({"role": "assistant", "content": reasoning})
            turns.append({"role": "observation", "content": step.observation})

        if trajectory.final_answer:
            turns.append({"role": "assistant", "content": trajectory.final_answer})

        return AugmentedSample(
            source_trajectory_id=trajectory.trajectory_id,
            format=OutputFormat.SHAREGPT,
            hindsight_prompt=relabeled.hindsight_prompt,
            chosen=turns,
            metadata={
                "original_prompt": trajectory.original_prompt,
                "relabel_confidence": relabeled.confidence,
            },
        )

    def augment(
        self,
        trajectory: FailedTrajectory,
        relabeled: RelabeledData,
        output_format: OutputFormat = OutputFormat.SFT,
    ) -> AugmentedSample:
        """Route to the appropriate format builder."""
        builders = {
            OutputFormat.SFT: self.to_sft,
            OutputFormat.DPO: self.to_dpo,
            OutputFormat.SHAREGPT: self.to_sharegpt,
        }
        return builders[output_format](trajectory, relabeled)

    @staticmethod
    def save_samples(
        samples: list[AugmentedSample],
        output_path: str | Path,
    ) -> Path:
        """Write augmented samples to a JSONL file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            for sample in samples:
                f.write(sample.model_dump_json() + "\n")

        logger.info("Saved %d samples to %s", len(samples), path)
        return path


def _build_assistant_response(trajectory: FailedTrajectory) -> str:
    """Reconstruct a coherent assistant response from trajectory steps."""
    parts: list[str] = []

    for i, step in enumerate(trajectory.steps, 1):
        parts.append(f"Step {i}: {step.thought}")
        action_args = json.dumps(step.action_input, ensure_ascii=False)
        parts.append(f"  Action: {step.action_name}({action_args})")
        obs_preview = step.observation[:300] if len(step.observation) > 300 else step.observation
        parts.append(f"  Result: {obs_preview}")

    if trajectory.final_answer:
        parts.append(f"\nFinal Answer: {trajectory.final_answer}")

    return "\n".join(parts)

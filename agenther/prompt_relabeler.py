"""Prompt Relabeler — Stage 3 of the AgentHER pipeline.

Reverse-engineers a natural user prompt such that the existing (failed)
trajectory becomes a successful demonstration of that new prompt.

This is the core intellectual contribution of AgentHER: the insight that
a trajectory that fails Goal A might perfectly succeed for Goal B.
"""

from __future__ import annotations

import logging

from agenther.llm_client import LLMClient
from agenther.models import FailedTrajectory, RelabeledData, ReplayOutcome
from agenther.prompts import PROMPT_RELABEL_SYSTEM, PROMPT_RELABEL_USER

logger = logging.getLogger(__name__)


class PromptRelabeler:
    """Generates hindsight prompts that transform failures into successes."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    def relabel(
        self,
        trajectory: FailedTrajectory,
        outcome: ReplayOutcome,
    ) -> RelabeledData:
        """Generate a hindsight prompt for a single trajectory.

        The new prompt is crafted so that:
        1. A human could plausibly have written it
        2. The trajectory's observations genuinely satisfy it
        3. The complexity matches the original prompt's style
        """
        user_prompt = PROMPT_RELABEL_USER.render(
            achievements=outcome.actual_achievements,
            observations=outcome.key_observations,
            original_prompt=trajectory.original_prompt,
            num_steps=len(trajectory.steps),
        )

        result = self._llm.call_structured(
            system_prompt=PROMPT_RELABEL_SYSTEM,
            user_prompt=user_prompt,
            output_schema=RelabeledData,
        )
        return result.model_copy(update={"trajectory_id": trajectory.trajectory_id})

    def relabel_with_validation(
        self,
        trajectory: FailedTrajectory,
        outcome: ReplayOutcome,
        min_confidence: float = 0.5,
        max_attempts: int = 3,
    ) -> RelabeledData | None:
        """Relabel with retry logic — discard low-confidence relabelings.

        Returns None if no attempt reaches the minimum confidence threshold.
        """
        best: RelabeledData | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                result = self.relabel(trajectory, outcome)

                if not result.is_valid_replay:
                    logger.info(
                        "Attempt %d/%d for %s: marked as invalid replay",
                        attempt, max_attempts, trajectory.trajectory_id,
                    )
                    continue

                if result.confidence >= min_confidence:
                    return result

                if best is None or result.confidence > best.confidence:
                    best = result

                logger.info(
                    "Attempt %d/%d for %s: confidence %.2f < %.2f threshold",
                    attempt, max_attempts, trajectory.trajectory_id,
                    result.confidence, min_confidence,
                )
            except Exception:
                logger.exception(
                    "Attempt %d/%d for %s failed",
                    attempt, max_attempts, trajectory.trajectory_id,
                )

        if best and best.confidence >= min_confidence * 0.8:
            logger.warning(
                "Accepting best-effort relabeling for %s (confidence=%.2f)",
                trajectory.trajectory_id, best.confidence,
            )
            return best

        return None

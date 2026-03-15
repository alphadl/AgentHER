"""Prompt Relabeler — Stage 3 of the AgentHER pipeline.

Reverse-engineers a natural user prompt such that the existing (failed)
trajectory becomes a successful demonstration of that new prompt.

Implements Algorithm 1 from the paper:
  - Temperature 0.3 on first attempt, 0.7 on retries (diversity)
  - Multi-judge verification: when c >= θ, a second independent LLM call
    at temperature=0 must also agree (c_2 >= θ); accepted confidence is
    the average (c + c_2) / 2
  - Fallback: retain best single-judge attempt if c >= 0.8θ
"""

from __future__ import annotations

import logging

from agenther.llm_client import LLMClient
from agenther.models import (
    FailedTrajectory,
    RelabeledData,
    ReplayOutcome,
    SecondJudgeVerdict,
)
from agenther.prompts import (
    PROMPT_RELABEL_SYSTEM,
    PROMPT_RELABEL_USER,
    SECOND_JUDGE_SYSTEM,
    SECOND_JUDGE_USER,
    steps_for_prompt,
)

logger = logging.getLogger(__name__)


class PromptRelabeler:
    """Generates hindsight prompts that transform failures into successes."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    def relabel(
        self,
        trajectory: FailedTrajectory,
        outcome: ReplayOutcome,
        temperature: float | None = None,
    ) -> RelabeledData:
        """Generate a hindsight prompt for a single trajectory.

        Args:
            trajectory: The failed trajectory to relabel.
            outcome:    Extracted achievements from Stage 2.
            temperature: LLM temperature override; None falls back to client default.

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
            temperature=temperature,
        )
        return result.model_copy(update={"trajectory_id": trajectory.trajectory_id})

    def _second_judge(
        self,
        hindsight_prompt: str,
        trajectory: FailedTrajectory,
    ) -> SecondJudgeVerdict:
        """Independent verification call at temperature=0 (deterministic).

        Implements the multi-judge step from Algorithm 1: a separate LLM
        call that must also assign confidence >= θ before the relabeling is
        accepted.
        """
        user_prompt = SECOND_JUDGE_USER.render(
            hindsight_prompt=hindsight_prompt,
            steps=steps_for_prompt(trajectory.steps),
            num_steps=len(trajectory.steps),
            final_answer=trajectory.final_answer,
        )
        return self._llm.call_structured(
            system_prompt=SECOND_JUDGE_SYSTEM,
            user_prompt=user_prompt,
            output_schema=SecondJudgeVerdict,
            temperature=0.0,
        )

    def relabel_with_validation(
        self,
        trajectory: FailedTrajectory,
        outcome: ReplayOutcome,
        min_confidence: float = 0.5,
        max_attempts: int = 3,
    ) -> RelabeledData | None:
        """Relabel with retry, temperature escalation, and multi-judge verification.

        Algorithm (from paper §3.3):
          - Attempt 1: temperature=0.3; attempts 2+: temperature=0.7
          - When b=1 and c >= θ: run second judge at temperature=0
            - If c_2 >= θ: accept with merged confidence (c + c_2) / 2
          - Track best b=1 attempt across all tries
          - Fallback: accept best if c >= 0.8θ

        Returns None if no attempt meets the 0.8θ fallback threshold.
        """
        best: RelabeledData | None = None

        for attempt in range(1, max_attempts + 1):
            # Temperature escalation: first attempt explores less, retries more
            temperature = 0.3 if attempt == 1 else 0.7
            try:
                result = self.relabel(trajectory, outcome, temperature=temperature)

                if not result.is_valid_replay:
                    logger.info(
                        "Attempt %d/%d for %s: marked as invalid replay",
                        attempt, max_attempts, trajectory.trajectory_id,
                    )
                    continue

                # Track best valid (b=1) attempt for the 0.8θ fallback
                if best is None or result.confidence > best.confidence:
                    best = result

                if result.confidence >= min_confidence:
                    # Multi-judge verification
                    try:
                        verdict = self._second_judge(result.hindsight_prompt, trajectory)
                        if verdict.confidence >= min_confidence:
                            merged = (result.confidence + verdict.confidence) / 2
                            logger.info(
                                "Attempt %d/%d for %s: accepted (c=%.2f, c2=%.2f → %.2f)",
                                attempt, max_attempts, trajectory.trajectory_id,
                                result.confidence, verdict.confidence, merged,
                            )
                            return result.model_copy(update={"confidence": merged})
                        logger.info(
                            "Attempt %d/%d for %s: second judge rejected (c2=%.2f < θ=%.2f)",
                            attempt, max_attempts, trajectory.trajectory_id,
                            verdict.confidence, min_confidence,
                        )
                    except Exception:
                        logger.warning(
                            "Attempt %d/%d for %s: second judge failed, will retry",
                            attempt, max_attempts, trajectory.trajectory_id,
                        )
                else:
                    logger.info(
                        "Attempt %d/%d for %s: confidence %.2f < θ=%.2f",
                        attempt, max_attempts, trajectory.trajectory_id,
                        result.confidence, min_confidence,
                    )

            except Exception:
                logger.exception(
                    "Attempt %d/%d for %s failed",
                    attempt, max_attempts, trajectory.trajectory_id,
                )

        # Fallback: accept best single-judge result if c >= 0.8θ
        if best and best.confidence >= min_confidence * 0.8:
            logger.warning(
                "Accepting best-effort relabeling for %s (confidence=%.2f, fallback 0.8θ=%.2f)",
                trajectory.trajectory_id, best.confidence, min_confidence * 0.8,
            )
            return best

        return None

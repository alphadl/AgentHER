"""Failure Detector — Stage 1 of the AgentHER pipeline.

Determines whether a trajectory is a genuine failure relative to the original
prompt, classifies the failure type, and assesses recoverability for hindsight
relabeling.

Supports two modes:
  - Rule-based: keyword/heuristic matching (fast, no LLM cost)
  - LLM-based:  uses an LLM judge for nuanced evaluation
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from agenther.llm_client import LLMClient
from agenther.models import FailedTrajectory, FailureAnalysis, FailureType
from agenther.prompts import FAILURE_DETECTION_SYSTEM, FAILURE_DETECTION_USER

logger = logging.getLogger(__name__)

_RULE_BASED_FAILURE_KEYWORDS: dict[FailureType, list[str]] = {
    FailureType.TOOL_ERROR: [
        "error",
        "exception",
        "traceback",
        "timeout",
        "connection refused",
        "404",
        "500",
    ],
    FailureType.INCOMPLETE: [
        "i could not",
        "unable to",
        "i don't know",
        "no results",
        "failed to",
    ],
    FailureType.HALLUCINATION: [
        "i believe",
        "i think",
        "probably",
        "it might be",
    ],
    FailureType.CONSTRAINT_VIOLATION: [
        "exceeds",
        "over budget",
        "does not match",
        "outside the range",
    ],
}


class FailureDetector:
    """Detects and classifies trajectory failures."""

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm = llm_client

    # ------------------------------------------------------------------
    # Rule-based detection (zero LLM cost)
    # ------------------------------------------------------------------

    def detect_rule_based(self, trajectory: FailedTrajectory) -> FailureAnalysis:
        """Heuristic failure detection using keyword matching on observations."""
        all_text = _aggregate_text(trajectory).lower()

        matched_type: FailureType | None = None
        max_hits = 0

        for ftype, keywords in _RULE_BASED_FAILURE_KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw in all_text)
            if hits > max_hits:
                max_hits = hits
                matched_type = ftype

        if trajectory.failure_reason:
            matched_type = matched_type or FailureType.WRONG_RESULT

        has_useful_observations = any(
            len(step.observation.strip()) > 20 for step in trajectory.steps
        )

        return FailureAnalysis(
            is_failure=True,
            failure_type=matched_type,
            severity=min(1.0, 0.3 + 0.1 * max_hits),
            explanation=trajectory.failure_reason or "Detected via rule-based heuristics",
            recoverable=has_useful_observations and matched_type != FailureType.TOOL_ERROR,
        )

    # ------------------------------------------------------------------
    # LLM-based detection (high accuracy)
    # ------------------------------------------------------------------

    def detect_llm(self, trajectory: FailedTrajectory) -> FailureAnalysis:
        """Use an LLM judge to analyze the trajectory failure in detail."""
        if self._llm is None:
            raise RuntimeError("LLM client required for LLM-based detection")

        user_prompt = FAILURE_DETECTION_USER.render(
            original_prompt=trajectory.original_prompt,
            steps=trajectory.steps,
            num_steps=len(trajectory.steps),
            final_answer=trajectory.final_answer,
            failure_reason=trajectory.failure_reason,
        )

        return self._llm.call_structured(
            system_prompt=FAILURE_DETECTION_SYSTEM,
            user_prompt=user_prompt,
            output_schema=FailureAnalysis,
        )

    # ------------------------------------------------------------------
    # Unified interface
    # ------------------------------------------------------------------

    def detect(
        self, trajectory: FailedTrajectory, use_llm: bool = False
    ) -> FailureAnalysis:
        """Detect failure using the selected strategy."""
        if use_llm:
            return self.detect_llm(trajectory)
        return self.detect_rule_based(trajectory)

    def batch_detect(
        self,
        trajectories: Sequence[FailedTrajectory],
        use_llm: bool = False,
    ) -> list[FailureAnalysis]:
        """Detect failures across a batch of trajectories."""
        results = []
        for traj in trajectories:
            try:
                results.append(self.detect(traj, use_llm=use_llm))
            except Exception:
                logger.exception("Failed to analyze trajectory %s", traj.trajectory_id)
                results.append(
                    FailureAnalysis(
                        is_failure=True,
                        failure_type=FailureType.TOOL_ERROR,
                        severity=1.0,
                        explanation="Analysis failed due to an internal error",
                        recoverable=False,
                    )
                )
        return results


def _aggregate_text(trajectory: FailedTrajectory) -> str:
    """Concatenate all textual content from a trajectory for keyword search."""
    parts = [trajectory.original_prompt, trajectory.failure_reason, trajectory.final_answer]
    for step in trajectory.steps:
        parts.extend([step.thought, step.observation])
    return " ".join(parts)

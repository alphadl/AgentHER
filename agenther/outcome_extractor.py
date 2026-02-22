"""Outcome Extractor — Stage 2 of the AgentHER pipeline.

Analyzes the terminal state and observation history of a failed trajectory
to summarize what was *actually* achieved, regardless of the original goal.

Supports:
  - Rule-based extraction: pulls structured facts from observations
  - LLM-based extraction: uses an LLM for nuanced understanding
"""

from __future__ import annotations

import logging
import re

from agenther.llm_client import LLMClient
from agenther.models import FailedTrajectory, ReplayOutcome
from agenther.prompts import OUTCOME_EXTRACTION_SYSTEM, OUTCOME_EXTRACTION_USER

logger = logging.getLogger(__name__)


class OutcomeExtractor:
    """Extracts actual achievements from failed trajectories."""

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm = llm_client

    def extract_rule_based(self, trajectory: FailedTrajectory) -> ReplayOutcome:
        """Heuristic extraction: treats each non-trivial observation as an achievement."""
        achievements: list[str] = []
        key_observations: list[str] = []

        for i, step in enumerate(trajectory.steps, 1):
            obs = step.observation.strip()
            if len(obs) < 15:
                continue

            if _looks_like_error(obs):
                continue

            summary = _truncate(obs, max_len=200)
            achievements.append(
                f"Step {i} ({step.action_name}): {summary}"
            )

            numbers = re.findall(r"\$?[\d,]+\.?\d*", obs)
            if numbers:
                key_observations.append(
                    f"Numeric data found in step {i}: {', '.join(numbers[:5])}"
                )

        if not achievements:
            achievements = ["Agent attempted actions but produced no substantive results"]

        limitations = trajectory.failure_reason or "Original goal was not met"

        return ReplayOutcome(
            actual_achievements=achievements,
            key_observations=key_observations,
            limitations=limitations,
        )

    def extract_llm(self, trajectory: FailedTrajectory) -> ReplayOutcome:
        """Use an LLM to produce a nuanced summary of actual outcomes."""
        if self._llm is None:
            raise RuntimeError("LLM client required for LLM-based extraction")

        user_prompt = OUTCOME_EXTRACTION_USER.render(
            original_prompt=trajectory.original_prompt,
            steps=trajectory.steps,
            final_answer=trajectory.final_answer,
        )

        return self._llm.call_structured(
            system_prompt=OUTCOME_EXTRACTION_SYSTEM,
            user_prompt=user_prompt,
            output_schema=ReplayOutcome,
        )

    def extract(
        self, trajectory: FailedTrajectory, use_llm: bool = False
    ) -> ReplayOutcome:
        if use_llm:
            return self.extract_llm(trajectory)
        return self.extract_rule_based(trajectory)


def _looks_like_error(text: str) -> bool:
    """Quick check if an observation is purely an error message."""
    error_signals = ["traceback", "error:", "exception:", "errno", "status code: 5"]
    lower = text.lower()
    return any(sig in lower for sig in error_signals)


def _truncate(text: str, max_len: int = 200) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit(" ", 1)[0] + "..."

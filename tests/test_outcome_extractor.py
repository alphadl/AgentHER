"""Tests for the Outcome Extractor module."""

from __future__ import annotations

from agenther.models import AgentStep, FailedTrajectory
from agenther.outcome_extractor import OutcomeExtractor


class TestRuleBasedExtraction:
    def test_extracts_achievements_from_observations(
        self, sample_trajectory: FailedTrajectory
    ) -> None:
        extractor = OutcomeExtractor()
        outcome = extractor.extract(sample_trajectory, use_llm=False)
        assert len(outcome.actual_achievements) > 0
        assert outcome.limitations != ""

    def test_filters_error_observations(self, crash_trajectory: FailedTrajectory) -> None:
        extractor = OutcomeExtractor()
        outcome = extractor.extract(crash_trajectory, use_llm=False)
        assert any("no substantive" in a.lower() for a in outcome.actual_achievements)

    def test_extracts_numeric_data(self) -> None:
        trajectory = FailedTrajectory(
            trajectory_id="numeric",
            original_prompt="Find prices",
            steps=[
                AgentStep(
                    thought="searching",
                    action_name="search",
                    action_input={},
                    observation="Product costs $42.99 with shipping $5.00 to 90210",
                ),
            ],
            failure_reason="wrong product",
        )
        extractor = OutcomeExtractor()
        outcome = extractor.extract(trajectory, use_llm=False)
        assert len(outcome.key_observations) > 0
        assert any("42.99" in obs for obs in outcome.key_observations)

    def test_handles_short_observations(self) -> None:
        trajectory = FailedTrajectory(
            trajectory_id="short",
            original_prompt="Do task",
            steps=[
                AgentStep(
                    thought="ok",
                    action_name="noop",
                    action_input={},
                    observation="OK",
                ),
            ],
            failure_reason="too short",
        )
        extractor = OutcomeExtractor()
        outcome = extractor.extract(trajectory, use_llm=False)
        assert len(outcome.actual_achievements) >= 1

"""Tests for Pydantic data models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agenther.models import (
    AgentStep,
    AugmentedSample,
    FailedTrajectory,
    FailureAnalysis,
    OutputFormat,
    RelabeledData,
    ReplayOutcome,
)


class TestAgentStep:
    def test_basic_creation(self) -> None:
        step = AgentStep(
            thought="thinking",
            action_name="search",
            action_input={"q": "test"},
            observation="result",
        )
        assert step.action_name == "search"
        assert step.action_input == {"q": "test"}

    def test_empty_action_input_default(self) -> None:
        step = AgentStep(thought="t", action_name="a", action_input={}, observation="o")
        assert step.action_input == {}


class TestFailedTrajectory:
    def test_auto_id_generation(self) -> None:
        t = FailedTrajectory(
            original_prompt="test",
            steps=[AgentStep(thought="t", action_name="a", action_input={}, observation="o")],
            failure_reason="failed",
        )
        assert len(t.trajectory_id) == 12

    def test_requires_at_least_one_step(self) -> None:
        with pytest.raises(ValidationError):
            FailedTrajectory(
                original_prompt="test",
                steps=[],
                failure_reason="failed",
            )

    def test_serialization_roundtrip(self, sample_trajectory: FailedTrajectory) -> None:
        json_str = sample_trajectory.model_dump_json()
        restored = FailedTrajectory.model_validate_json(json_str)
        assert restored.trajectory_id == sample_trajectory.trajectory_id
        assert len(restored.steps) == len(sample_trajectory.steps)


class TestReplayOutcome:
    def test_creation(self) -> None:
        outcome = ReplayOutcome(
            actual_achievements=["did X", "did Y"],
            limitations="could not do Z",
        )
        assert len(outcome.actual_achievements) == 2
        assert outcome.key_observations == []


class TestRelabeledData:
    def test_confidence_bounds(self) -> None:
        with pytest.raises(ValidationError):
            RelabeledData(
                trajectory_id="x",
                hindsight_prompt="p",
                is_valid_replay=True,
                rationale="r",
                confidence=1.5,
            )

    def test_valid_relabeling(self, sample_relabeled: RelabeledData) -> None:
        assert sample_relabeled.is_valid_replay is True
        assert 0 <= sample_relabeled.confidence <= 1


class TestFailureAnalysis:
    def test_severity_bounds(self) -> None:
        analysis = FailureAnalysis(is_failure=True, severity=0.8)
        assert analysis.severity == 0.8

        with pytest.raises(ValidationError):
            FailureAnalysis(is_failure=True, severity=2.0)


class TestAugmentedSample:
    def test_sft_sample(self, sample_trajectory: FailedTrajectory) -> None:
        sample = AugmentedSample(
            source_trajectory_id=sample_trajectory.trajectory_id,
            format=OutputFormat.SFT,
            hindsight_prompt="test prompt",
            chosen=[
                {"role": "user", "content": "test prompt"},
                {"role": "assistant", "content": "test response"},
            ],
        )
        assert sample.format == OutputFormat.SFT
        assert len(sample.chosen) == 2
        assert sample.rejected == []

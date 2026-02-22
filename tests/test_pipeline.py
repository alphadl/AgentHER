"""Tests for the AgentHER pipeline (rule-based stages only, no LLM required)."""

from __future__ import annotations

from agenther.failure_detector import FailureDetector
from agenther.models import FailedTrajectory
from agenther.outcome_extractor import OutcomeExtractor


class TestPipelineStages:
    """Integration test for the rule-based portion of the pipeline."""

    def test_detect_then_extract(self, sample_trajectory: FailedTrajectory) -> None:
        detector = FailureDetector()
        analysis = detector.detect(sample_trajectory, use_llm=False)
        assert analysis.is_failure
        assert analysis.recoverable

        extractor = OutcomeExtractor()
        outcome = extractor.extract(sample_trajectory, use_llm=False)
        assert len(outcome.actual_achievements) > 0

    def test_irrecoverable_skips_extraction(
        self, crash_trajectory: FailedTrajectory
    ) -> None:
        detector = FailureDetector()
        analysis = detector.detect(crash_trajectory, use_llm=False)
        assert analysis.is_failure
        assert not analysis.recoverable


class TestPipelineConfig:
    def test_default_config(self) -> None:
        from agenther.pipeline import PipelineConfig

        config = PipelineConfig()
        assert config.model == "gpt-4o"
        assert config.min_confidence == 0.5
        assert config.use_llm_detector is False

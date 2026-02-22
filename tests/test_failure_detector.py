"""Tests for the Failure Detector module."""

from __future__ import annotations

from agenther.failure_detector import FailureDetector
from agenther.models import FailedTrajectory, FailureType


class TestRuleBasedDetection:
    def test_detects_constraint_violation(self, sample_trajectory: FailedTrajectory) -> None:
        detector = FailureDetector()
        analysis = detector.detect(sample_trajectory, use_llm=False)
        assert analysis.is_failure is True
        assert analysis.recoverable is True

    def test_detects_tool_error(self, crash_trajectory: FailedTrajectory) -> None:
        detector = FailureDetector()
        analysis = detector.detect(crash_trajectory, use_llm=False)
        assert analysis.is_failure is True
        assert analysis.failure_type == FailureType.TOOL_ERROR
        assert analysis.recoverable is False

    def test_severity_increases_with_more_error_keywords(self) -> None:
        from agenther.models import AgentStep

        trajectory = FailedTrajectory(
            trajectory_id="multi_error",
            original_prompt="Do something",
            steps=[
                AgentStep(
                    thought="trying",
                    action_name="api",
                    action_input={},
                    observation="Error: timeout. Exception occurred. Traceback follows.",
                ),
            ],
            failure_reason="Multiple errors",
        )
        detector = FailureDetector()
        analysis = detector.detect(trajectory, use_llm=False)
        assert analysis.severity > 0.5

    def test_batch_detect(
        self,
        sample_trajectory: FailedTrajectory,
        crash_trajectory: FailedTrajectory,
    ) -> None:
        detector = FailureDetector()
        results = detector.batch_detect([sample_trajectory, crash_trajectory])
        assert len(results) == 2
        assert all(r.is_failure for r in results)

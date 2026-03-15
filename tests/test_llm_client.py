"""Tests for LLMClient and AsyncLLMClient."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agenther.llm_client import LLMClient, _parse_structured_response
from agenther.models import ReplayOutcome


class TestLLMClientTemperature:
    """temperature=0.0 must not be silently replaced by the instance default."""

    def _make_client(self, default_temp: float = 0.7) -> LLMClient:
        client = LLMClient(model="gpt-4o", api_key="test-key", temperature=default_temp)
        return client

    def test_explicit_zero_temperature_is_passed_through(self) -> None:
        client = self._make_client(default_temp=0.7)
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "hello"

        completions = client._client.chat.completions
        with patch.object(completions, "create", return_value=mock_response) as mock_create:
            client.call("sys", "user", temperature=0.0)
            _, kwargs = mock_create.call_args
            assert kwargs["temperature"] == 0.0, (
                "temperature=0.0 was silently replaced; falsy-or bug still present"
            )

    def test_none_temperature_falls_back_to_instance_default(self) -> None:
        client = self._make_client(default_temp=0.3)
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "hello"

        completions = client._client.chat.completions
        with patch.object(completions, "create", return_value=mock_response) as mock_create:
            client.call("sys", "user", temperature=None)
            _, kwargs = mock_create.call_args
            assert kwargs["temperature"] == 0.3

    def test_omitted_temperature_falls_back_to_instance_default(self) -> None:
        client = self._make_client(default_temp=0.5)
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "hello"

        completions = client._client.chat.completions
        with patch.object(completions, "create", return_value=mock_response) as mock_create:
            client.call("sys", "user")
            _, kwargs = mock_create.call_args
            assert kwargs["temperature"] == 0.5


class TestParseStructuredResponse:
    """Unit tests for JSON extraction fallback logic."""

    def test_direct_json_parse(self) -> None:
        raw = '{"actual_achievements": ["did X"], "key_observations": [], "limitations": "none"}'
        result = _parse_structured_response(raw, ReplayOutcome)
        assert result.actual_achievements == ["did X"]

    def test_json_in_markdown_fence(self) -> None:
        raw = (
            'Here is the result:\n'
            '```json\n'
            '{"actual_achievements": ["done"], "key_observations": [], "limitations": "n/a"}\n'
            '```'
        )
        result = _parse_structured_response(raw, ReplayOutcome)
        assert result.limitations == "n/a"

    def test_json_embedded_in_prose(self) -> None:
        payload = (
            '{"actual_achievements": ["found data"],'
            ' "key_observations": [], "limitations": "partial"}'
        )
        raw = f"Sure, here you go: {payload} Hope that helps."
        result = _parse_structured_response(raw, ReplayOutcome)
        assert result.limitations == "partial"

    def test_raises_on_unparseable_response(self) -> None:
        with pytest.raises(ValueError, match="Failed to parse"):
            _parse_structured_response("this is not json at all", ReplayOutcome)


class TestPublicAPIExports:
    """Confirm that community-facing types are importable from the top-level package."""

    def test_augmented_sample_importable(self) -> None:
        from agenther import AugmentedSample  # noqa: F401

    def test_failure_analysis_importable(self) -> None:
        from agenther import FailureAnalysis  # noqa: F401

    def test_pipeline_result_importable(self) -> None:
        from agenther import PipelineResult  # noqa: F401

    def test_second_judge_verdict_importable(self) -> None:
        from agenther import SecondJudgeVerdict  # noqa: F401

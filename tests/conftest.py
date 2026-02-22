"""Shared test fixtures for AgentHER."""

from __future__ import annotations

import pytest

from agenther.models import AgentStep, FailedTrajectory, RelabeledData, ReplayOutcome


@pytest.fixture()
def simple_step() -> AgentStep:
    return AgentStep(
        thought="I should search for information",
        action_name="web_search",
        action_input={"query": "test query"},
        observation="Found 3 results: A ($10), B ($20), C ($15)",
    )


@pytest.fixture()
def error_step() -> AgentStep:
    return AgentStep(
        thought="Let me try again",
        action_name="api_call",
        action_input={"url": "https://example.com"},
        observation="Error: Connection timeout. Traceback: ...",
    )


@pytest.fixture()
def sample_trajectory(simple_step: AgentStep) -> FailedTrajectory:
    return FailedTrajectory(
        trajectory_id="test_001",
        original_prompt="Find product X under $5",
        steps=[
            simple_step,
            AgentStep(
                thought="Let me check prices in detail",
                action_name="get_details",
                action_input={"product": "A"},
                observation="Product A: price $10, rating 4.5/5, in stock",
            ),
        ],
        final_answer="The cheapest option is Product A at $10, which exceeds your $5 budget.",
        failure_reason="No product found under $5",
    )


@pytest.fixture()
def crash_trajectory(error_step: AgentStep) -> FailedTrajectory:
    return FailedTrajectory(
        trajectory_id="test_crash",
        original_prompt="Look up weather in Tokyo",
        steps=[error_step],
        final_answer="",
        failure_reason="Tool API crashed",
    )


@pytest.fixture()
def sample_outcome() -> ReplayOutcome:
    return ReplayOutcome(
        actual_achievements=[
            "Found Product A at $10 with rating 4.5/5",
            "Confirmed Product A is in stock",
        ],
        key_observations=["Numeric data: $10, $20, $15"],
        limitations="No product under $5 was found",
    )


@pytest.fixture()
def sample_relabeled() -> RelabeledData:
    return RelabeledData(
        trajectory_id="test_001",
        hindsight_prompt="Find product X and compare prices across vendors",
        is_valid_replay=True,
        rationale="The trajectory successfully compared prices and found detailed product info",
        confidence=0.85,
    )

"""Core data models for AgentHER.

Defines the Pydantic schemas that flow through the entire pipeline:
  FailedTrajectory -> ReplayOutcome -> RelabeledData -> AugmentedSample
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AgentStep(BaseModel):
    """A single step in an agent's execution trajectory."""

    thought: str = Field(..., description="Agent's chain-of-thought reasoning")
    action_name: str = Field(..., description="Tool or action invoked")
    action_input: dict[str, Any] = Field(
        default_factory=dict, description="Arguments to the action"
    )
    observation: str = Field(..., description="Environment response after the action")


class FailedTrajectory(BaseModel):
    """A trajectory where the agent did NOT satisfy the original user intent."""

    trajectory_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex[:12],
        description="Unique identifier for this trajectory",
    )
    original_prompt: str = Field(..., description="The user prompt the agent was trying to fulfill")
    steps: list[AgentStep] = Field(..., min_length=1, description="Ordered list of agent steps")
    final_answer: str = Field(default="", description="Agent's final response, if any")
    failure_reason: str = Field(..., description="Why this trajectory is considered a failure")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")


class ReplayOutcome(BaseModel):
    """Summarizes what the agent *actually* achieved, regardless of the original goal."""

    actual_achievements: list[str] = Field(
        ..., description="Factual summary of what the agent actually found/did"
    )
    key_observations: list[str] = Field(
        default_factory=list,
        description="Notable observations extracted from the trajectory",
    )
    limitations: str = Field(..., description="What the agent failed to do")


class RelabeledData(BaseModel):
    """The result of hindsight relabeling: a new prompt paired with the original trajectory."""

    trajectory_id: str
    hindsight_prompt: str = Field(
        ..., description="Reverse-engineered user prompt that makes this trajectory a success"
    )
    is_valid_replay: bool = Field(
        ...,
        description="False if the trajectory is a catastrophic tool crash with no useful output",
    )
    rationale: str = Field(..., description="Why this relabeling is considered valid")
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence score for the relabeling quality"
    )


class FailureType(str, Enum):
    """Classification of trajectory failure modes."""

    CONSTRAINT_VIOLATION = "constraint_violation"
    WRONG_RESULT = "wrong_result"
    INCOMPLETE = "incomplete"
    TOOL_ERROR = "tool_error"
    HALLUCINATION = "hallucination"
    OFF_TOPIC = "off_topic"


class FailureAnalysis(BaseModel):
    """Detailed analysis of why and how a trajectory failed."""

    is_failure: bool = Field(..., description="Whether this trajectory is actually a failure")
    failure_type: FailureType | None = Field(None, description="Classification of the failure")
    severity: float = Field(
        default=0.5, ge=0.0, le=1.0, description="0=mild deviation, 1=catastrophic failure"
    )
    explanation: str = Field(default="", description="Human-readable failure explanation")
    recoverable: bool = Field(
        default=True, description="Whether hindsight relabeling can salvage this trajectory"
    )


class OutputFormat(str, Enum):
    """Supported output formats for augmented training data."""

    SFT = "sft"
    DPO = "dpo"
    SHAREGPT = "sharegpt"


class AugmentedSample(BaseModel):
    """Final training sample produced by AgentHER."""

    source_trajectory_id: str
    format: OutputFormat
    hindsight_prompt: str
    chosen: list[dict[str, str]] = Field(
        default_factory=list, description="Chosen conversation turns (for SFT/DPO)"
    )
    rejected: list[dict[str, str]] = Field(
        default_factory=list, description="Rejected conversation turns (DPO only)"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

"""AgentHER: Hindsight Experience Replay for LLM Agents.

Transforms failed agent trajectories into high-quality training data
by reverse-engineering prompts that match achieved outcomes.
"""

__version__ = "0.1.0"

from agenther.models import (
    AgentStep,
    AugmentedSample,
    FailedTrajectory,
    FailureAnalysis,
    FailureType,
    OutputFormat,
    RelabeledData,
    ReplayOutcome,
    SecondJudgeVerdict,
)
from agenther.pipeline import AgentHERPipeline, PipelineConfig, PipelineResult

__all__ = [
    "AgentHERPipeline",
    "AgentStep",
    "AugmentedSample",
    "FailedTrajectory",
    "FailureAnalysis",
    "FailureType",
    "OutputFormat",
    "PipelineConfig",
    "PipelineResult",
    "RelabeledData",
    "ReplayOutcome",
    "SecondJudgeVerdict",
]

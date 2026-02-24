"""AgentHER: Hindsight Experience Replay for LLM Agents.

Transforms failed agent trajectories into high-quality training data
by reverse-engineering prompts that match achieved outcomes.
"""

__version__ = "0.1.0"

from agenther.models import (
    AgentStep,
    FailedTrajectory,
    FailureType,
    OutputFormat,
    RelabeledData,
    ReplayOutcome,
)
from agenther.pipeline import AgentHERPipeline, PipelineConfig

__all__ = [
    "AgentStep",
    "FailedTrajectory",
    "FailureType",
    "OutputFormat",
    "RelabeledData",
    "ReplayOutcome",
    "AgentHERPipeline",
    "PipelineConfig",
]

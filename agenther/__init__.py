"""AgentHER: Hindsight Experience Replay for LLM Agents.

Transforms failed agent trajectories into high-quality training data
by reverse-engineering prompts that match achieved outcomes.
"""

__version__ = "0.1.0"

from agenther.models import (
    AgentStep,
    FailedTrajectory,
    RelabeledData,
    ReplayOutcome,
)
from agenther.pipeline import AgentHERPipeline

__all__ = [
    "AgentStep",
    "FailedTrajectory",
    "RelabeledData",
    "ReplayOutcome",
    "AgentHERPipeline",
]

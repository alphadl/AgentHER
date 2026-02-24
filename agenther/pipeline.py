"""AgentHER Pipeline — end-to-end orchestrator.

Wires together all four stages into a single, configurable pipeline:
  FailedTrajectory -> Detect -> Extract -> Relabel -> Augment -> AugmentedSample

Supports both synchronous (single-threaded) and batch processing modes.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from agenther.data_augmenter import DataAugmenter
from agenther.failure_detector import FailureDetector
from agenther.llm_client import LLMClient
from agenther.models import (
    AugmentedSample,
    FailedTrajectory,
    OutputFormat,
    RelabeledData,
    ReplayOutcome,
)
from agenther.outcome_extractor import OutcomeExtractor
from agenther.prompt_relabeler import PromptRelabeler

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the AgentHER pipeline."""

    model: str = "gpt-4o"
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.3

    use_llm_detector: bool = False
    use_llm_extractor: bool = True
    output_format: OutputFormat = OutputFormat.SFT

    min_confidence: float = 0.5
    relabel_max_attempts: int = 3

    output_dir: str = "outputs"


@dataclass
class PipelineResult:
    """Result of processing a single trajectory through the pipeline."""

    trajectory_id: str
    stage_reached: str
    success: bool
    sample: AugmentedSample | None = None
    outcome: ReplayOutcome | None = None
    relabeled: RelabeledData | None = None
    error: str | None = None


class AgentHERPipeline:
    """Main pipeline that orchestrates the four AgentHER stages."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()

        llm = LLMClient(
            model=self.config.model,
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            temperature=self.config.temperature,
        )

        self.detector = FailureDetector(llm_client=llm)
        self.extractor = OutcomeExtractor(llm_client=llm)
        self.relabeler = PromptRelabeler(llm_client=llm)
        self.augmenter = DataAugmenter()

    def process(self, trajectory: FailedTrajectory) -> PipelineResult:
        """Run a single trajectory through all four pipeline stages.

        Args:
            trajectory: A single failed trajectory to relabel.

        Returns:
            PipelineResult with success=True and sample/outcome/relabeled when all
            stages succeed; otherwise success=False and error/stage_reached set.
        """
        tid = trajectory.trajectory_id

        # Stage 1: Failure Detection
        try:
            analysis = self.detector.detect(
                trajectory, use_llm=self.config.use_llm_detector
            )
        except Exception as e:
            logger.exception("Trajectory %s failed at failure_detection", tid)
            return PipelineResult(
                trajectory_id=tid, stage_reached="failure_detection",
                success=False, error=str(e),
            )

        if not analysis.is_failure:
            logger.info("Trajectory %s is not a failure — skipping", tid)
            return PipelineResult(
                trajectory_id=tid, stage_reached="failure_detection",
                success=False, error="Trajectory is not a failure",
            )

        if not analysis.recoverable:
            logger.info("Trajectory %s is not recoverable — skipping", tid)
            return PipelineResult(
                trajectory_id=tid, stage_reached="failure_detection",
                success=False, error=f"Irrecoverable failure: {analysis.failure_type}",
            )

        # Stage 2: Outcome Extraction
        try:
            outcome = self.extractor.extract(
                trajectory, use_llm=self.config.use_llm_extractor
            )
        except Exception as e:
            logger.exception("Trajectory %s failed at outcome_extraction", tid)
            return PipelineResult(
                trajectory_id=tid, stage_reached="outcome_extraction",
                success=False, error=str(e),
            )

        # Stage 3: Prompt Relabeling
        try:
            relabeled = self.relabeler.relabel_with_validation(
                trajectory,
                outcome,
                min_confidence=self.config.min_confidence,
                max_attempts=self.config.relabel_max_attempts,
            )
        except Exception as e:
            logger.exception("Trajectory %s failed at prompt_relabeling", tid)
            return PipelineResult(
                trajectory_id=tid, stage_reached="prompt_relabeling",
                success=False, outcome=outcome, error=str(e),
            )

        if relabeled is None:
            return PipelineResult(
                trajectory_id=tid, stage_reached="prompt_relabeling",
                success=False, outcome=outcome,
                error="Relabeling failed to meet confidence threshold",
            )

        # Stage 4: Data Augmentation
        try:
            sample = self.augmenter.augment(
                trajectory, relabeled, output_format=self.config.output_format,
            )
        except Exception as e:
            logger.exception("Trajectory %s failed at data_augmentation", tid)
            return PipelineResult(
                trajectory_id=tid, stage_reached="data_augmentation",
                success=False, outcome=outcome, relabeled=relabeled, error=str(e),
            )

        return PipelineResult(
            trajectory_id=tid, stage_reached="complete",
            success=True, sample=sample, outcome=outcome, relabeled=relabeled,
        )

    def process_batch(
        self, trajectories: Sequence[FailedTrajectory]
    ) -> list[PipelineResult]:
        """Process multiple trajectories and return all results."""
        results: list[PipelineResult] = []
        total = len(trajectories)

        for i, traj in enumerate(trajectories, 1):
            logger.info("Processing trajectory %d/%d: %s", i, total, traj.trajectory_id)
            result = self.process(traj)
            results.append(result)

            status = "OK" if result.success else f"SKIP ({result.error})"
            logger.info("  -> %s", status)

        return results

    def run_and_save(
        self,
        trajectories: Sequence[FailedTrajectory],
        output_path: str | None = None,
    ) -> tuple[list[PipelineResult], Path | None]:
        """Process trajectories and save successful samples to disk.

        Args:
            trajectories: List of failed trajectories to relabel.
            output_path: Optional path for output JSONL; if None, uses
                config.output_dir / augmented_{format}.jsonl.

        Returns:
            (results, path): results for every trajectory; path is the saved file
            path, or None if no trajectory succeeded.
        """
        results = self.process_batch(trajectories)

        samples = [r.sample for r in results if r.success and r.sample is not None]

        if not samples:
            logger.warning("No successful relabelings — nothing to save")
            return results, None

        default = f"{self.config.output_dir}/augmented_{self.config.output_format.value}.jsonl"
        path = output_path or default
        saved = DataAugmenter.save_samples(samples, path)

        success_rate = len(samples) / len(trajectories) * 100
        logger.info(
            "Pipeline complete: %d/%d trajectories relabeled (%.1f%%), saved to %s",
            len(samples), len(trajectories), success_rate, saved,
        )
        return results, saved

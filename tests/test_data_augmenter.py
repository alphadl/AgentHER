"""Tests for the Data Augmenter module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from agenther.data_augmenter import DataAugmenter
from agenther.models import (
    FailedTrajectory,
    OutputFormat,
    RelabeledData,
)


class TestSFTFormat:
    def test_creates_two_turn_conversation(
        self,
        sample_trajectory: FailedTrajectory,
        sample_relabeled: RelabeledData,
    ) -> None:
        augmenter = DataAugmenter()
        sample = augmenter.to_sft(sample_trajectory, sample_relabeled)

        assert sample.format == OutputFormat.SFT
        assert len(sample.chosen) == 2
        assert sample.chosen[0]["role"] == "user"
        assert sample.chosen[0]["content"] == sample_relabeled.hindsight_prompt
        assert sample.chosen[1]["role"] == "assistant"
        assert sample.rejected == []

    def test_metadata_includes_original_prompt(
        self,
        sample_trajectory: FailedTrajectory,
        sample_relabeled: RelabeledData,
    ) -> None:
        augmenter = DataAugmenter()
        sample = augmenter.to_sft(sample_trajectory, sample_relabeled)
        assert sample.metadata["original_prompt"] == sample_trajectory.original_prompt


class TestDPOFormat:
    def test_creates_chosen_and_rejected(
        self,
        sample_trajectory: FailedTrajectory,
        sample_relabeled: RelabeledData,
    ) -> None:
        augmenter = DataAugmenter()
        sample = augmenter.to_dpo(sample_trajectory, sample_relabeled)

        assert sample.format == OutputFormat.DPO
        assert len(sample.chosen) == 2
        assert len(sample.rejected) == 2
        assert sample.chosen[0]["content"] == sample_relabeled.hindsight_prompt
        assert sample.rejected[0]["content"] == sample_trajectory.original_prompt


class TestShareGPTFormat:
    def test_creates_multi_turn(
        self,
        sample_trajectory: FailedTrajectory,
        sample_relabeled: RelabeledData,
    ) -> None:
        augmenter = DataAugmenter()
        sample = augmenter.to_sharegpt(sample_trajectory, sample_relabeled)

        assert sample.format == OutputFormat.SHAREGPT
        assert sample.chosen[0]["role"] == "user"
        assert len(sample.chosen) > 2


class TestFormatRouting:
    def test_routes_to_correct_builder(
        self,
        sample_trajectory: FailedTrajectory,
        sample_relabeled: RelabeledData,
    ) -> None:
        augmenter = DataAugmenter()

        for fmt in OutputFormat:
            sample = augmenter.augment(sample_trajectory, sample_relabeled, fmt)
            assert sample.format == fmt


class TestSaveSamples:
    def test_saves_jsonl(
        self,
        sample_trajectory: FailedTrajectory,
        sample_relabeled: RelabeledData,
    ) -> None:
        augmenter = DataAugmenter()
        sample = augmenter.to_sft(sample_trajectory, sample_relabeled)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = DataAugmenter.save_samples([sample], Path(tmpdir) / "out.jsonl")
            assert path.exists()

            lines = path.read_text().strip().split("\n")
            assert len(lines) == 1

            loaded = json.loads(lines[0])
            assert loaded["source_trajectory_id"] == sample_trajectory.trajectory_id

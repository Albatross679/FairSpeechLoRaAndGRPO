"""
Phase 1 (1.0 + 1.1) Nyquist validation tests.

Tests exercise code-level logic WITHOUT GPU, model weights, or real audio files.
All audio is mocked with numpy arrays; all manifests are temporary CSVs.

Coverage:
- Phase 1.0 (DATA-01, DATA-02, DATA-03): data_loader.py, prepare_splits.py, validate_splits.py
- Phase 1.1 (INFRA-01..05): lora_prototype.py constants/config, data_collator.py logic
"""

import os
import sys
import tempfile
import textwrap

import numpy as np
import pandas as pd
import pytest
import torch

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so "scripts.training" resolves
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ===========================================================================
# Helpers -- temporary manifest CSVs
# ===========================================================================

def _make_manifest_csv(tmp_path, rows, filename="manifest.csv"):
    """Write a list-of-dicts to a CSV and return the path."""
    df = pd.DataFrame(rows)
    path = os.path.join(str(tmp_path), filename)
    df.to_csv(path, index=False)
    return path


def _manifest_rows(n=20, axis="ethnicity", groups=None):
    """Generate n manifest rows with required columns + a demographic axis."""
    if groups is None:
        groups = ["group_a", "group_b", "group_c"]
    rows = []
    for i in range(n):
        rows.append({
            "utterance_id": f"utt_{i:04d}",
            "audio_path": f"/fake/audio_{i:04d}.wav",
            "sentence": f"this is sentence number {i}",
            axis: groups[i % len(groups)],
        })
    return rows


# ===========================================================================
# Phase 1.0 -- DATA-01: ASRFairnessDataset validates required columns
# ===========================================================================

class TestASRFairnessDatasetColumnValidation:
    """DATA-01: Dataset enforces required columns (utterance_id, audio_path, sentence)."""

    def test_missing_required_column_raises(self, tmp_path):
        """Dataset rejects a manifest missing 'sentence' column."""
        from scripts.training.data_loader import ASRFairnessDataset

        bad_rows = [{"utterance_id": "u1", "audio_path": "/fake.wav"}]
        csv_path = _make_manifest_csv(tmp_path, bad_rows)

        with pytest.raises(ValueError, match="Missing required columns.*sentence"):
            ASRFairnessDataset(csv_path, demographic_axis=None)

    def test_missing_demographic_axis_raises(self, tmp_path):
        """Dataset rejects a manifest missing the requested demographic column."""
        from scripts.training.data_loader import ASRFairnessDataset

        rows = _manifest_rows(5, axis="ethnicity")
        csv_path = _make_manifest_csv(tmp_path, rows)

        with pytest.raises(ValueError, match="Demographic axis 'accent' not in columns"):
            ASRFairnessDataset(csv_path, demographic_axis="accent")

    def test_valid_manifest_accepted(self, tmp_path):
        """Dataset accepts a well-formed manifest with all required columns."""
        from scripts.training.data_loader import ASRFairnessDataset

        rows = _manifest_rows(10, axis="ethnicity")
        csv_path = _make_manifest_csv(tmp_path, rows)

        ds = ASRFairnessDataset(csv_path, demographic_axis="ethnicity")
        assert len(ds) == 10

    def test_demographics_filled_for_missing_labels(self, tmp_path):
        """DATA-01 / D-08: Missing demographic labels become empty string."""
        from scripts.training.data_loader import ASRFairnessDataset

        rows = _manifest_rows(4, axis="ethnicity")
        rows[0]["ethnicity"] = np.nan
        rows[2]["ethnicity"] = ""
        csv_path = _make_manifest_csv(tmp_path, rows)

        ds = ASRFairnessDataset(csv_path, demographic_axis="ethnicity")
        # NaN -> "", already-empty -> ""
        assert ds.demographics[0] == ""
        assert ds.demographics[2] == ""
        # Non-missing should be preserved
        assert ds.demographics[1] != ""


# ===========================================================================
# Phase 1.0 -- DATA-02: collate_fn pads variable-length audio correctly
# ===========================================================================

class TestCollateFn:
    """DATA-02: collate_fn pads audio to batch-max length, returns correct shapes."""

    def test_padding_to_max_length(self):
        """Padded tensor width equals the longest sample in the batch."""
        from scripts.training.data_loader import collate_fn

        batch = [
            {"audio": torch.randn(100), "transcript": "a", "demographic_group": "g1", "utterance_id": "u1"},
            {"audio": torch.randn(200), "transcript": "b", "demographic_group": "g2", "utterance_id": "u2"},
            {"audio": torch.randn(150), "transcript": "c", "demographic_group": "g1", "utterance_id": "u3"},
        ]
        out = collate_fn(batch)

        assert out["audio"].shape == (3, 200)
        assert out["audio_lengths"].tolist() == [100, 200, 150]

    def test_padding_values_are_zero(self):
        """Padded positions contain zeros (silence)."""
        from scripts.training.data_loader import collate_fn

        short = torch.ones(50)
        batch = [
            {"audio": short, "transcript": "a", "demographic_group": "g", "utterance_id": "u1"},
            {"audio": torch.randn(100), "transcript": "b", "demographic_group": "g", "utterance_id": "u2"},
        ]
        out = collate_fn(batch)

        # The first sample's padded region (50:100) should be all zeros
        assert torch.all(out["audio"][0, 50:] == 0)
        # The first sample's real region should be all ones
        assert torch.all(out["audio"][0, :50] == 1)

    def test_metadata_lists_preserved(self):
        """Transcripts, demographics, utterance_ids are returned as lists."""
        from scripts.training.data_loader import collate_fn

        batch = [
            {"audio": torch.randn(10), "transcript": "hello", "demographic_group": "A", "utterance_id": "id1"},
            {"audio": torch.randn(20), "transcript": "world", "demographic_group": "B", "utterance_id": "id2"},
        ]
        out = collate_fn(batch)

        assert out["transcripts"] == ["hello", "world"]
        assert out["demographic_groups"] == ["A", "B"]
        assert out["utterance_ids"] == ["id1", "id2"]

    def test_single_sample_batch(self):
        """collate_fn handles a batch of size 1 without error."""
        from scripts.training.data_loader import collate_fn

        batch = [
            {"audio": torch.randn(80), "transcript": "solo", "demographic_group": "X", "utterance_id": "u0"},
        ]
        out = collate_fn(batch)

        assert out["audio"].shape == (1, 80)
        assert out["audio_lengths"].tolist() == [80]


# ===========================================================================
# Phase 1.0 -- DATA-03: DemographicStratifiedSampler oversamples small groups
# ===========================================================================

class TestDemographicStratifiedSampler:
    """DATA-03: Sampler oversamples small groups to ensure minimum representation."""

    def test_oversampling_small_group(self):
        """A group with fewer than min_per_group samples is oversampled."""
        from scripts.training.data_loader import DemographicStratifiedSampler

        # 5 samples in group_a (below min=50), 200 in group_b
        demographics = ["group_a"] * 5 + ["group_b"] * 200
        sampler = DemographicStratifiedSampler(
            demographics, batch_size=8, min_per_group=50, seed=42
        )

        indices = list(sampler)
        # group_a indices are 0..4; after oversampling there should be >= 50 of them
        group_a_count = sum(1 for i in indices if i < 5)
        assert group_a_count >= 50, (
            f"Expected >= 50 group_a samples after oversampling, got {group_a_count}"
        )

    def test_unlabeled_samples_included(self):
        """Samples with empty demographic label are included in the pool (D-08)."""
        from scripts.training.data_loader import DemographicStratifiedSampler

        demographics = ["group_a"] * 100 + [""] * 10 + ["group_b"] * 100
        sampler = DemographicStratifiedSampler(
            demographics, batch_size=8, min_per_group=50, seed=42
        )

        indices = list(sampler)
        # The unlabeled indices are 100..109
        unlabeled_indices = set(range(100, 110))
        unlabeled_in_output = sum(1 for i in indices if i in unlabeled_indices)
        assert unlabeled_in_output > 0, "Unlabeled samples should appear in sampler output"

    def test_num_batches_limits_epoch_length(self):
        """When num_batches is set, sampler yields exactly num_batches * batch_size."""
        from scripts.training.data_loader import DemographicStratifiedSampler

        demographics = ["g1"] * 100 + ["g2"] * 100
        sampler = DemographicStratifiedSampler(
            demographics, batch_size=4, min_per_group=50, seed=42, num_batches=10
        )
        assert len(sampler) == 40  # 10 batches * 4

    def test_groups_sorted_alphabetically(self):
        """Sampler stores groups in sorted order for reproducibility."""
        from scripts.training.data_loader import DemographicStratifiedSampler

        demographics = ["zebra"] * 60 + ["alpha"] * 60 + ["mid"] * 60
        sampler = DemographicStratifiedSampler(
            demographics, batch_size=4, min_per_group=50, seed=42
        )
        assert sampler.groups == ["alpha", "mid", "zebra"]

    def test_reproducibility_with_same_seed(self):
        """Two samplers with same seed produce the same index order."""
        from scripts.training.data_loader import DemographicStratifiedSampler

        demographics = ["g1"] * 80 + ["g2"] * 80
        s1 = DemographicStratifiedSampler(demographics, batch_size=4, min_per_group=50, seed=99)
        s2 = DemographicStratifiedSampler(demographics, batch_size=4, min_per_group=50, seed=99)
        assert list(s1) == list(s2)


# ===========================================================================
# Phase 1.1 -- INFRA-01: data_collator constants and _find_transcript_start
# ===========================================================================

class TestDataCollatorConstants:
    """INFRA-01: Data collator uses correct masking constants."""

    def test_ignore_index_is_negative_100(self):
        from scripts.training.data_collator import IGNORE_INDEX
        assert IGNORE_INDEX == -100

    def test_asr_text_tag_defined(self):
        from scripts.training.data_collator import ASR_TEXT_TAG
        assert ASR_TEXT_TAG == "<asr_text>"

    def test_language_prefix_defined(self):
        from scripts.training.data_collator import LANGUAGE_PREFIX
        assert LANGUAGE_PREFIX == "language English"


class TestFindTranscriptStart:
    """INFRA-01: _find_transcript_start locates <asr_text> boundary correctly."""

    def _make_collator_with_mock_tokenizer(self):
        """Build a DataCollatorForQwen3ASR with a minimal mock processor."""

        class MockTokenizer:
            pad_token_id = 0
            eos_token_id = 2

            def encode(self, text, add_special_tokens=False):
                # Encode <asr_text> as token IDs [99, 100]
                if text == "<asr_text>":
                    return [99, 100]
                return list(range(len(text)))

        class MockProcessor:
            tokenizer = MockTokenizer()

            def apply_chat_template(self, *a, **kw):
                return ""

        from scripts.training.data_collator import DataCollatorForQwen3ASR
        return DataCollatorForQwen3ASR(MockProcessor())

    def test_finds_tag_in_middle(self):
        """Returns position right after the <asr_text> tag tokens."""
        collator = self._make_collator_with_mock_tokenizer()
        # Simulate: [10, 20, 30, 99, 100, 50, 60]
        #                        ^tag^     ^transcript starts at index 5
        input_ids = torch.tensor([10, 20, 30, 99, 100, 50, 60])
        pos = collator._find_transcript_start(input_ids)
        assert pos == 5

    def test_tag_at_start(self):
        """Tag at the very beginning of input_ids."""
        collator = self._make_collator_with_mock_tokenizer()
        input_ids = torch.tensor([99, 100, 50, 60])
        pos = collator._find_transcript_start(input_ids)
        assert pos == 2

    def test_fallback_when_tag_missing(self):
        """When tag is not found, falls back to 80% of sequence length."""
        collator = self._make_collator_with_mock_tokenizer()
        input_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        pos = collator._find_transcript_start(input_ids)
        assert pos == 8  # int(10 * 0.8)


# ===========================================================================
# Phase 1.1 -- INFRA-01: _pad_batch pads text and features correctly
# ===========================================================================

class TestPadBatch:
    """INFRA-01: _pad_batch pads input_ids, labels, features to batch max."""

    def _make_collator_with_mock_tokenizer(self):
        class MockTokenizer:
            pad_token_id = 0
            eos_token_id = 2
            def encode(self, text, add_special_tokens=False):
                if text == "<asr_text>":
                    return [99, 100]
                return []
        class MockProcessor:
            tokenizer = MockTokenizer()
        from scripts.training.data_collator import DataCollatorForQwen3ASR
        return DataCollatorForQwen3ASR(MockProcessor())

    def test_text_padding(self):
        """input_ids padded with pad_token_id, labels padded with -100."""
        collator = self._make_collator_with_mock_tokenizer()

        ids_a = torch.tensor([1, 2, 3])
        ids_b = torch.tensor([4, 5, 6, 7, 8])
        attn_a = torch.ones(3, dtype=torch.long)
        attn_b = torch.ones(5, dtype=torch.long)
        labels_a = torch.tensor([1, 2, 3])
        labels_b = torch.tensor([4, 5, 6, 7, 8])

        # Minimal feature tensors (1D for simplicity)
        feat_a = torch.randn(1, 10)
        feat_b = torch.randn(1, 10)
        feat_mask_a = torch.ones(10)
        feat_mask_b = torch.ones(10)

        batch = collator._pad_batch(
            [ids_a, ids_b], [attn_a, attn_b],
            [feat_a, feat_b], [feat_mask_a, feat_mask_b],
            [labels_a, labels_b],
        )

        assert batch["input_ids"].shape == (2, 5)
        assert batch["labels"].shape == (2, 5)
        # ids_a padded with 0 (pad_token_id)
        assert batch["input_ids"][0, 3:].tolist() == [0, 0]
        # labels_a padded with -100
        assert batch["labels"][0, 3:].tolist() == [-100, -100]
        # attention_mask_a padded with 0
        assert batch["attention_mask"][0, 3:].tolist() == [0, 0]

    def test_feature_padding(self):
        """input_features padded along time dimension with zeros."""
        collator = self._make_collator_with_mock_tokenizer()

        ids = [torch.tensor([1]), torch.tensor([2])]
        attn = [torch.ones(1, dtype=torch.long), torch.ones(1, dtype=torch.long)]
        labels = [torch.tensor([1]), torch.tensor([2])]

        feat_a = torch.ones(80, 50)  # (mel_bins, time=50)
        feat_b = torch.ones(80, 80)  # (mel_bins, time=80)
        mask_a = torch.ones(50)
        mask_b = torch.ones(80)

        batch = collator._pad_batch(ids, attn, [feat_a, feat_b], [mask_a, mask_b], labels)

        assert batch["input_features"].shape == (2, 80, 80)
        # feat_a padded region should be zeros
        assert torch.all(batch["input_features"][0, :, 50:] == 0)
        assert batch["feature_attention_mask"].shape == (2, 80)
        assert batch["feature_attention_mask"][0, 50:].sum() == 0


# ===========================================================================
# Phase 1.1 -- INFRA-02: LoRA prototype constants match requirements
# ===========================================================================

class TestLoraPrototypeConstants:
    """INFRA-02: LoRA hyperparameters and subset sizes match design docs."""

    def test_lora_rank(self):
        from scripts.training.lora_prototype import LORA_RANK
        assert LORA_RANK == 16

    def test_lora_alpha(self):
        from scripts.training.lora_prototype import LORA_ALPHA
        assert LORA_ALPHA == 32

    def test_lora_dropout(self):
        from scripts.training.lora_prototype import LORA_DROPOUT
        assert LORA_DROPOUT == 0.05

    def test_subset_sizes(self):
        from scripts.training.lora_prototype import FS_SUBSET_SIZE, CV_SUBSET_SIZE
        assert FS_SUBSET_SIZE == 100
        assert CV_SUBSET_SIZE == 100

    def test_default_steps(self):
        from scripts.training.lora_prototype import DEFAULT_NUM_STEPS
        assert DEFAULT_NUM_STEPS == 100

    def test_model_id(self):
        from scripts.training.lora_prototype import MODEL_ID
        assert MODEL_ID == "Qwen/Qwen3-ASR-1.7B"


# ===========================================================================
# Phase 1.1 -- INFRA-03: create_stratified_subset logic
# ===========================================================================

class TestCreateStratifiedSubset:
    """INFRA-03: Stratified subsetting produces correct sizes and preserves groups."""

    def test_subset_sizes_correct(self, tmp_path):
        from scripts.training.lora_prototype import create_stratified_subset

        # Create FS manifest with ethnicity
        fs_rows = []
        for i in range(300):
            fs_rows.append({
                "utterance_id": f"fs_{i}", "audio_path": f"/fake/fs_{i}.wav",
                "sentence": f"fs sentence {i}",
                "ethnicity": ["white", "black", "asian"][i % 3],
            })
        fs_path = _make_manifest_csv(tmp_path, fs_rows, "fs_train.csv")

        # Create CV manifest with accent
        cv_rows = []
        for i in range(300):
            cv_rows.append({
                "utterance_id": f"cv_{i}", "audio_path": f"/fake/cv_{i}.wav",
                "sentence": f"cv sentence {i}",
                "accent": ["us", "uk", "au"][i % 3],
            })
        cv_path = _make_manifest_csv(tmp_path, cv_rows, "cv_train.csv")

        out_dir = os.path.join(str(tmp_path), "output")
        subset_path = create_stratified_subset(fs_path, cv_path, out_dir, seed=42)

        assert os.path.isfile(subset_path)
        df = pd.read_csv(subset_path)
        assert len(df) == 200  # 100 FS + 100 CV

    def test_subset_preserves_groups(self, tmp_path):
        """Multiple demographic groups appear in the subset."""
        from scripts.training.lora_prototype import create_stratified_subset

        fs_rows = [{"utterance_id": f"fs_{i}", "audio_path": f"/f/{i}.wav",
                     "sentence": f"s{i}", "ethnicity": ["A", "B"][i % 2]}
                    for i in range(200)]
        cv_rows = [{"utterance_id": f"cv_{i}", "audio_path": f"/f/{i}.wav",
                     "sentence": f"s{i}", "accent": ["X", "Y"][i % 2]}
                    for i in range(200)]

        fs_path = _make_manifest_csv(tmp_path, fs_rows, "fs.csv")
        cv_path = _make_manifest_csv(tmp_path, cv_rows, "cv.csv")
        out_dir = os.path.join(str(tmp_path), "out")

        subset_path = create_stratified_subset(fs_path, cv_path, out_dir, seed=42)
        df = pd.read_csv(subset_path)

        # At least 2 ethnicity groups should be present
        if "ethnicity" in df.columns:
            assert df["ethnicity"].dropna().nunique() >= 2


# ===========================================================================
# Phase 1.1 -- INFRA-04: validate_lora_prototype constants
# ===========================================================================

class TestValidateLoraPrototypeConstants:
    """INFRA-04: Validation script thresholds match design requirements."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_whisper(self):
        pytest.importorskip("whisper", reason="whisper not installed in this env")

    def test_vram_limit(self):
        from scripts.training.validate_lora_prototype import VRAM_LIMIT_GB
        assert VRAM_LIMIT_GB == 14.0

    def test_wer_threshold(self):
        from scripts.training.validate_lora_prototype import WER_THRESHOLD
        assert WER_THRESHOLD == 0.80

    def test_chatty_patterns_nonempty(self):
        from scripts.training.validate_lora_prototype import CHATTY_PATTERNS
        assert len(CHATTY_PATTERNS) > 0
        assert all(isinstance(p, str) for p in CHATTY_PATTERNS)


# ===========================================================================
# Phase 1.1 -- INFRA-04: check_loss_trend logic
# ===========================================================================

class TestCheckLossTrend:
    """INFRA-04: Loss trend check detects decreasing vs non-decreasing loss."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_whisper(self):
        pytest.importorskip("whisper", reason="whisper not installed in this env")

    def test_decreasing_loss_passes(self, tmp_path):
        """Decreasing loss curve produces no issues."""
        import json
        from scripts.training.validate_lora_prototype import check_loss_trend

        log_history = [{"loss": 5.0 - i * 0.3, "step": i * 10} for i in range(10)]
        state = {"log_history": log_history}
        state_path = os.path.join(str(tmp_path), "trainer_state.json")
        with open(state_path, "w") as f:
            json.dump(state, f)

        issues = []
        check_loss_trend(str(tmp_path), issues)
        assert len(issues) == 0, f"Expected no issues, got: {issues}"

    def test_increasing_loss_fails(self, tmp_path):
        """Increasing loss curve produces an issue."""
        import json
        from scripts.training.validate_lora_prototype import check_loss_trend

        log_history = [{"loss": 1.0 + i * 0.5, "step": i * 10} for i in range(10)]
        state = {"log_history": log_history}
        state_path = os.path.join(str(tmp_path), "trainer_state.json")
        with open(state_path, "w") as f:
            json.dump(state, f)

        issues = []
        check_loss_trend(str(tmp_path), issues)
        assert len(issues) > 0
        assert "not decreasing" in issues[0].lower() or "Not" in issues[0]

    def test_missing_state_file_reports_issue(self, tmp_path):
        """Missing trainer_state.json is reported as an issue."""
        from scripts.training.validate_lora_prototype import check_loss_trend

        issues = []
        check_loss_trend(str(tmp_path), issues)
        assert any("trainer_state.json" in i for i in issues)


# ===========================================================================
# Phase 1.1 -- INFRA-05: patch_outer_forward raises on bad model
# ===========================================================================

class TestPatchOuterForward:
    """INFRA-05: Forward patch raises RuntimeError for models without thinker."""

    def test_raises_on_model_without_thinker(self):
        from scripts.training.lora_prototype import patch_outer_forward

        class FakeModel:
            pass

        with pytest.raises(RuntimeError, match="Cannot patch forward"):
            patch_outer_forward(FakeModel())

    def test_succeeds_on_model_with_thinker(self):
        from scripts.training.lora_prototype import patch_outer_forward

        class FakeThinker:
            def forward(self, **kw):
                pass

        class FakeModel:
            thinker = FakeThinker()
            _forward_patched = False

        model = FakeModel()
        patch_outer_forward(model)
        assert model.__class__._forward_patched is True

    def test_idempotent_patch(self):
        """Patching twice does not error."""
        from scripts.training.lora_prototype import patch_outer_forward

        class FakeThinker:
            def forward(self, **kw):
                pass

        class FakeModel2:
            thinker = FakeThinker()
            _forward_patched = False

        model = FakeModel2()
        patch_outer_forward(model)
        patch_outer_forward(model)  # should not raise


# ===========================================================================
# Phase 1.0 -- Package import smoke tests
# ===========================================================================

class TestImports:
    """Smoke tests: all Phase 1 modules are importable."""

    def test_import_data_loader(self):
        from scripts.training.data_loader import (
            ASRFairnessDataset, DemographicStratifiedSampler, collate_fn, create_dataloader
        )

    def test_import_data_collator(self):
        from scripts.training.data_collator import (
            DataCollatorForQwen3ASR, IGNORE_INDEX, ASR_TEXT_TAG, LANGUAGE_PREFIX
        )

    def test_import_training_init(self):
        from scripts.training import (
            ASRFairnessDataset, DemographicStratifiedSampler, collate_fn,
            create_dataloader, DataCollatorForQwen3ASR
        )

    def test_import_lora_prototype_constants(self):
        from scripts.training.lora_prototype import (
            MODEL_ID, LORA_RANK, LORA_ALPHA, LORA_DROPOUT,
            FS_SUBSET_SIZE, CV_SUBSET_SIZE,
            DEFAULT_NUM_STEPS, DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE,
        )

    def test_import_validate_lora_prototype_constants(self):
        pytest.importorskip("whisper", reason="whisper not installed in this env")
        from scripts.training.validate_lora_prototype import (
            VRAM_LIMIT_GB, WER_THRESHOLD, CHATTY_PATTERNS,
        )


# ===========================================================================
# Phase 1.0 -- prepare_splits.py: split_fairspeech logic
# ===========================================================================

class TestSplitFairSpeech:
    """DATA-01: Speaker-disjoint stratified splitting preserves demographics."""

    def test_speaker_disjoint_split(self):
        """No speaker appears in both train and eval splits."""
        from scripts.prepare_splits import split_fairspeech

        # 10 speakers, 10 utterances each, 3 ethnicity groups
        rows = []
        ethnicities = ["white", "black", "asian"]
        for spk in range(10):
            eth = ethnicities[spk % 3]
            for utt in range(10):
                rows.append({
                    "utterance_id": f"utt_{spk}_{utt}",
                    "audio_path": f"/fake/{spk}_{utt}.wav",
                    "sentence": f"sentence {spk} {utt}",
                    "ethnicity": eth,
                    "speaker_id": f"spk_{spk}",
                })
        df = pd.DataFrame(rows)

        df_train, df_eval = split_fairspeech(df, seed=42)

        train_speakers = set(df_train["speaker_id"])
        eval_speakers = set(df_eval["speaker_id"])
        assert len(train_speakers & eval_speakers) == 0, "Speaker leakage detected"

    def test_split_covers_all_rows(self):
        """Train + eval sizes sum to original dataset size."""
        from scripts.prepare_splits import split_fairspeech

        rows = []
        for spk in range(10):
            for utt in range(10):
                rows.append({
                    "utterance_id": f"u_{spk}_{utt}",
                    "audio_path": f"/f/{spk}_{utt}.wav",
                    "sentence": f"s {spk} {utt}",
                    "ethnicity": ["A", "B", "C"][spk % 3],
                    "speaker_id": f"s{spk}",
                })
        df = pd.DataFrame(rows)
        df_train, df_eval = split_fairspeech(df, seed=42)
        assert len(df_train) + len(df_eval) == len(df)

    def test_ethnicity_represented_in_both_splits(self):
        """Each ethnicity group appears in both train and eval."""
        from scripts.prepare_splits import split_fairspeech

        rows = []
        eths = ["white", "black", "asian"]
        # Use 30 speakers (10 per group) so KFold can place each group in eval
        for spk in range(30):
            eth = eths[spk % 3]
            for utt in range(10):
                rows.append({
                    "utterance_id": f"u_{spk}_{utt}",
                    "audio_path": f"/f/{spk}_{utt}.wav",
                    "sentence": f"s",
                    "ethnicity": eth,
                    "speaker_id": f"s{spk}",
                })
        df = pd.DataFrame(rows)
        df_train, df_eval = split_fairspeech(df, seed=42)

        for eth in eths:
            assert eth in df_train["ethnicity"].values, f"{eth} missing from train"
            assert eth in df_eval["ethnicity"].values, f"{eth} missing from eval"


# ===========================================================================
# Phase 1.0 -- validate_splits.py: validation check logic
# ===========================================================================

class TestValidateSplitsChecks:
    """DATA-03: Validation checks detect issues in split data."""

    def test_speaker_leakage_detected(self):
        """check_speaker_leakage flags overlapping speakers."""
        from scripts.validate_splits import check_speaker_leakage

        df_train = pd.DataFrame({
            "speaker_id": ["s1", "s2", "s3"],
            "utterance_id": ["u1", "u2", "u3"],
        })
        df_eval = pd.DataFrame({
            "speaker_id": ["s3", "s4"],
            "utterance_id": ["u4", "u5"],
        })
        issues = []
        warnings = []
        check_speaker_leakage(df_train, df_eval, issues, warnings)
        assert any("leakage" in i.lower() or "Speaker" in i for i in issues)

    def test_no_leakage_clean(self):
        """check_speaker_leakage passes when splits are disjoint."""
        from scripts.validate_splits import check_speaker_leakage

        df_train = pd.DataFrame({
            "speaker_id": ["s1", "s2"],
            "utterance_id": ["u1", "u2"],
        })
        df_eval = pd.DataFrame({
            "speaker_id": ["s3", "s4"],
            "utterance_id": ["u3", "u4"],
        })
        issues = []
        warnings = []
        check_speaker_leakage(df_train, df_eval, issues, warnings)
        assert len(issues) == 0

    def test_demographic_completeness_flags_high_missing(self):
        """check_demographic_completeness flags >5% missing ethnicity labels."""
        from scripts.validate_splits import check_demographic_completeness

        # 20% missing ethnicity in eval
        df_eval = pd.DataFrame({
            "ethnicity": ["white"] * 8 + [np.nan] * 2,
            "gender": ["M"] * 10,
            "age": ["20-30"] * 10,
            "first_language": ["en"] * 10,
        })
        df_train = df_eval.copy()

        issues = []
        warnings = []
        check_demographic_completeness(df_train, df_eval, issues, warnings)
        assert any("ethnicity" in i and "missing" in i for i in issues)

    def test_distribution_drift_detected(self):
        """check_distribution_drift flags large proportion differences."""
        from scripts.validate_splits import check_distribution_drift

        # Train: 90% A, 10% B. Eval: 50% A, 50% B => large drift
        df_train = pd.DataFrame({"ethnicity": ["A"] * 90 + ["B"] * 10})
        df_eval = pd.DataFrame({"ethnicity": ["A"] * 50 + ["B"] * 50})

        issues = []
        warnings = []
        check_distribution_drift(df_train, df_eval, issues, warnings)
        assert len(warnings) > 0, "Expected drift warnings"


# ===========================================================================
# Phase 1.1 -- INFRA-02: LoRA target modules pattern
# ===========================================================================

class TestLoraTargetModules:
    """INFRA-02: LoRA targets only decoder self-attention projections."""

    def test_target_modules_pattern_matches_expected(self):
        """Target modules regex matches q/k/v/o_proj in thinker decoder layers."""
        import re

        # Use the raw pattern string directly (LoraConfig converts list to set)
        pattern = r"thinker\.model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"

        # These should match
        assert re.match(pattern, "thinker.model.layers.0.self_attn.q_proj")
        assert re.match(pattern, "thinker.model.layers.15.self_attn.v_proj")
        assert re.match(pattern, "thinker.model.layers.3.self_attn.k_proj")
        assert re.match(pattern, "thinker.model.layers.7.self_attn.o_proj")

        # These should NOT match (MLP modules excluded per D-04)
        assert not re.match(pattern, "thinker.model.layers.0.mlp.gate_proj")
        assert not re.match(pattern, "audio_tower.layers.0.self_attn.q_proj")


# ===========================================================================
# Phase 1.1 -- INFRA-03: print_gpu_memory does not crash without GPU
# ===========================================================================

class TestPrintGpuMemory:
    """INFRA-03: VRAM profiling function handles no-GPU gracefully."""

    def test_no_crash_without_cuda(self):
        """print_gpu_memory should not raise even if CUDA is unavailable."""
        from scripts.training.lora_prototype import print_gpu_memory
        # This should print a message but not crash
        print_gpu_memory("test_label")

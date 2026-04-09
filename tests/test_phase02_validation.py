"""
Phase 2 (02-standard-lora-baseline-evaluation-bridge) Nyquist validation tests.

Validates artifact existence, structure, and code patterns WITHOUT GPU or model loading.

Coverage:
- Plan 02-01 (BASE-01): HP sweep artifacts, training script, adapter weights
- Plan 02-02 (EVAL-01..04): evaluation bridge script, prediction CSVs, analysis JSON
"""

import ast
import csv
import json
import os
import sys

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HP_SWEEP_DIR = os.path.join(PROJECT_ROOT, "outputs", "hp-sweep")
STANDARD_LORA_DIR = os.path.join(PROJECT_ROOT, "outputs", "standard-lora")
ADAPTER_DIR = os.path.join(STANDARD_LORA_DIR, "adapter")
EVAL_DIR = os.path.join(STANDARD_LORA_DIR, "eval")

TRAIN_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "training", "train_standard_lora.py")
EVAL_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "training", "evaluate_adapter.py")

# Expected HP keys that should appear in config JSONs
EXPECTED_HP_KEYS = {"lr", "rank", "alpha_ratio", "dropout", "target_mlp", "weight_decay"}

# Prediction CSV columns matching run_inference.py format
EXPECTED_PRED_COLUMNS = {
    "utterance_id", "reference", "hypothesis", "wer",
    "num_hyp_words", "num_ref_words", "perturbation",
    "model", "generation", "architecture",
}

# Demographic columns (may be empty but should exist)
DEMOGRAPHIC_COLUMNS = {"gender", "accent", "age", "ethnicity", "first_language"}


# ===========================================================================
# HP Sweep Artifacts (Plan 02-01, Task 1)
# ===========================================================================

class TestHPSweepArtifacts:
    """Validate HP sweep outputs from 20 Optuna trials."""

    def test_best_params_exists_and_parses(self):
        path = os.path.join(HP_SWEEP_DIR, "best_params.json")
        assert os.path.isfile(path), f"Missing {path}"
        data = json.load(open(path))
        assert "params" in data, "best_params.json must contain 'params' key"
        params = data["params"]
        assert EXPECTED_HP_KEYS.issubset(set(params.keys())), (
            f"Missing HP keys: {EXPECTED_HP_KEYS - set(params.keys())}"
        )

    def test_top3_configs_has_three_entries(self):
        path = os.path.join(HP_SWEEP_DIR, "top3_configs.json")
        assert os.path.isfile(path), f"Missing {path}"
        data = json.load(open(path))
        assert isinstance(data, list), "top3_configs.json must be a JSON array"
        assert len(data) == 3, f"Expected 3 configs, got {len(data)}"

    def test_top3_configs_each_has_hp_keys(self):
        path = os.path.join(HP_SWEEP_DIR, "top3_configs.json")
        data = json.load(open(path))
        for i, config in enumerate(data):
            assert "params" in config, f"Config {i} missing 'params'"
            assert "eval_loss" in config, f"Config {i} missing 'eval_loss'"
            params = config["params"]
            assert EXPECTED_HP_KEYS.issubset(set(params.keys())), (
                f"Config {i} missing HP keys: {EXPECTED_HP_KEYS - set(params.keys())}"
            )

    def test_top3_configs_sorted_by_loss(self):
        path = os.path.join(HP_SWEEP_DIR, "top3_configs.json")
        data = json.load(open(path))
        losses = [c["eval_loss"] for c in data]
        assert losses == sorted(losses), (
            f"Top 3 configs should be sorted by eval_loss ascending: {losses}"
        )

    def test_all_trials_csv_has_20_rows(self):
        path = os.path.join(HP_SWEEP_DIR, "all_trials.csv")
        assert os.path.isfile(path), f"Missing {path}"
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 20, f"Expected 20 trial rows, got {len(rows)}"

    def test_all_trials_csv_has_expected_columns(self):
        path = os.path.join(HP_SWEEP_DIR, "all_trials.csv")
        with open(path) as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames)
        # Must have at least: number, value, state
        for col in ["number", "value", "state"]:
            assert col in fieldnames, f"all_trials.csv missing column '{col}'"

    def test_all_trials_all_complete(self):
        path = os.path.join(HP_SWEEP_DIR, "all_trials.csv")
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        states = {r["state"] for r in rows}
        assert states == {"COMPLETE"}, f"Expected all COMPLETE, got states: {states}"


# ===========================================================================
# Training Script Code Patterns (Plan 02-01, Task 2)
# ===========================================================================

class TestTrainScriptCodePatterns:
    """Validate train_standard_lora.py code structure without execution."""

    def test_script_parses_without_syntax_errors(self):
        with open(TRAIN_SCRIPT) as f:
            source = f.read()
        ast.parse(source)  # raises SyntaxError if broken

    def test_script_has_validate_and_train_modes(self):
        with open(TRAIN_SCRIPT) as f:
            source = f.read()
        assert "validate" in source, "Script must reference 'validate' mode"
        assert '"train"' in source or "'train'" in source, "Script must reference 'train' mode"
        # Check argparse mode choice
        assert "choices=" in source, "Script must have argparse choices for mode"

    def test_script_imports_data_loader(self):
        with open(TRAIN_SCRIPT) as f:
            source = f.read()
        assert "data_loader" in source, "Script must import from data_loader"
        assert "ASRFairnessDataset" in source or "data_loader" in source

    def test_script_imports_data_collator(self):
        with open(TRAIN_SCRIPT) as f:
            source = f.read()
        assert "data_collator" in source, "Script must import from data_collator"
        assert "DataCollatorForQwen3ASR" in source or "data_collator" in source

    def test_script_has_argparse_cli(self):
        with open(TRAIN_SCRIPT) as f:
            source = f.read()
        assert "argparse" in source
        assert "--mode" in source
        assert "--fs_manifest" in source
        assert "--output_dir" in source

    def test_script_has_vram_monitoring(self):
        with open(TRAIN_SCRIPT) as f:
            source = f.read()
        assert "print_gpu_memory" in source or "gpu_memory" in source or "nvidia-smi" in source, (
            "Script must include VRAM monitoring"
        )

    def test_script_has_gradient_checkpointing(self):
        with open(TRAIN_SCRIPT) as f:
            source = f.read()
        assert "gradient_checkpointing" in source, (
            "Script must enable gradient checkpointing"
        )

    def test_script_references_locked_config(self):
        with open(TRAIN_SCRIPT) as f:
            source = f.read()
        assert "locked_config" in source, (
            "Script must read/write locked_config.json"
        )

    def test_script_minimum_line_count(self):
        with open(TRAIN_SCRIPT) as f:
            lines = f.readlines()
        assert len(lines) >= 250, (
            f"Script has {len(lines)} lines, expected >= 250 for full implementation"
        )


# ===========================================================================
# Validation + Locked Config (Plan 02-01, Tasks 3-4)
# ===========================================================================

class TestValidationAndLockedConfig:
    """Validate medium-scale validation results and locked config."""

    def test_validation_results_exists_and_parses(self):
        path = os.path.join(STANDARD_LORA_DIR, "validation_results.json")
        assert os.path.isfile(path), f"Missing {path}"
        data = json.load(open(path))
        assert isinstance(data, list), "validation_results.json must be a JSON array"
        assert len(data) >= 2, f"Expected >= 2 validation results, got {len(data)}"

    def test_validation_results_loss_decreased(self):
        path = os.path.join(STANDARD_LORA_DIR, "validation_results.json")
        data = json.load(open(path))
        for i, result in enumerate(data):
            first = result.get("train_loss_first")
            last = result.get("train_loss_last")
            if first is not None and last is not None:
                assert last < first, (
                    f"Config {i}: training loss did not decrease "
                    f"(first={first}, last={last})"
                )

    def test_locked_config_exists_and_has_hp_keys(self):
        path = os.path.join(STANDARD_LORA_DIR, "locked_config.json")
        assert os.path.isfile(path), f"Missing {path}"
        data = json.load(open(path))
        assert "params" in data, "locked_config.json must contain 'params'"
        params = data["params"]
        assert EXPECTED_HP_KEYS.issubset(set(params.keys())), (
            f"Missing HP keys in locked config: {EXPECTED_HP_KEYS - set(params.keys())}"
        )

    def test_locked_config_lr_in_reasonable_range(self):
        path = os.path.join(STANDARD_LORA_DIR, "locked_config.json")
        data = json.load(open(path))
        lr = data["params"]["lr"]
        assert 1e-5 <= lr <= 1e-2, f"LR {lr} outside expected range [1e-5, 1e-2]"

    def test_locked_config_rank_is_valid(self):
        path = os.path.join(STANDARD_LORA_DIR, "locked_config.json")
        data = json.load(open(path))
        rank = data["params"]["rank"]
        assert rank in (4, 8, 16, 32), f"Rank {rank} not in expected set {{4, 8, 16, 32}}"


# ===========================================================================
# Adapter Artifacts (Plan 02-01, Task 4)
# ===========================================================================

class TestAdapterArtifacts:
    """Validate trained LoRA adapter files."""

    def test_adapter_safetensors_exists_and_large_enough(self):
        path = os.path.join(ADAPTER_DIR, "adapter_model.safetensors")
        assert os.path.isfile(path), f"Missing {path}"
        size = os.path.getsize(path)
        assert size > 1024, f"Adapter too small: {size} bytes (expected > 1KB)"

    def test_adapter_safetensors_is_megabytes(self):
        """Adapter for rank=4 LoRA should be in the MB range."""
        path = os.path.join(ADAPTER_DIR, "adapter_model.safetensors")
        size_mb = os.path.getsize(path) / (1024 * 1024)
        assert size_mb > 1.0, f"Adapter only {size_mb:.2f} MB, expected > 1 MB"

    def test_adapter_config_exists_and_parses(self):
        path = os.path.join(ADAPTER_DIR, "adapter_config.json")
        assert os.path.isfile(path), f"Missing {path}"
        data = json.load(open(path))
        assert data.get("peft_type") == "LORA", (
            f"Expected peft_type=LORA, got {data.get('peft_type')}"
        )

    def test_adapter_config_matches_locked_config(self):
        """Adapter config LoRA rank should match locked config."""
        adapter_cfg = json.load(
            open(os.path.join(ADAPTER_DIR, "adapter_config.json"))
        )
        locked_cfg = json.load(
            open(os.path.join(STANDARD_LORA_DIR, "locked_config.json"))
        )
        # adapter_config.json uses "r" for rank
        assert adapter_cfg["r"] == locked_cfg["params"]["rank"], (
            f"Rank mismatch: adapter has r={adapter_cfg['r']}, "
            f"locked config has rank={locked_cfg['params']['rank']}"
        )

    def test_adapter_config_base_model(self):
        path = os.path.join(ADAPTER_DIR, "adapter_config.json")
        data = json.load(open(path))
        assert "Qwen" in data.get("base_model_name_or_path", ""), (
            "Adapter base_model should reference Qwen"
        )


# ===========================================================================
# Training Config + Metrics (Plan 02-01, Task 4)
# ===========================================================================

class TestTrainingConfig:
    """Validate training_config.json from final training run."""

    def test_training_config_exists_and_parses(self):
        path = os.path.join(STANDARD_LORA_DIR, "training_config.json")
        assert os.path.isfile(path), f"Missing {path}"
        data = json.load(open(path))
        assert "params" in data, "training_config.json must contain 'params'"
        assert "max_steps" in data, "training_config.json must contain 'max_steps'"

    def test_training_config_loss_decreased(self):
        path = os.path.join(STANDARD_LORA_DIR, "training_config.json")
        data = json.load(open(path))
        final_loss = data.get("final_train_loss")
        assert final_loss is not None, "training_config.json missing final_train_loss"
        # Final loss should be well below initial (typically starts > 1.0)
        assert final_loss < 1.0, (
            f"Final train loss {final_loss} seems too high (expected < 1.0)"
        )

    def test_training_config_vram_within_budget(self):
        path = os.path.join(STANDARD_LORA_DIR, "training_config.json")
        data = json.load(open(path))
        peak_vram = data.get("peak_vram_gb")
        if peak_vram is not None:
            assert peak_vram < 14.0, (
                f"Peak VRAM {peak_vram:.1f} GB exceeds 14 GB budget"
            )

    def test_training_used_expected_subset_size(self):
        path = os.path.join(STANDARD_LORA_DIR, "training_config.json")
        data = json.load(open(path))
        subset = data.get("subset_size")
        assert subset is not None, "training_config.json missing subset_size"
        assert subset >= 1000, f"Subset size {subset} seems too small"

    def test_train_and_eval_subsets_exist(self):
        train_path = os.path.join(STANDARD_LORA_DIR, "train_subset.csv")
        eval_path = os.path.join(STANDARD_LORA_DIR, "eval_subset.csv")
        assert os.path.isfile(train_path), f"Missing {train_path}"
        assert os.path.isfile(eval_path), f"Missing {eval_path}"


# ===========================================================================
# Evaluate Adapter Script Code Patterns (Plan 02-02, Task 1)
# ===========================================================================

class TestEvalScriptCodePatterns:
    """Validate evaluate_adapter.py code structure without execution."""

    def test_script_parses_without_syntax_errors(self):
        with open(EVAL_SCRIPT) as f:
            source = f.read()
        ast.parse(source)

    def test_script_uses_english_text_normalizer(self):
        with open(EVAL_SCRIPT) as f:
            source = f.read()
        assert "EnglishTextNormalizer" in source, (
            "evaluate_adapter.py must use EnglishTextNormalizer for text normalization"
        )

    def test_script_uses_peft_model(self):
        with open(EVAL_SCRIPT) as f:
            source = f.read()
        assert "PeftModel" in source, (
            "evaluate_adapter.py must use PeftModel to load LoRA adapter"
        )
        assert "from_pretrained" in source, (
            "evaluate_adapter.py must call PeftModel.from_pretrained"
        )

    def test_script_has_min_group_size(self):
        with open(EVAL_SCRIPT) as f:
            source = f.read()
        assert "MIN_GROUP_SIZE" in source, (
            "evaluate_adapter.py must define MIN_GROUP_SIZE for group filtering"
        )
        # Check the value is 50
        assert "50" in source, "MIN_GROUP_SIZE should be 50"

    def test_script_computes_fairness_metrics(self):
        with open(EVAL_SCRIPT) as f:
            source = f.read()
        assert "compute_fairness_metrics" in source or "fairness_metrics" in source, (
            "evaluate_adapter.py must compute fairness metrics"
        )

    def test_script_has_argparse_cli(self):
        with open(EVAL_SCRIPT) as f:
            source = f.read()
        assert "argparse" in source
        assert "--adapter_path" in source
        assert "--model_name" in source
        assert "--output_dir" in source

    def test_script_minimum_line_count(self):
        with open(EVAL_SCRIPT) as f:
            lines = f.readlines()
        assert len(lines) >= 300, (
            f"Script has {len(lines)} lines, expected >= 300 for full implementation"
        )


# ===========================================================================
# Prediction CSVs (Plan 02-02, Task 2)
# ===========================================================================

class TestPredictionCSVs:
    """Validate prediction CSV format matches run_inference.py output."""

    @pytest.fixture(params=[
        "predictions_standard-lora_fairspeech.csv",
        "predictions_standard-lora_commonvoice.csv",
        "predictions_standard-lora_librispeech.csv",
    ])
    def pred_csv_path(self, request):
        return os.path.join(EVAL_DIR, request.param)

    def test_prediction_csv_exists(self, pred_csv_path):
        assert os.path.isfile(pred_csv_path), f"Missing {pred_csv_path}"

    def test_prediction_csv_has_expected_columns(self, pred_csv_path):
        with open(pred_csv_path) as f:
            reader = csv.DictReader(f)
            columns = set(reader.fieldnames)
        missing = EXPECTED_PRED_COLUMNS - columns
        assert not missing, (
            f"Prediction CSV missing columns: {missing}"
        )

    def test_prediction_csv_has_demographic_columns(self, pred_csv_path):
        with open(pred_csv_path) as f:
            reader = csv.DictReader(f)
            columns = set(reader.fieldnames)
        missing = DEMOGRAPHIC_COLUMNS - columns
        assert not missing, (
            f"Prediction CSV missing demographic columns: {missing}"
        )

    def test_prediction_csv_has_rows(self, pred_csv_path):
        with open(pred_csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) > 0, "Prediction CSV is empty"

    def test_prediction_csv_wer_values_in_range(self, pred_csv_path):
        with open(pred_csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        for i, row in enumerate(rows[:50]):  # check first 50 rows
            wer = float(row["wer"])
            assert 0.0 <= wer <= 5.0, (
                f"Row {i}: WER {wer} outside range [0, 5.0]"
            )

    def test_prediction_csv_model_name_is_standard_lora(self, pred_csv_path):
        with open(pred_csv_path) as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert row["model"] == "standard-lora", (
            f"Expected model='standard-lora', got '{row['model']}'"
        )

    def test_prediction_csv_perturbation_is_clean(self, pred_csv_path):
        with open(pred_csv_path) as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert row["perturbation"] == "clean", (
            f"Expected perturbation='clean', got '{row['perturbation']}'"
        )


# ===========================================================================
# Analysis JSON (Plan 02-02, Tasks 2-3)
# ===========================================================================

class TestAnalysisJSON:
    """Validate analysis JSON structure and content."""

    @pytest.fixture
    def analysis(self):
        path = os.path.join(EVAL_DIR, "analysis_standard-lora.json")
        assert os.path.isfile(path), f"Missing {path}"
        return json.load(open(path))

    def test_analysis_has_model_name(self, analysis):
        assert analysis.get("model_name") == "standard-lora"

    def test_analysis_has_adapter_path(self, analysis):
        assert "adapter_path" in analysis

    def test_analysis_has_evaluation_date(self, analysis):
        assert "evaluation_date" in analysis

    def test_analysis_has_both_datasets(self, analysis):
        datasets = analysis.get("datasets", {})
        assert "fairspeech" in datasets, "Analysis missing fairspeech dataset"
        assert "commonvoice" in datasets, "Analysis missing commonvoice dataset"

    def test_fairspeech_uses_ethnicity_axis(self, analysis):
        fs = analysis["datasets"]["fairspeech"]
        assert fs.get("demographic_axis") == "ethnicity"

    def test_commonvoice_uses_accent_axis(self, analysis):
        cv = analysis["datasets"]["commonvoice"]
        assert cv.get("demographic_axis") == "accent"

    def test_each_dataset_has_overall_wer(self, analysis):
        for name, ds in analysis["datasets"].items():
            wer = ds.get("overall_wer")
            assert wer is not None, f"{name} missing overall_wer"
            assert 0.0 <= wer <= 1.0, f"{name} overall_wer={wer} out of range"

    def test_each_dataset_has_per_group(self, analysis):
        for name, ds in analysis["datasets"].items():
            groups = ds.get("per_group")
            assert groups is not None, f"{name} missing per_group"
            if name != "librispeech":
                # LS has no demographic axis so per_group is empty
                assert len(groups) > 0, f"{name} per_group is empty"

    def test_per_group_entries_have_wer_and_n(self, analysis):
        for name, ds in analysis["datasets"].items():
            for group, info in ds["per_group"].items():
                assert "wer" in info, f"{name}/{group} missing 'wer'"
                assert "n" in info, f"{name}/{group} missing 'n'"

    def test_fairspeech_has_ethnicity_groups(self, analysis):
        groups = set(analysis["datasets"]["fairspeech"]["per_group"].keys())
        # At minimum we expect some of these groups
        expected_some = {"White", "Black/AA", "Asian"}
        found = expected_some & groups
        assert len(found) >= 2, (
            f"Expected at least 2 of {expected_some} in FS groups, found {found}"
        )

    def test_commonvoice_has_accent_groups(self, analysis):
        groups = set(analysis["datasets"]["commonvoice"]["per_group"].keys())
        assert len(groups) >= 2, (
            f"Expected at least 2 accent groups in CV, found {len(groups)}"
        )

    def test_each_dataset_has_fairness_metrics_key(self, analysis):
        for name, ds in analysis["datasets"].items():
            assert "fairness_metrics" in ds, (
                f"{name} missing fairness_metrics key"
            )

    def test_fairness_metrics_fully_computed(self, analysis):
        """Fairness metrics should have actual values for FS and CV (not 'error').
        LibriSpeech has no demographic axis, so fairness metrics are expected to error.
        """
        for name, ds in analysis["datasets"].items():
            fm = ds["fairness_metrics"]
            if name == "librispeech":
                # LS has no demographic axis — "Too few valid groups" is expected
                continue
            assert "error" not in fm, (
                f"{name} fairness_metrics has error: {fm.get('error')}"
            )
            assert "max_min_ratio" in fm, f"{name} missing max_min_ratio"
            assert "relative_gap_pct" in fm, f"{name} missing relative_gap_pct"
            assert "wer_std" in fm, f"{name} missing wer_std"


# ===========================================================================
# Known Deviations (documented as xfail/skip)
# ===========================================================================

class TestBridgedGaps:
    """Tests for previously-xfailed gaps that are now bridged."""

    def test_cv_manifest_has_speaker_id(self):
        """Common Voice manifest should include speaker_id from client_id."""
        import pandas as pd
        cv = pd.read_csv(os.path.join(PROJECT_ROOT, "outputs", "manifests", "cv_train.csv"))
        assert "speaker_id" in cv.columns, "cv_train.csv missing speaker_id column"
        non_empty = cv["speaker_id"].notna() & (cv["speaker_id"].astype(str).str.strip() != "")
        assert non_empty.sum() > 0, "speaker_id column is all empty"

    def test_train_script_handles_speaker_disjoint(self):
        """train_standard_lora.py should handle speaker_id for disjoint splits."""
        with open(TRAIN_SCRIPT) as f:
            source = f.read()
        assert "GroupShuffleSplit" in source, "Script must use GroupShuffleSplit"
        assert "__unknown_speaker_" in source, (
            "Script must handle missing speaker_id with pseudo-IDs"
        )

    def test_bootstrap_cis_present(self):
        """Bootstrap CIs should be present for groups >= MIN_GROUP_SIZE."""
        path = os.path.join(EVAL_DIR, "analysis_standard-lora.json")
        data = json.load(open(path))
        for name, ds in data["datasets"].items():
            if name == "librispeech":
                continue  # LS has no demographic groups
            for group, info in ds["per_group"].items():
                if info.get("n", 0) >= 50:
                    assert info.get("ci_lower") is not None, (
                        f"{name}/{group} (n={info.get('n')}) missing bootstrap CI"
                    )

    def test_librispeech_evaluation_present(self):
        """LibriSpeech should now be included in evaluation."""
        path = os.path.join(EVAL_DIR, "analysis_standard-lora.json")
        data = json.load(open(path))
        assert "librispeech" in data["datasets"], (
            "Analysis JSON should include LibriSpeech evaluation"
        )

    def test_librispeech_manifest_exists(self):
        """LibriSpeech test-clean manifest should exist."""
        path = os.path.join(PROJECT_ROOT, "outputs", "manifests", "ls_test_clean.csv")
        assert os.path.isfile(path), f"Missing {path}"
        import pandas as pd
        df = pd.read_csv(path)
        assert len(df) > 2000, f"LS manifest too small: {len(df)} rows"
        assert "speaker_id" in df.columns, "LS manifest missing speaker_id"

    def test_librispeech_overall_wer_reasonable(self):
        """LS test-clean WER should be reasonable (< 10%)."""
        path = os.path.join(EVAL_DIR, "analysis_standard-lora.json")
        data = json.load(open(path))
        ls = data["datasets"].get("librispeech", {})
        wer = ls.get("overall_wer")
        assert wer is not None, "LibriSpeech missing overall_wer"
        assert 0.0 < wer < 0.10, f"LS WER {wer} outside expected range (0, 0.10)"

    def test_three_datasets_evaluated(self):
        """All three datasets (FS, CV, LS) should be in analysis."""
        path = os.path.join(EVAL_DIR, "analysis_standard-lora.json")
        data = json.load(open(path))
        datasets = set(data["datasets"].keys())
        expected = {"fairspeech", "commonvoice", "librispeech"}
        assert expected == datasets, f"Expected {expected}, got {datasets}"


class TestRemainingDeviations:
    """Deviations that are true data limitations (not bridgeable)."""

    @pytest.mark.xfail(
        reason="Fair-Speech has no speaker_id — hash_name is 1:1 per utterance, "
               "not a speaker identifier. This is a dataset limitation."
    )
    def test_fairspeech_has_speaker_id(self):
        """Fair-Speech manifest should have speaker_id for disjoint splits."""
        import pandas as pd
        fs = pd.read_csv(os.path.join(PROJECT_ROOT, "outputs", "manifests", "fs_train.csv"))
        assert "speaker_id" in fs.columns, "fs_train.csv missing speaker_id"

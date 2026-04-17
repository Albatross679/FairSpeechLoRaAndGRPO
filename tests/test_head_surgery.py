# tests for scripts/head_surgery — see tasks/prd-head-surgery-diagnosis.md

import json
from pathlib import Path

import pytest
import torch
from transformers import WhisperForConditionalGeneration

from scripts.head_surgery.head_mask_hook import BatchedHeadMaskHook, SerialHeadMaskHook
from scripts.head_surgery import repro_config as rc


@pytest.fixture(scope="module")
def whisper_cpu():
    """Lightweight Whisper model on CPU for hook-correctness tests.

    Uses whisper-tiny (same architecture shape, much smaller) so the test runs
    in seconds without a GPU. The hook code is shape-general and does not
    depend on Whisper-large-v3's specific dimensions.
    """
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    model.eval()
    return model


def _decoder_self_attn_out_proj_input(model, input_features, decoder_input_ids, layer_idx):
    """Capture the tensor that enters out_proj of decoder layer `layer_idx`."""
    captured = {}
    target = model.model.decoder.layers[layer_idx].self_attn.out_proj
    handle = target.register_forward_pre_hook(
        lambda _mod, args: captured.setdefault("x", args[0].detach().clone())
    )
    with torch.no_grad():
        model(input_features=input_features, decoder_input_ids=decoder_input_ids)
    handle.remove()
    return captured["x"]


def test_serial_head_mask_zeros_target_head(whisper_cpu):
    model = whisper_cpu
    num_heads = model.config.decoder_attention_heads
    head_dim = model.config.d_model // num_heads
    bsz = 2
    input_features = torch.randn(bsz, model.config.num_mel_bins, 3000)
    decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]] * bsz)
    ref_x = _decoder_self_attn_out_proj_input(model, input_features, decoder_input_ids, layer_idx=0)

    hook = SerialHeadMaskHook(model, layer_idx=0, head_idx=3)
    hook.install()
    try:
        masked_x = _decoder_self_attn_out_proj_input(
            model, input_features, decoder_input_ids, layer_idx=0
        )
    finally:
        hook.remove()

    ref_r = ref_x.view(bsz, -1, num_heads, head_dim)
    masked_r = masked_x.view(bsz, -1, num_heads, head_dim)

    assert torch.allclose(masked_r[:, :, 3, :], torch.zeros_like(masked_r[:, :, 3, :])), \
        "head 3 was not zeroed"
    for h in range(num_heads):
        if h == 3:
            continue
        assert torch.allclose(ref_r[:, :, h, :], masked_r[:, :, h, :]), \
            f"non-target head {h} was modified"


def test_serial_hook_remove_restores_behavior(whisper_cpu):
    model = whisper_cpu
    bsz = 1
    input_features = torch.randn(bsz, model.config.num_mel_bins, 3000)
    decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]])

    ref = _decoder_self_attn_out_proj_input(model, input_features, decoder_input_ids, 0)
    hook = SerialHeadMaskHook(model, layer_idx=0, head_idx=7)
    hook.install(); hook.remove()
    after = _decoder_self_attn_out_proj_input(model, input_features, decoder_input_ids, 0)
    assert torch.allclose(ref, after), "hook.remove() did not fully detach the hook"


def test_batched_hook_per_sample_zeros_correct_heads(whisper_cpu):
    model = whisper_cpu
    num_heads = model.config.decoder_attention_heads
    head_dim = model.config.d_model // num_heads
    bsz = 3
    input_features = torch.randn(bsz, model.config.num_mel_bins, 3000)
    decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]] * bsz)

    # Build a per-sample mask: sample 0 masks head 2; sample 1 masks head 5; sample 2 masks none.
    per_sample_mask = torch.ones(bsz, num_heads)
    per_sample_mask[0, 2] = 0.0
    per_sample_mask[1, 5] = 0.0

    hook = BatchedHeadMaskHook(model, layer_idx=0)
    hook.install()
    hook.set_batch_mask(per_sample_mask)
    try:
        captured = []
        target = model.model.decoder.layers[0].self_attn.out_proj
        h2 = target.register_forward_pre_hook(
            lambda _m, args: captured.append(args[0].detach().clone())
        )
        with torch.no_grad():
            model(input_features=input_features, decoder_input_ids=decoder_input_ids)
        h2.remove()
    finally:
        hook.remove()

    # The second hook registered sees the already-masked tensor (hooks run in
    # registration order; our batched hook registered first).
    x = captured[0].view(bsz, -1, num_heads, head_dim)
    assert torch.allclose(x[0, :, 2, :], torch.zeros_like(x[0, :, 2, :])), "sample 0 head 2 not zero"
    assert torch.allclose(x[1, :, 5, :], torch.zeros_like(x[1, :, 5, :])), "sample 1 head 5 not zero"
    # Sample 2 has no masked heads — no head should be all-zero by coincidence.
    for h in range(num_heads):
        assert not torch.allclose(x[2, :, h, :], torch.zeros_like(x[2, :, h, :])), \
            f"sample 2 head {h} unexpectedly zero"


def test_batched_hook_requires_set_batch_mask_before_forward(whisper_cpu):
    model = whisper_cpu
    bsz = 1
    input_features = torch.randn(bsz, model.config.num_mel_bins, 3000)
    decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]])
    hook = BatchedHeadMaskHook(model, layer_idx=0).install()
    try:
        with pytest.raises(RuntimeError, match="set_batch_mask"):
            with torch.no_grad():
                model(input_features=input_features, decoder_input_ids=decoder_input_ids)
    finally:
        hook.remove()


def test_model_revision_pinned():
    assert rc.MODEL_ID == "openai/whisper-large-v3"
    assert isinstance(rc.MODEL_REVISION, str) and len(rc.MODEL_REVISION) >= 7, \
        "MODEL_REVISION must be a HuggingFace revision (commit SHA or branch)"


def test_seed_is_deterministic():
    assert rc.SEED == 20260417


def test_generate_config_pinned_to_midterm_defaults():
    g = rc.GENERATE_CONFIG
    assert g["max_new_tokens"] == 440
    assert g["language"] == "en"
    assert g["task"] == "transcribe"
    assert g["num_beams"] == 1
    assert g["do_sample"] is False
    assert g["temperature"] == 0.0
    assert g["repetition_penalty"] == 1.0
    assert g["no_repeat_ngram_size"] == 0
    assert g["length_penalty"] == 1.0


def test_indian_accent_ids_count():
    # Adapted for CV25: strict single-label match gives 510 rows (CV24 had 511).
    ids = rc.load_indian_accent_ids()
    assert len(ids) == 510
    assert len(set(ids)) == 510, "utterance IDs must be unique"


def test_indian_accent_ids_sorted_and_stable():
    ids = rc.load_indian_accent_ids()
    assert ids == sorted(ids), "IDs must be sorted for stable iteration"
    assert ids == rc.load_indian_accent_ids()

# tests for scripts/head_surgery — see tasks/prd-head-surgery-diagnosis.md

import pytest
import torch
from transformers import WhisperForConditionalGeneration

from scripts.head_surgery.head_mask_hook import SerialHeadMaskHook


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

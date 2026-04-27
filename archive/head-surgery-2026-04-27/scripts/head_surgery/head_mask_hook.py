"""Per-head attention-mask forward hooks for Whisper decoder (T1).

Two variants:
  SerialHeadMaskHook  — zeros head `h` at layer `L` uniformly across the batch.
  BatchedHeadMaskHook — zeros different heads per sample in a batch
                        (per-sample (layer, head) pairs).

Hook point: forward_pre_hook on decoder.layers[L].self_attn.out_proj.
The tensor at that point has shape [batch, tgt_len, num_heads * head_dim].
We reshape to [batch, tgt_len, num_heads, head_dim], apply the mask, and
reshape back. This matches the Calm-Whisper semantics of "zero a head's
contribution" (arxiv 2505.12969, §Methods).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


def _resolve_out_proj(model, layer_idx: int) -> nn.Module:
    return model.model.decoder.layers[layer_idx].self_attn.out_proj


def _head_dims(model) -> Tuple[int, int]:
    num_heads = model.config.decoder_attention_heads
    head_dim = model.config.d_model // num_heads
    return num_heads, head_dim


class SerialHeadMaskHook:
    """Zero a single (layer, head) uniformly across the batch."""

    def __init__(self, model, layer_idx: int, head_idx: int):
        self.model = model
        self.layer_idx = layer_idx
        self.head_idx = head_idx
        self._handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self._num_heads, self._head_dim = _head_dims(model)

    def install(self) -> "SerialHeadMaskHook":
        if self._handle is not None:
            raise RuntimeError("Hook already installed")
        target = _resolve_out_proj(self.model, self.layer_idx)
        self._handle = target.register_forward_pre_hook(self._hook)
        return self

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _hook(self, _module, args):
        (x,) = args
        # x: [bsz, tgt_len, num_heads * head_dim]
        bsz, tgt_len, _ = x.shape
        x_r = x.view(bsz, tgt_len, self._num_heads, self._head_dim)
        mask = torch.ones(self._num_heads, device=x.device, dtype=x.dtype)
        mask[self.head_idx] = 0.0
        x_masked = x_r * mask.view(1, 1, self._num_heads, 1)
        return (x_masked.view(bsz, tgt_len, self._num_heads * self._head_dim),)

    def __enter__(self):
        return self.install()

    def __exit__(self, *exc):
        self.remove()


class BatchedHeadMaskHook:
    """Per-sample head masking at a fixed layer.

    Call set_batch_mask(mask) once per batch before forward. `mask` has shape
    [batch, num_heads] with 1=keep, 0=zero. The mask tensor is reused across
    every autoregressive decoding step within the same batch.
    """

    def __init__(self, model, layer_idx: int):
        self.model = model
        self.layer_idx = layer_idx
        self._handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self._num_heads, self._head_dim = _head_dims(model)
        self._mask: Optional[torch.Tensor] = None

    def install(self) -> "BatchedHeadMaskHook":
        if self._handle is not None:
            raise RuntimeError("Hook already installed")
        target = _resolve_out_proj(self.model, self.layer_idx)
        self._handle = target.register_forward_pre_hook(self._hook)
        return self

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self._mask = None

    def set_batch_mask(self, mask: torch.Tensor) -> None:
        """mask: [batch, num_heads] float tensor with 1=keep, 0=zero."""
        if mask.ndim != 2 or mask.shape[1] != self._num_heads:
            raise ValueError(
                f"mask must be [batch, {self._num_heads}]; got {tuple(mask.shape)}"
            )
        self._mask = mask

    def _hook(self, _module, args):
        if self._mask is None:
            raise RuntimeError(
                "BatchedHeadMaskHook: set_batch_mask(mask) must be called before forward"
            )
        (x,) = args
        bsz, tgt_len, _ = x.shape
        if bsz != self._mask.shape[0]:
            raise RuntimeError(
                f"batch size {bsz} != mask batch {self._mask.shape[0]}"
            )
        x_r = x.view(bsz, tgt_len, self._num_heads, self._head_dim)
        m = self._mask.to(device=x.device, dtype=x.dtype).view(bsz, 1, self._num_heads, 1)
        x_masked = x_r * m
        return (x_masked.view(bsz, tgt_len, self._num_heads * self._head_dim),)

    def __enter__(self):
        return self.install()

    def __exit__(self, *exc):
        self.remove()

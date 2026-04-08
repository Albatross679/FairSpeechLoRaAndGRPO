"""
Data collator for Qwen3-ASR fine-tuning with prefix masking.

Implements DataCollatorForQwen3ASR which converts raw audio + transcript pairs
from ASRFairnessDataset into the chat-format inputs expected by Qwen3-ASR,
with labels masked to -100 for all non-transcript tokens (system/user prompt).

This ensures the training loss is computed only on the transcript portion,
preventing the model from learning to predict the prompt template.

API: Qwen3ASRProcessor takes (text=..., audio=...) where text comes from
apply_chat_template (user-only) + manually appended assistant response.

Usage:
    processor = asr_wrapper.processor
    collator = DataCollatorForQwen3ASR(processor)
    batch = collator(features)  # features from ASRFairnessDataset
"""

import torch
import numpy as np


# -- Constants ----------------------------------------------------------------

LANGUAGE_PREFIX = "language English"
ASR_TEXT_TAG = "<asr_text>"
IGNORE_INDEX = -100


# -- Data Collator ------------------------------------------------------------

class DataCollatorForQwen3ASR:
    """Collates ASRFairnessDataset items into Qwen3-ASR chat-format training batches.

    For each sample, builds the chat message in Qwen3-ASR format:
        user: [audio content]
        assistant: language English<asr_text>{transcript}

    Uses apply_chat_template for the user message, then manually appends the
    assistant response (since the template strips assistant content). Processes
    through the processor to get input_ids, attention_mask, input_features,
    and feature_attention_mask. Creates labels by masking prefix tokens with -100.

    Args:
        processor: Qwen3ASRProcessor from qwen-asr wrapper.
    """

    def __init__(self, processor):
        self.processor = processor
        self.tokenizer = processor.tokenizer

        # Cache the <asr_text> token ID for boundary detection
        self._asr_tag_ids = self.tokenizer.encode(
            ASR_TEXT_TAG, add_special_tokens=False
        )

    def __call__(self, features):
        """Collate a list of dataset items into a training batch.

        Args:
            features: List of dicts from ASRFairnessDataset.__getitem__:
                {"audio": tensor(samples,), "transcript": str,
                 "demographic_group": str, "utterance_id": str}

        Returns:
            Dict with keys: input_ids, attention_mask, input_features,
            feature_attention_mask, labels. All tensors are padded to the
            same length within the batch.
        """
        all_input_ids = []
        all_attention_mask = []
        all_input_features = []
        all_feature_attention_mask = []
        all_labels = []

        for feature in features:
            # Extract audio as numpy array (processor expects numpy)
            audio = feature["audio"]
            if isinstance(audio, torch.Tensor):
                audio_np = audio.numpy()
            else:
                audio_np = np.asarray(audio)

            transcript = feature["transcript"]

            # Step 1: Get user-only template with generation prompt
            conversation = [
                {
                    "role": "user",
                    "content": [{"type": "audio", "audio": audio_np}],
                },
            ]
            user_text = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False,
            )

            # Step 2: Append assistant response manually
            assistant_text = f"{LANGUAGE_PREFIX}{ASR_TEXT_TAG}{transcript}"
            full_text = user_text + assistant_text + "<|im_end|>\n"

            # Step 3: Process through Qwen3ASR processor
            processed = self.processor(
                text=full_text,
                audio=[audio_np],
                return_tensors="pt",
                padding=False,
            )

            input_ids = processed["input_ids"].squeeze(0)
            attention_mask = processed["attention_mask"].squeeze(0)
            input_feats = processed["input_features"].squeeze(0)
            feat_attn_mask = processed["feature_attention_mask"].squeeze(0)

            # Step 4: Create labels with prefix masking
            labels = input_ids.clone()
            prefix_end = self._find_transcript_start(input_ids)
            labels[:prefix_end] = IGNORE_INDEX

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_input_features.append(input_feats)
            all_feature_attention_mask.append(feat_attn_mask)
            all_labels.append(labels)

        # Pad sequences to same length within batch
        batch = self._pad_batch(
            all_input_ids, all_attention_mask,
            all_input_features, all_feature_attention_mask,
            all_labels,
        )

        return batch

    def _find_transcript_start(self, input_ids):
        """Find the token position where the actual transcript starts.

        Searches for the <asr_text> tag token in input_ids and returns
        the position right after it, so everything before (including the tag)
        is masked with -100.

        Args:
            input_ids: 1D tensor of token IDs.

        Returns:
            int: Index of the first transcript token in input_ids.
        """
        input_list = input_ids.tolist()
        tag_len = len(self._asr_tag_ids)

        for i in range(len(input_list) - tag_len + 1):
            if input_list[i:i + tag_len] == self._asr_tag_ids:
                return i + tag_len

        # Fallback: mask everything up to 80% (should not happen)
        print(f"  WARNING: Could not find <asr_text> boundary, "
              f"masking first 80% of {len(input_ids)} tokens")
        return int(len(input_ids) * 0.8)

    def _pad_batch(self, all_input_ids, all_attention_mask,
                   all_input_features, all_feature_attention_mask,
                   all_labels):
        """Pad all sequences in the batch to the same length.

        Args:
            all_input_ids: List of 1D tensors (variable length).
            all_attention_mask: List of 1D tensors (variable length).
            all_input_features: List of 2D/3D tensors (mel features).
            all_feature_attention_mask: List of 1D tensors (variable length).
            all_labels: List of 1D tensors (variable length).

        Returns:
            Dict with padded batch tensors.
        """
        # Pad input_ids, attention_mask, labels to max text length
        max_text_len = max(ids.shape[0] for ids in all_input_ids)

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        for input_ids, attn_mask, labels in zip(
            all_input_ids, all_attention_mask, all_labels
        ):
            text_pad_len = max_text_len - input_ids.shape[0]
            if text_pad_len > 0:
                padded_input_ids.append(torch.cat([
                    input_ids,
                    torch.full((text_pad_len,), pad_token_id, dtype=input_ids.dtype),
                ]))
                padded_attention_mask.append(torch.cat([
                    attn_mask,
                    torch.zeros(text_pad_len, dtype=attn_mask.dtype),
                ]))
                padded_labels.append(torch.cat([
                    labels,
                    torch.full((text_pad_len,), IGNORE_INDEX, dtype=labels.dtype),
                ]))
            else:
                padded_input_ids.append(input_ids)
                padded_attention_mask.append(attn_mask)
                padded_labels.append(labels)

        # Pad input_features (mel spectrograms) to max feature length
        max_feat_len = max(f.shape[-1] for f in all_input_features)

        padded_input_features = []
        padded_feature_attention_mask = []

        for input_feats, feat_mask in zip(
            all_input_features, all_feature_attention_mask
        ):
            feat_pad_len = max_feat_len - input_feats.shape[-1]
            if feat_pad_len > 0:
                # Pad along the time dimension (last dim)
                pad_shape = list(input_feats.shape)
                pad_shape[-1] = feat_pad_len
                padded_input_features.append(torch.cat([
                    input_feats,
                    torch.zeros(pad_shape, dtype=input_feats.dtype),
                ], dim=-1))
                padded_feature_attention_mask.append(torch.cat([
                    feat_mask,
                    torch.zeros(feat_pad_len, dtype=feat_mask.dtype),
                ]))
            else:
                padded_input_features.append(input_feats)
                padded_feature_attention_mask.append(feat_mask)

        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "input_features": torch.stack(padded_input_features),
            "feature_attention_mask": torch.stack(padded_feature_attention_mask),
            "labels": torch.stack(padded_labels),
        }

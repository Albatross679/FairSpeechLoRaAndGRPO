"""
Data collator for Qwen3-ASR fine-tuning with prefix masking.

Implements DataCollatorForQwen3ASR which converts raw audio + transcript pairs
from ASRFairnessDataset into the chat-format inputs expected by Qwen3-ASR,
with labels masked to -100 for all non-transcript tokens (system/user prompt).

This ensures the training loss is computed only on the transcript portion,
preventing the model from learning to predict the prompt template.

Based on the official Qwen3-ASR SFT script data collation pattern.
Reference: RESEARCH.md Pitfall 6 (prefix masking is critical).

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

    Then applies the processor to get input_ids, attention_mask, input_features,
    and feature_attention_mask. Creates labels by copying input_ids and masking
    prefix tokens (everything before the transcript) with -100.

    Args:
        processor: Qwen3ASRProcessor from qwen-asr wrapper.
    """

    def __init__(self, processor):
        self.processor = processor
        self.tokenizer = processor.tokenizer

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

            # Build chat messages in Qwen3-ASR format
            # User message contains the audio content
            # Assistant message contains the language prefix + transcript
            assistant_text = f"{LANGUAGE_PREFIX}{ASR_TEXT_TAG}{transcript}"

            messages = [
                {
                    "role": "user",
                    "content": [{"type": "audio", "audio": audio_np}],
                },
                {
                    "role": "assistant",
                    "content": assistant_text,
                },
            ]

            # Apply processor with chat template (no generation prompt)
            text = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=False,
                tokenize=False,
            )

            # Process text + audio through the processor
            processed = self.processor(
                text=text,
                audios=[audio_np],
                return_tensors="pt",
                padding=False,
            )

            input_ids = processed["input_ids"].squeeze(0)
            attention_mask = processed["attention_mask"].squeeze(0)
            input_feats = processed["input_features"].squeeze(0)
            feat_attn_mask = processed["feature_attention_mask"].squeeze(0)

            # Create labels with prefix masking
            # Loss should only be computed on transcript tokens, not on
            # system/user prompt tokens (Pitfall 6 from RESEARCH.md)
            labels = input_ids.clone()

            # Find where the assistant response transcript starts
            # Tokenize the assistant prefix to find the masking boundary
            prefix_end = self._find_transcript_start(input_ids, transcript)
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

    def _find_transcript_start(self, input_ids, transcript):
        """Find the token position where the actual transcript starts.

        Searches for the <asr_text> tag token(s) in the input_ids and returns
        the position right after them, so everything before (including the tag)
        is masked with -100.

        Args:
            input_ids: 1D tensor of token IDs.
            transcript: The raw transcript string (for fallback heuristic).

        Returns:
            int: Index of the first transcript token in input_ids.
        """
        # Tokenize the asr_text tag to find it in the sequence
        asr_tag_ids = self.tokenizer.encode(
            ASR_TEXT_TAG, add_special_tokens=False
        )

        # Search for the tag sequence in input_ids
        input_list = input_ids.tolist()
        tag_len = len(asr_tag_ids)

        for i in range(len(input_list) - tag_len + 1):
            if input_list[i:i + tag_len] == asr_tag_ids:
                # Found the tag; transcript starts right after it
                return i + tag_len

        # Fallback: if we cannot find the exact tag, tokenize the full
        # assistant prefix and mask up to that length.
        # This handles cases where the tokenizer merges the tag differently.
        prefix_text = f"{LANGUAGE_PREFIX}{ASR_TEXT_TAG}"
        prefix_ids = self.tokenizer.encode(prefix_text, add_special_tokens=False)
        prefix_len = len(prefix_ids)

        # Search for the prefix sequence
        for i in range(len(input_list) - prefix_len + 1):
            if input_list[i:i + prefix_len] == prefix_ids:
                return i + prefix_len

        # Last resort: mask the first 80% of tokens (conservative estimate)
        # This should not happen with well-formed inputs
        print(f"  WARNING: Could not find transcript boundary, "
              f"masking first 80% of {len(input_ids)} tokens")
        return int(len(input_ids) * 0.8)

    def _pad_batch(self, all_input_ids, all_attention_mask,
                   all_input_features, all_feature_attention_mask,
                   all_labels):
        """Pad all sequences in the batch to the same length.

        Args:
            all_input_ids: List of 1D tensors (variable length).
            all_attention_mask: List of 1D tensors (variable length).
            all_input_features: List of 2D tensors (mel features).
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

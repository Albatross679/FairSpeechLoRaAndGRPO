"""
Reward computation for GRPO fairness-aware ASR fine-tuning.

Implements composite reward: R = (1-lambda)(1-WER) + lambda(-|WER_g - WER_mean|)
where the accuracy term rewards low WER and the fairness term penalizes
demographic WER gaps.

Uses jiwer for WER computation and whisper_normalizer for text normalization
(matching the evaluation pipeline in evaluate_adapter.py).

Usage:
    reward_computer = RewardComputer(lambda_=0.3)
    rewards = reward_computer(candidates, references, demographics)
    # rewards: (batch_size, G) tensor
"""

import numpy as np
import torch
import jiwer

try:
    from whisper.normalizers import EnglishTextNormalizer
except ImportError:
    from whisper_normalizer.english import EnglishTextNormalizer

_normalizer = EnglishTextNormalizer()


class RewardComputer:
    """Composite accuracy + fairness reward for GRPO.

    R_acc = (1 - lambda_) * (1 - WER_i)
    R_fair = lambda_ * (-|WER_g - WER_mean|)
    R = R_acc + R_fair

    For lambda_=0, this is pure accuracy reward (standard SFT baseline).
    For lambda_>0, the fairness penalty encourages equal WER across groups.

    Args:
        lambda_: Fairness weight in [0, 1]. 0 = accuracy only.
        normalize: Whether to apply English text normalization.
        fairness_window: Number of recent batches to accumulate for
            fairness reward computation. Mitigates Pitfall #5 (single-batch
            fairness signal is trivially zero with small batch sizes).
        wer_floor: If mean WER exceeds baseline_wer * wer_floor_factor,
            fairness bonus is zeroed to prevent reward hacking (Pitfall #3).
        baseline_wer: Reference WER from SFT baseline for floor check.
        wer_floor_factor: Multiplier for baseline WER floor (default 1.1).
    """

    def __init__(
        self,
        lambda_: float = 0.0,
        normalize: bool = True,
        fairness_window: int = 10,
        baseline_wer: float | None = None,
        wer_floor_factor: float = 1.1,
    ):
        self.lambda_ = lambda_
        self.normalize = normalize
        self.fairness_window = fairness_window
        self.baseline_wer = baseline_wer
        self.wer_floor_factor = wer_floor_factor

        # Rolling window for group WER accumulation (Pitfall #5)
        self._group_wer_history: list[dict[str, list[float]]] = []

    def _normalize(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""
        if self.normalize:
            return _normalizer(text)
        return text.strip()

    def _compute_utterance_wer(self, hypothesis: str, reference: str) -> float:
        """Compute WER for a single utterance."""
        ref = self._normalize(reference)
        hyp = self._normalize(hypothesis)
        if not ref:
            return 0.0 if not hyp else 1.0
        if not hyp:
            return 1.0
        try:
            return jiwer.wer(ref, hyp)
        except Exception:
            return 1.0

    def _compute_fairness_penalty(
        self, wers: list[float], demographics: list[str]
    ) -> torch.Tensor:
        """Compute per-sample fairness penalty using rolling window.

        Accumulates group WERs across recent batches to ensure
        multiple demographic groups contribute to the signal.

        Returns:
            Tensor of shape (batch_size,) with fairness penalties (<= 0).
        """
        batch_size = len(wers)

        # Build group WER map for this batch
        batch_group_wers: dict[str, list[float]] = {}
        for wer_val, demo in zip(wers, demographics):
            demo = str(demo).strip()
            if demo and demo != "nan":
                batch_group_wers.setdefault(demo, []).append(wer_val)

        # Add to rolling window
        self._group_wer_history.append(batch_group_wers)
        if len(self._group_wer_history) > self.fairness_window:
            self._group_wer_history.pop(0)

        # Aggregate group WERs across window
        agg_group_wers: dict[str, list[float]] = {}
        for entry in self._group_wer_history:
            for group, group_wers in entry.items():
                agg_group_wers.setdefault(group, []).extend(group_wers)

        # Need at least 2 groups for fairness signal
        if len(agg_group_wers) < 2:
            return torch.zeros(batch_size)

        # Mean WER across all groups (group-level, not utterance-level)
        group_means = {g: np.mean(ws) for g, ws in agg_group_wers.items()}
        overall_mean = np.mean(list(group_means.values()))

        # Per-sample fairness penalty: -|WER_g - WER_mean|
        r_fair = torch.zeros(batch_size)
        for i, demo in enumerate(demographics):
            demo = str(demo).strip()
            if demo and demo != "nan" and demo in group_means:
                r_fair[i] = -abs(group_means[demo] - overall_mean)

        return r_fair

    def __call__(
        self,
        candidates: list[list[str]],
        references: list[str],
        demographics: list[str],
    ) -> tuple[torch.Tensor, dict]:
        """Compute composite rewards for G candidate sets.

        Args:
            candidates: List of G lists, each containing batch_size
                hypothesis strings. candidates[g][i] is the g-th candidate
                for the i-th input.
            references: List of batch_size reference transcripts.
            demographics: List of batch_size demographic group labels.

        Returns:
            rewards: Tensor of shape (batch_size, G).
            metrics: Dict with diagnostic values for logging.
        """
        batch_size = len(references)
        G = len(candidates)
        rewards = torch.zeros(batch_size, G)

        all_wers = []  # (G, batch_size) for diagnostics

        for g in range(G):
            # Per-utterance WER for this candidate set
            wers = [
                self._compute_utterance_wer(candidates[g][i], references[i])
                for i in range(batch_size)
            ]
            all_wers.append(wers)
            wers_tensor = torch.tensor(wers, dtype=torch.float32)

            # Accuracy reward: (1 - lambda)(1 - WER)
            r_acc = (1.0 - self.lambda_) * (1.0 - wers_tensor)

            if self.lambda_ > 0:
                # Fairness penalty
                r_fair = self._compute_fairness_penalty(wers, demographics)

                # WER floor check (Pitfall #3: reward hacking)
                mean_wer = wers_tensor.mean().item()
                if (
                    self.baseline_wer is not None
                    and mean_wer > self.baseline_wer * self.wer_floor_factor
                ):
                    r_fair = torch.zeros_like(r_fair)

                rewards[:, g] = r_acc + self.lambda_ * r_fair
            else:
                rewards[:, g] = r_acc

        # Diagnostics
        all_wers_np = np.array(all_wers)  # (G, batch_size)
        per_candidate_mean_wer = all_wers_np.mean(axis=1)  # (G,)
        reward_per_candidate = rewards.mean(dim=0)  # (G,)

        # Fraction of prompts where all G candidates get identical reward
        # (Pitfall #1: if > 50%, GRPO is not getting signal)
        reward_stds = rewards.std(dim=1)  # (batch_size,)
        frac_zero_std = (reward_stds < 1e-6).float().mean().item()

        metrics = {
            "reward/mean": rewards.mean().item(),
            "reward/std": rewards.std().item(),
            "reward/frac_zero_std": frac_zero_std,
            "wer/mean": all_wers_np.mean(),
            "wer/std_across_candidates": all_wers_np.std(axis=0).mean(),
            "wer/per_candidate_mean": per_candidate_mean_wer.tolist(),
        }

        # Per-group WER diagnostics (for monitoring reward hacking)
        group_wer_means = {}
        for g_idx in range(G):
            for i, demo in enumerate(demographics):
                demo = str(demo).strip()
                if demo and demo != "nan":
                    group_wer_means.setdefault(demo, []).append(all_wers[g_idx][i])
        for group, wers_list in group_wer_means.items():
            metrics[f"wer/group_{group}"] = np.mean(wers_list)

        return rewards, metrics

    def reset_window(self):
        """Reset the rolling fairness window (e.g., between lambda runs)."""
        self._group_wer_history.clear()

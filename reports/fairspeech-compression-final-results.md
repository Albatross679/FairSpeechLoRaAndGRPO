# FairSpeech Compression Final Results Memo

Updated: 2026-04-30

## Completion State

The FairSpeech compression evaluation is data-complete.

| Item | Result |
|---|---:|
| Models included | 9 |
| Models excluded | 0 |
| Audio variants per model | 6 |
| Completed inference runs | 54/54 |
| Validated prediction CSVs | 54/54 |
| Rows per CSV | 26,471 |
| Total prediction rows | 1,429,434 |
| Ethnicity metric cells | 378 |
| Bootstrap CI rows | 378 |
| Bootstrap comparison rows | 108 |

Artifacts:

- Prediction validation: `datasets/fairspeech_compression/full_eval/full_prediction_validation.json`
- Final metrics: `/workspace/fairspeech-full-eval/results/full_metrics/`
- Final plots: `/workspace/fairspeech-full-eval/results/full_plots/`
- Bootstrap CIs: `datasets/fairspeech_compression/full_eval/bootstrap_ci_200/`
- Project-state dashboard: `docs/project-state.html`

## Topline Finding

Compression and bandwidth loss mostly increase WER, but the effect is not uniformly distributed across ethnicity groups. The strongest and clearest disparity amplification appears under the severe stressors: 8 kHz bottleneck and MP3 16 kbps. Mild stressors, 12 kHz and MP3 64 kbps, are usually small on average, though individual model/group cells can still move.

The best overall accuracy/fairness slot remains `qwen3-asr-1.7b / MP3 64 kbps` with WER 4.7%, MMR 2.82x, and MGD 1.23 pp. The Granite 8B baseline adds a useful middle-frontier point: WER 8.0%, MMR 2.21x, and MGD 1.68 pp.

## Sample-Rate Bottlenecks

| Variant | Mean WER delta vs baseline | Mean MGD delta | Mean MMR delta | Interpretation |
|---|---:|---:|---:|---|
| 12 kHz bottleneck | +0.33 pp | +0.07 pp | +0.03x | Mild average degradation |
| 8 kHz bottleneck | +1.14 pp | +0.21 pp | +0.09x | Clearer degradation and disparity amplification |

The 12 kHz bottleneck is a light stressor. Average WER rises by only +0.33 pp, but the group deltas are not perfectly uniform: Black/AA averages +0.54 pp while Asian averages -0.03 pp across models.

The 8 kHz bottleneck is substantially stronger. Average WER rises +1.14 pp, and Black/AA has the largest average group increase at +2.12 pp, compared with +0.52 pp for White and +0.61 pp for Asian. This supports the hypothesis that narrowband loss can amplify existing group error gaps rather than just shift every group upward equally.

## MP3 Codec Artifacts

| Variant | Mean WER delta vs baseline | Mean MGD delta | Mean MMR delta | Interpretation |
|---|---:|---:|---:|---|
| MP3 64 kbps | +0.34 pp | +0.06 pp | +0.02x | Mild average degradation |
| MP3 32 kbps | +0.77 pp | +0.16 pp | +0.01x | Moderate degradation |
| MP3 16 kbps | +2.30 pp | +0.39 pp | -0.06x | Severe degradation; ratio alone can hide harm |

MP3 64 kbps behaves similarly to 12 kHz on average, but with model-specific exceptions. MP3 32 kbps is a clearer stressor, raising average WER by +0.77 pp and increasing MGD by +0.16 pp.

MP3 16 kbps is the strongest codec stressor. Mean WER rises +2.30 pp, with Black/AA averaging +4.11 pp versus +1.01 pp for White. The MMR delta is slightly negative on average, which is a useful warning: when all groups degrade but the best group degrades too, a ratio metric can look stable or improve while absolute harm increases. MGD and group-level WER deltas are better for this part of the story.

Largest single group increases include:

- `whisper-small / MP3 16 kbps / Black/AA`: +6.77 pp
- `granite-speech-3.3-2b / MP3 16 kbps / Black/AA`: +6.54 pp
- `granite-speech-3.3-2b / MP3 32 kbps / Black/AA`: +5.41 pp
- `granite-speech-3.3-2b / 8 kHz bottleneck / Black/AA`: +5.14 pp
- `granite-speech-3.3-8b / MP3 16 kbps / Black/AA`: +4.94 pp

## Model-Family Readout

| Model | Baseline WER | Baseline MMR | Baseline MGD |
|---|---:|---:|---:|
| `wav2vec2-large` | 31.7% | 1.58x | 3.37 pp |
| `whisper-small` | 11.4% | 4.08x | 2.95 pp |
| `whisper-medium` | 8.8% | 3.25x | 2.27 pp |
| `whisper-large-v3` | 7.8% | 3.15x | 2.20 pp |
| `qwen3-asr-0.6b` | 5.9% | 3.66x | 1.88 pp |
| `qwen3-asr-1.7b` | 4.7% | 3.03x | 1.35 pp |
| `canary-qwen-2.5b` | 6.6% | 3.45x | 1.80 pp |
| `granite-speech-3.3-2b` | 9.4% | 2.50x | 2.33 pp |
| `granite-speech-3.3-8b` | 8.0% | 2.21x | 1.68 pp |

`wav2vec2-large` has the lowest MMR but very high WER, so it is an example of low-accuracy parity rather than a desirable fairness/accuracy point. Whisper improves accuracy but keeps large relative group ratios. Qwen3 1.7B is the strongest accuracy point and also has the lowest MGD slot. Granite 8B is not the most accurate, but it is a useful lower-disparity middle-frontier model.

## Confidence-Interval Check

The 200-resample bootstrap CI check covers all 54 completed runs and 378 ethnicity cells.

- Worst-vs-best ethnicity gap: 54/54 CI-separated.
- Black/AA-vs-White gap: 49/54 CI-separated.
- The 5 Black/AA-vs-White overlaps are all `whisper-large-v3` slots.

## Figure QA

All 12 plot PNGs were generated under `/workspace/fairspeech-full-eval/results/full_plots/`.

- Group WER bars: 9 model-specific figures, each 2400 x 1200.
- Degradation, fairness heatmap, and insertion subtype figures: 2000 x 1200.
- Image channel ranges are nonzero across all figures, so none are blank.
- Variant labels are present for baseline, 12 kHz, 8 kHz, MP3 64, MP3 32, and MP3 16 views.

## Limitations

The results are FairSpeech-only and use ethnicity as the main demographic axis. The audio transformations are controlled bottlenecks and codec passes, not full real-world channel simulations. The memo reports model outputs as scored by the existing WER pipeline; blank hypotheses are treated as model outputs, not missing WER. Runtime and batching reflect the local single-GPU setup and the selected `160s / max16` guard.

## Paper Use

The cleanest claim is:

> Controlled audio degradation can amplify ASR fairness gaps, especially under severe bandwidth or codec loss. Mild degradation has small average effects, but severe 8 kHz and MP3 16 kbps conditions increase absolute WER gaps, with Black/AA speakers often seeing the largest WER increases. Accuracy-only and ratio-only fairness summaries can hide this, so paired group deltas and MGD should be reported alongside WER and MMR.

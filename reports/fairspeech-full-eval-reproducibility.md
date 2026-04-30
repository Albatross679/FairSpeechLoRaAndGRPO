# FairSpeech Full Evaluation Reproducibility Notes

Updated: 2026-04-30

## Run Identity

| Item | Value |
|---|---|
| Repo branch | `master` |
| Repo commit at run environment | `099f675` |
| Python | `3.11.10` |
| Evaluation root | `/workspace/fairspeech-full-eval` |
| Project root | `/workspace/FairSpeechLoRaAndGRPO` |
| Prediction root | `/workspace/fairspeech-full-eval/results/full` |
| Metrics root | `/workspace/fairspeech-full-eval/results/full_metrics` |
| Plot root | `/workspace/fairspeech-full-eval/results/full_plots` |
| Derived audio root | `/workspace/fairspeech-full-eval/variants/full` |
| HF cache root | `/workspace/fairspeech-full-eval/hf-cache/hub` |

## Hardware

| Item | Value |
|---|---|
| CPU | AMD EPYC 7352 24-Core Processor |
| Logical CPUs | 48 |
| System RAM | 251 GiB |
| GPU | NVIDIA RTX 4000 Ada Generation |
| GPU memory | 20,475 MiB |
| NVIDIA driver | 550.127.05 |
| Workspace filesystem | `/workspace`, 2.1P total, 617T available at final check |
| Root overlay | `/`, 20G total, 18G available at final check |

## Runtime Package Versions

Key packages in the project venv:

| Package | Version |
|---|---|
| `torch` | 2.6.0 |
| `torchaudio` | 2.6.0 |
| `transformers` | 4.57.6 |
| `accelerate` | 1.12.0 |
| `pandas` | 3.0.2 |
| `numpy` | 2.4.4 |
| `matplotlib` | 3.10.9 |
| `seaborn` | 0.13.2 |
| `librosa` | 0.11.0 |
| `soundfile` | 0.13.1 |
| `jiwer` | 3.1.0 |
| `editdistance` | 0.8.1 |
| `datasets` | 4.8.5 |

The editable project package in the venv points at commit `099f6754838b20a74e89d12d712fc3a002f3a60b`.

## Preserved Metadata

| Artifact | Path |
|---|---|
| Full run matrix | `datasets/fairspeech_compression/full_eval/full_eval_run_matrix.json` |
| Full status ledger | `datasets/fairspeech_compression/full_eval/full_eval_status.jsonl` |
| Full prediction validation | `datasets/fairspeech_compression/full_eval/full_prediction_validation.json` |
| Batch-plan validation | `datasets/fairspeech_compression/full_eval/full_batch_plan_validation.json` |
| Gen 1/2 model cache manifest | `datasets/fairspeech_compression/model_cache/whisper_cache_manifest.json` |
| Gen 3 model cache manifest | `datasets/fairspeech_compression/model_cache/gen3_cache_manifest.json` |
| Metrics summary | `/workspace/fairspeech-full-eval/results/full_metrics/fairspeech_compression_metrics_summary.json` |
| Bootstrap summary | `datasets/fairspeech_compression/full_eval/bootstrap_ci_200/fairspeech_compression_metrics_summary.json` |
| Result memo | `reports/fairspeech-compression-final-results.md` |

Model cache manifests preserve model keys, Hugging Face IDs, snapshot paths, snapshot sizes, download status, smoke status inputs, and any recorded errors. Per-run metadata JSONs under `/workspace/fairspeech-full-eval/results/full/<model>/` preserve model key, device, batch plan, audio variant, elapsed time, and overall WER.

## Preserved Logs

| Log family | Path |
|---|---|
| Audio generation | `datasets/fairspeech_compression/logs/full_eval/full_audio_generation.log` |
| Audio validation | `datasets/fairspeech_compression/logs/full_eval/full_audio_validation.log` |
| Full matrix tmux log | `datasets/fairspeech_compression/logs/full_eval/run_matrix_tmux.log` |
| Per-run inference logs | `datasets/fairspeech_compression/logs/full_eval/infer_<model>_<variant>.log` |
| Metrics log | `datasets/fairspeech_compression/logs/full_eval/full_metrics.log` |
| Plot log | `datasets/fairspeech_compression/logs/full_eval/full_plots.log` |
| Historical postprocess watcher log | `datasets/fairspeech_compression/logs/full_eval/postprocess_watcher.log` |

## Heavy Artifact Policy

Heavy audio and prediction payloads are outside the repository tree:

| Payload | Location | Size at final check |
|---|---|---:|
| Derived full audio variants | `/workspace/fairspeech-full-eval/variants/full` | 36G |
| Full prediction CSVs and per-run metadata | `/workspace/fairspeech-full-eval/results/full` | 470M |
| Final metrics | `/workspace/fairspeech-full-eval/results/full_metrics` | 1.1M |
| Final plots | `/workspace/fairspeech-full-eval/results/full_plots` | 4.0M |

Repo-side changes are limited to scripts, planning docs, small metrics summaries, logs, and reports. The generated audio and prediction payloads should stay out of Git.

## Reproduction Commands

The final run used the full-run supervisor with the prebuilt mixed matrix and selected `160s / max16` guards:

```bash
.venv/bin/python scripts/setup/run_fairspeech_full_eval.py validate-predictions \
  --result-root /workspace/fairspeech-full-eval/results \
  --matrix-json datasets/fairspeech_compression/full_eval/full_eval_run_matrix.json \
  --expected-count 26471
```

```bash
.venv/bin/python scripts/metrics/compute_fairspeech_compression_metrics.py \
  --predictions-dir /workspace/fairspeech-full-eval/results/full \
  --output-dir /workspace/fairspeech-full-eval/results/full_metrics \
  --min-group-size 50 \
  --recursive \
  --expected-rows 26471
```

```bash
.venv/bin/python scripts/metrics/compute_fairspeech_compression_metrics.py \
  --predictions-dir /workspace/fairspeech-full-eval/results/full \
  --output-dir datasets/fairspeech_compression/full_eval/bootstrap_ci_200 \
  --min-group-size 50 \
  --recursive \
  --expected-rows 26471 \
  --bootstrap-resamples 200 \
  --bootstrap-seed 42
```

```bash
.venv/bin/python scripts/setup/run_fairspeech_full_eval.py plots \
  --result-root /workspace/fairspeech-full-eval/results
```

## Validation Result

Final validation passed:

- `validated_csvs`: 54
- `passed_csvs`: 54
- `failed_csvs`: 0
- Rows per prediction CSV: 26,471
- Total prediction rows used in metrics: 1,429,434

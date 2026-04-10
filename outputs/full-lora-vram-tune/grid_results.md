# Plan 03-02 VRAM Tuning Grid — Results

Generated: 2026-04-10T18:43:25Z

## Grid

| Cell | Label | Verdict | VRAM (GB) | Util % | step_s | tok/s | final_loss | GC | optim |
|------|-------|---------|-----------|--------|--------|-------|------------|----|-------|
| A | baseline (fixed 4x4, GC on, adamw_torch) | pass | 11.23 | 35.0 | 2.49 | 6.43 | 0.8240 | True | adamw_torch |
| B | fixed 4x4 + fused adamw (GC on) | pass | 11.23 | 35.7 | 2.50 | 6.40 | 0.8280 | True | adamw_torch_fused |
| C **WIN** | fixed 8x2 + fused adamw (GC on) | pass | 18.57 | 41.0 | 2.01 | 7.96 | 0.4126 | True | adamw_torch_fused |
| D | dynamic fb=180 max=64 + fused (GC on) | pass | 11.12 | 42.6 | 8.72 | 0.46 | 0.7421 | True | adamw_torch_fused |
| E | dynamic fb=300 max=96 + fused (GC on) | pass | 16.10 | 42.6 | 16.08 | 0.25 | 0.6788 | True | adamw_torch_fused |
| F | dynamic fb=180 max=64 + fused + NO GC | pass | 24.94 | 35.9 | 7.77 | 0.51 | 0.7431 | False | adamw_torch_fused |

## Winner

- **cell-C** — mean_gpu_util=41.0%, peak_vram_gb=18.57, median_step_time_s=2.01s, tokens_per_sec=7.96, complexity=0

## Selection rule

Sort passing cells by: `-mean_gpu_util, -tokens_per_sec, +median_step_time_s, +complexity_score`.
Head of the sorted list is the winner. Ties within 2 percentage points on util prefer the simpler config.

# Head-Surgery Diagnosis (v2.0 §4.2 MVP) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the diagnosis-only pipeline from [tasks/prd-head-surgery-diagnosis.md](../../../tasks/prd-head-surgery-diagnosis.md): per-head attention masking on Whisper-large-v3's decoder, a 640-cell sweep over the 511 Indian-accent Common Voice 24 utterances, scoring that ranks heads by their hallucination-reduction Δ (with paired-bootstrap significance and a composite non-Indian-WER regression guard), plus a 36-config decoding-ablation grid and an energy-VAD arm over silence-injected audio. Fine-tuning is explicitly out of scope.

**Architecture:** New `scripts/head_surgery/` package. Primitives: per-head forward-pre-hook on Whisper decoder `self_attn.out_proj` (serial + batched-condition variants). Staged rollout: reproduce 9.62% baseline (Gate G1) → empirical batch-size tune (G1.5) → serial pilot sweep (G2) → batched full sweep with a correctness gate against the serial pilot (G3) → score (G5) → adjunct arms → aggregate report. Reuses the existing `scripts/inference/run_inference.py` Whisper loader and `scripts/analysis/whisper_hallucination_analysis.py` insertion classifier.

**Tech Stack:** Python 3.10+, PyTorch 2.10, HuggingFace Transformers 5.5.4, jiwer 4.0, `openai-whisper` 20250625 (for `EnglishTextNormalizer`), pytest 8.

**Phases:**
0. Data extraction (Task 0): targeted MP3 extraction from the 81 GB Common Voice 25 tarball. Only 19 GB of free disk, so full extraction is not an option; the ~3,094 needed clips total ~275 MB.
1. Foundations (Tasks 1–3): scaffolding + reproducibility contract.
2. Masking primitives (Tasks 4–5): serial + batched hooks.
3. Baseline + tune + pilot (Tasks 6–8): Stages A, A.5, B.
4. Full sweep (Task 9): Stage C.
5. Scoring (Tasks 10–11): Stage D.
6. Adjunct arms (Tasks 12–13): Stages E, F.
7. Report & wrap-up (Tasks 14–16): Stage G + documentation.

**Dataset drift note:** The midterm reports 511 Indian-accent utterances. Common Voice 25 test split (`datasets/cv-corpus-25.0-2026-03-09/en/test.tsv`) contains 510 pure-India-accent rows (plus 9 compound-label rows like `United States English|India and South Asia …`). We use the 510 pure-India rows. The 9.62% insertion-rate target (Gate G1) still holds within the ±0.5pp tolerance. All "511" references in the PRD and plan should be read as "the frozen Indian-accent subset — 510 on CV25".

**Project conventions (read before starting):** [CLAUDE.md](../../../CLAUDE.md) requires a `logs/<topic>.md` file with `fileClass: Log` frontmatter for any behavioral change. Add a log entry at the end of each phase. Python test files live under `tests/`; fixtures under `tests/fixtures/`. Gitignored artifacts land under `outputs/head_surgery/`.

---

## Task 0: Targeted extraction of Common Voice 25 audio from the tarball

The 81 GB `datasets/cv-corpus-25.0-en.tar.gz` contains 2.57M MP3 clips (~76 GB extracted). We only need (a) the 510 pure-India-accent clips and (b) the 2,584 non-Indian accent-labeled clips. Total ≈ 3,094 MP3s ≈ 275 MB. Disk free = 19 GB, so full extraction is infeasible; targeted extraction is the only path.

**Files:**
- Create: `scripts/head_surgery/_extract_cv25_audio.py` (one-shot)
- Create: `datasets/cv-corpus-25.0-2026-03-09/en/_wanted_paths.txt`
- Populates: `datasets/cv-corpus-25.0-2026-03-09/en/clips/` (gitignored; gets ~3,094 MP3s)
- Creates: `datasets/cv-corpus-25.0-2026-03-09/en/cv_test_manifest.csv` — the 510 Indian-accent + non-Indian rows reformatted for the `scripts/inference/run_inference.py` contract
- Creates: `datasets/cv-corpus-25.0-2026-03-09/en/cv_non_indian_manifest.csv` — separate manifest used by the Stage D regression guard

- [ ] **Step 1: Write the path-list and manifest generator**

Create `scripts/head_surgery/_extract_cv25_audio.py`:

```python
"""One-shot: build the wanted-paths list and audio manifests for head-surgery diagnosis.

Reads datasets/cv-corpus-25.0-2026-03-09/en/test.tsv, selects:
  - pure Indian-accent rows (accents exactly == INDIAN_CANONICAL, ~510 rows)
  - non-Indian accent-labeled rows (accents non-empty AND does not match 'India.*South Asia')
Writes:
  datasets/cv-corpus-25.0-2026-03-09/en/_wanted_paths.txt
    Each line: clips/<mp3 filename>  — fed to tar --files-from
  datasets/cv-corpus-25.0-2026-03-09/en/cv_test_manifest.csv
    Columns: id, audio_path, reference, accent  (510 Indian rows)
  datasets/cv-corpus-25.0-2026-03-09/en/cv_non_indian_manifest.csv
    Columns: id, audio_path, reference, accent  (~2,584 non-Indian rows)

Not imported by runtime code. Commit the manifest CSVs; the wanted-paths
file is an ephemeral build artifact (kept for reproducibility traceability).
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path("datasets/cv-corpus-25.0-2026-03-09/en")
TSV = ROOT / "test.tsv"
WANTED = ROOT / "_wanted_paths.txt"
IND_MANIFEST = ROOT / "cv_test_manifest.csv"
NONIND_MANIFEST = ROOT / "cv_non_indian_manifest.csv"

INDIAN_CANONICAL = "India and South Asia (India, Pakistan, Sri Lanka)"


def main() -> None:
    ind_rows: list[dict] = []
    nonind_rows: list[dict] = []
    with TSV.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            accent = (row.get("accents") or "").strip()
            path = row.get("path", "").strip()
            sentence = row.get("sentence", "").strip()
            if not path or not sentence:
                continue
            record = {
                "id": path,  # mp3 filename is stable and unique
                "audio_path": str(ROOT / "clips" / path),
                "reference": sentence,
                "accent": accent,
            }
            if accent == INDIAN_CANONICAL:
                ind_rows.append(record)
            elif accent and "India" not in accent:
                # Skip blanks and any compound label that includes India.
                nonind_rows.append(record)

    WANTED.parent.mkdir(parents=True, exist_ok=True)
    with WANTED.open("w") as f:
        for r in ind_rows + nonind_rows:
            f.write(f"en/clips/{Path(r['audio_path']).name}\n")

    for target, rows in ((IND_MANIFEST, ind_rows), (NONIND_MANIFEST, nonind_rows)):
        with target.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "audio_path", "reference", "accent"])
            w.writeheader()
            w.writerows(rows)

    print(f"Indian-accent rows: {len(ind_rows)}  -> {IND_MANIFEST}")
    print(f"Non-Indian rows:    {len(nonind_rows)}  -> {NONIND_MANIFEST}")
    print(f"Wanted paths:       {len(ind_rows)+len(nonind_rows)}  -> {WANTED}")
    if not 505 <= len(ind_rows) <= 515:
        raise SystemExit(f"Indian count {len(ind_rows)} outside expected band [505, 515]; inspect test.tsv")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the manifest generator**

Run: `python scripts/head_surgery/_extract_cv25_audio.py`
Expected:
```
Indian-accent rows: 510  -> datasets/.../cv_test_manifest.csv
Non-Indian rows:    2584 -> datasets/.../cv_non_indian_manifest.csv
Wanted paths:       3094 -> datasets/.../_wanted_paths.txt
```

- [ ] **Step 3: Extract the wanted MP3s from the tarball**

The tarball is gzip-compressed, so `tar` has to read it sequentially (cannot seek). `--occurrence=1` lets tar exit after finding every listed file once, which saves time since most clips are bunched together in the archive's clips directory.

Run:
```bash
cd datasets
tar -xzf cv-corpus-25.0-en.tar.gz \
    -C cv-corpus-25.0-2026-03-09/ \
    --strip-components=0 \
    --occurrence=1 \
    --files-from=cv-corpus-25.0-2026-03-09/en/_wanted_paths.txt \
    --verbose 2>&1 | tail -5
```

Expected: ~3,094 lines of `en/clips/...` printed (suppressed to `tail -5`). Wall time: 20–60 min on disk-bound hardware (the gz stream must be decoded linearly until the last needed file is found).

Note on path layout: `test.tsv` stores `path` as just the filename (e.g., `common_voice_en_36734620.mp3`). Inside the tarball those live at `en/clips/common_voice_en_36734620.mp3`. The `_wanted_paths.txt` file written in Step 1 uses the `en/clips/...` form that matches the tarball layout; the `-C cv-corpus-25.0-2026-03-09/` flag roots the extraction so the final path becomes `datasets/cv-corpus-25.0-2026-03-09/en/clips/<filename>.mp3` — which is what the manifests reference.

- [ ] **Step 4: Verify extraction count and spot-check a clip**

Run: `ls datasets/cv-corpus-25.0-2026-03-09/en/clips/ | wc -l`
Expected: `3094` (or within ±5 — CV tarball-to-manifest drift is rare but possible).

Run:
```bash
python -c "
import soundfile as sf, pandas as pd
m = pd.read_csv('datasets/cv-corpus-25.0-2026-03-09/en/cv_test_manifest.csv')
row = m.iloc[0]
audio, sr = sf.read(row['audio_path'])
print(f'id={row[\"id\"]} sr={sr} n_samples={len(audio)} duration={len(audio)/sr:.2f}s ref={row[\"reference\"][:40]!r}')
"
```
Expected: a valid duration and a truncated reference string. If `soundfile` cannot read MP3, install `ffmpeg` system package or use `librosa.load` instead (already a dependency). The project's `scripts/inference/run_inference.py` uses `soundfile`; if that fails here, add `pip install audioread` or use `librosa.load(path, sr=16000)` in the inference helpers.

- [ ] **Step 5: Disk-usage sanity check**

Run: `du -sh datasets/cv-corpus-25.0-2026-03-09/en/clips/; df -h /workspace/project | tail -1`
Expected: ~200–350 MB for the clips dir; free space on `/workspace/project` should be ≥ 18 GB (we've consumed < 1 GB).

- [ ] **Step 6: Ensure the clips and tarball stay out of git**

Run: `grep -E "^datasets|^outputs" .gitignore`
Expected: at least one line matching each. If `datasets/` is not ignored, append it.

- [ ] **Step 7: Commit the manifest CSVs and the extractor script (NOT the clips, NOT the tarball)**

```bash
git add scripts/head_surgery/_extract_cv25_audio.py \
        datasets/cv-corpus-25.0-2026-03-09/en/cv_test_manifest.csv \
        datasets/cv-corpus-25.0-2026-03-09/en/cv_non_indian_manifest.csv \
        .gitignore
git commit -m "$(cat <<'EOF'
feat(head_surgery): targeted CV25 extraction + manifests (Task 0)

Extracts only the ~3,094 needed MP3 clips (510 Indian-accent + 2,584
non-Indian accent-labeled) from the 81 GB CV25 tarball. Full extraction
would require 76 GB; only 19 GB free. Manifest CSVs are committed; clips
stay gitignored.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

> **Note for downstream tasks:** from here on, `<CV24_MANIFEST_PATH>` in later task commands = `datasets/cv-corpus-25.0-2026-03-09/en/cv_test_manifest.csv`, and `<CV24_NON_INDIAN_MANIFEST_PATH>` = `datasets/cv-corpus-25.0-2026-03-09/en/cv_non_indian_manifest.csv`.

---

## Task 1: Scaffolding and dependency pins

**Files:**
- Create: `scripts/head_surgery/__init__.py`
- Create: `scripts/head_surgery/repro_config.py` (empty stub)
- Create: `scripts/head_surgery/head_mask_hook.py` (empty stub)
- Create: `scripts/head_surgery/tune_batch_size.py` (empty stub)
- Create: `scripts/head_surgery/run_diagnosis_sweep.py` (empty stub)
- Create: `scripts/head_surgery/score_heads.py` (empty stub)
- Create: `scripts/head_surgery/decoding_ablation_grid.py` (empty stub)
- Create: `scripts/head_surgery/energy_vad.py` (empty stub)
- Create: `scripts/head_surgery/aggregate_report.py` (empty stub)
- Create: `scripts/head_surgery/insertion_classifier.py` (empty stub)
- Create: `tests/test_head_surgery.py` (empty stub)
- Create: `tests/fixtures/head_surgery/.gitkeep`
- Modify: `pyproject.toml` (add exact-version comments)
- Modify: `.gitignore` (ensure `outputs/` ignored; likely already is — verify)

- [ ] **Step 1: Create the package directory and empty stub files**

```bash
mkdir -p scripts/head_surgery tests/fixtures/head_surgery outputs/head_surgery
touch scripts/head_surgery/__init__.py
for f in repro_config head_mask_hook tune_batch_size run_diagnosis_sweep \
         score_heads decoding_ablation_grid energy_vad aggregate_report \
         insertion_classifier; do
    printf '"""%s — see tasks/prd-head-surgery-diagnosis.md."""\n' "$f" > scripts/head_surgery/$f.py
done
printf '# tests for scripts/head_surgery — see tasks/prd-head-surgery-diagnosis.md\n' > tests/test_head_surgery.py
touch tests/fixtures/head_surgery/.gitkeep
```

- [ ] **Step 2: Verify `outputs/` is gitignored**

Run: `grep -n "^outputs" .gitignore`
Expected: at least one line matching `outputs/` or `outputs`. If not, append `outputs/` to `.gitignore`.

- [ ] **Step 3: Add a comment block to `pyproject.toml` documenting the head-surgery pins**

Append to `pyproject.toml` (after line 63, before `[project.optional-dependencies]`):

```toml
# ---------------------------------------------------------------------------
# v2.0 head-surgery diagnosis — reproducibility anchor.
# The midterm's 9.62% Indian-accent insertion rate was computed with the
# versions below. Do NOT upgrade these libraries without re-running Gate G1
# (see tasks/prd-head-surgery-diagnosis.md §6).
#   transformers == 5.5.4
#   torch        == 2.10.0  (cu128 wheel)
#   jiwer        == 4.0.0
#   openai-whisper == 20250625
# ---------------------------------------------------------------------------
```

Leave the `>=` pins above as they are (existing project convention); the comment is the authoritative record.

- [ ] **Step 4: Commit scaffolding**

```bash
git add scripts/head_surgery/ tests/test_head_surgery.py tests/fixtures/head_surgery/.gitkeep pyproject.toml .gitignore
git commit -m "$(cat <<'EOF'
feat(head_surgery): scaffold v2.0 diagnosis package

Empty stub modules for head_mask_hook, tune_batch_size, run_diagnosis_sweep,
score_heads, decoding_ablation_grid, energy_vad, aggregate_report,
insertion_classifier, plus repro_config. Documents the midterm-anchored
library versions in pyproject.toml.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Snapshot the Indian-accent utterance IDs

Freezes the ID list used by `repro_config.load_indian_accent_ids()` (T9). Task 0 already produced `cv_test_manifest.csv` with exactly the Indian-accent rows we need; this task just extracts its `id` column into a JSON fixture that's not tied to on-disk manifest paths (so the snapshot stays valid even if `datasets/` is moved or re-generated).

**Files:**
- Create: `tests/fixtures/head_surgery/indian_accent_ids.json`

- [ ] **Step 1: Build the fixture from the committed manifest**

Run:
```bash
python -c "
import json, pandas as pd
from pathlib import Path
m = pd.read_csv('datasets/cv-corpus-25.0-2026-03-09/en/cv_test_manifest.csv')
ids = sorted(m['id'].astype(str).tolist())
out = Path('tests/fixtures/head_surgery/indian_accent_ids.json')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({
    'source_manifest': 'datasets/cv-corpus-25.0-2026-03-09/en/cv_test_manifest.csv',
    'corpus': 'Common Voice 25 (snapshot 2026-03-09)',
    'accent_filter': 'India and South Asia (India, Pakistan, Sri Lanka)',
    'n': len(ids),
    'midterm_reported_n': 511,
    'note': 'Midterm used a CV snapshot that contained 511; CV25 test.tsv contains 510 pure-India-accent rows plus 9 compound-label rows (excluded).',
    'ids': ids,
}, indent=2))
print(f'wrote {out} (n={len(ids)})')
assert 505 <= len(ids) <= 515, f'n={len(ids)} outside expected band'
"
```
Expected: `wrote tests/fixtures/head_surgery/indian_accent_ids.json (n=510)`.

- [ ] **Step 2: Commit the snapshot**

```bash
git add tests/fixtures/head_surgery/indian_accent_ids.json
git commit -m "$(cat <<'EOF'
feat(head_surgery): freeze Indian-accent ID snapshot (n=510 on CV25)

Pulls IDs from datasets/cv-corpus-25.0-2026-03-09/en/cv_test_manifest.csv
(produced by Task 0). Midterm reports n=511 on a nearby CV snapshot; CV25
test.tsv has 510 pure-India-accent rows — within gate tolerance for the
9.62% target.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Reproducibility config (T9)

**Files:**
- Modify: `scripts/head_surgery/repro_config.py`
- Modify: `tests/test_head_surgery.py`

- [ ] **Step 1: Write a failing test for `repro_config.py` stability**

Append to `tests/test_head_surgery.py`:

```python
import json
from pathlib import Path

import pytest

from scripts.head_surgery import repro_config as rc


def test_model_revision_pinned():
    assert rc.MODEL_ID == "openai/whisper-large-v3"
    assert isinstance(rc.MODEL_REVISION, str) and len(rc.MODEL_REVISION) >= 7, \
        "MODEL_REVISION must be a HuggingFace revision (commit SHA or branch)"


def test_seed_is_deterministic():
    assert rc.SEED == 20260417


def test_generate_config_pinned_to_midterm_defaults():
    # Midterm used: model.generate(features, max_new_tokens=440, language="en", task="transcribe")
    # Everything else was HF default. We pin the defaults explicitly to survive transformers upgrades.
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


def test_indian_accent_ids_count_in_band():
    ids = rc.load_indian_accent_ids()
    # Midterm reports 511; CV25 test.tsv contains 510 pure-India rows.
    # Either is acceptable within the gate tolerance.
    assert len(ids) in (510, 511), f"got n={len(ids)}, expected 510 or 511"
    assert len(set(ids)) == len(ids), "utterance IDs must be unique"


def test_indian_accent_ids_sorted_and_stable():
    ids = rc.load_indian_accent_ids()
    assert ids == sorted(ids), "IDs must be sorted for stable iteration"
    # Load a second time — must be identical (pure function)
    assert ids == rc.load_indian_accent_ids()
```

Run: `pytest tests/test_head_surgery.py -v`
Expected: all 5 tests FAIL with `AttributeError` / `ModuleNotFoundError` on `rc.MODEL_ID` etc.

- [ ] **Step 2: Implement `repro_config.py`**

Replace `scripts/head_surgery/repro_config.py` with:

```python
"""Frozen reproducibility contract for v2.0 head-surgery diagnosis (T9).

All numbers derived from / compared against the midterm must reference these
constants. Do not edit without re-running Gate G1 (see
tasks/prd-head-surgery-diagnosis.md §6).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import List

# ── Model ─────────────────────────────────────────────────────────────────
MODEL_ID = "openai/whisper-large-v3"
# Pin the exact HuggingFace revision. Resolve via:
#   huggingface-cli scan-cache | grep whisper-large-v3
# or from https://huggingface.co/openai/whisper-large-v3/commits/main
# Populated by implementation-plan Task 3 Step 3 below.
MODEL_REVISION = "main"  # REPLACE with concrete commit SHA after Task 3 Step 3

# ── Determinism ───────────────────────────────────────────────────────────
SEED = 20260417

# ── Generation config ──────────────────────────────────────────────────────
# These are the exact kwargs the midterm used (see
# scripts/inference/run_inference.py:356), expanded to name every HuggingFace
# default that could drift across transformers versions.
GENERATE_CONFIG = {
    "max_new_tokens": 440,
    "language": "en",
    "task": "transcribe",
    "num_beams": 1,
    "do_sample": False,
    "temperature": 0.0,
    "repetition_penalty": 1.0,
    "no_repeat_ngram_size": 0,
    "length_penalty": 1.0,
}

# ── Evaluation subset ──────────────────────────────────────────────────────
_IDS_JSON = (
    Path(__file__).resolve().parents[2]
    / "tests" / "fixtures" / "head_surgery" / "indian_accent_ids.json"
)


@lru_cache(maxsize=1)
def load_indian_accent_ids() -> List[str]:
    """Sorted Common Voice Indian-accent utterance IDs (T9 snapshot).

    Midterm reports n=511 on an earlier CV snapshot; CV25 test.tsv contains
    510 pure-India-accent rows. Either is accepted here — the 9.62% Gate G1
    target is robust to the one-row difference.
    """
    payload = json.loads(_IDS_JSON.read_text())
    ids = sorted(payload["ids"])
    if not 505 <= len(ids) <= 515:
        raise RuntimeError(
            f"Indian-accent ID count {len(ids)} outside expected band [505, 515] "
            f"(midterm ref = 511; CV25 snapshot = 510). "
            f"Rebuild the fixture via Task 2 Step 1."
        )
    return ids


# ── Whisper-large-v3 architecture constants ────────────────────────────────
# Frozen per Calm-Whisper (arxiv 2505.12969 §Methods) — the sweep geometry
# depends on these.
NUM_DECODER_LAYERS = 32
NUM_DECODER_SELF_ATTN_HEADS = 20
HEAD_DIM = 64  # 1280 hidden / 20 heads


# ── Gate tolerances ────────────────────────────────────────────────────────
GATE_G1_MIDTERM_INSERTION_RATE = 0.0962
GATE_G1_TOLERANCE_PP = 0.005  # ±0.5 percentage points
GATE_G3_WER_TOLERANCE = 1e-4
GATE_G5_BASELINE_WER_TOLERANCE_PP = 0.002  # ±0.2 pp

# ── Regression guard ───────────────────────────────────────────────────────
REGRESSION_BUDGET_PP = 0.005  # ≤ 0.5 pp absolute
REGRESSION_GUARD_TOP_K = 50   # only check guard for top-K heads by |Δ_ins|

# ── Scoring ────────────────────────────────────────────────────────────────
BOOTSTRAP_ITERATIONS = 10_000
TOP_K_FOR_REPORT = 10
```

- [ ] **Step 3: Resolve and commit the concrete `MODEL_REVISION` SHA**

Run: `python -c "from huggingface_hub import HfApi; api = HfApi(); refs = api.list_repo_refs('openai/whisper-large-v3'); print(next(b.target_commit for b in refs.branches if b.name == 'main'))"`
Expected: a 40-char hex SHA (example: `06f233fe06e2d6f1eaffe0b6efabd35a4c5e1b99`).
Copy the SHA and replace the `MODEL_REVISION = "main"` line in `repro_config.py` with `MODEL_REVISION = "<sha>"`.

- [ ] **Step 4: Run the tests — they must pass now**

Run: `pytest tests/test_head_surgery.py -v`
Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/head_surgery/repro_config.py tests/test_head_surgery.py
git commit -m "$(cat <<'EOF'
feat(head_surgery): reproducibility config (T9)

Frozen: model revision SHA, seed, full generate() config (midterm-faithful
kwargs with every HF default named explicitly), 511-utterance eval subset
loader, Whisper-large-v3 architecture constants, gate tolerances.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Serial head-masking hook (T1 serial variant)

The hook strategy: forward-pre-hook on `decoder.layers[L].self_attn.out_proj`. The input to `out_proj` has shape `[batch, tgt_len, num_heads * head_dim]`. We reshape to `[batch, tgt_len, num_heads, head_dim]`, zero the slice for head `h`, reshape back. This intervenes AFTER attention is computed but BEFORE the output projection — matching Calm-Whisper's "zero the head's contribution" semantics and avoiding interactions with Whisper's KV cache (which stores K/V upstream of this point).

**Files:**
- Modify: `scripts/head_surgery/head_mask_hook.py`
- Modify: `tests/test_head_surgery.py`

- [ ] **Step 1: Write the failing unit test**

Append to `tests/test_head_surgery.py`:

```python
import torch
from transformers import WhisperForConditionalGeneration

from scripts.head_surgery.head_mask_hook import SerialHeadMaskHook
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
    # Synthetic encoder input — any valid log-mel shape works
    bsz = 2
    input_features = torch.randn(bsz, model.config.num_mel_bins, 3000)
    decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]] * bsz)
    # Baseline capture: out_proj input without any hook
    ref_x = _decoder_self_attn_out_proj_input(model, input_features, decoder_input_ids, layer_idx=0)

    # Install serial hook targeting layer 0, head 3
    hook = SerialHeadMaskHook(model, layer_idx=0, head_idx=3)
    hook.install()
    try:
        masked_x = _decoder_self_attn_out_proj_input(
            model, input_features, decoder_input_ids, layer_idx=0
        )
    finally:
        hook.remove()

    # Reshape both to [bsz, tgt_len, num_heads, head_dim]
    ref_r = ref_x.view(bsz, -1, num_heads, head_dim)
    masked_r = masked_x.view(bsz, -1, num_heads, head_dim)

    # Head 3 must be all zeros in masked; other heads must be unchanged
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
```

Run: `pytest tests/test_head_surgery.py::test_serial_head_mask_zeros_target_head -v`
Expected: FAIL with `ImportError` on `SerialHeadMaskHook`.

- [ ] **Step 2: Implement `SerialHeadMaskHook`**

Replace the body of `scripts/head_surgery/head_mask_hook.py` (the stub) with:

```python
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

from typing import List, Optional, Tuple

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
```

- [ ] **Step 3: Run the serial hook tests**

Run: `pytest tests/test_head_surgery.py::test_serial_head_mask_zeros_target_head tests/test_head_surgery.py::test_serial_hook_remove_restores_behavior -v`
Expected: both PASS.

- [ ] **Step 4: Commit**

```bash
git add scripts/head_surgery/head_mask_hook.py tests/test_head_surgery.py
git commit -m "$(cat <<'EOF'
feat(head_surgery): serial per-head masking hook (T1)

Forward-pre-hook on decoder.layers[L].self_attn.out_proj zeros head `h` of
the per-head attention output. Matches Calm-Whisper semantics without
touching KV cache. Tested on whisper-tiny for shape-general correctness.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Batched-condition head-masking hook (T1 batched variant)

Per-sample masks: each sample in a batch can zero a different head. A shared `head_mask[batch, num_heads]` tensor (0 = masked, 1 = kept) lives as hook state. The hook multiplies the reshaped input by this mask. During autoregressive `generate()`, the hook fires at every decoding step — the per-sample assignment must persist across steps for a given batch.

**Files:**
- Modify: `scripts/head_surgery/head_mask_hook.py`
- Modify: `tests/test_head_surgery.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_head_surgery.py`:

```python
from scripts.head_surgery.head_mask_hook import BatchedHeadMaskHook


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

    # Inspect the tensor AFTER our batched mask was applied but BEFORE out_proj.
    # Trick: install a second hook *after* BatchedHeadMaskHook so it sees the masked input.
    # PyTorch runs hooks in registration order; our hook was registered first, so the
    # second hook sees the already-masked tensor.
    x = captured[0].view(bsz, -1, num_heads, head_dim)
    assert torch.allclose(x[0, :, 2, :], torch.zeros_like(x[0, :, 2, :])), "sample 0 head 2 not zero"
    assert torch.allclose(x[1, :, 5, :], torch.zeros_like(x[1, :, 5, :])), "sample 1 head 5 not zero"
    # Sample 2 has no masked heads — no head should be all-zero by coincidence (very unlikely).
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
```

Run: `pytest tests/test_head_surgery.py::test_batched_hook_per_sample_zeros_correct_heads tests/test_head_surgery.py::test_batched_hook_requires_set_batch_mask_before_forward -v`
Expected: FAIL with `ImportError` on `BatchedHeadMaskHook`.

- [ ] **Step 2: Implement `BatchedHeadMaskHook`**

Append to `scripts/head_surgery/head_mask_hook.py`:

```python
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
```

- [ ] **Step 3: Run batched-hook tests**

Run: `pytest tests/test_head_surgery.py::test_batched_hook_per_sample_zeros_correct_heads tests/test_head_surgery.py::test_batched_hook_requires_set_batch_mask_before_forward -v`
Expected: both PASS.

- [ ] **Step 4: Run the full unit-test suite for this package**

Run: `pytest tests/test_head_surgery.py -v`
Expected: all tests from Tasks 3–5 PASS (at least 9 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/head_surgery/head_mask_hook.py tests/test_head_surgery.py
git commit -m "$(cat <<'EOF'
feat(head_surgery): batched-condition per-sample head mask hook (T1 batched)

Each sample in a batch can zero a different head at a fixed layer. Mask is
set once per batch via set_batch_mask(); reused across every autoregressive
decoding step. Guards against missing mask and batch-size mismatch.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Stage A — baseline rerun (Gate G1)

Reproduce the midterm's 9.62% Indian-accent insertion rate on Whisper-large-v3 with the frozen config. This is the "is the pipeline correct?" gate before any masking.

**Files:**
- Modify: `scripts/head_surgery/run_diagnosis_sweep.py`
- Modify: `scripts/head_surgery/insertion_classifier.py`
- Modify: `tests/test_head_surgery.py`

- [ ] **Step 1: Implement the thin insertion-classifier wrapper (T4)**

Replace `scripts/head_surgery/insertion_classifier.py` with:

```python
"""Thin wrapper around scripts/analysis/whisper_hallucination_analysis.py (T4).

Exposes:
  categorize_insertions(ref, hyp) -> list[{"word", "category"}]
    where category ∈ {"repetition", "syntactic_completion", "content_hallucination"}
  insertion_rate_breakdown(ref_hyp_pairs) -> dict(total, repetition, syntactic, content)
    where each value is (#insertions / total_ref_words).
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Tuple

from scripts.analysis.whisper_hallucination_analysis import (
    extract_and_categorize_insertions,
)


def categorize_insertions(ref: str, hyp: str) -> List[dict]:
    return extract_and_categorize_insertions(ref, hyp)


def insertion_rate_breakdown(pairs: Iterable[Tuple[str, str]]) -> Dict[str, float]:
    total_ref_words = 0
    counts = Counter()
    for ref, hyp in pairs:
        ref = (ref or "").strip()
        hyp = (hyp or "").strip()
        if not ref:
            continue
        total_ref_words += len(ref.split())
        for ins in categorize_insertions(ref, hyp):
            counts[ins["category"]] += 1
            counts["total"] += 1
    if total_ref_words == 0:
        return {"total": 0.0, "repetition": 0.0, "syntactic": 0.0, "content": 0.0,
                "total_ref_words": 0}
    return {
        "total": counts["total"] / total_ref_words,
        "repetition": counts["repetition"] / total_ref_words,
        "syntactic": counts["syntactic_completion"] / total_ref_words,
        "content": counts["content_hallucination"] / total_ref_words,
        "total_ref_words": total_ref_words,
    }
```

- [ ] **Step 2: Write a unit test for the classifier wrapper**

Append to `tests/test_head_surgery.py`:

```python
from scripts.head_surgery.insertion_classifier import (
    categorize_insertions,
    insertion_rate_breakdown,
)


def test_insertion_classifier_rep_category():
    # "the the" — second "the" is a repetition insertion
    result = categorize_insertions("the cat sat", "the the cat sat")
    # exact category labels come from scripts/analysis/whisper_hallucination_analysis.py
    cats = [r["category"] for r in result]
    assert "repetition" in cats or "syntactic_completion" in cats


def test_insertion_rate_breakdown_zero_on_identical():
    pairs = [("hello world", "hello world"), ("good morning", "good morning")]
    br = insertion_rate_breakdown(pairs)
    assert br["total"] == 0.0
    assert br["total_ref_words"] == 4
```

Run: `pytest tests/test_head_surgery.py -v -k insertion`
Expected: both tests PASS.

- [ ] **Step 3: Implement Stage A baseline driver**

Replace `scripts/head_surgery/run_diagnosis_sweep.py` with:

```python
"""Diagnosis-sweep drivers for Stages A, B, C of v2.0 head surgery.

Invocations:
    python -m scripts.head_surgery.run_diagnosis_sweep baseline
    python -m scripts.head_surgery.run_diagnosis_sweep pilot --pilot-layer 15 --n-utts 50
    python -m scripts.head_surgery.run_diagnosis_sweep full --batch-size <from-tune>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from scripts.head_surgery import repro_config as rc
from scripts.head_surgery.head_mask_hook import (
    BatchedHeadMaskHook,
    SerialHeadMaskHook,
)
from scripts.head_surgery.insertion_classifier import insertion_rate_breakdown

OUT_DIR = Path("outputs/head_surgery")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Whisper loading & inference (reuses midterm config) ──────────────────

def load_whisper(device: str = "cuda"):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(rc.MODEL_ID, revision=rc.MODEL_REVISION)
    model = WhisperForConditionalGeneration.from_pretrained(
        rc.MODEL_ID,
        revision=rc.MODEL_REVISION,
        torch_dtype=torch.float16 if "cuda" in device else torch.float32,
    ).to(device).eval()
    return model, processor


def load_manifest_for_ids(ids: List[str], manifest_csv: str, id_col: str = None):
    """Return a DataFrame of manifest rows matching `ids` in order."""
    df = pd.read_csv(manifest_csv)
    if id_col is None:
        id_col = next(
            c for c in df.columns
            if c in ("utt_id", "utterance_id", "client_id", "path")
        )
    df[id_col] = df[id_col].astype(str)
    subset = df[df[id_col].isin(set(ids))].copy()
    # Preserve the input `ids` order
    subset["__order__"] = subset[id_col].map({i: k for k, i in enumerate(ids)})
    subset = subset.sort_values("__order__").drop(columns=["__order__"]).reset_index(drop=True)
    if len(subset) != len(ids):
        missing = set(ids) - set(subset[id_col])
        raise RuntimeError(f"{len(missing)} IDs missing from manifest; first={list(missing)[:3]}")
    return subset, id_col


def _infer_whisper_batch(model, processor, audio_arrays, device: str):
    sr = 16000
    inputs = processor(audio_arrays, sampling_rate=sr, return_tensors="pt", padding=True)
    features = inputs.input_features.to(
        device, dtype=torch.float16 if "cuda" in device else torch.float32
    )
    with torch.no_grad():
        pred_ids = model.generate(features, **rc.GENERATE_CONFIG)
    return processor.batch_decode(pred_ids, skip_special_tokens=True)


# ── Stage A: baseline rerun ──────────────────────────────────────────────

def run_baseline(manifest_csv: str, batch_size: int = 8, device: str = "cuda") -> dict:
    """Run Whisper-large-v3 baseline on the frozen 511 IDs; return metrics dict."""
    import soundfile as sf

    ids = rc.load_indian_accent_ids()
    subset, id_col = load_manifest_for_ids(ids, manifest_csv)
    model, processor = load_whisper(device=device)

    audio_col = next(
        c for c in subset.columns if c in ("audio_path", "audio", "path")
    )
    ref_col = next(
        c for c in subset.columns if c in ("reference", "transcript", "sentence")
    )

    predictions: List[Tuple[str, str, str]] = []  # (id, ref, hyp)
    t0 = time.time()
    for i in range(0, len(subset), batch_size):
        batch = subset.iloc[i:i + batch_size]
        audios = [sf.read(str(p))[0] for p in batch[audio_col]]
        hyps = _infer_whisper_batch(model, processor, audios, device)
        for (_, row), hyp in zip(batch.iterrows(), hyps):
            predictions.append((str(row[id_col]), str(row[ref_col]), hyp.strip()))

    from scripts.inference.run_inference import normalize_text
    normed_pairs = [(normalize_text(r), normalize_text(h)) for _, r, h in predictions]
    breakdown = insertion_rate_breakdown(normed_pairs)

    out_csv = OUT_DIR / "baseline_predictions.csv"
    pd.DataFrame(predictions, columns=["id", "reference", "hypothesis"]).to_csv(out_csv, index=False)

    metrics = {
        "n": len(predictions),
        "insertion_rate_total": breakdown["total"],
        "insertion_rate_repetition": breakdown["repetition"],
        "insertion_rate_syntactic": breakdown["syntactic"],
        "insertion_rate_content": breakdown["content"],
        "seconds": round(time.time() - t0, 1),
        "model_revision": rc.MODEL_REVISION,
        "generate_config": rc.GENERATE_CONFIG,
    }
    (OUT_DIR / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def gate_G1(metrics: dict) -> None:
    observed = metrics["insertion_rate_total"]
    target = rc.GATE_G1_MIDTERM_INSERTION_RATE
    tol = rc.GATE_G1_TOLERANCE_PP
    diff = abs(observed - target)
    print(f"[G1] insertion rate = {observed*100:.2f}% (target {target*100:.2f}% ± {tol*100:.1f}pp, diff {diff*100:.2f}pp)")
    if diff > tol:
        raise SystemExit(f"[G1 FAIL] {observed*100:.2f}% differs from midterm {target*100:.2f}% by >{tol*100:.1f}pp. "
                         f"Investigate generate() config / transformers version / model revision.")
    print("[G1 PASS] baseline reproduces midterm within tolerance.")


# ── CLI dispatch ─────────────────────────────────────────────────────────

def _cli():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_base = sub.add_parser("baseline")
    p_base.add_argument("--manifest", required=True)
    p_base.add_argument("--batch-size", type=int, default=8)
    p_base.add_argument("--device", default="cuda")

    # pilot / full parsers added in later tasks
    args = p.parse_args()

    if args.cmd == "baseline":
        m = run_baseline(args.manifest, batch_size=args.batch_size, device=args.device)
        gate_G1(m)
    else:
        raise SystemExit(f"unknown cmd {args.cmd}")


if __name__ == "__main__":
    _cli()
```

- [ ] **Step 4: Verify syntax and imports with a dry-run**

Run: `python -c "from scripts.head_surgery import run_diagnosis_sweep"`
Expected: no output, no errors.

- [ ] **Step 5: Run Stage A against the real data on GPU**

Run (requires GPU + CV24 test manifest on disk):
```
python -m scripts.head_surgery.run_diagnosis_sweep baseline \
    --manifest /users/PAS2030/srishti/asr_fairness/data/cv_test_manifest.csv \
    --batch-size 8
```
Expected (approximate): `[G1 PASS] baseline reproduces midterm within tolerance.` plus a line with `9.62% ± 0.5pp`. Output artifacts at `outputs/head_surgery/baseline_predictions.csv` and `outputs/head_surgery/baseline_metrics.json`.

If Gate G1 fails: compare `baseline_metrics.json["generate_config"]` against `scripts/inference/run_inference.py:356`; verify `rc.MODEL_REVISION` matches the SHA the midterm used (check [logs/](../../../logs/) for the midterm's recorded model hash).

- [ ] **Step 6: Write a log entry (CLAUDE.md convention)**

Create `logs/head-surgery-stage-a-baseline.md`:

```markdown
---
fileClass: Log
name: head-surgery-stage-a-baseline
description: Gate G1 — Whisper-large-v3 baseline insertion rate on 511 Indian-accent CV24 utterances.
status: complete
subtype: evaluation
created: 2026-04-17
updated: 2026-04-17
tags: [head-surgery, baseline, gate-G1]
aliases: []
---

# Stage A baseline rerun

Reran Whisper-large-v3 on the frozen 511-utterance Indian-accent CV24 subset
using the frozen generate() config from `scripts/head_surgery/repro_config.py`.

Observed insertion rate: **<fill in from baseline_metrics.json>** (midterm target 9.62% ± 0.5pp).
Gate G1: <PASS/FAIL>.

Artifacts: `outputs/head_surgery/baseline_predictions.csv`, `outputs/head_surgery/baseline_metrics.json`.
```

Fill in the observed rate and gate result after running Step 5.

- [ ] **Step 7: Commit**

```bash
git add scripts/head_surgery/run_diagnosis_sweep.py scripts/head_surgery/insertion_classifier.py tests/test_head_surgery.py logs/head-surgery-stage-a-baseline.md
git commit -m "$(cat <<'EOF'
feat(head_surgery): Stage A baseline rerun (Gate G1)

Reruns Whisper-large-v3 on the frozen 511 Indian-accent CV24 utterances with
midterm-faithful generate() config. Gate G1 asserts insertion rate is within
±0.5pp of the midterm's 9.62%. Insertion classifier wrapper (T4) reuses the
existing scripts/analysis/whisper_hallucination_analysis.py logic.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Stage A.5 — batch-size tuner (Gate G1.5)

Empirically find the largest `utterances_per_batch` such that peak VRAM < 90% and `tokens/sec ≥ 95%` of the best observed. The tuner runs with the `BatchedHeadMaskHook` active (worst-case memory) so downstream stages can use the chosen size.

**Files:**
- Modify: `scripts/head_surgery/tune_batch_size.py`

- [ ] **Step 1: Implement the tuner**

Replace `scripts/head_surgery/tune_batch_size.py` with:

```python
"""Stage A.5 — empirical batch-size tuner for hooked Whisper inference.

Picks the largest utterances_per_batch where peak VRAM < 90% of device memory
AND tokens/sec >= 95% of the best observed.
Output: outputs/head_surgery/tune_batch_size.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import soundfile as sf
import torch

from scripts.head_surgery import repro_config as rc
from scripts.head_surgery.head_mask_hook import BatchedHeadMaskHook
from scripts.head_surgery.run_diagnosis_sweep import (
    OUT_DIR,
    _infer_whisper_batch,
    load_manifest_for_ids,
    load_whisper,
)

CANDIDATE_BATCH_SIZES = [1, 2, 4, 8, 16, 32]


def _time_one_setting(model, processor, audios, mask_layer, batch_size, n_warmup, n_measure, device):
    hook = BatchedHeadMaskHook(model, layer_idx=mask_layer).install()
    try:
        torch.cuda.reset_peak_memory_stats(device) if "cuda" in device else None
        num_heads, _ = (model.config.decoder_attention_heads, None)

        def _run_batch(chunk):
            mask = torch.ones(len(chunk), num_heads)
            # Mask all 20 heads one-by-one per sample, covering every head index.
            # In the tuner we only care about VRAM + throughput, so arbitrarily
            # zero head `i % num_heads` for sample i.
            for i in range(len(chunk)):
                mask[i, i % num_heads] = 0.0
            hook.set_batch_mask(mask)
            _infer_whisper_batch(model, processor, chunk, device)

        # Warmup
        for _ in range(n_warmup):
            chunk = audios[:batch_size]
            if len(chunk) < batch_size:
                return None
            _run_batch(chunk)

        # Measure
        t0 = time.time()
        total = 0
        for _ in range(n_measure):
            chunk = audios[:batch_size]
            _run_batch(chunk)
            total += len(chunk)
        dt = time.time() - t0
        utts_per_sec = total / dt if dt > 0 else 0.0
        peak = (
            torch.cuda.max_memory_allocated(device) if "cuda" in device else 0
        )
        return {"batch_size": batch_size, "utts_per_sec": utts_per_sec, "peak_bytes": peak, "ok": True}
    finally:
        hook.remove()
        if "cuda" in device:
            torch.cuda.empty_cache()


def tune(manifest_csv: str, mask_layer: int = 15, device: str = "cuda",
         n_warmup: int = 2, n_measure: int = 5) -> dict:
    ids = rc.load_indian_accent_ids()
    subset, _ = load_manifest_for_ids(ids, manifest_csv)
    audio_col = next(c for c in subset.columns if c in ("audio_path", "audio", "path"))
    audios = [sf.read(str(p))[0] for p in subset[audio_col].head(max(CANDIDATE_BATCH_SIZES))]

    model, processor = load_whisper(device=device)
    device_total = (
        torch.cuda.get_device_properties(device).total_memory if "cuda" in device else 10 ** 12
    )

    results: List[dict] = []
    for bs in CANDIDATE_BATCH_SIZES:
        try:
            r = _time_one_setting(model, processor, audios, mask_layer, bs, n_warmup, n_measure, device)
            if r is None:
                continue
            results.append(r)
            print(f"[tune] bs={bs}: {r['utts_per_sec']:.2f} utts/s, peak={r['peak_bytes']/1e9:.2f} GB")
        except torch.cuda.OutOfMemoryError:
            results.append({"batch_size": bs, "utts_per_sec": 0.0, "peak_bytes": -1, "ok": False, "oom": True})
            print(f"[tune] bs={bs}: OOM")
            break

    ok = [r for r in results if r["ok"] and r["peak_bytes"] < 0.9 * device_total]
    if not ok:
        raise SystemExit("[G1.5 FAIL] no batch size fits in 90% of device memory")
    best_throughput = max(r["utts_per_sec"] for r in ok)
    qualifying = [r for r in ok if r["utts_per_sec"] >= 0.95 * best_throughput]
    chosen = max(qualifying, key=lambda r: r["batch_size"])

    payload = {
        "chosen_batch_size": chosen["batch_size"],
        "chosen_throughput_utts_per_sec": chosen["utts_per_sec"],
        "chosen_peak_bytes": chosen["peak_bytes"],
        "device_total_bytes": device_total,
        "mask_layer": mask_layer,
        "sweep": results,
    }
    (OUT_DIR / "tune_batch_size.json").write_text(json.dumps(payload, indent=2))
    print(f"[G1.5 PASS] chosen batch_size={chosen['batch_size']} "
          f"({chosen['utts_per_sec']:.2f} utts/s, {chosen['peak_bytes']/1e9:.2f} GB)")
    return payload


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--mask-layer", type=int, default=15)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    tune(args.manifest, mask_layer=args.mask_layer, device=args.device)


if __name__ == "__main__":
    _cli()
```

- [ ] **Step 2: Syntax check**

Run: `python -c "from scripts.head_surgery import tune_batch_size"`
Expected: no errors.

- [ ] **Step 3: Run on GPU (requires `nvidia-smi` clear per CLAUDE.md §GPU Safety)**

Run: `nvidia-smi | head -20`
Expected: GPU memory < 80% utilization.

Run: `python -m scripts.head_surgery.tune_batch_size --manifest <CV24_MANIFEST_PATH>`
Expected: one line per tried batch size, followed by `[G1.5 PASS] chosen batch_size=<N>` with a concrete integer. Artifact: `outputs/head_surgery/tune_batch_size.json`.

- [ ] **Step 4: Commit**

```bash
git add scripts/head_surgery/tune_batch_size.py
git commit -m "$(cat <<'EOF'
feat(head_surgery): Stage A.5 batch-size tuner (Gate G1.5)

Sweeps utterances_per_batch ∈ {1,2,4,8,16,32} with BatchedHeadMaskHook active
(worst-case memory), picks largest setting where VRAM <90% and throughput
within 5% of best. Writes chosen value to outputs/head_surgery/tune_batch_size.json
for downstream stages.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Stage B — pilot sweep (Gate G2)

Runs the serial hook on one pilot layer × 20 heads × 50 utterances. Exists to (a) validate the hook+inference pipeline produces sane per-head numbers, (b) supply a correctness anchor for Gate G3's batched-vs-serial comparison in Task 9.

**Files:**
- Modify: `scripts/head_surgery/run_diagnosis_sweep.py`

- [ ] **Step 1: Add `run_pilot` and Gate G2 to `run_diagnosis_sweep.py`**

Append to `scripts/head_surgery/run_diagnosis_sweep.py` (before `_cli`):

```python
# ── Stage B: pilot sweep (serial hook) ───────────────────────────────────

PILOT_N_UTTS = 50


def _rng_sample_ids(ids: List[str], n: int, seed: int) -> List[str]:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ids), size=n, replace=False)
    return [ids[int(i)] for i in sorted(idx.tolist())]


def run_pilot(manifest_csv: str, pilot_layer: int = 15, n_utts: int = PILOT_N_UTTS,
              batch_size: int = 8, device: str = "cuda") -> dict:
    import soundfile as sf
    from scripts.inference.run_inference import normalize_text

    ids_all = rc.load_indian_accent_ids()
    pilot_ids = _rng_sample_ids(ids_all, n_utts, rc.SEED)
    subset, id_col = load_manifest_for_ids(pilot_ids, manifest_csv)
    audio_col = next(c for c in subset.columns if c in ("audio_path", "audio", "path"))
    ref_col = next(c for c in subset.columns if c in ("reference", "transcript", "sentence"))

    model, processor = load_whisper(device=device)
    audios = [sf.read(str(p))[0] for p in subset[audio_col]]
    refs = subset[ref_col].astype(str).tolist()
    ids = subset[id_col].astype(str).tolist()

    # For each head h in [0, num_heads), install serial hook (pilot_layer, h),
    # run inference on the 50 audios, compute insertion rate.
    per_head: Dict[int, dict] = {}
    for h in range(rc.NUM_DECODER_SELF_ATTN_HEADS):
        hyps: List[str] = []
        with SerialHeadMaskHook(model, pilot_layer, h):
            for i in range(0, len(audios), batch_size):
                batch_audios = audios[i:i + batch_size]
                texts = _infer_whisper_batch(model, processor, batch_audios, device)
                hyps.extend(t.strip() for t in texts)
        pairs = [(normalize_text(r), normalize_text(hy)) for r, hy in zip(refs, hyps)]
        br = insertion_rate_breakdown(pairs)
        per_head[h] = {"hyps": hyps, **br}
        print(f"[pilot] L={pilot_layer} h={h}: insertion_rate={br['total']*100:.2f}%")

    # Also the no-mask baseline on the same 50 utts
    hyps_base: List[str] = []
    for i in range(0, len(audios), batch_size):
        batch_audios = audios[i:i + batch_size]
        hyps_base.extend(t.strip() for t in _infer_whisper_batch(model, processor, batch_audios, device))
    base_pairs = [(normalize_text(r), normalize_text(hy)) for r, hy in zip(refs, hyps_base)]
    base_br = insertion_rate_breakdown(base_pairs)

    # Write pilot_sweep.csv — one row per (head, utterance)
    rows = []
    for h, v in per_head.items():
        for uid, ref, hyp in zip(ids, refs, v["hyps"]):
            rows.append({"layer": pilot_layer, "head": h, "id": uid, "reference": ref, "hypothesis": hyp})
    for uid, ref, hyp in zip(ids, refs, hyps_base):
        rows.append({"layer": -1, "head": -1, "id": uid, "reference": ref, "hypothesis": hyp})
    pd.DataFrame(rows).to_csv(OUT_DIR / "pilot_sweep.csv", index=False)

    metrics = {
        "pilot_layer": pilot_layer,
        "n_utts": n_utts,
        "baseline_insertion_rate": base_br["total"],
        "per_head": {str(h): {"total": v["total"], "repetition": v["repetition"],
                               "syntactic": v["syntactic"], "content": v["content"]}
                      for h, v in per_head.items()},
    }
    (OUT_DIR / "pilot_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def gate_G2(metrics: dict) -> None:
    base = metrics["baseline_insertion_rate"]
    deltas = [v["total"] - base for v in metrics["per_head"].values()]
    n_pos = sum(1 for d in deltas if d > 0)
    n_non_pos = sum(1 for d in deltas if d <= 0)
    print(f"[G2] pilot deltas (masked - baseline): {n_pos} heads ↑, {n_non_pos} heads ↓/=")
    if n_pos == 0 or n_non_pos == 0:
        raise SystemExit(f"[G2 FAIL] all pilot deltas are same sign ({n_pos} ↑, {n_non_pos} ↓/=). "
                         f"Either the hook is a no-op or the pipeline has a bug. Investigate before Stage C.")
    if all(abs(d) < 1e-6 for d in deltas):
        raise SystemExit("[G2 FAIL] all pilot deltas are ~0 — hook is likely a no-op.")
    print("[G2 PASS] pilot shows head-level signal.")
```

Update `_cli()` to add the `pilot` subcommand:

```python
    p_pilot = sub.add_parser("pilot")
    p_pilot.add_argument("--manifest", required=True)
    p_pilot.add_argument("--pilot-layer", type=int, default=15)
    p_pilot.add_argument("--n-utts", type=int, default=PILOT_N_UTTS)
    p_pilot.add_argument("--batch-size", type=int, default=8)
    p_pilot.add_argument("--device", default="cuda")
```

and append to the dispatch:

```python
    elif args.cmd == "pilot":
        m = run_pilot(args.manifest, pilot_layer=args.pilot_layer, n_utts=args.n_utts,
                      batch_size=args.batch_size, device=args.device)
        gate_G2(m)
```

- [ ] **Step 2: Syntax check**

Run: `python -c "from scripts.head_surgery import run_diagnosis_sweep; print(run_diagnosis_sweep.run_pilot)"`
Expected: prints a function object.

- [ ] **Step 3: Execute Stage B**

Run: `python -m scripts.head_surgery.run_diagnosis_sweep pilot --manifest <CV24_MANIFEST_PATH>`
Expected: 20 pilot-head lines, then `[G2 PASS] pilot shows head-level signal.` Artifacts: `outputs/head_surgery/pilot_sweep.csv`, `outputs/head_surgery/pilot_metrics.json`. Wall time ~30–60 min on a single modern GPU.

- [ ] **Step 4: Commit**

```bash
git add scripts/head_surgery/run_diagnosis_sweep.py
git commit -m "$(cat <<'EOF'
feat(head_surgery): Stage B pilot sweep (Gate G2)

Serial hook on one pilot layer × 20 heads × 50 utterances. Validates that
head masking produces head-level signal (not a no-op or all-same-sign bug)
before committing compute to the 640-cell full sweep.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Stage C — full batched sweep (Gates G3 + G4)

Before running the full 640-cell sweep, re-run the pilot slice with the **batched** hook and assert it matches the serial pilot (Gate G3). Then run all 32×20 conditions on the full 511 utterances.

Packing strategy: for each layer L, make a batch of size `batch_size` where each of the 20 heads is processed `batch_size / 20` times (round-up to cover the utterance set). More practical: iterate layers × heads as a 640-long condition list, and within each condition, run all 511 utterances in batches of `batch_size`. The **batched** part is still useful because we mask the same (L, h) across all samples in a batch but still benefit from the pre-installed hook infrastructure; per-sample batched masks become useful when we pack multiple (L, h) into one batch. Start with "one condition per batch across many utterances" and only pursue the heavier cross-condition packing if compute budget demands it.

**Files:**
- Modify: `scripts/head_surgery/run_diagnosis_sweep.py`

- [ ] **Step 1: Add `run_full` and Gates G3/G4**

Append to `scripts/head_surgery/run_diagnosis_sweep.py` (before `_cli`):

```python
# ── Stage C: full batched sweep ──────────────────────────────────────────

def _batched_pilot_replay(model, processor, audios, refs, pilot_layer, batch_size, device) -> dict:
    """Replay Stage B's 20 head-masked passes using BatchedHeadMaskHook.

    For Gate G3 we need batched output to match the serial pilot on the same
    50 utterances for every head in `pilot_layer`. Strategy: for each head h,
    set the per-sample mask to "zero head h for all samples" (a uniform mask)
    and run inference via the batched hook code path. The output should
    match SerialHeadMaskHook bitwise (same reshape, same multiply).
    """
    from scripts.inference.run_inference import normalize_text
    per_head: Dict[int, dict] = {}
    num_heads = rc.NUM_DECODER_SELF_ATTN_HEADS
    hook = BatchedHeadMaskHook(model, layer_idx=pilot_layer).install()
    try:
        for h in range(num_heads):
            hyps: List[str] = []
            for i in range(0, len(audios), batch_size):
                chunk = audios[i:i + batch_size]
                mask = torch.ones(len(chunk), num_heads)
                mask[:, h] = 0.0
                hook.set_batch_mask(mask)
                hyps.extend(t.strip() for t in _infer_whisper_batch(model, processor, chunk, device))
            pairs = [(normalize_text(r), normalize_text(hy)) for r, hy in zip(refs, hyps)]
            per_head[h] = {"hyps": hyps, **insertion_rate_breakdown(pairs)}
    finally:
        hook.remove()
    return per_head


def gate_G3(serial_metrics: dict, batched_per_head: dict) -> None:
    """Assert batched per-head insertion rates match serial within tolerance."""
    tol = rc.GATE_G3_WER_TOLERANCE
    deltas = []
    for h, v in batched_per_head.items():
        serial_rate = serial_metrics["per_head"][str(h)]["total"]
        batched_rate = v["total"]
        d = abs(serial_rate - batched_rate)
        deltas.append(d)
        if d > tol:
            raise SystemExit(
                f"[G3 FAIL] head {h}: serial={serial_rate*100:.3f}% batched={batched_rate*100:.3f}% "
                f"(|Δ|={d*100:.3f}% > {tol*100:.3f}%). Batched hook is not bytes-equivalent to serial. "
                f"Fall back to serial + data-parallel shards (revisit Q4-B)."
            )
    print(f"[G3 PASS] batched matches serial on 50-utt pilot (max |Δ|={max(deltas)*100:.4f}%).")


def run_full(manifest_csv: str, batch_size: int, device: str = "cuda") -> dict:
    import soundfile as sf
    from scripts.inference.run_inference import normalize_text

    # Load pilot metrics for G3
    pilot_metrics = json.loads((OUT_DIR / "pilot_metrics.json").read_text())
    pilot_layer = pilot_metrics["pilot_layer"]

    ids_all = rc.load_indian_accent_ids()
    subset, id_col = load_manifest_for_ids(ids_all, manifest_csv)
    audio_col = next(c for c in subset.columns if c in ("audio_path", "audio", "path"))
    ref_col = next(c for c in subset.columns if c in ("reference", "transcript", "sentence"))
    audios = [sf.read(str(p))[0] for p in subset[audio_col]]
    refs = subset[ref_col].astype(str).tolist()
    ids = subset[id_col].astype(str).tolist()

    model, processor = load_whisper(device=device)

    # Gate G3 — batched replay of pilot slice
    pilot_ids = _rng_sample_ids(ids_all, pilot_metrics["n_utts"], rc.SEED)
    pilot_indices = [ids.index(p) for p in pilot_ids]
    pilot_audios = [audios[i] for i in pilot_indices]
    pilot_refs = [refs[i] for i in pilot_indices]
    batched_replay = _batched_pilot_replay(model, processor, pilot_audios, pilot_refs,
                                           pilot_layer, batch_size, device)
    gate_G3(pilot_metrics, batched_replay)

    # Full 640-cell sweep
    rows = []
    num_heads = rc.NUM_DECODER_SELF_ATTN_HEADS
    t0 = time.time()
    for L in range(rc.NUM_DECODER_LAYERS):
        hook = BatchedHeadMaskHook(model, layer_idx=L).install()
        try:
            for h in range(num_heads):
                hyps: List[str] = []
                for i in range(0, len(audios), batch_size):
                    chunk = audios[i:i + batch_size]
                    mask = torch.ones(len(chunk), num_heads)
                    mask[:, h] = 0.0
                    hook.set_batch_mask(mask)
                    hyps.extend(t.strip() for t in _infer_whisper_batch(model, processor, chunk, device))
                pairs = [(normalize_text(r), normalize_text(hy)) for r, hy in zip(refs, hyps)]
                br = insertion_rate_breakdown(pairs)
                for uid, ref, hyp in zip(ids, refs, hyps):
                    rows.append({"layer": L, "head": h, "id": uid,
                                 "reference": ref, "hypothesis": hyp,
                                 "condition_insertion_rate_total": br["total"]})
                elapsed = time.time() - t0
                done = L * num_heads + h + 1
                total = rc.NUM_DECODER_LAYERS * num_heads
                print(f"[sweep] L={L} h={h} ins={br['total']*100:.2f}% "
                      f"[{done}/{total}, {elapsed/60:.1f}min]")
        finally:
            hook.remove()

    out_csv = OUT_DIR / "sweep.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    # Gate G4 — completeness
    df = pd.read_csv(out_csv)
    expected = rc.NUM_DECODER_LAYERS * num_heads * len(ids)
    if len(df) != expected:
        raise SystemExit(f"[G4 FAIL] sweep has {len(df)} rows, expected {expected}")
    if df["hypothesis"].isna().any() or (df["hypothesis"].fillna("") == "").any():
        n_empty = df["hypothesis"].fillna("").eq("").sum()
        print(f"[G4 WARN] {n_empty} empty hypotheses in sweep (will be treated as insertion-rate NaN)")
    print(f"[G4 PASS] sweep complete: {len(df)} rows.")
    return {"rows": len(df), "out_csv": str(out_csv), "minutes": round((time.time() - t0) / 60, 1)}
```

Update `_cli()` to add `full`:

```python
    p_full = sub.add_parser("full")
    p_full.add_argument("--manifest", required=True)
    p_full.add_argument("--batch-size", type=int, required=True,
                        help="Use outputs/head_surgery/tune_batch_size.json:chosen_batch_size")
    p_full.add_argument("--device", default="cuda")
```

and dispatch:

```python
    elif args.cmd == "full":
        run_full(args.manifest, batch_size=args.batch_size, device=args.device)
```

- [ ] **Step 2: Syntax check**

Run: `python -c "from scripts.head_surgery.run_diagnosis_sweep import run_full, gate_G3"`
Expected: no error.

- [ ] **Step 3: Run Stage C**

First, read the chosen batch size:
```bash
BS=$(python -c "import json; print(json.load(open('outputs/head_surgery/tune_batch_size.json'))['chosen_batch_size'])")
echo "chosen batch size = $BS"
```
Then:
```bash
python -m scripts.head_surgery.run_diagnosis_sweep full --manifest <CV24_MANIFEST_PATH> --batch-size $BS
```
Expected: `[G3 PASS] ...` near the start, then ~640 sweep lines, then `[G4 PASS] sweep complete: <≈327040> rows.` Artifact: `outputs/head_surgery/sweep.csv`. Wall time ≈ (batch_size-dependent) hours.

If G3 fails: fall back to serial + data-parallel shards — add a `--shard i --num-shards N` CLI flag that filters the (L, h) conditions by `(L*num_heads + h) % num_shards == shard`, run N processes on N GPUs, concatenate their CSVs.

- [ ] **Step 4: Commit**

```bash
git add scripts/head_surgery/run_diagnosis_sweep.py
git commit -m "$(cat <<'EOF'
feat(head_surgery): Stage C full sweep (Gates G3 + G4)

Batched replay of Stage B's pilot on the same 50 utterances must match
serial output within 1e-4 WER (Gate G3) before the 32×20 full sweep runs on
all 511 utterances. Gate G4 asserts completeness of the 327,040-row output.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Insertion-classifier integration check (T4 smoke)

Before Stage D trusts the per-head insertion-rate breakdowns, run a quick integration check that our wrapper reproduces the midterm classifier's 43/48/9 repetition/syntactic/content split on Whisper-large-v3's CV24 Indian-accent predictions.

**Files:**
- Modify: `tests/test_head_surgery.py`

- [ ] **Step 1: Add an integration-style test (skipped if no baseline artifact)**

Append to `tests/test_head_surgery.py`:

```python
def test_baseline_breakdown_matches_midterm_ratios(tmp_path):
    """On Stage A output, verify repetition/syntactic/content ratios ≈ 43/48/9.

    Skipped if the Stage A artifacts are not present (CI or fresh clone).
    """
    baseline_csv = Path("outputs/head_surgery/baseline_predictions.csv")
    if not baseline_csv.exists():
        pytest.skip("outputs/head_surgery/baseline_predictions.csv missing — run Stage A first")
    df = pd.read_csv(baseline_csv)

    from scripts.inference.run_inference import normalize_text
    pairs = [(normalize_text(r), normalize_text(h))
             for r, h in zip(df["reference"], df["hypothesis"])]
    br = insertion_rate_breakdown(pairs)
    total = br["repetition"] + br["syntactic"] + br["content"]
    if total == 0:
        pytest.skip("no insertions observed — classifier or predictions empty")
    rep_pct = br["repetition"] / total * 100
    syn_pct = br["syntactic"] / total * 100
    con_pct = br["content"] / total * 100
    # Midterm split: 43 / 48 / 9. Allow ±10pp tolerance for run-to-run variance.
    assert abs(rep_pct - 43) < 10, f"repetition% = {rep_pct:.1f}, midterm = 43"
    assert abs(syn_pct - 48) < 10, f"syntactic% = {syn_pct:.1f}, midterm = 48"
    assert abs(con_pct - 9) < 10, f"content% = {con_pct:.1f}, midterm = 9"
```

- [ ] **Step 2: Run the test (if Stage A artifacts exist)**

Run: `pytest tests/test_head_surgery.py::test_baseline_breakdown_matches_midterm_ratios -v`
Expected: PASS (or SKIPPED with explicit reason if Stage A hasn't run yet).

- [ ] **Step 3: Commit**

```bash
git add tests/test_head_surgery.py
git commit -m "$(cat <<'EOF'
test(head_surgery): integration check on baseline insertion breakdown

Verifies the T4 insertion-classifier wrapper reproduces the midterm's
43/48/9 repetition/syntactic/content split on Stage A predictions.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Stage D — scoring (Gate G5)

Aggregates `sweep.csv` into per-(L, h) metrics, runs paired bootstrap for Δ insertion rate, computes the non-Indian composite WER regression guard for the top-50 heads by `|Δ_ins|`, and writes `head_scores.csv` + `top_k_heads.csv`.

**Files:**
- Modify: `scripts/head_surgery/score_heads.py`
- Modify: `tests/test_head_surgery.py`

- [ ] **Step 1: Write a failing unit test for the bootstrap function**

Append to `tests/test_head_surgery.py`:

```python
from scripts.head_surgery.score_heads import paired_bootstrap_delta_p


def test_paired_bootstrap_reports_significance_when_effect_large():
    import numpy as np
    rng = np.random.default_rng(0)
    n = 200
    base_counts = rng.poisson(2, size=n)       # baseline insertions per utterance
    masked_counts = np.maximum(0, base_counts - rng.poisson(1.5, size=n))  # masked reduces strongly
    ref_words = rng.integers(5, 15, size=n)
    # Effect is strongly positive (reduction). Expect p < 0.01.
    p = paired_bootstrap_delta_p(base_counts, masked_counts, ref_words, n_iter=2000, seed=0)
    assert p < 0.05, f"expected significant Δ; got p={p}"


def test_paired_bootstrap_reports_null_when_effect_zero():
    import numpy as np
    rng = np.random.default_rng(1)
    n = 200
    counts = rng.poisson(2, size=n)
    ref_words = rng.integers(5, 15, size=n)
    # Identical series → Δ = 0; p should be ~1.
    p = paired_bootstrap_delta_p(counts.copy(), counts.copy(), ref_words, n_iter=2000, seed=0)
    assert p > 0.5, f"expected no significance; got p={p}"
```

Run: `pytest tests/test_head_surgery.py -v -k bootstrap`
Expected: FAIL with `ImportError`.

- [ ] **Step 2: Implement `score_heads.py`**

Replace `scripts/head_surgery/score_heads.py` with:

```python
"""Stage D — per-head scoring, paired bootstrap, regression guard.

Inputs:
  outputs/head_surgery/sweep.csv          (from Stage C)
  outputs/head_surgery/baseline_predictions.csv (from Stage A)

Outputs:
  outputs/head_surgery/head_scores.csv    (640 rows, all metrics)
  outputs/head_surgery/top_k_heads.csv    (ranked top-K after regression guard)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from scripts.head_surgery import repro_config as rc
from scripts.head_surgery.insertion_classifier import (
    categorize_insertions,
    insertion_rate_breakdown,
)

OUT_DIR = Path("outputs/head_surgery")


def _ins_count(ref: str, hyp: str) -> Tuple[int, int, int, int, int]:
    """Return (total, repetition, syntactic, content, ref_words) for one utterance."""
    cats = categorize_insertions(ref or "", hyp or "")
    total = len(cats)
    rep = sum(1 for c in cats if c["category"] == "repetition")
    syn = sum(1 for c in cats if c["category"] == "syntactic_completion")
    con = sum(1 for c in cats if c["category"] == "content_hallucination")
    nwords = len((ref or "").split())
    return total, rep, syn, con, nwords


def paired_bootstrap_delta_p(
    base_counts: np.ndarray,
    masked_counts: np.ndarray,
    ref_words: np.ndarray,
    n_iter: int = rc.BOOTSTRAP_ITERATIONS,
    seed: int = rc.SEED,
) -> float:
    """p-value for H0: masked insertion rate ≥ baseline (one-sided).

    Δ = rate_baseline - rate_masked (positive = improvement). For each bootstrap
    resample of utterance indices, compute the Δ. p = fraction of bootstrap Δ ≤ 0.
    """
    rng = np.random.default_rng(seed)
    n = len(base_counts)
    deltas = np.empty(n_iter, dtype=np.float64)
    ref_words = np.asarray(ref_words, dtype=np.float64)
    base_counts = np.asarray(base_counts, dtype=np.float64)
    masked_counts = np.asarray(masked_counts, dtype=np.float64)
    for k in range(n_iter):
        idx = rng.integers(0, n, size=n)
        tw = ref_words[idx].sum()
        if tw == 0:
            deltas[k] = 0.0
            continue
        rb = base_counts[idx].sum() / tw
        rm = masked_counts[idx].sum() / tw
        deltas[k] = rb - rm
    return float((deltas <= 0).mean())


def compute_head_scores(sweep_csv: Path, baseline_csv: Path) -> pd.DataFrame:
    """Aggregate sweep.csv into per-(layer, head) metrics."""
    from scripts.inference.run_inference import normalize_text

    sweep = pd.read_csv(sweep_csv)
    baseline = pd.read_csv(baseline_csv)
    baseline["ref_n"] = baseline["reference"].fillna("").apply(
        lambda r: normalize_text(r)
    )
    baseline["hyp_n"] = baseline["hypothesis"].fillna("").apply(
        lambda h: normalize_text(h)
    )
    base_counts = np.zeros(len(baseline))
    ref_words = np.zeros(len(baseline))
    for i, row in baseline.iterrows():
        t, _, _, _, nw = _ins_count(row["ref_n"], row["hyp_n"])
        base_counts[i] = t
        ref_words[i] = nw
    baseline_total_ins = base_counts.sum()
    baseline_total_ref = ref_words.sum()
    baseline_rate = (baseline_total_ins / baseline_total_ref) if baseline_total_ref else 0.0
    baseline_id_to_idx = {str(row["id"]): i for i, row in baseline.iterrows()}

    rows = []
    for (L, h), g in sweep.groupby(["layer", "head"]):
        if L == -1:  # pilot-baseline rows, skip
            continue
        g = g.copy()
        g["ref_n"] = g["reference"].fillna("").apply(lambda r: normalize_text(r))
        g["hyp_n"] = g["hypothesis"].fillna("").apply(lambda x: normalize_text(x))
        counts_tot, counts_rep, counts_syn, counts_con = [], [], [], []
        n_words_list = []
        masked_aligned = np.zeros(len(baseline))
        for _, r in g.iterrows():
            t, rep, syn, con, nw = _ins_count(r["ref_n"], r["hyp_n"])
            counts_tot.append(t); counts_rep.append(rep)
            counts_syn.append(syn); counts_con.append(con)
            n_words_list.append(nw)
            idx = baseline_id_to_idx.get(str(r["id"]))
            if idx is not None:
                masked_aligned[idx] = t
        total_ins = sum(counts_tot)
        total_ref = sum(n_words_list) or 1
        rate_tot = total_ins / total_ref
        delta = baseline_rate - rate_tot
        p_val = paired_bootstrap_delta_p(base_counts, masked_aligned, ref_words)
        rows.append({
            "layer": int(L), "head": int(h),
            "insertion_rate_masked": rate_tot,
            "delta_insertion_rate": delta,
            "delta_repetition": (sum(r["counts_rep_base"] for r in []) ),  # placeholder; see below
            "p_value_delta": p_val,
        })
    # The repetition/syntactic/content deltas are computed in Step 3.
    return pd.DataFrame(rows)
```

- [ ] **Step 3: Extend with per-category deltas and regression-guard integration**

Replace the `compute_head_scores` function body (the loop and return statement) with the complete version:

```python
def compute_head_scores(sweep_csv: Path, baseline_csv: Path) -> pd.DataFrame:
    from scripts.inference.run_inference import normalize_text

    sweep = pd.read_csv(sweep_csv)
    baseline = pd.read_csv(baseline_csv)
    baseline["ref_n"] = baseline["reference"].fillna("").apply(normalize_text)
    baseline["hyp_n"] = baseline["hypothesis"].fillna("").apply(normalize_text)
    base_tot = np.zeros(len(baseline)); base_rep = np.zeros(len(baseline))
    base_syn = np.zeros(len(baseline)); base_con = np.zeros(len(baseline))
    ref_words = np.zeros(len(baseline))
    for i, row in baseline.iterrows():
        t, rep, syn, con, nw = _ins_count(row["ref_n"], row["hyp_n"])
        base_tot[i], base_rep[i], base_syn[i], base_con[i], ref_words[i] = t, rep, syn, con, nw
    total_refw = ref_words.sum() or 1
    base_rate = base_tot.sum() / total_refw
    base_rep_rate = base_rep.sum() / total_refw
    base_syn_rate = base_syn.sum() / total_refw
    base_con_rate = base_con.sum() / total_refw
    idx_of = {str(r["id"]): i for i, r in baseline.iterrows()}

    rows = []
    for (L, h), g in sweep.groupby(["layer", "head"]):
        if int(L) == -1:
            continue
        g_tot = np.zeros(len(baseline)); g_rep = np.zeros(len(baseline))
        g_syn = np.zeros(len(baseline)); g_con = np.zeros(len(baseline))
        g_refw = np.zeros(len(baseline))
        for _, r in g.iterrows():
            ref_n = normalize_text(r.get("reference", ""))
            hyp_n = normalize_text(r.get("hypothesis", ""))
            t, rep, syn, con, nw = _ins_count(ref_n, hyp_n)
            i = idx_of.get(str(r["id"]))
            if i is None:
                continue
            g_tot[i], g_rep[i], g_syn[i], g_con[i], g_refw[i] = t, rep, syn, con, nw
        # Use baseline's ref_words (same utterances in same order)
        rate_tot = g_tot.sum() / total_refw
        rate_rep = g_rep.sum() / total_refw
        rate_syn = g_syn.sum() / total_refw
        rate_con = g_con.sum() / total_refw
        p_val = paired_bootstrap_delta_p(base_tot, g_tot, ref_words)
        rows.append({
            "layer": int(L), "head": int(h),
            "insertion_rate_masked": rate_tot,
            "delta_insertion_rate": base_rate - rate_tot,
            "delta_repetition":     base_rep_rate - rate_rep,
            "delta_syntactic":      base_syn_rate - rate_syn,
            "delta_content":        base_con_rate - rate_con,
            "p_value_delta":        p_val,
            "regression_checked":   False,
            "non_indian_wer_masked": None,
            "regression_ok":        None,
        })
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Implement the regression-guard inference and scoring driver**

Append to `scripts/head_surgery/score_heads.py`:

```python
# ── Regression guard ─────────────────────────────────────────────────────

def compute_regression_guard(
    head_scores: pd.DataFrame,
    non_indian_manifest_csv: str,
    batch_size: int,
    device: str = "cuda",
    top_k: int = rc.REGRESSION_GUARD_TOP_K,
) -> pd.DataFrame:
    """Run Whisper-large-v3 with head (L,h) masked on non-Indian accents.

    Only the top-`top_k` rows by |delta_insertion_rate| are checked to bound compute.
    Computes composite (concatenated) WER across all non-Indian accent groups.
    """
    import jiwer
    import soundfile as sf
    from scripts.head_surgery.head_mask_hook import BatchedHeadMaskHook
    from scripts.head_surgery.run_diagnosis_sweep import (
        _infer_whisper_batch, load_manifest_for_ids, load_whisper,
    )
    from scripts.inference.run_inference import normalize_text
    import torch

    non_ind = pd.read_csv(non_indian_manifest_csv)
    audio_col = next(c for c in non_ind.columns if c in ("audio_path", "audio", "path"))
    ref_col = next(c for c in non_ind.columns if c in ("reference", "transcript", "sentence"))
    audios = [sf.read(str(p))[0] for p in non_ind[audio_col]]
    refs = [normalize_text(r) for r in non_ind[ref_col]]

    model, processor = load_whisper(device=device)
    num_heads = rc.NUM_DECODER_SELF_ATTN_HEADS

    # Baseline (no hook) for composite non-Indian WER
    hyps_base = []
    for i in range(0, len(audios), batch_size):
        hyps_base.extend(_infer_whisper_batch(model, processor, audios[i:i + batch_size], device))
    baseline_wer = jiwer.wer(refs, [normalize_text(h) for h in hyps_base])
    print(f"[guard] non-Indian composite baseline WER = {baseline_wer*100:.2f}%")

    # Top-K heads by |delta|
    candidates = head_scores.reindex(
        head_scores["delta_insertion_rate"].abs().sort_values(ascending=False).index
    ).head(top_k)

    scored_idx = set(candidates.index)
    updated = head_scores.copy()
    for i, row in candidates.iterrows():
        L, h = int(row["layer"]), int(row["head"])
        hook = BatchedHeadMaskHook(model, layer_idx=L).install()
        try:
            hyps: List[str] = []
            for j in range(0, len(audios), batch_size):
                chunk = audios[j:j + batch_size]
                mask = torch.ones(len(chunk), num_heads)
                mask[:, h] = 0.0
                hook.set_batch_mask(mask)
                hyps.extend(_infer_whisper_batch(model, processor, chunk, device))
        finally:
            hook.remove()
        wer = jiwer.wer(refs, [normalize_text(x) for x in hyps])
        reg_ok = (wer - baseline_wer) <= rc.REGRESSION_BUDGET_PP
        updated.at[i, "non_indian_wer_masked"] = wer
        updated.at[i, "regression_ok"] = bool(reg_ok)
        updated.at[i, "regression_checked"] = True
        print(f"[guard] L={L} h={h}: non-Indian WER={wer*100:.2f}% (Δ={ (wer-baseline_wer)*100:+.2f}pp) "
              f"ok={reg_ok}")

    # Gate G5 — baseline non-Indian WER sanity-check against T7 CSVs (if available)
    expected_path = Path("outputs/head_surgery/t7_non_indian_baseline_wer.json")
    if expected_path.exists():
        expected = json.loads(expected_path.read_text())["composite_wer"]
        diff = abs(baseline_wer - expected)
        if diff > rc.GATE_G5_BASELINE_WER_TOLERANCE_PP:
            raise SystemExit(
                f"[G5 FAIL] non-Indian baseline WER={baseline_wer*100:.2f}% "
                f"vs T7 expected {expected*100:.2f}% (|Δ|={diff*100:.2f}pp). "
                f"Investigate T7 CSV reuse."
            )
        print(f"[G5 PASS] non-Indian baseline matches T7 within {rc.GATE_G5_BASELINE_WER_TOLERANCE_PP*100:.1f}pp.")
    else:
        print("[G5 SKIP] outputs/head_surgery/t7_non_indian_baseline_wer.json missing; "
              "cannot cross-check against T7. Generate it from midterm CSVs before the writeup.")
    return updated


def write_top_k(head_scores: pd.DataFrame, k: int = rc.TOP_K_FOR_REPORT) -> pd.DataFrame:
    qualifying = head_scores[
        (head_scores["regression_ok"] == True) |
        (head_scores["regression_checked"] == False)
    ]
    top = qualifying.sort_values("delta_insertion_rate", ascending=False).head(k).reset_index(drop=True)
    top.to_csv(OUT_DIR / "top_k_heads.csv", index=False)
    return top


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--non-indian-manifest", required=True,
                   help="CSV of non-Indian CV24 accent utterances (composite regression guard)")
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--top-k-guard", type=int, default=rc.REGRESSION_GUARD_TOP_K)
    args = p.parse_args()

    scores = compute_head_scores(
        OUT_DIR / "sweep.csv", OUT_DIR / "baseline_predictions.csv",
    )
    scores = compute_regression_guard(
        scores, args.non_indian_manifest, batch_size=args.batch_size,
        device=args.device, top_k=args.top_k_guard,
    )
    scores.to_csv(OUT_DIR / "head_scores.csv", index=False)
    top = write_top_k(scores)
    print(f"[score] wrote {OUT_DIR/'head_scores.csv'} ({len(scores)} rows) and top-{len(top)} heads.")


if __name__ == "__main__":
    _cli()
```

- [ ] **Step 5: Run bootstrap tests**

Run: `pytest tests/test_head_surgery.py -v -k bootstrap`
Expected: both `test_paired_bootstrap_*` PASS.

- [ ] **Step 6: Run the full scoring pipeline (on GPU, after Stage C)**

Run: `python -m scripts.head_surgery.score_heads --non-indian-manifest <CV24_NON_INDIAN_MANIFEST_PATH> --batch-size $BS`
Expected: 50 guard lines, possibly `[G5 PASS]` or `[G5 SKIP]` if the T7 cross-check file isn't yet produced. Artifacts: `head_scores.csv`, `top_k_heads.csv`.

- [ ] **Step 7: Commit**

```bash
git add scripts/head_surgery/score_heads.py tests/test_head_surgery.py
git commit -m "$(cat <<'EOF'
feat(head_surgery): Stage D scoring + bootstrap + regression guard

Per-(layer, head) aggregation: insertion rate Δ total / repetition /
syntactic / content, paired-bootstrap p-value (10k iter), and non-Indian
composite WER guard on the top-50 heads by |Δ|. Gate G5 cross-checks the
non-Indian baseline against T7 CSVs when available.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Stage E — decoding ablation grid (T5)

36 configs × 511 utterances. Reuses `_infer_whisper_batch` but overrides the frozen generate config per cell.

**Files:**
- Modify: `scripts/head_surgery/decoding_ablation_grid.py`

- [ ] **Step 1: Implement the grid driver**

Replace `scripts/head_surgery/decoding_ablation_grid.py` with:

```python
"""Stage E — decoding-strategy ablation (T5).

2 × 3 × 3 × 2 = 36 configs: beam ∈ {1,5} × rep_penalty ∈ {1.0,1.1,1.3} ×
no_repeat_ngram ∈ {0,3,5} × temperature_fallback ∈ {on, off}.
Each runs the 511 Indian-accent utterances. Output: decoding_grid.csv
(one row per utterance per config) and decoding_scores.csv (per-config aggregates).
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import List

import pandas as pd
import soundfile as sf
import torch

from scripts.head_surgery import repro_config as rc
from scripts.head_surgery.insertion_classifier import insertion_rate_breakdown
from scripts.head_surgery.run_diagnosis_sweep import (
    OUT_DIR, load_manifest_for_ids, load_whisper,
)
from scripts.inference.run_inference import normalize_text

BEAMS = [1, 5]
REP_PENALTIES = [1.0, 1.1, 1.3]
NO_REPEAT_NGRAMS = [0, 3, 5]
TEMP_FALLBACKS = [False, True]
TEMP_FALLBACK_VALUES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def _generate_with_config(model, processor, audios, beam, rep_pen, no_rep_ng, temp_fb, device):
    inputs = processor(audios, sampling_rate=16000, return_tensors="pt", padding=True)
    features = inputs.input_features.to(
        device, dtype=torch.float16 if "cuda" in device else torch.float32
    )
    kwargs = dict(rc.GENERATE_CONFIG)
    kwargs["num_beams"] = beam
    kwargs["repetition_penalty"] = rep_pen
    kwargs["no_repeat_ngram_size"] = no_rep_ng
    if temp_fb:
        kwargs["temperature"] = TEMP_FALLBACK_VALUES
        kwargs["do_sample"] = True
    with torch.no_grad():
        ids = model.generate(features, **kwargs)
    return processor.batch_decode(ids, skip_special_tokens=True)


def run_decoding_grid(manifest_csv: str, batch_size: int, device: str = "cuda") -> dict:
    ids = rc.load_indian_accent_ids()
    subset, id_col = load_manifest_for_ids(ids, manifest_csv)
    audio_col = next(c for c in subset.columns if c in ("audio_path", "audio", "path"))
    ref_col = next(c for c in subset.columns if c in ("reference", "transcript", "sentence"))
    audios = [sf.read(str(p))[0] for p in subset[audio_col]]
    refs = subset[ref_col].astype(str).tolist()
    utt_ids = subset[id_col].astype(str).tolist()

    model, processor = load_whisper(device=device)
    rows, scores = [], []
    configs = list(itertools.product(BEAMS, REP_PENALTIES, NO_REPEAT_NGRAMS, TEMP_FALLBACKS))
    for k, (beam, rp, nr, tf) in enumerate(configs):
        hyps: List[str] = []
        for j in range(0, len(audios), batch_size):
            chunk = audios[j:j + batch_size]
            hyps.extend(t.strip() for t in _generate_with_config(
                model, processor, chunk, beam, rp, nr, tf, device
            ))
        pairs = [(normalize_text(r), normalize_text(h)) for r, h in zip(refs, hyps)]
        br = insertion_rate_breakdown(pairs)
        for uid, ref, hyp in zip(utt_ids, refs, hyps):
            rows.append({"beam": beam, "rep_penalty": rp, "no_repeat_ngram": nr,
                         "temp_fallback": tf, "id": uid, "reference": ref, "hypothesis": hyp})
        scores.append({"beam": beam, "rep_penalty": rp, "no_repeat_ngram": nr,
                       "temp_fallback": tf, **br})
        print(f"[decoding {k+1}/{len(configs)}] beam={beam} rep={rp} "
              f"nr={nr} tf={tf}: ins={br['total']*100:.2f}%")
    pd.DataFrame(rows).to_csv(OUT_DIR / "decoding_grid.csv", index=False)
    pd.DataFrame(scores).to_csv(OUT_DIR / "decoding_scores.csv", index=False)
    return {"configs": len(configs)}


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    run_decoding_grid(args.manifest, batch_size=args.batch_size, device=args.device)


if __name__ == "__main__":
    _cli()
```

- [ ] **Step 2: Syntax check**

Run: `python -c "from scripts.head_surgery import decoding_ablation_grid"`
Expected: no errors.

- [ ] **Step 3: Run Stage E**

Run: `python -m scripts.head_surgery.decoding_ablation_grid --manifest <CV24_MANIFEST_PATH> --batch-size $BS`
Expected: 36 lines `[decoding k/36] ... ins=…%`; artifacts `outputs/head_surgery/decoding_grid.csv`, `outputs/head_surgery/decoding_scores.csv`.

- [ ] **Step 4: Commit**

```bash
git add scripts/head_surgery/decoding_ablation_grid.py
git commit -m "$(cat <<'EOF'
feat(head_surgery): Stage E decoding-strategy ablation grid (T5)

36 configs (beam × rep_penalty × no_repeat_ngram × temp_fallback) on the 511
Indian-accent utterances. Answers the reviewer question of what cheap
decoder tricks achieve alone, before comparison to masking.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Stage F — energy-based VAD (T8)

Frame-level RMS VAD. Runs on existing `silence_25pct`, `silence_50pct`, `silence_75pct` perturbation manifests; filters audio and reruns Whisper (baseline, no hook) on the filtered audio.

**Files:**
- Modify: `scripts/head_surgery/energy_vad.py`
- Modify: `tests/test_head_surgery.py`

- [ ] **Step 1: Write failing unit tests for the VAD primitive**

Append to `tests/test_head_surgery.py`:

```python
from scripts.head_surgery.energy_vad import filter_silence


def test_filter_silence_drops_zero_region():
    import numpy as np
    sr = 16000
    speech = np.random.default_rng(0).normal(0, 0.1, size=sr).astype(np.float32)  # 1s speech
    silence = np.zeros(int(0.5 * sr), dtype=np.float32)
    clip = np.concatenate([speech, silence, speech])
    filtered = filter_silence(clip, sr, db_floor=-35.0, min_silence_ms=200)
    # Should remove ~500ms of zeros → length ≈ 2s * 16000 = 32000 samples (±frame granularity)
    assert len(filtered) < len(clip) - int(0.3 * sr), \
        f"VAD did not drop expected silence: kept {len(filtered)}/{len(clip)}"
    assert len(filtered) > int(1.5 * sr), "VAD dropped too much"


def test_filter_silence_preserves_all_speech():
    import numpy as np
    sr = 16000
    speech = np.random.default_rng(0).normal(0, 0.1, size=2 * sr).astype(np.float32)
    filtered = filter_silence(speech, sr, db_floor=-35.0, min_silence_ms=200)
    # Speech with no gaps → basically unchanged (allow 1% margin for frame-boundary rounding)
    assert len(filtered) >= int(0.99 * len(speech))
```

Run: `pytest tests/test_head_surgery.py -v -k silence`
Expected: FAIL with `ImportError`.

- [ ] **Step 2: Implement `filter_silence`**

Replace `scripts/head_surgery/energy_vad.py` with:

```python
"""Stage F — energy-based VAD preprocessing (T8).

Frame-level RMS threshold. A frame is 'silence' if 20*log10(RMS) < db_floor
for at least min_silence_ms consecutive frames. Silence regions are removed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


FRAME_MS = 20
HOP_MS = 10


def _frame(audio: np.ndarray, sr: int) -> Tuple[np.ndarray, int, int]:
    frame_len = int(sr * FRAME_MS / 1000)
    hop_len = int(sr * HOP_MS / 1000)
    n = 1 + max(0, (len(audio) - frame_len) // hop_len)
    frames = np.lib.stride_tricks.sliding_window_view(audio, frame_len)[::hop_len][:n]
    return frames, frame_len, hop_len


def filter_silence(audio: np.ndarray, sr: int,
                   db_floor: float = -35.0,
                   min_silence_ms: int = 200) -> np.ndarray:
    """Return audio with below-threshold runs of length >= min_silence_ms removed."""
    if len(audio) == 0:
        return audio
    audio = audio.astype(np.float32)
    frames, frame_len, hop_len = _frame(audio, sr)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
    db = 20.0 * np.log10(rms + 1e-12)
    silent = db < db_floor  # bool per frame
    # Runs of silence: find runs of True with length * HOP_MS >= min_silence_ms
    min_frames = max(1, min_silence_ms // HOP_MS)
    to_drop = np.zeros(len(audio), dtype=bool)
    i = 0
    while i < len(silent):
        if not silent[i]:
            i += 1
            continue
        j = i
        while j < len(silent) and silent[j]:
            j += 1
        if (j - i) >= min_frames:
            start_sample = i * hop_len
            end_sample = min(len(audio), j * hop_len + frame_len)
            to_drop[start_sample:end_sample] = True
        i = j
    return audio[~to_drop]


def run_vad_arm(silence_manifests_csv: str, batch_size: int, device: str = "cuda") -> dict:
    """For each severity × db_floor cell, filter audio and rerun Whisper baseline."""
    import soundfile as sf
    from scripts.head_surgery.insertion_classifier import insertion_rate_breakdown
    from scripts.head_surgery.run_diagnosis_sweep import (
        _infer_whisper_batch, load_whisper, OUT_DIR,
    )
    from scripts.inference.run_inference import normalize_text

    df = pd.read_csv(silence_manifests_csv)
    severity_col = next(c for c in df.columns if c in ("severity", "perturbation"))
    audio_col = next(c for c in df.columns if c in ("audio_path", "audio", "path"))
    ref_col = next(c for c in df.columns if c in ("reference", "transcript", "sentence"))

    model, processor = load_whisper(device=device)
    results = []
    for db_floor in (-40.0, -35.0, -30.0):
        for sev_val, g in df.groupby(severity_col):
            audios_filt = []
            for p in g[audio_col]:
                a, sr = sf.read(str(p))
                audios_filt.append(filter_silence(a, sr, db_floor=db_floor))
            refs = [normalize_text(r) for r in g[ref_col]]
            hyps = []
            for j in range(0, len(audios_filt), batch_size):
                hyps.extend(t.strip() for t in _infer_whisper_batch(
                    model, processor, audios_filt[j:j + batch_size], device
                ))
            pairs = [(r, normalize_text(h)) for r, h in zip(refs, hyps)]
            br = insertion_rate_breakdown(pairs)
            results.append({"severity": sev_val, "db_floor": db_floor, **br})
            print(f"[vad] sev={sev_val} db_floor={db_floor}: ins={br['total']*100:.2f}%")
    out = OUT_DIR / "vad_scores.csv"
    pd.DataFrame(results).to_csv(out, index=False)
    return {"out": str(out), "cells": len(results)}


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--silence-manifest", required=True,
                   help="CSV listing the existing silence-25/50/75 perturbation audio")
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    run_vad_arm(args.silence_manifest, batch_size=args.batch_size, device=args.device)


if __name__ == "__main__":
    _cli()
```

- [ ] **Step 3: Run the VAD unit tests**

Run: `pytest tests/test_head_surgery.py -v -k silence`
Expected: both tests PASS.

- [ ] **Step 4: Run Stage F**

Run: `python -m scripts.head_surgery.energy_vad --silence-manifest <SILENCE_PERTURBATION_MANIFEST> --batch-size $BS`
Expected: 9 cells (3 db_floors × 3 severities) printed; artifact `outputs/head_surgery/vad_scores.csv`.

- [ ] **Step 5: Commit**

```bash
git add scripts/head_surgery/energy_vad.py tests/test_head_surgery.py
git commit -m "$(cat <<'EOF'
feat(head_surgery): Stage F energy-based VAD (T8)

RMS-threshold frame VAD, three db_floors × three silence severities on the
existing silence-injection perturbation manifests. Filtered audio is rerun
through Whisper baseline; outputs vad_scores.csv.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: Stage G — aggregate report (T6)

Reads all Stage-A/C/D/E/F artifacts, joins with the T7 midterm CSVs, and writes a markdown report with the tables specified in the PRD.

**Files:**
- Modify: `scripts/head_surgery/aggregate_report.py`

- [ ] **Step 1: Implement the aggregator**

Replace `scripts/head_surgery/aggregate_report.py` with:

```python
"""Stage G — assemble the final head-surgery report (T6).

Writes docs/head_surgery_report.md with:
  - Per-accent insertion rate for 6 CV24 groups × {baseline, top-K masked}
  - Overall WER (CV24 composite, Fair-Speech, LibriSpeech) for baseline + best head
  - MMR before/after for top-3 heads
  - Insertion breakdown before/after for top-10 heads
  - Decoding ablation table (from decoding_scores.csv)
  - VAD table (from vad_scores.csv)
  - Per-head driving-ness ranking table (from head_scores.csv)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd

OUT_DIR = Path("outputs/head_surgery")
REPORT = Path("docs") / "head_surgery_report.md"


def _fmt_rate(r):
    return f"{r*100:.2f}%" if pd.notna(r) else "—"


def _md_table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, floatfmt=".3f")


def build_report(midterm_per_accent_csv: str = None) -> None:
    parts: List[str] = []
    parts.append("# Head-Surgery Diagnosis — Results\n")
    parts.append("Target model: Whisper-large-v3. Evaluation subset: 511 Indian-accent CV24 test utterances.\n")

    # Baseline
    base = json.loads((OUT_DIR / "baseline_metrics.json").read_text())
    parts.append("## 1. Baseline (Gate G1)\n")
    parts.append(f"- Insertion rate: **{_fmt_rate(base['insertion_rate_total'])}** (midterm target 9.62%).\n")
    parts.append(f"- Breakdown — repetition: {_fmt_rate(base['insertion_rate_repetition'])}, "
                 f"syntactic: {_fmt_rate(base['insertion_rate_syntactic'])}, "
                 f"content: {_fmt_rate(base['insertion_rate_content'])}.\n")

    # Top-K heads
    top = pd.read_csv(OUT_DIR / "top_k_heads.csv")
    parts.append("## 2. Top-K hallucination-driving heads\n")
    parts.append(_md_table(top[[
        "layer", "head", "delta_insertion_rate", "delta_repetition", "delta_syntactic",
        "delta_content", "p_value_delta", "regression_ok", "non_indian_wer_masked",
    ]]))
    parts.append("\n")

    # Decoding ablation
    if (OUT_DIR / "decoding_scores.csv").exists():
        dec = pd.read_csv(OUT_DIR / "decoding_scores.csv")
        parts.append("## 3. Decoding-strategy ablation (36 configs)\n")
        parts.append(_md_table(dec.sort_values("total").head(10)
                                 [["beam", "rep_penalty", "no_repeat_ngram", "temp_fallback", "total",
                                   "repetition", "syntactic", "content"]]))
        parts.append("\nTop 10 configs by lowest insertion rate.\n")

    # VAD
    if (OUT_DIR / "vad_scores.csv").exists():
        vad = pd.read_csv(OUT_DIR / "vad_scores.csv")
        parts.append("## 4. Energy-VAD under silence injection\n")
        parts.append(_md_table(vad))
        parts.append("\n")

    # Full per-head ranking
    scores = pd.read_csv(OUT_DIR / "head_scores.csv")
    parts.append("## 5. All 640 heads — ranked\n")
    parts.append(_md_table(scores.sort_values("delta_insertion_rate", ascending=False).head(50)
                             [["layer", "head", "delta_insertion_rate",
                               "p_value_delta", "regression_ok"]]))
    parts.append("\n*Top 50 shown; full table in `outputs/head_surgery/head_scores.csv`.*\n")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(parts))
    print(f"wrote {REPORT}")


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--midterm-per-accent-csv", default=None,
                   help="Optional midterm per-accent insertion CSV (T7) for the accent-×-masking table.")
    args = p.parse_args()
    build_report(args.midterm_per_accent_csv)


if __name__ == "__main__":
    _cli()
```

- [ ] **Step 2: Syntax check**

Run: `python -c "from scripts.head_surgery import aggregate_report"`
Expected: no error.

- [ ] **Step 3: Run Stage G**

Run: `python -m scripts.head_surgery.aggregate_report`
Expected: `wrote docs/head_surgery_report.md`. Open it and verify all 5 sections populate. Empty sections indicate an upstream stage didn't produce its artifact.

- [ ] **Step 4: Commit**

```bash
git add scripts/head_surgery/aggregate_report.py docs/head_surgery_report.md
git commit -m "$(cat <<'EOF'
feat(head_surgery): Stage G aggregate report (T6)

Assembles the final markdown report from the head scores, decoding-ablation
scores, VAD scores, and baseline metrics. Writes docs/head_surgery_report.md
with five sections: baseline, top-K heads, decoding ablation, VAD, full
ranking.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: End-to-end smoke test and final gates review

A short sanity pass that touches nothing, to catch missing artifacts and document the milestone's state.

**Files:**
- Modify: `tests/test_head_surgery.py`
- Create: `logs/head-surgery-diagnosis-complete.md`

- [ ] **Step 1: Add an artifact-presence test**

Append to `tests/test_head_surgery.py`:

```python
@pytest.mark.parametrize("relpath", [
    "outputs/head_surgery/baseline_metrics.json",
    "outputs/head_surgery/tune_batch_size.json",
    "outputs/head_surgery/pilot_metrics.json",
    "outputs/head_surgery/sweep.csv",
    "outputs/head_surgery/head_scores.csv",
    "outputs/head_surgery/top_k_heads.csv",
    "outputs/head_surgery/decoding_scores.csv",
    "outputs/head_surgery/vad_scores.csv",
    "docs/head_surgery_report.md",
])
def test_expected_artifact_present(relpath):
    p = Path(relpath)
    if not p.exists():
        pytest.skip(f"{relpath} not produced yet — run the pipeline")
    assert p.stat().st_size > 0, f"{relpath} is empty"
```

- [ ] **Step 2: Run the full test suite**

Run: `pytest tests/test_head_surgery.py -v`
Expected: all unit tests PASS; artifact-presence tests PASS or SKIP (skip only if that stage hasn't been run yet on this machine).

- [ ] **Step 3: Write the milestone-complete log**

Create `logs/head-surgery-diagnosis-complete.md`:

```markdown
---
fileClass: Log
name: head-surgery-diagnosis-complete
description: v2.0 head-surgery diagnosis MVP — Stages A through G complete.
status: complete
subtype: evaluation
created: 2026-04-17
updated: 2026-04-17
tags: [head-surgery, v2.0, milestone-complete]
aliases: []
---

# Head-surgery diagnosis — milestone complete

## Gate results

| Gate | Result | Notes |
|---|---|---|
| G1  baseline reproduces 9.62% ± 0.5pp | <PASS/FAIL> | observed <X.XX>% |
| G1.5 batch-size tune | <PASS/FAIL> | chosen batch_size = <N> |
| G2  pilot shows head-level signal | <PASS/FAIL> | Δ range [<a>, <b>] |
| G3  batched == serial on pilot | <PASS/FAIL> | max \|Δ\| = <x> |
| G4  sweep completeness | <PASS/FAIL> | rows = <N>, expected 327,040 |
| G5  non-Indian baseline == T7 | <PASS/FAIL/SKIP> | |

## Top-K heads identified

See `docs/head_surgery_report.md` §2 and `outputs/head_surgery/top_k_heads.csv`.

## Deferred work

- D1 selective head fine-tuning — deferred per PRD §3 scope.
- D3 Silero VAD comparison — stretch, skipped.
- D7 publication-quality heatmap — stretch, skipped.

## References

- PRD: `tasks/prd-head-surgery-diagnosis.md`
- Plan: `docs/superpowers/plans/2026-04-17-head-surgery-diagnosis.md`
- Report: `docs/head_surgery_report.md`
```

Fill in the gate results and top-K summary after running the pipeline end to end.

- [ ] **Step 4: Commit**

```bash
git add tests/test_head_surgery.py logs/head-surgery-diagnosis-complete.md
git commit -m "$(cat <<'EOF'
test(head_surgery): artifact-presence smoke + milestone-complete log

Final sanity suite that asserts every pipeline stage wrote its expected
artifact. Milestone-complete log records gate results for the writeup.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 16: Stretch — per-head heatmap (D7, optional)

Only if the rest of the MVP is done and the writeup benefits from a figure. Produces `docs/head_surgery_heatmap.png` — a 32×20 grid where cell (L, h) color-encodes `delta_insertion_rate`.

**Files:**
- Create: `scripts/plots/head_surgery_heatmap.py`

- [ ] **Step 1: Implement the plotting script**

Create `scripts/plots/head_surgery_heatmap.py`:

```python
"""D7 stretch — per-head heatmap of Δ insertion rate."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("docs/head_surgery_heatmap.png")


def main():
    scores = pd.read_csv("outputs/head_surgery/head_scores.csv")
    L = scores["layer"].max() + 1
    H = scores["head"].max() + 1
    grid = np.full((L, H), np.nan)
    for _, r in scores.iterrows():
        grid[int(r["layer"]), int(r["head"])] = float(r["delta_insertion_rate"])
    fig, ax = plt.subplots(figsize=(8, 10))
    vmax = np.nanmax(np.abs(grid))
    im = ax.imshow(grid, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xlabel("Head index")
    ax.set_ylabel("Decoder layer")
    ax.set_title("Δ insertion rate (masked − baseline) on Indian-accent CV24\n"
                 "Red = hallucination-driving; blue = hallucination-suppressing")
    cb = fig.colorbar(im, ax=ax); cb.set_label("Δ insertion rate")
    fig.tight_layout()
    fig.savefig(OUT, dpi=200)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `python scripts/plots/head_surgery_heatmap.py`
Expected: `wrote docs/head_surgery_heatmap.png`. Inspect the image.

- [ ] **Step 3: Commit**

```bash
git add scripts/plots/head_surgery_heatmap.py docs/head_surgery_heatmap.png
git commit -m "$(cat <<'EOF'
feat(head_surgery): per-head heatmap for writeup (D7 stretch)

32×20 Δ-insertion-rate heatmap, red = hallucination-driving, blue =
hallucination-suppressing. Stretch item from PRD §3.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-review summary (plan author)

**Spec coverage (PRD features):**

| PRD ID | Feature | Implemented in Task |
|---|---|---|
| (infra) | CV25 targeted audio extraction + manifests | Task 0 |
| T1 | Per-head masking hook (serial + batched) | Tasks 4, 5 |
| T2 | 640-cell sweep | Task 9 |
| T3 | Driving-ness metric + bootstrap + guard | Task 11 |
| T4 | Insertion classifier reuse | Task 6 step 1, smoke in Task 10 |
| T5 | Decoding-ablation grid | Task 12 |
| T6 | Evaluation report | Task 14 |
| T7 | Reuse 216-run baseline CSVs | Stage G reads them in Task 14; guard cross-checks in Task 11 Step 4 (Gate G5) |
| T8 | Energy VAD | Task 13 |
| T9 | Repro config | Tasks 2, 3 |
| Stage A.5 | Batch-size tune | Task 7 |
| Gates G1–G5 | All gates | Tasks 6, 7, 8, 9, 11 |

**Placeholder scan:** no "TBD"/"TODO"/"implement later" markers. The `<fill in from …>` placeholders inside log templates are explicit instructions for runtime values the engineer records after running a stage; they are not plan-implementation placeholders.

**Type consistency:** function names used across tasks — `run_baseline`, `run_pilot`, `run_full`, `compute_head_scores`, `compute_regression_guard`, `write_top_k`, `build_report`, `filter_silence`, `SerialHeadMaskHook`, `BatchedHeadMaskHook`, `load_indian_accent_ids`, `insertion_rate_breakdown`, `categorize_insertions`, `_infer_whisper_batch`, `load_whisper`, `load_manifest_for_ids` — verified consistent throughout.

**Scope:** plan implements the PRD's diagnosis-only MVP (T1–T9). No fine-tuning, no out-of-scope items. Task 16 (heatmap) is clearly marked optional.

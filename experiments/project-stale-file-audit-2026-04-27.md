---
fileClass: Experiment
name: Project Stale File Audit 2026-04-27
description: Repository-wide audit for stale, generated, duplicated, and locally derived files in FairSpeechLoRaAndGRPO.
status: complete
created: 2026-04-27
updated: 2026-04-27
tags:
  - project-audit
  - stale-files
  - cleanup
aliases:
  - stale file audit
---

# Project Stale File Audit 2026-04-27

## Scope

Audited `/Users/qifanwen/Desktop/Vault1/projects/FairSpeechLoRaAndGRPO` without deleting or moving files.

## Checks run

- `git status --short` and tracked/untracked file inventory.
- Top-level and runtime directory sizes via `du`/Python traversal.
- `.gitignore` gap check against Python/ML/project-audit patterns.
- Duplicate archive extraction check for `datasets/asr_fairness_share.zip`.
- Generated deployment duplicate check for `docs/_deploy-head-surgery/index.html` versus `docs/head-surgery-results.html`.
- Head-surgery/hallucination pivot scan given the current project direction toward compression/accent fairness.

## Key findings

- `datasets/asr_fairness_share.zip` duplicates the extracted `datasets/data/` and `datasets/results/` contents exactly by size for all 331 archive files.
- `docs/_deploy-head-surgery/index.html` is byte-identical to `docs/head-surgery-results.html`; the deploy directory is generated packaging, not a source artifact.
- `graphify-out/` is already deleted in the working tree but still tracked in Git; it is generated graph/cache output.
- `outputs/head_surgery/` is the largest tracked artifact area at about 58 MB and belongs to the now-negative hallucination/head-surgery direction.
- `.pkm/pkm-index.db` and `.codex/hooks.json` are local machine/tooling state and should not be treated as project source.

## No cleanup executed

This audit only produced recommendations. Cleanup should be done selectively after approval.

---
fileClass: Log
name: Project Cleanup Duplicate Artifacts
description: Removed duplicate local artifacts after verifying the shared ASR archive was extracted and the deploy HTML copied the source page exactly.
status: complete
subtype: setup
created: 2026-04-27
updated: 2026-04-27
tags:
  - cleanup
  - duplicate-artifacts
  - datasets
aliases:
  - duplicate artifact cleanup
---

# Project Cleanup Duplicate Artifacts

Removed duplicate local artifacts identified by the stale-file audit.

## Removed

- `datasets/asr_fairness_share.zip` — the archive duplicated extracted `datasets/data/` and `datasets/results/` contents; all 331 archive files matched the extracted files by size before deletion.
- `docs/_deploy-head-surgery/` — generated deployment package; `index.html` was byte-identical to `docs/head-surgery-results.html`, and `.vercel/` was deployment metadata.

No source code was changed in this step.

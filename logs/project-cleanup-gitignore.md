---
fileClass: Log
name: Project Cleanup Gitignore
description: Added ignore rules for local Codex/PKM/graphify state, generated deploy output, Python caches, and ML binary artifacts.
status: complete
subtype: setup
created: 2026-04-27
updated: 2026-04-27
tags:
  - cleanup
  - gitignore
  - local-state
aliases:
  - cleanup gitignore
---

# Project Cleanup Gitignore

Updated `.gitignore` as part of the stale-file cleanup pass.

## Added coverage

- Python/test/cache outputs: `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, `.cache/`, coverage outputs.
- Local environment state: `.env`.
- ML artifacts: model/checkpoint directories and common weight formats.
- Local tool state: `.codex/`, `.pkm/*.db`, `graphify-out/`, `.graphify_detect.json`, `.graphify_python`.
- Generated deployment copy: `docs/_deploy-head-surgery/`.
- Local-only archive payloads: `archive-local/`.

This keeps local machine state and generated artifacts out of future project diffs.

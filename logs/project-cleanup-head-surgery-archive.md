---
fileClass: Log
name: Project Cleanup Head Surgery Archive
description: Moved stale head-surgery and hallucination artifacts out of active source, docs, tests, outputs, and dataset result paths.
status: complete
subtype: refactor
created: 2026-04-27
updated: 2026-04-27
tags:
  - cleanup
  - archive
  - head-surgery
  - hallucination
aliases:
  - head surgery archive cleanup
---

# Project Cleanup Head Surgery Archive

Archived the stale hallucination/head-surgery line so the active project tree can focus on the requested compression/accent-fairness direction.

## Tracked archive

Moved tracked source, tests, docs, logs, task spec, knowledge note, plotting utility, workflow explainer, and output artifacts under:

- `archive/head-surgery-2026-04-27/`

This preserves reproducibility while removing the head-surgery implementation and artifacts from active `scripts/`, `tests/`, `docs/`, `logs/`, `knowledge/`, `tasks/`, and `outputs/` paths. `pyproject.toml` was also updated to remove the active head-surgery reproducibility anchor and exclude archive directories from package/tool discovery.

## Local-only archive

Moved ignored/local hallucination result payloads under:

- `archive-local/head-surgery-2026-04-27/`

`archive-local/` is gitignored because those files are local data/result payloads rather than source artifacts.

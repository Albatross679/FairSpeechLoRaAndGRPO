---
fileClass: Issue
name: Pytest Unavailable in Local Environment
description: The cleanup pass could not run the test suite because pytest is not installed in the active Python environment.
status: open
severity: low
subtype: compatibility
created: 2026-04-27
updated: 2026-04-27
tags:
  - pytest
  - environment
  - cleanup
aliases:
  - pytest unavailable
---

# Pytest Unavailable in Local Environment

During the stale-file cleanup verification pass, both test commands failed because `pytest` is unavailable in the active environment.

## Commands attempted

```bash
pytest -q
python3 -m pytest -q
```

## Observed errors

- `pytest`: command not found
- `python3 -m pytest`: `No module named pytest`

## Impact

Repository cleanup changes were applied, but the local test suite could not be executed in this environment without installing developer dependencies.

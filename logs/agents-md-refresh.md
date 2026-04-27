---
fileClass: Log
name: AGENTS.md Refresh
description: Updated AGENTS.md to match the current FairSpeechLoRaAndGRPO repository layout and available tooling.
status: complete
subtype: setup
created: 2026-04-27
updated: 2026-04-27
tags:
  - agents
  - documentation
  - repository-layout
aliases:
  - Agent instructions refresh
---

# AGENTS.md Refresh

Updated `AGENTS.md` so the agent instructions reflect the current repository state:

- Replaced stale `NLPClassProject/` root with `FairSpeechLoRaAndGRPO/`.
- Removed the nonexistent `.Codex/` directory reference and documented current `.claude/` and `.pkm/` directories.
- Added `CLAUDE.md`, `autoresearch/`, and `scripts/head_surgery/` to the documented layout.
- Marked `datasets/` as locally created/gitignored and noted that `issues/` and `references/` should be created when needed.
- Updated VRAM guidance to reference `docs/maximize-vram-playbook.html` when a dedicated skill is unavailable.
- Made graphify guidance conditional on `graphify-out/` existing in the environment.

---
fileClass: Log
name: Graphify Codex Install
description: Installed Graphify integration for Codex and registered the project hook/instructions.
status: complete
subtype: setup
created: 2026-04-27
updated: 2026-04-27
tags:
  - graphify
  - codex
  - setup
aliases:
  - Graphify install for Codex
---

# Graphify Codex Install

Installed Graphify for Codex in three places:

- Ran `graphify install --platform codex`, which installed the packaged Codex skill at `/Users/qifanwen/.agents/skills/graphify/SKILL.md`.
- Staged and installed the same Graphify Codex skill into Codex Desktop's skill directory at `/Users/qifanwen/.codex/skills/graphify/SKILL.md`.
- Ran `graphify codex install` in this project. `AGENTS.md` already had the graphify section, and `.codex/hooks.json` was created with a PreToolUse reminder hook for existing `graphify-out/graph.json` graphs.
- Mirrored `.codex/hooks.json` through `/Users/qifanwen/Desktop/FairSpeechLoRaAndGRPO-codex-stage/install.py` so the project-local `.codex/` change has a staged, repeatable installer.

Verification:

- `graphify --help` works from `/Users/qifanwen/.local/bin/graphify`.
- `/Users/qifanwen/.codex/skills/graphify/SKILL.md` exists and begins with `name: graphify`.
- `/Users/qifanwen/.agents/skills/graphify/SKILL.md` exists with a `.graphify_version` stamp.

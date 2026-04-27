---
fileClass: Log
name: Codex PreToolUse Hook Cleanup
description: Removed the project-local Claude-style Graphify PreToolUse hook that caused Codex Desktop hook failure notices.
status: complete
subtype: fix
created: 2026-04-27
updated: 2026-04-27
tags:
  - codex
  - hooks
  - graphify
  - local-config
aliases:
  - PreToolUse hook cleanup
---

# Codex PreToolUse Hook Cleanup

Removed `/Users/qifanwen/Desktop/Vault1/projects/FairSpeechLoRaAndGRPO/.codex/hooks.json`, which contained a Claude-style `PreToolUse`/`Bash` hook. Codex Desktop was surfacing that hook as failed before shell commands.

## Notes

- Original hook was backed up under `/Users/qifanwen/Desktop/FairSpeechLoRaAndGRPO-codex-stage/`.
- Graphify guidance remains in `AGENTS.md`, so the hook was redundant.
- Restarting the Codex session may be needed if hooks were loaded at session startup.

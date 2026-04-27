---
fileClass: Issue
name: Codex PreToolUse Hook Failure
description: Project-local Claude-style Graphify hook caused Codex Desktop to report PreToolUse failures before shell commands.
status: resolved
severity: low
subtype: system
created: 2026-04-27
updated: 2026-04-27
tags:
  - codex
  - hooks
  - graphify
  - local-config
aliases:
  - PreToolUse failed
---

# Codex PreToolUse Hook Failure

Codex Desktop displayed `PreToolUse failed` before shell commands because the project-local `.codex/hooks.json` used a Claude-style `PreToolUse`/`Bash` hook shape.

## Resolution

Removed the project-local `.codex/hooks.json` hook file. Graphify guidance remains available in `AGENTS.md`, so the failing hook is redundant.

---
fileClass: Log
name: Graphify Active Code Map
description: Scoped Graphify away from local dataset/audio payloads, enabled artifact synchronization, and rebuilt the active code graph.
status: complete
subtype: setup
created: 2026-04-27
updated: 2026-04-27
tags:
  - graphify
  - setup
  - code-map
aliases:
  - Active code graph map
---

# Graphify Active Code Map

Added `.graphifyignore` so Graphify maps the active project corpus instead of local/generated/heavy payloads.

Ignored paths include local datasets, `archive-local/`, generated `graphify-out/`, archived stale work, and large model/audio artifacts.

Updated `.gitignore` so Graphify artifacts are not ignored by Git. The generated `graphify-out/` outputs and root `.graphify_detect.json` are intended to be synchronized with the repository.

Rebuilt the no-LLM AST code graph with:

```bash
graphify update .
```

Result:

- Detection after ignore: 137 supported files, about 166,700 words.
- Code graph: 1,041 nodes, 2,089 edges, 34 communities.
- Outputs written to `graphify-out/graph.json`, `graphify-out/graph.html`, and `graphify-out/GRAPH_REPORT.md`.

Note: this was an AST-only code graph. Docs, PDFs, and images are detected but still require the LLM-backed `/graphify --update` semantic pass if a full multimodal graph is needed.

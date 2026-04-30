# Layout audit for `project_state_acl_report.tex`

Audited: 2026-04-29

## Source inventory

| Item | Count | Notes |
|---|---:|---|
| `table` floats | 6 | Compact single-column tables. |
| `table*` floats | 2 | Used only for WER matrix and baseline group disparity. |
| `tabular` blocks | 8 | All begin/end pairs balanced. |
| `figure` floats | 0 | No external figure dependencies. |
| `landscape` pages | 0 | No landscape waste risk. |
| `resizebox` usage | 0 | Avoids unreadably scaled tables. |
| Cited bibliography keys | 9 | All keys are present in the `.bib` file. |

## Compile status

Compilation was attempted from `reports/acl-project-state/`, but this VM does not have a TeX engine installed:

```text
latexmk: command not found
pdflatex: command not found
```

## Layout risk notes

- The document is intentionally short: about 1,087 source words plus compact tables.
- The table design avoids long paragraph blocks and uses matrices/bar cells for the main findings.
- Wide tables are limited to two `table*` floats to reduce float-placement pressure in ACL two-column format.
- No PDF/log overfull audit could be completed until `latexmk` or `pdflatex` is installed.

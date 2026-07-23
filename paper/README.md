# Manuscript

LaTeX source for the paper. Target: IEEE journal (IEEEtran, `journal` mode).

## Files

- `main.tex` — the manuscript.
- `refs.bib` — bibliography. **Verified** method references at the top; **placeholder**
  domain references (marked `TODO`) that you must replace from your own literature review.
- `figures/` — self-contained copies of the figures the paper embeds, so it compiles on
  Overleaf without the rest of the repo. Regenerate the originals under `analysis/` and
  re-copy if they change (see below).

## Draft markers (search for these before submission)

- `\TODO{...}` — content you must supply or verify (rendered in **red**). Includes every
  placeholder citation and the entire Related Work section.
- `\PENDING{...}` — a number that fills in once the multi-seed runs finish (rendered in
  **blue**), e.g. the mean ± std in the main results table.

Before submission, resolve all `\TODO`/`\PENDING`, then either delete them or neutralise the
macros by redefining them to `{}` at the top of `main.tex`.

## Building

**Overleaf (easiest, no install).** Create a project, upload `main.tex`, `refs.bib`, and the
`figures/` folder, set the compiler to pdfLaTeX, and compile. The `figures/` copies mean no
other repo files are needed.

**Locally** (needs TeX Live or MiKTeX; not installed in this repo's environment):

```bash
cd paper
latexmk -pdf main.tex        # runs pdflatex + bibtex as needed
# or manually:
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`\graphicspath` also points at `../analysis/` and `../experiments/results/`, so a local build
finds figures even without the `figures/` copies.

## Keeping figures in sync

The embedded figures currently are:

| In paper | Source |
|---|---|
| `tf_family_snr_robustness.png` | `analysis/compare_all_experiments.py` |
| `qualitative_tf_lfm.png` | `analysis/qualitative_tf_illustration.py` |

If you regenerate them, copy the updated files into `paper/figures/`. Other figures available
to add: `overall_accuracy_ci.png`, `wvd_lowsnr_confusion_custom_cnn.png`,
`all9_snr_robustness.png`, and the per-experiment confusion matrices under
`experiments/results/`.

## Source material

The prose draws directly on:

- `docs/decisions.md` — Methods / implementation details (the design-decision log).
- `docs/results_summary.md` — Results / Discussion (numbers, figures, significance).
- `docs/dataset.md` — Dataset and Signal Model section.

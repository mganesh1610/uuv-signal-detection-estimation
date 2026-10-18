# Source provenance and validation

## Existing source preserved

- `uuv_loader.py`: recovered from the author's local project files and copied without edits. It handles MAT binary loading, expected dimensions/transposition and optional Google Drive downloads.
- [Study 1 excerpts](study-1.md#original-code-excerpts): four Python blocks copied from the author's public [matched-filter study](https://mganesh.notion.site/33818c605907812c8268c7ea63da699a).
- [Study 2 excerpts](study-2.md#original-code-excerpts): three Python blocks copied from the author's public [GLRT/MLE study](https://mganesh.notion.site/3c018c60590781c2b5f6cb2084acb416).

The excerpts reference context variables and helper functions from their original scripts. They are presented as excerpts, not standalone executable programs. No missing original implementation has been invented or substituted.

## New packaging

The README, study summaries, this provenance note and `examples/inspect_mat.py` were prepared for this public archive. The example is a small local-only interface to the existing loader.

## Components still needed for full study reproduction

- Complete matched-filter/ROC/waveform analysis script and configuration.
- `Marimuthu_prob2_prob4_GLRT_MLE.py`.
- `Marimuthu_prob5_Jammer_GLRT.py`.
- Original definitions of referenced helper functions, parameter grids, normalization and threshold conventions.
- The corresponding study-specific input data, random seeds and run instructions.

The study pages list the latter two script names, but their complete files were not available in the sources recovered for this archive. Dataset labels such as "Set 1" are local to each study; they should not be assumed to identify the same file across studies.

## Results provenance

Numerical study outcomes are transcribed from the linked write-ups, not newly measured by this repository. The study pages retain the original plots and broader discussion. Detection probability and ROC evidence belong to Study 1; Study 2's case-specific GLRT score comparisons are not a substitute for threshold calibration or a multi-condition ROC evaluation.

No course datasets, private file paths, personal contact records or unrelated research data are committed.

## Validation

The original loader is checked on the author's locally recovered MAT arrays, with input files kept outside the repository. A synthetic transpose and invalid-shape check verifies the loader's documented behavior. This validates data loading only, not the reported detection/estimation experiments.

Checked on October 18, 2026 with Python 3.15.0rc3, NumPy 2.5.4 and SciPy 1.18.1:

- Both locally recovered arrays loaded as finite float64 matrices of shape (1024, 200), identical to the values read directly by SciPy.
- A synthetic transposed MAT array was oriented correctly.
- Incompatible expected dimensions raised ValueError.
- The loader copy matched the original byte-for-byte. SHA-256: `7d7c631dc00956d3653ce419588fc4513245ef3b83dd54d0e2e1acef156158eb`.

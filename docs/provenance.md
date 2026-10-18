# Source provenance and validation

Full source was recovered from the author's coursework archive. [source-manifest.json](source-manifest.json) records source-relative paths, original/published SHA-256 hashes and adaptations.

## Source mapping

| Published file | Recovered source |
|---|---|
| study1/white_noise_detection.py | Project 1/Submisison file/problem_2_code.py |
| study1/roc_analysis.py | Project 1/Submisison file/problem_3_code.py |
| study1/correlated_noise_detection.py | Project 1/Submisison file/problem_4_code.py |
| study1/waveform_optimization.py | Project 1/Submisison file/problem_5_code.py |
| study1/data_validation.py | Project 1/Submisison file/data_check_code.py |
| uuv_loader_set34.py | Project 1/Submisison file/uuv_data loader_set 3 and 4_code.py |
| study2/glrt_mle.py | Project 2/old/Marimuthu_prob2 final old.py |
| study2/jammer_glrt.py | Project 2/old/5 old/marimuthu_prob6.py |

The original `uuv_loader.py` is unchanged, with SHA-256 `7d7c631dc00956d3653ce419588fc4513245ef3b83dd54d0e2e1acef156158eb`. It matches the supplied Study 1 submission loader.

## Study 2 version boundary

The OneDrive export's error log identifies the final Project 2 submission scripts, data and source zip as failed downloads. Earlier complete scripts were available and are published here. The portfolio's display names `Marimuthu_prob2_prob4_GLRT_MLE.py` and `Marimuthu_prob5_Jammer_GLRT.py` were not recovered as files with those exact names; no missing implementation was fabricated.

The available Study 2 MAT files cover Sets 1 and 2. Original jammer Sets 3 and 4 were unavailable. Their reported metrics remain sourced from the original write-up, separate from the synthetic checks below.

## Portability changes

- Replace original Drive/Colab resource links with local input defaults.
- Resolve bundled loaders locally rather than downloading executable helper code.
- Accept local MAT paths in the white-noise script's existing input branches.
- Configure Study 2 input/output directories through environment variables.
- Normalize line endings in adapted scripts.

Numerical analysis function syntax trees were compared with the originals and preserved. Only loader resolution and the white-noise main input-loading branches are excluded from that equality check. Original files remain unchanged outside the repository. Documentation and `examples/` wrappers are new packaging.

## Validation — October 18, 2026

Environment: Python 3.15.0rc3, NumPy 2.5.4 and SciPy 1.18.1, with noninteractive Matplotlib.

### Study 1: actual supplied datasets

All five programs completed successfully.

- White-noise threshold: **7.550756**. Set 2 empirical detection probability: **0.90**.
- Theoretical white-noise probability at the stated 10.9 dB: **0.913975**. Exact template energy corresponds to 10.9439 dB and yields **0.916735**, explaining the small distinction.
- Correlated-noise probability: **0.886246 theoretical / 0.88 empirical**.
- Waveform deflection gain: **13.166656×**.
- Analytical/empirical ROC plots generated successfully.
- Paired-data check: Set 2 ≈ Set 1 + template, scale 1.0 and negligible numerical residual. These are paired observations, not independent validation cohorts.

The original loader also passed a direct-value comparison with SciPy, a synthetic transpose check and rejection of incompatible dimensions.

### Study 2: available Sets 1 and 2

The recovered GLRT/MLE program ran on local MAT-to-CSV conversions and reproduced:

- GLRT peaks **1.009281** and **1.233437**.
- Target delay **371 samples**, FM-rate shift **10.192323**, amplitude **1.371551** and phase **0.639725 rad**.
- Signal-fit residual variance **2.300785**, noise-only variance **2.309201**.
- Agreement of the unknown-variance and known-variance signal-parameter estimates.

### Study 2: synthetic jammer check

A deterministic check (seed 556) used a 101 Hz sinusoid, known HFM target and independent Gaussian noise across 40 channels. The recovered code correctly classified the constructed absent/present cases, estimated jammer frequency within 0.0011 Hz and amplitude within 0.03, and recovered target delay at sample 500 with amplitude error below 0.03 on a small FM grid around the truth.

This checks implementation behavior; it does not regenerate the original 6.42× jammer-score comparison or original recovery results.

## Distribution and interpretation

The study documents preserve seven original code highlights from the two linked Notion write-ups. Original reported metrics and current execution checks are distinguished above. Study 2 score comparisons do not replace threshold calibration or multi-condition ROC evaluation.

No input datasets, credentials, private resource URLs, assignment prompts, course PDFs, virtual environments or unrelated research files are committed.

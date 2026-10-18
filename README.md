# Underwater Acoustic Signal Detection & Estimation

Two Arizona State University EEE 556 studies by **Ganeshkumar Marimuthu**, connecting statistical detection theory with Python analysis of noisy underwater acoustic sensor recordings.

**Study 1:** matched filtering, ROC analysis, correlated noise and fixed-energy waveform optimization.  
**Study 2:** generalized likelihood ratio testing (GLRT), maximum likelihood estimation (MLE) and sinusoidal jammer suppression.

[Data Scientist portfolio](https://mganesh.notion.site/) · [LinkedIn](https://www.linkedin.com/in/mganesh-ds/)

## Source and reproducibility status

This repository contains the **original UUV MAT-file loader** and **seven existing analysis-code excerpts** recovered from the two portfolio studies. The loader is preserved byte-for-byte; the excerpts are labeled and retain their original code.

The complete experiment scripts and their parameter/configuration files have not yet been recovered. Accordingly, this is a source archive and technical case study, **not a complete reproduction package**. Reported study metrics below come from the linked original write-ups; they have not been regenerated from the partial source in this repository. See [source provenance](docs/provenance.md).

## The two studies

| Study | Question | Methods | Reported evidence |
|---|---|---|---|
| [1 — Detection, ROC and waveform optimization](docs/study-1.md) | How does a known acoustic signal behave under white and correlated Gaussian noise? | Matched filters, covariance-aware filtering, analytical/empirical ROC, Monte Carlo and bootstrap evaluation, covariance eigendecomposition | Correlated-noise detection probability: **0.886246 theoretical / 0.88 empirical**. Fixed-energy waveform optimization: **13.1667× deflection gain**. |
| [2 — GLRT, MLE and jammer suppression](docs/study-2.md) | Can an HFM target signal be detected and its parameters recovered when delay, FM rate and interference are unknown? | Closed-form least-squares MLE, grid search, residual analysis and sinusoidal interference fitting | Evaluated jammer cases: **6.42× GLRT peak-score separation**. Recovered target delay: **0-sample error in the evaluated recovery case**. |

The 13.1667× figure describes the detector's deflection term, and the 6.42× figure compares two GLRT peak scores. Neither is classification accuracy or a general performance guarantee.

## Repository layout

```text
uuv_loader.py                 Original MATLAB data-loading/download helper
requirements.txt             Loader dependencies
docs/
  study-1.md                 Matched-filter study and four original code excerpts
  study-2.md                 GLRT/MLE study and three original code excerpts
  provenance.md              Source inventory, validation and missing components
examples/
  inspect_mat.py             New local-only data inspection wrapper
```

## Use the existing loader

Python 3.10+ is recommended. Create an isolated environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Supply a local MATLAB v5-compatible file containing a two-dimensional numeric sample-by-channel array:

```bash
python examples/inspect_mat.py /path/to/UUV_Set1.mat --samples 1024 --channels 200
```

The wrapper loads the file through the original loader, accepts a transposed channel-by-sample array and prints its dimensions and basic checks. Set the dimensions to match your own dataset; the example dimensions describe the locally checked study files.

Direct Python usage:

```python
from pathlib import Path
from uuv_loader import load_uuv_mat_from_bytes

R = load_uuv_mat_from_bytes(
    Path("data/UUV_Set1.mat").read_bytes(),
    expected_N=1024,
    expected_L=200,
)
# R.shape == (samples, channels)
```

The original helper also supports Google Drive downloads through requests/gdown. The local example makes no network requests. Course datasets are not distributed here.

## Data science skills demonstrated

- Statistical hypothesis testing and false-alarm/detection trade-offs
- Covariance-aware modeling and numerical linear algebra
- Maximum likelihood estimation and least-squares regression
- Parameter search, residual validation and interference modeling
- Monte Carlo/bootstrap evaluation and interpretable scientific reporting

## Original studies

1. [UUV Mine Detection, ROC Analytics & Waveform Optimization](https://mganesh.notion.site/33818c605907812c8268c7ea63da699a)
2. [Jammer-Resilient Underwater Threat Detection — GLRT, MLE & Signal Recovery](https://mganesh.notion.site/3c018c60590781c2b5f6cb2084acb416)

These are academic analyses of supplied acoustic data. The available archive does not establish field performance, real-time deployment or a calibrated operational decision threshold for the jammer study.

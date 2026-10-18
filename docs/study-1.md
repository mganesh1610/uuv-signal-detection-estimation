# Study 1 — UUV Detection, ROC Analytics & Waveform Optimization

[Original study and figures](https://mganesh.notion.site/33818c605907812c8268c7ea63da699a) · [Repository overview](../README.md)

## Problem and approach

Evaluate a known HFM acoustic target template in multi-hydrophone recordings under white Gaussian noise, then account for correlated noise through its covariance matrix. Compare empirical detection behavior with theoretical predictions and examine waveform design under a fixed energy budget.

For template x and covariance C, the covariance-aware statistic is T = rᵀC⁻¹x, with deflection d² = xᵀC⁻¹x. Analytical thresholds and ROC curves are paired with empirical and bootstrap checks. Under a fixed energy constraint, the optimal waveform follows the minimum-eigenvalue direction of C.

## Reported results

| Measurement | Value | Interpretation |
|---|---:|---|
| White-noise detection probability | 0.913975 theoretical / 0.90 empirical | Agreement for the evaluated setup |
| Correlated-noise detection probability | 0.886246 theoretical / 0.88 empirical | Covariance-aware detector validation |
| Fixed-energy waveform improvement | 13.1667× | Gain in the deflection term, not a multiplier on accuracy |

The original page reports additional ROC operating points, energy-to-noise settings and false-alarm estimates. Read them in their original context; values from different settings should not be combined into one operating point.

## Original code excerpts

These four blocks are copied from the existing study. Full implementations, helper functions and parameter defaults are available in [study1/](../study1/).

### White-noise detector

```python
Tvals = (R.T @ x) / sigma2
d2 = float((x @ x) / sigma2)
gamma = math.sqrt(d2) * Qinv(PFA)

detections = Tvals > gamma
K = int(np.sum(detections))
frac = K / L

Tsum = float(np.sum(Tvals))
gamma_sum = math.sqrt(L * d2) * Qinv(PFA_set)
mine_present_global = (Tsum > gamma_sum)
```

### ROC generation

```python
PFA_hat_emp, PD_hat_emp, _ = empirical_roc_from_datasets(
    R_h0=R1, R_h1=R2, x=x, num_thresh=args.num_thresh
)

pfa_grid = np.concatenate([
    np.logspace(-5, -2, 250),
    np.linspace(1e-2, 0.999, 750),
])
pd_true_curve = theoretical_roc(args.enr_true_db, pfa_grid)
pd_alt_curve = theoretical_roc(args.enr_alt_db, pfa_grid)
```

### Correlated-noise GMF

```python
Cw = build_correlated_covariance(N=N, a=0.99, sigma2=10.0)
cinv_x, d2 = prepare_correlated_gmf(x, Cw)
Tvals = R.T @ cinv_x
gamma = math.sqrt(d2) * Qinv(PFA)
detections = Tvals > gamma
PD_true = Q(Qinv(PFA) - math.sqrt(d2))
```

### Waveform optimization

```python
design = design_optimal_waveform_fixed_energy(x, a=a, sigma2=sigma2)
xhat = design["xhat"]
lam_min_C = design["lam_min_C"]
d2_new = design["d2_max"]
```

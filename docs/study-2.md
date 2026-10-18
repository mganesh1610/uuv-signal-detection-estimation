# Study 2 — Jammer-Resilient Detection, GLRT, MLE & Signal Recovery

[Original study and figures](https://mganesh.notion.site/3c018c60590781c2b5f6cb2084acb416) · [Repository overview](../README.md)

## Problem and approach

Detect an HFM target return when its amplitude, phase, delay, FM-rate shift and noise variance are unknown. Combine a grid search over delay/FM rate with a closed-form least-squares fit of cosine and sine coefficients. Convert those coefficients into amplitude and phase estimates.

The GLRT compares total received energy with the residual energy after fitting the signal. A second analysis fits sinusoidal jammer interference, suppresses it, and re-estimates the target parameters. The original write-up includes GLRT surfaces and jammer-frequency profiles.

## Reported results

| Measurement | Value | Scope |
|---|---:|---|
| Noise-only / signal-present GLRT peaks | 1.009281 / 1.233437 | 22.21% peak-score lift in the evaluated cases |
| Noise-only variance / signal-fit residual variance | 2.309201 / 2.300785 | About 0.36% relative difference |
| Jammer cases, target absent / present | 0.386449 / 2.479464 | 6.42× GLRT peak-score separation |
| Estimated jammer frequency | ~101 Hz | Evaluated interference scenario |
| Recovered amplitude | 1.748555 vs true 1.780000 | About 1.77% relative error |
| Recovered phase | 0.932714 vs true 0.942478 rad | 0.009764 rad absolute error |
| Recovered FM-rate shift | 45.001 vs true 45.000 | 0.001 absolute error |
| Recovered delay | 500 vs true 500 samples | Zero-sample error in this recovery case |

GLRT scores are model-fit statistics, not calibrated probabilities. The archive does not provide a validated operating threshold or general detection accuracy for the jammer study. Broader ROC validation and sensitivity analysis remain future work.

## Original code excerpts

These three blocks are copied from the existing study. Full recovered implementations are available in [GLRT/MLE](../study2/glrt_mle.py) and [jammer detection/recovery](../study2/jammer_glrt.py). These are earlier source versions from the author's archive; see [provenance](provenance.md) for the version and data boundaries.

### GLRT map generation for unknown HFM parameters

```python
for beta_index, beta_value in enumerate(fm_rate_beta_grid):
    phase = 2.0 * np.pi * beta_value * hfm_time

    b1 = np.cos(phase)
    b2 = np.sin(phase)
    H = np.column_stack((b1, b2))
    HtH_inverse = np.linalg.pinv(H.T @ H)

    for delay_option_index, start_sample in enumerate(delay_sample_options):
        active_signal_window = received_trace[
            start_sample:start_sample + given_signal_template_sample_count
        ]

        fitted_coefficients = HtH_inverse @ (H.T @ active_signal_window)
        signal_fit = H @ fitted_coefficients

        region1_energy = regionsplit[start_sample]
        region2_energy = (active_signal_window - signal_fit).T @ (active_signal_window - signal_fit)
        region3_energy = regionsplit[total_sample_count] - regionsplit[start_sample + given_signal_template_sample_count]

        glrt_denominator_part = region1_energy + region2_energy + region3_energy
        glrt_score_surface[delay_option_index, beta_index] = glrt_numerator_part / glrt_denominator_part
```

### Parameter recovery from fitted coefficients

```python
estimated_amplitude = math.sqrt(
    estimated_cosine_weight**2 + estimated_sine_weight**2
)

estimated_phase = wrap_phase(
    math.atan2(-estimated_sine_weight, estimated_cosine_weight)
)

estimated_fm_rate_shift = estimated_fm_rate - ALPHA
estimated_delay_seconds = estimated_delay_sample * TS
```

### Jammer frequency grid search

```python
H_d = np.cos(
    2.0 * math.pi * TS * np.outer(sample_index, jammer_frequency_chunk)
)

Ht_r = H_d.T @ received_trace
Ht_H = np.sum(H_d * H_d, axis=0)

estimated_jammer_amplitude = Ht_r / Ht_H
glrt_denominator_part = received_signal_energy - (Ht_r * Ht_r) / Ht_H
```

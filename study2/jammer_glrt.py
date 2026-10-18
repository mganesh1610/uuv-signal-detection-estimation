import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LAST_NAME_PREFIX = "Marimuthu"

BASE_DIR = Path(os.environ.get("UUV_OUTPUT_DIR", "results/study2")).resolve()
DATA_DIR = Path(os.environ.get("UUV_DATA_DIR", "data/study2")).resolve()
OUTPUT_DIR = BASE_DIR / "problem5_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SET3_FILE = DATA_DIR / "Threat_Detection_Set3.csv"
SET4_FILE = DATA_DIR / "Threat_Detection_Set4.csv"

FS = 1000.0
TS = 1.0 / FS
TTX = 1.0
TRX = 2.0
TR = 0.1

A_TRUE = 1.78
PHI_TRUE = 0.3 * math.pi
BETA_TRUE = 45.0
T0_TRUE = 0.5
N0_TRUE = int(round(T0_TRUE / TS))

given_hydrophone_count = 40
given_received_sample_count = int(round(TRX / TS)) + 1
given_signal_template_sample_count = int(round(TTX / TS)) + 1

NU_STEP_INCREMENT = 0.10
NU_ZOOM_HALF_WIDTH = 0.20
NU_STEP_FINE = 0.001
NU_MIN = 0.0
NU_MAX = 0.5 * FS

BETA_MIN = 32.5
BETA_MAX = 0.5 * FS * TR
BETA_STEP_INCREMENT = 0.05
BETA_ZOOM_HALF_WIDTH = 0.15
BETA_STEP_FINE = 0.001

PLOT_DPI = 220

GLRT_PROFILE_LAYOUT = {
    "left": 0.075,
    "right": 0.965,
    "bottom": 0.14,
    "top": 0.88,
}

GLRT_PROFILE_EXPORT_OPTIONS = {
    "dpi": PLOT_DPI,
    "facecolor": "white",
    "bbox_inches": "tight",
    "pad_inches": 0.20,
}

GLRT_SURFACE_LAYOUT = {
    "left": 0.02,
    "right": 0.835,
    "bottom": 0.06,
    "top": 0.91,
}

GLRT_SURFACE_EXPORT_OPTIONS = {
    "dpi": PLOT_DPI,
    "facecolor": "white",
    "bbox_inches": "tight",
    "pad_inches": 0.20,
}

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
    }
)


def read_csv_numeric(csv_path):
    csv_path = Path(csv_path)
    return pd.read_csv(csv_path, header=None).to_numpy(dtype=float)


def load_dataset(csv_path, label):
    matrix = read_csv_numeric(csv_path)
    print(f"[{label}] loaded {Path(csv_path).name} with shape {matrix.shape}")
    return matrix


def wrap_phase(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def nice_label(label):
    if label.startswith("Set") and label[3:].isdigit():
        return f"Set {label[3:]}"
    return label


def make_known_hfm_signal(number_of_samples):
    known_hfm_signal = np.zeros(number_of_samples, dtype=float)

    sample_index = np.arange(given_signal_template_sample_count, dtype=float)
    hfm_time = np.log((TR + sample_index * TS) / TR)

    active_signal_window = A_TRUE * np.cos(
        2.0 * math.pi * BETA_TRUE * hfm_time + PHI_TRUE
    )

    start_sample = N0_TRUE
    stop_sample = start_sample + given_signal_template_sample_count
    known_hfm_signal[start_sample:stop_sample] = active_signal_window

    return known_hfm_signal


# ============================================================
# Problem 5(b): GLRT with known HFM parameters and unknown jammer
# ============================================================

def fit_jammer_on_frequency_grid(received_trace, jammer_frequency_options):
    received_trace = np.asarray(received_trace, dtype=float).reshape(-1)
    jammer_frequency_options = np.asarray(
        jammer_frequency_options,
        dtype=float,
    ).reshape(-1)

    total_sample_count = received_trace.size
    sample_index = np.arange(total_sample_count, dtype=float)

    received_signal_energy = float(received_trace.T @ received_trace)
    glrt_denominator_surface = np.empty(
        len(jammer_frequency_options),
        dtype=float,
    )
    estimated_jammer_amplitude_values = np.empty(
        len(jammer_frequency_options),
        dtype=float,
    )

    chunk_size = 800

    for start_index in range(0, len(jammer_frequency_options), chunk_size):
        stop_index = min(start_index + chunk_size, len(jammer_frequency_options))
        jammer_frequency_chunk = jammer_frequency_options[start_index:stop_index]

        H_d = np.cos(
            2.0 * math.pi * TS * np.outer(sample_index, jammer_frequency_chunk)
        )

        Ht_r = H_d.T @ received_trace
        Ht_H = np.sum(H_d * H_d, axis=0)

        estimated_jammer_amplitude = Ht_r / Ht_H
        glrt_denominator_part = received_signal_energy - (Ht_r * Ht_r) / Ht_H

        estimated_jammer_amplitude_values[start_index:stop_index] = (
            estimated_jammer_amplitude
        )
        glrt_denominator_surface[start_index:stop_index] = np.maximum(
            glrt_denominator_part,
            1e-14,
        )

    best_position = int(np.argmin(glrt_denominator_surface))

    return {
        "jammer_frequency_options": jammer_frequency_options,
        "glrt_denominator_surface": glrt_denominator_surface,
        "estimated_jammer_amplitude_values": estimated_jammer_amplitude_values,
        "estimated_jammer_frequency": float(jammer_frequency_options[best_position]),
        "estimated_jammer_amplitude": float(
            estimated_jammer_amplitude_values[best_position]
        ),
        "glrt_denominator_part": float(glrt_denominator_surface[best_position]),
        "noise_variance_from_signal_fit": float(
            glrt_denominator_surface[best_position] / total_sample_count
        ),
        "best_surface_location": best_position,
    }


def estimate_jammer_mle(received_trace):
    coarse_jammer_frequency_options = np.arange(
        NU_MIN,
        NU_MAX + NU_STEP_INCREMENT / 2.0,
        NU_STEP_INCREMENT,
    )

    coarse_result = fit_jammer_on_frequency_grid(
        received_trace,
        coarse_jammer_frequency_options,
    )

    center_frequency_estimate = coarse_result["estimated_jammer_frequency"]

    fine_jammer_frequency_low = max(
        NU_MIN,
        center_frequency_estimate - NU_ZOOM_HALF_WIDTH,
    )

    fine_jammer_frequency_high = min(
        NU_MAX,
        center_frequency_estimate + NU_ZOOM_HALF_WIDTH,
    )

    fine_jammer_frequency_options = np.arange(
        fine_jammer_frequency_low,
        fine_jammer_frequency_high + NU_STEP_FINE / 2.0,
        NU_STEP_FINE,
    )

    final_result = fit_jammer_on_frequency_grid(
        received_trace,
        fine_jammer_frequency_options,
    )

    return final_result


def make_profile_frequency_options(estimated_jammer_frequency):
    coarse_jammer_frequency_options = np.arange(
        NU_MIN,
        NU_MAX + NU_STEP_INCREMENT / 2.0,
        NU_STEP_INCREMENT,
    )

    fine_jammer_frequency_low = max(
        NU_MIN,
        estimated_jammer_frequency - NU_ZOOM_HALF_WIDTH,
    )

    fine_jammer_frequency_high = min(
        NU_MAX,
        estimated_jammer_frequency + NU_ZOOM_HALF_WIDTH,
    )

    fine_jammer_frequency_options = np.arange(
        fine_jammer_frequency_low,
        fine_jammer_frequency_high + NU_STEP_FINE / 2.0,
        NU_STEP_FINE,
    )

    return np.unique(
        np.r_[
            coarse_jammer_frequency_options,
            fine_jammer_frequency_options,
            estimated_jammer_frequency,
        ]
    )


def run_problem5b_glrt(matrix, label, known_hfm_signal):
    full_received_signal = np.mean(matrix, axis=1)

    h0_jammer_fit = estimate_jammer_mle(full_received_signal)
    h1_jammer_fit = estimate_jammer_mle(full_received_signal - known_hfm_signal)

    glrt_numerator_part = h0_jammer_fit["glrt_denominator_part"]
    glrt_denominator_part = h1_jammer_fit["glrt_denominator_part"]

    best_glrt_score = glrt_numerator_part / glrt_denominator_part
    decision = "Mine present" if best_glrt_score > 1.0 else "Mine absent"

    if decision == "Mine present":
        selected_jammer_fit = h1_jammer_fit
    else:
        selected_jammer_fit = h0_jammer_fit

    plot_jammer_frequency_options = make_profile_frequency_options(
        h1_jammer_fit["estimated_jammer_frequency"]
    )

    profile_h1_jammer_fit = fit_jammer_on_frequency_grid(
        full_received_signal - known_hfm_signal,
        plot_jammer_frequency_options,
    )

    glrt_score_profile = (
        glrt_numerator_part / profile_h1_jammer_fit["glrt_denominator_surface"]
    )

    return {
        "label": label,
        "full_received_signal": full_received_signal,
        "decision": decision,
        "best_glrt_score": float(best_glrt_score),
        "glrt_numerator_part": float(glrt_numerator_part),
        "glrt_denominator_part": float(glrt_denominator_part),
        "h0_jammer_fit": h0_jammer_fit,
        "h1_jammer_fit": h1_jammer_fit,
        "selected_jammer_amplitude": float(
            selected_jammer_fit["estimated_jammer_amplitude"]
        ),
        "selected_jammer_frequency": float(
            selected_jammer_fit["estimated_jammer_frequency"]
        ),
        "selected_noise_variance": float(
            selected_jammer_fit["noise_variance_from_signal_fit"]
        ),
        "plot_jammer_frequency_options": plot_jammer_frequency_options,
        "plot_glrt_score_profile": glrt_score_profile,
    }


def style_2d_axes(axis):
    axis.tick_params(axis="both", labelsize=9)
    axis.grid(True, alpha=0.25, linewidth=0.8)


def plot_problem5b_profile(result):
    label = result["label"]
    jammer_frequency_options = result["plot_jammer_frequency_options"]
    glrt_score_profile = result["plot_glrt_score_profile"]

    estimated_jammer_frequency = result["h1_jammer_fit"]["estimated_jammer_frequency"]
    best_glrt_score = result["best_glrt_score"]

    figure = plt.figure(figsize=(11.5, 5.3))
    axis = figure.add_subplot(111)

    axis.plot(
        jammer_frequency_options,
        glrt_score_profile,
        linewidth=1.7,
        label=r"$T(\bar{r};\nu)=J_0/J_1(\nu)$",
    )

    axis.axhline(
        1.0,
        linestyle="--",
        linewidth=1.0,
        label=r"reference $T=1$",
    )

    axis.axvline(
        estimated_jammer_frequency,
        linestyle=":",
        linewidth=2.0,
        label=rf"$\hat{{\nu}}_{{H1}}={estimated_jammer_frequency:.3f}$ Hz",
    )

    axis.scatter(
        [estimated_jammer_frequency],
        [best_glrt_score],
        linewidth=0.9,
        s=75,
        zorder=5,
    )

    axis.set_title(f"{nice_label(label)}: GLRT test statistic")
    axis.set_xlabel(r"jammer frequency $\nu$ (Hz)")
    axis.set_ylabel(r"GLRT ratio $T(\bar{r};\nu)$")

    axis.text(
        0.02,
        0.95,
        rf"$T_{{max}}={best_glrt_score:.6f}$",
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=10,
    )

    style_2d_axes(axis)
    axis.legend(loc="best", frameon=True)

    figure.subplots_adjust(**GLRT_PROFILE_LAYOUT)

    output_path = OUTPUT_DIR / f"{LAST_NAME_PREFIX}_{label}_problem5b_glrt_profile.png"
    figure.savefig(output_path, **GLRT_PROFILE_EXPORT_OPTIONS)
    plt.close(figure)

    result["profile_path"] = str(output_path)


# ============================================================
# Problem 5(c): jammer suppression and Problem 4(b)-style MLE
# ============================================================

def compute_known_variance_mle_map(full_received_signal, fm_rate_beta_grid, known_noise_variance):
    full_received_signal = np.asarray(full_received_signal, dtype=float).reshape(-1)
    fm_rate_beta_grid = np.asarray(fm_rate_beta_grid, dtype=float).reshape(-1)

    total_sample_count = full_received_signal.size

    delay_sample_options = np.arange(
        0,
        total_sample_count - given_signal_template_sample_count + 1,
        dtype=int,
    )

    known_variance_score_surface = np.zeros(
        (len(delay_sample_options), len(fm_rate_beta_grid)),
        dtype=float,
    )
    glrt_denominator_surface = np.zeros_like(known_variance_score_surface)
    scalar_b1_values = np.zeros_like(known_variance_score_surface)
    scalar_b2_values = np.zeros_like(known_variance_score_surface)

    glrt_numerator_part = float(full_received_signal.T @ full_received_signal)

    sample_index = np.arange(given_signal_template_sample_count, dtype=float)
    hfm_time = np.log((TR + sample_index * TS) / TR)

    constant_term = -0.5 * total_sample_count * math.log(
        2.0 * math.pi * known_noise_variance
    )

    for fm_rate_option_index, fm_rate in enumerate(fm_rate_beta_grid):
        phase = 2.0 * math.pi * fm_rate * hfm_time

        b1 = np.cos(phase)
        b2 = np.sin(phase)

        H = np.column_stack((b1, b2))
        HtH_inverse = np.linalg.pinv(H.T @ H)

        corr1 = np.correlate(full_received_signal, b1, mode="valid")
        corr2 = np.correlate(full_received_signal, b2, mode="valid")

        estimated_cosine_weight = (
            HtH_inverse[0, 0] * corr1 + HtH_inverse[0, 1] * corr2
        )
        estimated_sine_weight = (
            HtH_inverse[1, 0] * corr1 + HtH_inverse[1, 1] * corr2
        )

        fitted_projection_energy = (
            estimated_cosine_weight * corr1
            + estimated_sine_weight * corr2
        )

        glrt_denominator_part = glrt_numerator_part - fitted_projection_energy
        glrt_denominator_part = np.maximum(glrt_denominator_part, 1e-14)

        scalar_b1_values[:, fm_rate_option_index] = estimated_cosine_weight
        scalar_b2_values[:, fm_rate_option_index] = estimated_sine_weight
        glrt_denominator_surface[:, fm_rate_option_index] = glrt_denominator_part
        known_variance_score_surface[:, fm_rate_option_index] = (
            constant_term - 0.5 * glrt_denominator_part / known_noise_variance
        )

    best_position = int(np.argmax(known_variance_score_surface))
    delay_option_index, fm_rate_option_index = np.unravel_index(
        best_position,
        known_variance_score_surface.shape,
    )

    estimated_delay_sample = int(delay_sample_options[delay_option_index])
    estimated_fm_rate = float(fm_rate_beta_grid[fm_rate_option_index])
    estimated_cosine_weight = float(
        scalar_b1_values[delay_option_index, fm_rate_option_index]
    )
    estimated_sine_weight = float(
        scalar_b2_values[delay_option_index, fm_rate_option_index]
    )

    estimated_amplitude = math.sqrt(
        estimated_cosine_weight**2 + estimated_sine_weight**2
    )
    estimated_phase = wrap_phase(
        math.atan2(-estimated_sine_weight, estimated_cosine_weight)
    )
    estimated_delay_seconds = estimated_delay_sample * TS
    glrt_denominator_part = float(
        glrt_denominator_surface[delay_option_index, fm_rate_option_index]
    )

    return {
        "sample_count": total_sample_count,
        "given_signal_template_sample_count": given_signal_template_sample_count,
        "delay_sample_options": delay_sample_options,
        "fm_rate_beta_grid": fm_rate_beta_grid,
        "known_variance_score_surface": known_variance_score_surface,
        "best_known_variance_score": float(
            known_variance_score_surface[delay_option_index, fm_rate_option_index]
        ),
        "glrt_numerator_part": glrt_numerator_part,
        "glrt_denominator_part": glrt_denominator_part,
        "estimated_delay_sample": estimated_delay_sample,
        "estimated_delay_seconds": estimated_delay_seconds,
        "estimated_fm_rate": estimated_fm_rate,
        "estimated_cosine_weight": estimated_cosine_weight,
        "estimated_sine_weight": estimated_sine_weight,
        "estimated_amplitude": float(estimated_amplitude),
        "estimated_phase": float(estimated_phase),
        "known_noise_variance": float(known_noise_variance),
        "best_surface_location": (delay_option_index, fm_rate_option_index),
    }


def run_problem5c_known_variance_mle(full_received_signal, known_noise_variance):
    minimum_beta = BETA_MIN
    maximum_beta = BETA_MAX

    coarse_fm_rate_beta_grid = np.arange(
        minimum_beta,
        maximum_beta + BETA_STEP_INCREMENT / 2.0,
        BETA_STEP_INCREMENT,
    )

    coarse_result = compute_known_variance_mle_map(
        full_received_signal,
        coarse_fm_rate_beta_grid,
        known_noise_variance,
    )

    center_beta_estimate = coarse_result["estimated_fm_rate"]

    fine_beta_low = max(
        minimum_beta,
        center_beta_estimate - BETA_ZOOM_HALF_WIDTH,
    )

    fine_beta_high = min(
        maximum_beta,
        center_beta_estimate + BETA_ZOOM_HALF_WIDTH,
    )

    fine_fm_rate_beta_grid = np.arange(
        fine_beta_low,
        fine_beta_high + BETA_STEP_FINE / 2.0,
        BETA_STEP_FINE,
    )

    final_result = compute_known_variance_mle_map(
        full_received_signal,
        fine_fm_rate_beta_grid,
        known_noise_variance,
    )

    final_result["full_received_signal"] = full_received_signal
    final_result["plot_glrt_score_surface"] = coarse_result[
        "known_variance_score_surface"
    ]
    final_result["plot_fm_rate_beta_grid"] = coarse_result["fm_rate_beta_grid"]
    final_result["plot_delay_sample_options"] = coarse_result[
        "delay_sample_options"
    ]

    return final_result


def selected_indices(length, maximum, forced=None):
    if length <= maximum:
        indices = np.arange(length, dtype=int)
    else:
        indices = np.linspace(0, length - 1, maximum, dtype=int)

    if forced is not None and 0 <= forced < length:
        indices = np.r_[indices, forced]

    return np.unique(indices).astype(int)


def format_glrt_surface_axes(glrt_axis):
    glrt_axis.tick_params(axis="x", labelsize=9, pad=2)
    glrt_axis.tick_params(axis="y", labelsize=9, pad=2)
    glrt_axis.tick_params(axis="z", labelsize=9, pad=2)


def plot_problem5c_hfm_surface(result):
    full_map = result["plot_glrt_score_surface"]
    fm_rate_beta_values = result["plot_fm_rate_beta_grid"]
    delay_sample_values = result["plot_delay_sample_options"]

    peak_delay = int(
        np.argmin(np.abs(delay_sample_values - result["estimated_delay_sample"]))
    )
    peak_beta = int(
        np.argmin(np.abs(fm_rate_beta_values - result["estimated_fm_rate"]))
    )

    delay_choice = selected_indices(len(delay_sample_values), 260, peak_delay)
    beta_choice = selected_indices(len(fm_rate_beta_values), 210, peak_beta)

    delay_plot = delay_sample_values[delay_choice]
    beta_plot = fm_rate_beta_values[beta_choice]
    surface_values = full_map[np.ix_(delay_choice, beta_choice)]

    delay_mesh, beta_mesh = np.meshgrid(delay_plot, beta_plot)
    surface_values = surface_values.T

    surface_canvas = plt.figure(figsize=(12.0, 7.4))
    axis = surface_canvas.add_subplot(111, projection="3d")

    axis.plot_surface(
        delay_mesh,
        beta_mesh,
        surface_values,
        edgecolor="none",
        linewidth=0,
        antialiased=True,
        alpha=0.97,
    )

    axis.scatter(
        [delay_sample_values[peak_delay]],
        [fm_rate_beta_values[peak_beta]],
        [float(full_map[peak_delay, peak_beta])],
        s=75,
        marker="o",
        color="black",
        edgecolors="black",
        linewidths=2.0,
        depthshade=True,
    )

    axis.set_title(
        "Problem 5(c): GLRT test statistic after jammer suppression",
        pad=18,
        fontweight="bold",
    )

    axis.set_xlabel(r"time delay $n_0$ (samples)", labelpad=18)
    axis.set_ylabel(r"FM rate $\beta$ (unitless)", labelpad=18)
    axis.set_zlabel("")

    surface_canvas.text(
        0.795,
        0.53,
        r"known-variance score $T(\hat r;\beta,n_0)$",
        rotation=90,
        ha="center",
        va="center",
        fontsize=11,
    )

    axis.view_init(elev=28, azim=-45)
    axis.set_box_aspect((1.25, 1.0, 0.75))
    format_glrt_surface_axes(axis)
    surface_canvas.subplots_adjust(**GLRT_SURFACE_LAYOUT)

    output_path = OUTPUT_DIR / f"{LAST_NAME_PREFIX}_problem5c_hfm_surface.png"
    surface_canvas.savefig(output_path, **GLRT_SURFACE_EXPORT_OPTIONS)
    plt.close(surface_canvas)

    result["surface_path"] = str(output_path)


def plot_problem5c_suppressed_signal(full_received_signal, estimated_jammer_signal, interference_suppressed_signal):
    time_values = np.arange(given_received_sample_count, dtype=float) * TS

    figure, axis = plt.subplots(figsize=(11.0, 5.8))
    axis.plot(
        time_values,
        full_received_signal,
        linewidth=1.0,
        label=r"$\bar r[n]$ before suppression",
    )
    axis.plot(
        time_values,
        estimated_jammer_signal,
        linewidth=1.0,
        label=r"$\hat d[n]$",
    )
    axis.plot(
        time_values,
        interference_suppressed_signal,
        linewidth=1.0,
        label=r"$\hat r[n]$ after suppression",
    )

    axis.set_title("Signal-plus-noise data before and after jammer suppression")
    axis.set_xlabel("time (s)")
    axis.set_ylabel("amplitude")
    axis.grid(True, alpha=0.30)
    axis.legend(loc="upper right")

    output_path = OUTPUT_DIR / f"{LAST_NAME_PREFIX}_problem5c_suppressed_observation.png"
    figure.tight_layout()
    figure.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(figure)

    return str(output_path)


def make_problem5b_summary_table(results):
    glrt_decision_records = []

    for result in results:
        glrt_decision_records.append(
            {
                "Data_set": nice_label(result["label"]),
                "Decision": result["decision"],
                "Tmax_GLRT_ratio": result["best_glrt_score"],
                "GLRT_numerator_part": result["glrt_numerator_part"],
                "GLRT_denominator_part_at_MLE": result["glrt_denominator_part"],
                "sigma2_hat_H0": result["h0_jammer_fit"][
                    "noise_variance_from_signal_fit"
                ],
                "sigma2_hat_H1": result["h1_jammer_fit"][
                    "noise_variance_from_signal_fit"
                ],
                "D_hat_H0": result["h0_jammer_fit"][
                    "estimated_jammer_amplitude"
                ],
                "nu_hat_H0_Hz": result["h0_jammer_fit"][
                    "estimated_jammer_frequency"
                ],
                "D_hat_H1": result["h1_jammer_fit"][
                    "estimated_jammer_amplitude"
                ],
                "nu_hat_H1_Hz": result["h1_jammer_fit"][
                    "estimated_jammer_frequency"
                ],
                "Selected_D_hat": result["selected_jammer_amplitude"],
                "Selected_nu_hat_Hz": result["selected_jammer_frequency"],
                "Selected_sigma2_hat": result["selected_noise_variance"],
            }
        )

    return pd.DataFrame(glrt_decision_records)


def print_problem5b_tables(summary_table):
    print("\nProblem 5(b): GLRT result summary")
    print(
        summary_table[
            [
                "Data_set",
                "Decision",
                "Tmax_GLRT_ratio",
                "GLRT_numerator_part",
                "GLRT_denominator_part_at_MLE",
                "sigma2_hat_H0",
                "sigma2_hat_H1",
                "Selected_D_hat",
                "Selected_nu_hat_Hz",
                "Selected_sigma2_hat",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )

    print("\nProblem 5(b): H0 and H1 jammer MLE details")
    print(
        summary_table[
            [
                "Data_set",
                "D_hat_H0",
                "nu_hat_H0_Hz",
                "D_hat_H1",
                "nu_hat_H1_Hz",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )

    print("\nProblem 5(b): GLRT decisions")
    print(summary_table[["Data_set", "Decision"]].to_string(index=False))


def make_problem5c_comparison_table(hfm_result, known_noise_variance):
    return pd.DataFrame(
        [
            {
                "Parameter": "Amplitude A",
                "True_value": A_TRUE,
                "MLE_after_suppression": hfm_result["estimated_amplitude"],
                "Absolute_error": abs(hfm_result["estimated_amplitude"] - A_TRUE),
            },
            {
                "Parameter": "Phase phi, rad",
                "True_value": PHI_TRUE,
                "MLE_after_suppression": hfm_result["estimated_phase"],
                "Absolute_error": abs(
                    wrap_phase(hfm_result["estimated_phase"] - PHI_TRUE)
                ),
            },
            {
                "Parameter": "FM rate beta",
                "True_value": BETA_TRUE,
                "MLE_after_suppression": hfm_result["estimated_fm_rate"],
                "Absolute_error": abs(hfm_result["estimated_fm_rate"] - BETA_TRUE),
            },
            {
                "Parameter": "Time delay n0, samples",
                "True_value": N0_TRUE,
                "MLE_after_suppression": hfm_result["estimated_delay_sample"],
                "Absolute_error": abs(hfm_result["estimated_delay_sample"] - N0_TRUE),
            },
            {
                "Parameter": "Time delay t0, s",
                "True_value": T0_TRUE,
                "MLE_after_suppression": hfm_result["estimated_delay_seconds"],
                "Absolute_error": abs(hfm_result["estimated_delay_seconds"] - T0_TRUE),
            },
            {
                "Parameter": "Known variance used",
                "True_value": np.nan,
                "MLE_after_suppression": known_noise_variance,
                "Absolute_error": np.nan,
            },
        ]
    )


def main():
    set3_matrix = load_dataset(SET3_FILE, "Set3")
    set4_matrix = load_dataset(SET4_FILE, "Set4")

    print("\nData dimensions")
    print(f"Set 3: {set3_matrix.shape}")
    print(f"Set 4: {set4_matrix.shape}")

    known_hfm_signal = make_known_hfm_signal(given_received_sample_count)

    set3_result = run_problem5b_glrt(set3_matrix, "Set3", known_hfm_signal)
    set4_result = run_problem5b_glrt(set4_matrix, "Set4", known_hfm_signal)

    results = [set3_result, set4_result]

    plot_problem5b_profile(set3_result)
    plot_problem5b_profile(set4_result)

    problem5b_summary_table = make_problem5b_summary_table(results)
    problem5b_summary_csv = OUTPUT_DIR / f"{LAST_NAME_PREFIX}_problem5b_summary.csv"
    problem5b_summary_table.to_csv(problem5b_summary_csv, index=False)

    pd.options.display.float_format = "{:.6f}".format

    print_problem5b_tables(problem5b_summary_table)

    signal_result = max(results, key=lambda item: item["best_glrt_score"])
    noise_result = min(results, key=lambda item: item["best_glrt_score"])

    print("\nDecision summary")
    print(f"Mine-present data set: {nice_label(signal_result['label'])}")
    print(f"Mine-absent data set: {nice_label(noise_result['label'])}")

    estimated_jammer_amplitude = signal_result["h1_jammer_fit"][
        "estimated_jammer_amplitude"
    ]
    estimated_jammer_frequency = signal_result["h1_jammer_fit"][
        "estimated_jammer_frequency"
    ]
    known_noise_variance = signal_result["h1_jammer_fit"][
        "noise_variance_from_signal_fit"
    ]

    sample_index = np.arange(given_received_sample_count, dtype=float)
    estimated_jammer_signal = estimated_jammer_amplitude * np.cos(
        2.0 * math.pi * estimated_jammer_frequency * sample_index * TS
    )

    interference_suppressed_signal = (
        signal_result["full_received_signal"] - estimated_jammer_signal
    )

    problem5c_result = run_problem5c_known_variance_mle(
        interference_suppressed_signal,
        known_noise_variance,
    )

    problem5c_result["label"] = signal_result["label"]

    plot_problem5c_hfm_surface(problem5c_result)
    suppressed_signal_path = plot_problem5c_suppressed_signal(
        signal_result["full_received_signal"],
        estimated_jammer_signal,
        interference_suppressed_signal,
    )

    problem5c_comparison_table = make_problem5c_comparison_table(
        problem5c_result,
        known_noise_variance,
    )
    problem5c_comparison_csv = OUTPUT_DIR / f"{LAST_NAME_PREFIX}_problem5c_comparison.csv"
    problem5c_comparison_table.to_csv(problem5c_comparison_csv, index=False)

    print("\nProblem 5(c): interference suppression")
    print(f"Using signal-plus-noise data set: {nice_label(signal_result['label'])}")
    print(f"Using D_hat = {estimated_jammer_amplitude:.6f}")
    print(f"Using nu_hat = {estimated_jammer_frequency:.6f} Hz")
    print(f"Using sigma^2 = {known_noise_variance:.6f}")

    print("\nProblem 5(c): HFM MLE comparison")
    print(
        problem5c_comparison_table.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\nSaved files")
    print(set3_result["profile_path"])
    print(set4_result["profile_path"])
    print(problem5c_result["surface_path"])
    print(suppressed_signal_path)
    print(problem5b_summary_csv)
    print(problem5c_comparison_csv)


if __name__ == "__main__":
    main()

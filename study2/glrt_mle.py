
# Problem 2 code: GLRT implementation and MLE table

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
OUTPUT_DIR = BASE_DIR / "problem2_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SET1_FILE = DATA_DIR / "Threat_Detection_Set1.csv"
SET2_FILE = DATA_DIR / "Threat_Detection_Set2.csv"
SET3_FILE = DATA_DIR / "Threat_Detection_Set3.csv"
SET4_FILE = DATA_DIR / "Threat_Detection_Set4.csv"

FS = 1000.0
TS = 1.0 / FS
TTX = 1.2
TRX = 2.1
TR = 0.1
ALPHA = 32.2

given_hydrophone_count = 40
given_received_sample_count = int(round(TRX / TS)) + 1
given_signal_template_sample_count = int(round(TTX / TS)) + 1

BETA_STEP_INCREMENT = 0.012102022
BETA_ZOOM_HALF_WIDTH = 0.12102022
BETA_STEP_FINE = 0.0012102022

GLRT_SURFACE_LAYOUT = {
    "left": 0.02,
    "right": 0.835,
    "bottom": 0.06,
    "top": 0.91,
}

GLRT_SURFACE_EXPORT_OPTIONS = {
    "dpi": 220,
    "facecolor": "white",
    "bbox_inches": "tight",
    "pad_inches": 0.20,
}


def read_csv_numeric(csv_path):
    csv_path = Path(csv_path)
    return pd.read_csv(csv_path, header=None).to_numpy(dtype=float)


def load_dataset(csv_path, label, given_sample_count=None):
    matrix = read_csv_numeric(csv_path)

    if matrix.shape[1] != given_hydrophone_count:
        raise ValueError(
            f"{label} has {matrix.shape[1]} columns, given {given_hydrophone_count}"
        )

    if given_sample_count is not None and matrix.shape[0] != given_sample_count:
        raise ValueError(
            f"{label} has {matrix.shape[0]} samples, given {given_sample_count}"
        )

    print(f"[{label}] loaded {Path(csv_path).name} with shape {matrix.shape}")
    return matrix


def wrap_phase(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def compute_glrt_map(full_received_signal, fm_rate_options):
    full_received_signal = np.asarray(
        full_received_signal,
        dtype=float,
    ).reshape(-1)
    fm_rate_options = np.asarray(fm_rate_options, dtype=float).reshape(-1)

    total_sample_count = full_received_signal.size

    if total_sample_count != given_received_sample_count:
        raise ValueError(
            f"Given {given_received_sample_count} samples, received {total_sample_count}"
        )

    delay_sample_options = np.arange(
        0,
        total_sample_count - given_signal_template_sample_count + 1,
        dtype=int,
    )

    glrt_score_surface = np.zeros(
        (len(delay_sample_options), len(fm_rate_options)),
        dtype=float,
    )
    glrt_denominator_surface = np.zeros_like(glrt_score_surface)
    scalar_b1_values = np.zeros_like(glrt_score_surface)
    scalar_b2_values = np.zeros_like(glrt_score_surface)

    glrt_numerator_part = float(full_received_signal.T @ full_received_signal)

    sample_index = np.arange(given_signal_template_sample_count, dtype=float)
    hfm_time = np.log((TR + sample_index * TS) / TR)

    regionsplit = np.concatenate(
        ([0.0], np.cumsum(full_received_signal * full_received_signal))
    )

    for fm_rate_option_index, fm_rate in enumerate(fm_rate_options):
        phase = 2.0 * np.pi * fm_rate * hfm_time

        b1 = np.cos(phase)
        b2 = np.sin(phase)

        H = np.column_stack((b1, b2))
        HtH_inverse = np.linalg.pinv(H.T @ H)

        for delay_option_index, start_sample in enumerate(delay_sample_options):
            active_signal_window = full_received_signal[
                start_sample:start_sample + given_signal_template_sample_count
            ]
            fitted_coefficients = HtH_inverse @ (H.T @ active_signal_window)

            estimated_cosine_weight = float(fitted_coefficients[0])
            estimated_sine_weight = float(fitted_coefficients[1])

            scalar_b1_values[delay_option_index, fm_rate_option_index] = (
                estimated_cosine_weight
            )
            scalar_b2_values[delay_option_index, fm_rate_option_index] = (
                estimated_sine_weight
            )

            signal_fit = H @ fitted_coefficients

            region1_energy = float(regionsplit[start_sample])
            region2_energy = float(
                (active_signal_window - signal_fit).T
                @ (active_signal_window - signal_fit)
            )
            region3_energy = float(
                regionsplit[total_sample_count]
                - regionsplit[start_sample + given_signal_template_sample_count]
            )

            glrt_denominator_part = region1_energy + region2_energy + region3_energy
            glrt_denominator_part = max(glrt_denominator_part, 1e-14)

            glrt_denominator_surface[delay_option_index, fm_rate_option_index] = (
                glrt_denominator_part
            )
            glrt_score_surface[delay_option_index, fm_rate_option_index] = (
                glrt_numerator_part / glrt_denominator_part
            )

    best_position = int(np.argmax(glrt_score_surface))
    delay_option_index, fm_rate_option_index = np.unravel_index(
        best_position,
        glrt_score_surface.shape,
    )

    estimated_delay_sample = int(delay_sample_options[delay_option_index])
    estimated_fm_rate = float(fm_rate_options[fm_rate_option_index])
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
    estimated_fm_rate_shift = estimated_fm_rate - ALPHA
    estimated_delay_seconds = estimated_delay_sample * TS
    glrt_denominator_part = float(
        glrt_denominator_surface[delay_option_index, fm_rate_option_index]
    )

    return {
        "sample_count": total_sample_count,
        "given_signal_template_sample_count": given_signal_template_sample_count,
        "delay_sample_options": delay_sample_options,
        "fm_rate_options": fm_rate_options,
        "glrt_score_surface": glrt_score_surface,
        "best_glrt_score": float(
            glrt_score_surface[delay_option_index, fm_rate_option_index]
        ),
        "glrt_numerator_part": glrt_numerator_part,
        "glrt_denominator_part": glrt_denominator_part,
        "estimated_delay_sample": estimated_delay_sample,
        "estimated_delay_seconds": estimated_delay_seconds,
        "estimated_fm_rate": estimated_fm_rate,
        "estimated_fm_rate_shift": estimated_fm_rate_shift,
        "estimated_cosine_weight": estimated_cosine_weight,
        "estimated_sine_weight": estimated_sine_weight,
        "estimated_amplitude": float(estimated_amplitude),
        "estimated_phase": float(estimated_phase),
        "noise_variance_from_noise_only_fit": float(
            glrt_numerator_part / total_sample_count
        ),
        "noise_variance_from_signal_fit": float(
            glrt_denominator_part / total_sample_count
        ),
        "best_surface_location": (delay_option_index, fm_rate_option_index),
    }


def run_glrt(matrix, label):
    if matrix.shape != (given_received_sample_count, given_hydrophone_count):
        raise ValueError(
            f"{label} has shape {matrix.shape}, given {(given_received_sample_count, given_hydrophone_count)}"
        )

    full_received_signal = np.mean(matrix, axis=1)

    minimum_fm_rate = ALPHA
    maximum_fm_rate = 0.5 * FS * TR

    coarse_fm_rate_options = np.arange(
        minimum_fm_rate,
        maximum_fm_rate + BETA_STEP_INCREMENT / 2.0,
        BETA_STEP_INCREMENT,
    )

    coarse_result = compute_glrt_map(full_received_signal, coarse_fm_rate_options)

    center_fm_rate_estimate = coarse_result["estimated_fm_rate"]

    fine_fm_rate_low = max(
        minimum_fm_rate,
        center_fm_rate_estimate - BETA_ZOOM_HALF_WIDTH,
    )

    fine_fm_rate_high = min(
        maximum_fm_rate,
        center_fm_rate_estimate + BETA_ZOOM_HALF_WIDTH,
    )

    fine_fm_rate_options = np.arange(
        fine_fm_rate_low,
        fine_fm_rate_high + BETA_STEP_FINE / 2.0,
        BETA_STEP_FINE,
    )

    final_result = compute_glrt_map(full_received_signal, fine_fm_rate_options)

    final_result["label"] = label
    final_result["full_received_signal"] = full_received_signal
    final_result["plot_glrt_score_surface"] = coarse_result["glrt_score_surface"]
    final_result["plot_fm_rate_options"] = coarse_result["fm_rate_options"]
    final_result["plot_delay_sample_options"] = coarse_result["delay_sample_options"]

    return final_result


def nice_label(label):
    if label.startswith("Set") and label[3:].isdigit():
        return f"Set {label[3:]}"
    return label


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


def build_glrt_surface_path(dataset_label):
    surface_filename = f"{LAST_NAME_PREFIX}_{dataset_label}_GLRT_surface.png"
    return OUTPUT_DIR / surface_filename


def plot_glrt_surface(result, mark_peak):
    label = result["label"]

    full_map = result["plot_glrt_score_surface"]
    fm_rate_values = result["plot_fm_rate_options"]
    delay_sample_values = result["plot_delay_sample_options"]

    peak_delay = int(
        np.argmin(np.abs(delay_sample_values - result["estimated_delay_sample"]))
    )
    peak_beta = int(np.argmin(np.abs(fm_rate_values - result["estimated_fm_rate"])))

    delay_choice = selected_indices(
        len(delay_sample_values),
        260,
        peak_delay if mark_peak else None,
    )

    beta_choice = selected_indices(
        len(fm_rate_values),
        210,
        peak_beta if mark_peak else None,
    )

    delay_plot = delay_sample_values[delay_choice]
    beta_plot = fm_rate_values[beta_choice]
    surface_values = full_map[np.ix_(delay_choice, beta_choice)]

    beta_mesh, delay_mesh = np.meshgrid(beta_plot, delay_plot)

    surface_canvas = plt.figure(figsize=(12.0, 7.4))
    axis = surface_canvas.add_subplot(111, projection="3d")

    axis.plot_surface(
        beta_mesh,
        delay_mesh,
        surface_values,
        edgecolor="none",
        linewidth=0,
        antialiased=True,
        alpha=0.97,
    )

    if mark_peak:
        axis.scatter(
            [fm_rate_values[peak_beta]],
            [delay_sample_values[peak_delay]],
            [float(full_map[peak_delay, peak_beta])],
            s=75,
            marker="o",
            linewidths=2.0,
            depthshade=True,
        )

    axis.set_title(
        f"{nice_label(label)}: GLRT test statistic",
        pad=18,
    )

    axis.set_xlabel(r"FM rate $\beta$", labelpad=13)
    axis.set_ylabel(r"time delay $n_0$ (samples)", labelpad=13)
    axis.set_zlabel("")

    surface_canvas.text(
        0.795,
        0.53,
        r"GLRT ratio $T(\bar{r};\beta,n_0)$",
        rotation=90,
        ha="center",
        va="center",
        fontsize=11,
    )

    axis.view_init(elev=26, azim=-55)

    axis.set_box_aspect((1.25, 1.0, 0.75))

    format_glrt_surface_axes(axis)

    surface_canvas.subplots_adjust(**GLRT_SURFACE_LAYOUT)

    savePlotfile_ouputPath = build_glrt_surface_path(label)

    surface_canvas.savefig(
        savePlotfile_ouputPath,
        **GLRT_SURFACE_EXPORT_OPTIONS,
    )

    plt.close(surface_canvas)

    result["surface_path"] = str(savePlotfile_ouputPath)


def choose_signal_plus_noise_dataset(set1_glrt_result, set2_glrt_result):
    set1_peak_glrt = set1_glrt_result["best_glrt_score"]
    set2_peak_glrt = set2_glrt_result["best_glrt_score"]

    if set2_peak_glrt > set1_peak_glrt:
        return "Set 2", set2_glrt_result, set1_glrt_result

    return "Set 1", set1_glrt_result, set2_glrt_result


def make_glrt_decision_summary_table(set1_result, set2_result, signal_set):
    glrt_decision_records = []

    for dataset_display_label, glrt_estimates in (
        ("Set 1", set1_result),
        ("Set 2", set2_result),
    ):
        mine_presence_call = (
            "Mine present" if dataset_display_label == signal_set else "Mine absent"
        )

        glrt_decision_records.append(
            {
                "Threat_detection_data_set": dataset_display_label,
                "GLRT_mine_presence_call": mine_presence_call,
                "Peak_GLRT_statistic_T_r": glrt_estimates["best_glrt_score"],
                "MLE_time_delay_n0_samples": glrt_estimates[
                    "estimated_delay_sample"
                ],
                "MLE_time_delay_t0_seconds": glrt_estimates[
                    "estimated_delay_seconds"
                ],
                "MLE_beta_FM_rate": glrt_estimates["estimated_fm_rate"],
                "MLE_alpha0_FM_rate_shift": glrt_estimates[
                    "estimated_fm_rate_shift"
                ],
                "MLE_amplitude_A": glrt_estimates["estimated_amplitude"],
                "MLE_phase_phi_radians": glrt_estimates["estimated_phase"],
                "MLE_noise_variance_under_H1": glrt_estimates[
                    "noise_variance_from_signal_fit"
                ],
                "Noise_variance_estimate_under_H0": glrt_estimates[
                    "noise_variance_from_noise_only_fit"
                ],
                "GLRT_numerator_part": glrt_estimates["glrt_numerator_part"],
                "GLRT_denominator_part_at_MLE": glrt_estimates[
                    "glrt_denominator_part"
                ],
            }
        )

    return pd.DataFrame(glrt_decision_records)


def make_mle_table(result):
    return pd.DataFrame(
        {
            "Parameter": ["MLE estimate"],
            "Time-delay n0, samples": [result["estimated_delay_sample"]],
            "Time-delay t0, s": [result["estimated_delay_seconds"]],
            "FM rate change alpha0": [result["estimated_fm_rate_shift"]],
            "Amplitude A": [result["estimated_amplitude"]],
            "Phase phi": [result["estimated_phase"]],
            "Variance sigma^2": [result["noise_variance_from_signal_fit"]],
        }
    )


def save_outputs(set1_result, set2_result, signal_set, signal_result):
    summary_table = make_glrt_decision_summary_table(
        set1_result,
        set2_result,
        signal_set,
    )

    mle_table = make_mle_table(signal_result)

    pd.options.display.float_format = "{:.6f}".format

    print("\nSummary")
    print(summary_table.to_string(index=False))

    print("\nMine detection")
    print(f"{set1_result['best_glrt_score']:.6f} is Set 1's GLRT")
    print(f"{set2_result['best_glrt_score']:.6f} is Set 2's GLRT")
    print(f"Mine-present in {signal_set} dataset")

    print("\nMLE parameter table")
    print(mle_table.to_string(index=False))

    print("\nSaved files")
    print(set1_result["surface_path"])
    print(set2_result["surface_path"])



# Problem 4 code: known-noise-variance MLE implementation

def compute_known_variance_mle_map(
    full_received_signal,
    fm_rate_options,
    known_noise_variance,
):
    full_received_signal = np.asarray(
        full_received_signal,
        dtype=float,
    ).reshape(-1)
    fm_rate_options = np.asarray(fm_rate_options, dtype=float).reshape(-1)

    total_sample_count = full_received_signal.size

    if total_sample_count != given_received_sample_count:
        raise ValueError(
            f"Given {given_received_sample_count} samples, received {total_sample_count}"
        )

    if known_noise_variance <= 0:
        raise ValueError("Known noise variance must be positive.")

    delay_sample_options = np.arange(
        0,
        total_sample_count - given_signal_template_sample_count + 1,
        dtype=int,
    )

    known_variance_score_surface = np.zeros(
        (len(delay_sample_options), len(fm_rate_options)),
        dtype=float,
    )
    glrt_denominator_surface = np.zeros_like(known_variance_score_surface)
    scalar_b1_values = np.zeros_like(known_variance_score_surface)
    scalar_b2_values = np.zeros_like(known_variance_score_surface)

    sample_index = np.arange(given_signal_template_sample_count, dtype=float)
    hfm_time = np.log((TR + sample_index * TS) / TR)

    regionsplit = np.concatenate(
        ([0.0], np.cumsum(full_received_signal * full_received_signal))
    )

    constant_term = -0.5 * total_sample_count * np.log(
        2.0 * np.pi * known_noise_variance
    )

    for fm_rate_option_index, fm_rate in enumerate(fm_rate_options):
        phase = 2.0 * np.pi * fm_rate * hfm_time

        b1 = np.cos(phase)
        b2 = np.sin(phase)

        H = np.column_stack((b1, b2))
        HtH_inverse = np.linalg.pinv(H.T @ H)

        for delay_option_index, start_sample in enumerate(delay_sample_options):
            active_signal_window = full_received_signal[
                start_sample:start_sample + given_signal_template_sample_count
            ]
            fitted_coefficients = HtH_inverse @ (H.T @ active_signal_window)

            estimated_cosine_weight = float(fitted_coefficients[0])
            estimated_sine_weight = float(fitted_coefficients[1])

            scalar_b1_values[delay_option_index, fm_rate_option_index] = (
                estimated_cosine_weight
            )
            scalar_b2_values[delay_option_index, fm_rate_option_index] = (
                estimated_sine_weight
            )

            signal_fit = H @ fitted_coefficients

            region1_energy = float(regionsplit[start_sample])
            region2_energy = float(
                (active_signal_window - signal_fit).T
                @ (active_signal_window - signal_fit)
            )
            region3_energy = float(
                regionsplit[total_sample_count]
                - regionsplit[start_sample + given_signal_template_sample_count]
            )

            glrt_denominator_part = region1_energy + region2_energy + region3_energy
            glrt_denominator_part = max(glrt_denominator_part, 1e-14)

            glrt_denominator_surface[delay_option_index, fm_rate_option_index] = (
                glrt_denominator_part
            )
            known_variance_score_surface[delay_option_index, fm_rate_option_index] = (
                constant_term - 0.5 * glrt_denominator_part / known_noise_variance
            )

    best_position = int(np.argmax(known_variance_score_surface))
    delay_option_index, fm_rate_option_index = np.unravel_index(
        best_position,
        known_variance_score_surface.shape,
    )

    estimated_delay_sample = int(delay_sample_options[delay_option_index])
    estimated_fm_rate = float(fm_rate_options[fm_rate_option_index])
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
    estimated_fm_rate_shift = estimated_fm_rate - ALPHA
    estimated_delay_seconds = estimated_delay_sample * TS
    glrt_denominator_part = float(
        glrt_denominator_surface[delay_option_index, fm_rate_option_index]
    )

    return {
        "sample_count": total_sample_count,
        "given_signal_template_sample_count": given_signal_template_sample_count,
        "delay_sample_options": delay_sample_options,
        "fm_rate_options": fm_rate_options,
        "known_variance_score_surface": known_variance_score_surface,
        "best_known_variance_score": float(
            known_variance_score_surface[delay_option_index, fm_rate_option_index]
        ),
        "glrt_denominator_part": glrt_denominator_part,
        "estimated_delay_sample": estimated_delay_sample,
        "estimated_delay_seconds": estimated_delay_seconds,
        "estimated_fm_rate": estimated_fm_rate,
        "estimated_fm_rate_shift": estimated_fm_rate_shift,
        "estimated_cosine_weight": estimated_cosine_weight,
        "estimated_sine_weight": estimated_sine_weight,
        "estimated_amplitude": float(estimated_amplitude),
        "estimated_phase": float(estimated_phase),
        "known_noise_variance": float(known_noise_variance),
        "best_surface_location": (delay_option_index, fm_rate_option_index),
    }


def run_problem4_known_variance_mle(signal_result, known_noise_variance):
    full_received_signal = signal_result["full_received_signal"]

    minimum_fm_rate = ALPHA
    maximum_fm_rate = 0.5 * FS * TR

    coarse_fm_rate_options = np.arange(
        minimum_fm_rate,
        maximum_fm_rate + BETA_STEP_INCREMENT / 2.0,
        BETA_STEP_INCREMENT,
    )

    coarse_result = compute_known_variance_mle_map(
        full_received_signal,
        coarse_fm_rate_options,
        known_noise_variance,
    )

    center_fm_rate_estimate = coarse_result["estimated_fm_rate"]

    fine_fm_rate_low = max(
        minimum_fm_rate,
        center_fm_rate_estimate - BETA_ZOOM_HALF_WIDTH,
    )

    fine_fm_rate_high = min(
        maximum_fm_rate,
        center_fm_rate_estimate + BETA_ZOOM_HALF_WIDTH,
    )

    fine_fm_rate_options = np.arange(
        fine_fm_rate_low,
        fine_fm_rate_high + BETA_STEP_FINE / 2.0,
        BETA_STEP_FINE,
    )

    final_result = compute_known_variance_mle_map(
        full_received_signal,
        fine_fm_rate_options,
        known_noise_variance,
    )

    final_result["label"] = signal_result["label"]
    final_result["full_received_signal"] = full_received_signal

    return final_result


def make_problem4_mle_comparison_table(problem2_result, problem4_result):
    #Problem 4(b) comparison with the Problem 2 MLEs.
    mle_comparison_records = []

    problem4_mle_fields = [
        ("Time-delay n0, samples", "estimated_delay_sample"),
        ("Time-delay t0, s", "estimated_delay_seconds"),
        ("FM rate beta", "estimated_fm_rate"),
        ("FM rate change alpha0", "estimated_fm_rate_shift"),
        ("Amplitude A", "estimated_amplitude"),
        ("Phase phi, rad", "estimated_phase"),
    ]

    for parameter_label, estimate_key in problem4_mle_fields:
        problem2_estimate = float(problem2_result[estimate_key])
        known_sigma_estimate = float(problem4_result[estimate_key])

        mle_comparison_records.append(
            {
                "Estimated_HFM_parameter": parameter_label,
                "Problem_2_GLRT_MLE": problem2_estimate,
                "Problem_4_known_sigma_MLE": known_sigma_estimate,
                "Absolute_difference": abs(
                    known_sigma_estimate - problem2_estimate
                ),
            }
        )

    mle_comparison_records.append(
        {
            "Estimated_HFM_parameter": "Variance sigma^2",
            "Problem_2_GLRT_MLE": float(
                problem2_result["noise_variance_from_signal_fit"]
            ),
            "Problem_4_known_sigma_MLE": float(
                problem4_result["known_noise_variance"]
            ),
            "Absolute_difference": abs(
                float(problem4_result["known_noise_variance"])
                - float(problem2_result["noise_variance_from_signal_fit"])
            ),
        }
    )

    return pd.DataFrame(mle_comparison_records)


def run_problem4(set1_result, set2_result, signal_set, signal_result, noise_result):
    known_noise_variance = float(noise_result["noise_variance_from_noise_only_fit"])

    problem4_result = run_problem4_known_variance_mle(
        signal_result,
        known_noise_variance,
    )

    comparison_table = make_problem4_mle_comparison_table(
        signal_result,
        problem4_result,
    )

    print("\nProblem 4(b):")
    print(comparison_table.to_string(index=False))

    return problem4_result


def main():
    set1_matrix = load_dataset(SET1_FILE, "Set 1", given_received_sample_count)
    set2_matrix = load_dataset(SET2_FILE, "Set 2", given_received_sample_count)

    #print("\nData dimensions")
    #(f"Set 1: {set1_matrix.shape}")
    #print(f"Set 2: {set2_matrix.shape}")

    set1_result = run_glrt(set1_matrix, "Set1")
    set2_result = run_glrt(set2_matrix, "Set2")

    signal_set, signal_result, noise_result = choose_signal_plus_noise_dataset(
        set1_result,
        set2_result,
    )

    plot_glrt_surface(
        set1_result,
        mark_peak=(signal_set == "Set 1"),
    )

    plot_glrt_surface(
        set2_result,
        mark_peak=(signal_set == "Set 2"),
    )

    # Problem 2 output section.
    save_outputs(
        set1_result,
        set2_result,
        signal_set,
        signal_result,
    )

    # Problem 4 output section.
    run_problem4(
        set1_result,
        set2_result,
        signal_set,
        signal_result,
        noise_result,
    )


if __name__ == "__main__":
    main()

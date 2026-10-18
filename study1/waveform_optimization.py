
"""
Original file is located at
    [source-link-omitted]
"""

from __future__ import annotations

import argparse
import math
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import importlib

import numpy as np
import scipy.stats as st
from scipy.linalg import eigh_tridiagonal
import matplotlib.pyplot as plt
import matplotlib as mpl

try:
    from IPython.display import display, Image
    HAS_IPYTHON_DISPLAY = True
except Exception:
    HAS_IPYTHON_DISPLAY = False

# ------------------------------------------------------------
# Plot style
# ------------------------------------------------------------
mpl.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})

# ------------------------------------------------------------
# Default Google Drive links
# ------------------------------------------------------------
SET3_URL = "data/study1/UUV_Set3.mat"
SET4_URL = "data/study1/UUV_Set4.mat"
LOADER_URL = ""  # Bundled loader is used locally.
LOADER_LOCAL_NAME = "uuv_loader_set34.py"

# ------------------------------------------------------------
# Auto-download + import loader module (same style as Problem 4)
# ------------------------------------------------------------
def _ensure_pkg(pkg: str):
    try:
        __import__(pkg)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


def _ensure_loader_module():
    loader_path = Path(__file__).resolve().parents[1] / LOADER_LOCAL_NAME
    if not loader_path.is_file():
        raise FileNotFoundError(f"Bundled loader not found: {LOADER_LOCAL_NAME}")
    loader_root = str(loader_path.parent)
    if loader_root not in sys.path:
        sys.path.insert(0, loader_root)


def _import_loader():
    _ensure_loader_module()
    importlib.invalidate_caches()

    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    if "" not in sys.path:
        sys.path.insert(0, "")

    module_name = Path(LOADER_LOCAL_NAME).stem
    return importlib.import_module(module_name)


_loader = _import_loader()

# direct access to the same loader functions used in Problem 4
fetch_gdrive_mat_bytes = _loader.fetch_gdrive_mat_bytes
load_uuv_mat_from_bytes = _loader.load_uuv_mat_from_bytes
gdown_download = _loader.gdown_download
load_dataset_mat_only = _loader.load_dataset_mat_only

# ------------------------------------------------------------
# Basic stats helpers
# ------------------------------------------------------------
def Q(x: float) -> float:
    return float(st.norm.sf(x))


def Qinv(p: float) -> float:
    return float(st.norm.isf(p))

# ------------------------------------------------------------
# Template construction
# ------------------------------------------------------------
def build_hfm_echo_template(
    fs: float = 5000.0,
    tr: float = 0.01,
    alpha: float = 20.0,
    T: float = 0.1646,
    tau: float = 0.04,
    A: float = 0.95,
    Td: float = 0.2046,
):
    Ts = 1.0 / fs
    M = int(round(T / Ts)) + 1
    N = int(round(Td / Ts)) + 1
    n0 = int(round(tau / Ts))

    n = np.arange(M)
    t = tr + n * Ts
    s = np.cos(2.0 * math.pi * alpha * np.log(t / tr))

    x = np.zeros(N, dtype=float)
    start = n0
    end = n0 + M
    if end > N:
        raise ValueError(f"Echo support exceeds record length: end={end}, N={N}")
    x[start:end] = A * s

    meta = {
        "fs": fs,
        "Ts": Ts,
        "T": T,
        "Td": Td,
        "M": M,
        "N": N,
        "tau": tau,
        "n0": n0,
        "A": A,
        "tr": tr,
        "alpha": alpha,
        "support_start": start,
        "support_end": end - 1,
    }
    return x, meta


def cinv_tridiag_parts(N: int, a: float):
    diag = np.full(N, 1.0 + a * a, dtype=float)
    diag[0] = 1.0
    diag[-1] = 1.0
    offdiag = np.full(N - 1, -a, dtype=float)
    return diag, offdiag


def apply_cinv(v: np.ndarray, a: float, sigma2: float) -> np.ndarray:
    diag, offdiag = cinv_tridiag_parts(len(v), a)

    y = diag * v
    y[:-1] += offdiag * v[1:]
    y[1:] += offdiag * v[:-1]
    y /= sigma2
    return y


def d2_correlated(x: np.ndarray, a: float, sigma2: float) -> float:
    return float(x @ apply_cinv(x, a=a, sigma2=sigma2))


def project_noise_power(a: float, sigma2: float) -> float:
    return sigma2 / (1.0 - a * a)


def project_enr_db_from_signal(x: np.ndarray, a: float, sigma2: float) -> float:
    Es = float(x @ x)
    pw = project_noise_power(a, sigma2)
    return 10.0 * math.log10(Es / pw)

# ------------------------------------------------------------
# Problem 5(a): optimal fixed-energy waveform design
# ------------------------------------------------------------
def design_optimal_waveform_fixed_energy(
    x_reference: np.ndarray,
    a: float,
    sigma2: float,
) -> Dict[str, Any]:

    Ex = float(x_reference @ x_reference)
    N = len(x_reference)

    diag, offdiag = cinv_tridiag_parts(N, a)

    evals, evecs = eigh_tridiagonal(
        diag,
        offdiag,
        select="i",
        select_range=(N - 1, N - 1),
    )
    lam_max_Tinv = float(evals[0])
    v_max = evecs[:, 0]
    v_max /= np.linalg.norm(v_max)

    lam_max_Cinv = lam_max_Tinv / sigma2
    lam_min_C = 1.0 / lam_max_Cinv

    xhat = math.sqrt(Ex) * v_max
    d2_max = Ex * lam_max_Cinv

    return {
        "Ex": Ex,
        "xhat": xhat,
        "lam_max_Cinv": lam_max_Cinv,
        "lam_min_C": lam_min_C,
        "d2_max": d2_max,
    }

# ------------------------------------------------------------
# Plot helpers
# ------------------------------------------------------------
def save_waveform_comparison_clean(
    x: np.ndarray,
    xhat: np.ndarray,
    meta: Dict[str, Any],
    out_path: Path,
):
    n = np.arange(len(x))

    # normalize for display only
    x_plot = x / np.max(np.abs(x)) if np.max(np.abs(x)) > 0 else x.copy()
    xhat_plot = xhat / np.max(np.abs(xhat)) if np.max(np.abs(xhat)) > 0 else xhat.copy()

    n0 = meta["support_start"]
    n1 = meta["support_end"] + 1
    pad = 40
    z0 = max(0, n0 - pad)
    z1 = min(len(x), n1 + pad)

    fig, axes = plt.subplots(2, 2, figsize=(11, 6), constrained_layout=True)

    # Full record: original
    axes[0, 0].plot(n, x_plot, linewidth=1.2)
    axes[0, 0].set_title(r"Original waveform $x[n]$", fontsize=11)
    axes[0, 0].set_xlabel(r"Sample index $n$ (unitless)", fontsize=11)
    axes[0, 0].set_ylabel("Normalized amplitude (unitless)", fontsize=11)
    axes[0, 0].tick_params(axis="both", labelsize=11)

    # Full record: designed
    axes[0, 1].plot(n, xhat_plot, linewidth=0.8)
    axes[0, 1].set_title(r"Designed waveform $\hat{x}[n]$", fontsize=11)
    axes[0, 1].set_xlabel(r"Sample index $n$ (unitless)", fontsize=11)
    axes[0, 1].set_ylabel("Normalized amplitude (unitless)", fontsize=11)
    axes[0, 1].tick_params(axis="both", labelsize=11)

    # Zoom near original support: original
    axes[1, 0].plot(n[z0:z1], x_plot[z0:z1], linewidth=1.2)
    axes[1, 0].set_title(r"Original waveform near support", fontsize=11)
    axes[1, 0].set_xlabel(r"Sample index $n$ (unitless)", fontsize=11)
    axes[1, 0].set_ylabel("Normalized amplitude (unitless)", fontsize=11)
    axes[1, 0].tick_params(axis="both", labelsize=11)

    # Zoom near original support: designed
    axes[1, 1].plot(n[z0:z1], xhat_plot[z0:z1], linewidth=0.8)
    axes[1, 1].set_title(r"Designed waveform near support", fontsize=11)
    axes[1, 1].set_xlabel(r"Sample index $n$ (unitless)", fontsize=11)
    axes[1, 1].set_ylabel("Normalized amplitude (unitless)", fontsize=11)
    axes[1, 1].tick_params(axis="both", labelsize=11)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_waveform_overlay_better(
    x: np.ndarray,
    xhat: np.ndarray,
    out_path: Path,
):
    n = np.arange(len(x))

    x_plot = x / np.max(np.abs(x)) if np.max(np.abs(x)) > 0 else x.copy()
    xhat_plot = xhat / np.max(np.abs(xhat)) if np.max(np.abs(xhat)) > 0 else xhat.copy()

    plt.figure(figsize=(10, 4))

    # plot designed waveform first so it forms the filled transparent band
    plt.plot(
        n,
        xhat_plot,
        linewidth=1.0,
        alpha=0.58,
        label=r"Designed $\hat{x}[n]$",
        zorder=1,
    )

    # plot original waveform on top so it stays visible
    plt.plot(
        n,
        x_plot,
        linewidth=1.2,
        label=r"Original $x[n]$",
        zorder=2,
    )

    plt.xlabel(r"Sample index $n$ (unitless)", fontsize=11)
    plt.ylabel("Normalized amplitude (unitless)", fontsize=11)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)

    # legend order: original first, designed second
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1, 0]
    plt.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_test_stat_scatter(
    Tvals: np.ndarray,
    gamma: float,
    out_path: Path,
):
    L = len(Tvals)
    plt.figure()
    plt.plot(np.arange(1, L + 1), Tvals, ".", markersize=6)
    plt.axhline(gamma, linewidth=2)
    plt.xlabel(r"Hydrophone index $\ell$ (unitless)", fontsize=11)
    plt.ylabel(r"Test statistic $T(r_\ell)$ (unitless)", fontsize=11)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_test_stat_hist(
    Tvals: np.ndarray,
    gamma: float,
    out_path: Path,
):
    plt.figure()
    plt.hist(Tvals, bins=30)
    plt.axvline(gamma, linewidth=2)
    plt.xlabel(r"Test statistic $T(r_\ell)$ (unitless)", fontsize=11)
    plt.ylabel("Count (number of hydrophones)", fontsize=11)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_pd_vs_pfa_plot(
    pfa_grid: np.ndarray,
    pd_orig_curve: np.ndarray,
    pd_new_curve: np.ndarray,
    selected_pfa: float,
    pd_orig_sel: float,
    pd_new_sel: float,
    out_path: Path,
):
    plt.figure()
    plt.plot(pfa_grid, pd_orig_curve, linewidth=2, label="Original waveform")
    plt.plot(pfa_grid, pd_new_curve, linewidth=2, label="Designed waveform")
    plt.plot([selected_pfa], [pd_orig_sel], "o", markersize=6)
    plt.plot([selected_pfa], [pd_new_sel], "o", markersize=6)
    plt.xlabel(r"$P_{FA}$ (unitless)", fontsize=11)
    plt.ylabel(r"$P_D$ (unitless)", fontsize=11)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.xscale("log")
    plt.ylim(0.0, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_pd_vs_enr_plot(
    enr_db_grid: np.ndarray,
    curves: Dict[str, np.ndarray],
    out_path: Path,
):
    plt.figure()
    for label, vals in curves.items():
        plt.plot(enr_db_grid, vals, linewidth=2, label=label)
    plt.xlabel("ENR (dB)", fontsize=11)
    plt.ylabel(r"$P_D$ (unitless)", fontsize=11)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.ylim(0.0, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

# ------------------------------------------------------------
# Display helper
# ------------------------------------------------------------
def show_saved_plot(path: Path, label: str, width: int = 700):
    if HAS_IPYTHON_DISPLAY:
        print(f"\n[{label}] Plot:")
        display(Image(filename=str(path), width=width))

# ------------------------------------------------------------
# Detector on a dataset
# ------------------------------------------------------------
def run_detector_on_dataset(
    R: np.ndarray,
    xsig: np.ndarray,
    a: float,
    sigma2: float,
    PFA: float,
    PFA_set: float,
    out_prefix: str,
    out_dir: Path,
    show_inline: bool = True,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    N, L = R.shape
    cinv_x = apply_cinv(xsig, a=a, sigma2=sigma2)
    d2 = float(xsig @ cinv_x)

    Tvals = R.T @ cinv_x
    gamma = math.sqrt(d2) * Qinv(PFA)

    detections = Tvals > gamma
    K = int(np.sum(detections))
    frac = float(K / L)

    Tsum = float(np.sum(Tvals))
    gamma_sum = math.sqrt(L * d2) * Qinv(PFA_set)
    mine_present_global = bool(Tsum > gamma_sum)

    PD_true = Q(Qinv(PFA) - math.sqrt(d2))
    PD_set_true = Q(Qinv(PFA_set) - math.sqrt(L * d2))
    PD_hat = frac if mine_present_global else None

    scatter_path = out_dir / f"{out_prefix}_test_stats_scatter.png"
    hist_path = out_dir / f"{out_prefix}_test_stats_hist.png"

    save_test_stat_scatter(Tvals, gamma, scatter_path)
    save_test_stat_hist(Tvals, gamma, hist_path)

    if show_inline and HAS_IPYTHON_DISPLAY:
        print(f"\n[{out_prefix}] Plots:")
        display(Image(filename=str(scatter_path), width=650))
        display(Image(filename=str(hist_path), width=650))

    return {
        "N": N,
        "L": L,
        "gamma_per_hydrophone": gamma,
        "d2_per_hydrophone": d2,
        "K_detections_per_hydrophone": K,
        "fraction_above_gamma": frac,
        "Tsum_global": Tsum,
        "gamma_sum_global": gamma_sum,
        "mine_present_global": mine_present_global,
        "PD_true_per_hydrophone": PD_true,
        "PD_hat_per_hydrophone_from_data": PD_hat,
        "PD_true_global_sum_test": PD_set_true,
        "scatter_plot": str(scatter_path),
        "hist_plot": str(hist_path),
    }

# ------------------------------------------------------------
# ROC / PD-vs-ENR helpers
# ------------------------------------------------------------
def pd_from_d2(pfa: np.ndarray, d2: float) -> np.ndarray:
    return st.norm.sf(st.norm.isf(pfa) - math.sqrt(d2))


def shape_factor_beta(xsig: np.ndarray, a: float, sigma2: float) -> float:
    Es = float(xsig @ xsig)
    u = xsig / math.sqrt(Es)
    d2_u = float(u @ apply_cinv(u, a=a, sigma2=sigma2))
    pw = project_noise_power(a, sigma2)
    return pw * d2_u


def pd_vs_enr_db(enr_db_grid: np.ndarray, pfa: float, beta: float) -> np.ndarray:
    enr_lin = 10.0 ** (enr_db_grid / 10.0)
    d2_vals = beta * enr_lin
    return st.norm.sf(st.norm.isf(pfa) - np.sqrt(d2_vals))

# ------------------------------------------------------------
# Pretty console print
# ------------------------------------------------------------
def print_summary_block(title: str, items: Dict[str, Any]):
    print(f"\n=== {title} ===")
    for k, v in items.items():
        print(f"{k}: {v}")

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(
        description="EEE 556 Project 1 - Problem 5 waveform design in correlated Gaussian noise"
    )
    parser.add_argument("--set3", type=str, default=SET3_URL, help="URL or local path to UUV_Set3.mat")
    parser.add_argument("--set4", type=str, default=SET4_URL, help="URL or local path to UUV_Set4.mat")
    parser.add_argument("--pfa", type=float, default=0.016101995, help="Selected PFA from Problem 1")
    parser.add_argument("--pfa_set", type=float, default=None, help="Dataset-level PFA; defaults to same as --pfa")
    parser.add_argument("--pfa_alt", type=float, default=0.005, help="Second PFA for part (d)")
    parser.add_argument("--a", type=float, default=0.99, help="Pole location")
    parser.add_argument("--sigma2", type=float, default=10.0, help="Driving white-noise variance")
    parser.add_argument("--out_dir", type=str, default="outputs_prob5", help="Output folder")
    parser.add_argument("--debug_mat", action="store_true", help="Print MAT debug info")
    parser.add_argument("--no_inline", action="store_true", help="Do not display images inline in notebook")
    args, _unknown = parser.parse_known_args(argv)

    PFA = args.pfa
    PFA_set = PFA if args.pfa_set is None else args.pfa_set
    a = args.a
    sigma2 = args.sigma2
    show_inline = not args.no_inline

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Original waveform and baseline quantities
    # --------------------------------------------------------
    x, meta = build_hfm_echo_template(fs=5000.0)
    N = meta["N"]
    L_expected = 200

    Ex = float(x @ x)
    d2_orig = d2_correlated(x, a=a, sigma2=sigma2)
    gamma_orig = math.sqrt(d2_orig) * Qinv(PFA)
    pd_orig = Q(Qinv(PFA) - math.sqrt(d2_orig))
    enr_db_project = project_enr_db_from_signal(x, a=a, sigma2=sigma2)

    # --------------------------------------------------------
    # Problem 5(a): designed waveform
    # --------------------------------------------------------
    design = design_optimal_waveform_fixed_energy(x, a=a, sigma2=sigma2)
    xhat = design["xhat"]
    lam_min_C = design["lam_min_C"]
    d2_new = design["d2_max"]
    gamma_new = math.sqrt(d2_new) * Qinv(PFA)
    pd_new = Q(Qinv(PFA) - math.sqrt(d2_new))
    improve_factor = d2_new / d2_orig

    waveform_plot_clean = out_dir / "prob5_waveform_comparison_clean.png"
    save_waveform_comparison_clean(x, xhat, meta, waveform_plot_clean)

    waveform_plot_overlay = out_dir / "prob5_waveform_overlay_better.png"
    save_waveform_overlay_better(x, xhat, waveform_plot_overlay)

    if show_inline:
        show_saved_plot(waveform_plot_clean, "prob5_waveform_comparison_clean")
        show_saved_plot(waveform_plot_overlay, "prob5_waveform_overlay_better")

    # --------------------------------------------------------
    # Problem 5(b): PD vs PFA ROC
    # --------------------------------------------------------
    pfa_grid = np.concatenate([
        np.logspace(-5, -2, 300),
        np.linspace(1e-2, 0.999, 900),
    ])
    pd_orig_curve = pd_from_d2(pfa_grid, d2_orig)
    pd_new_curve = pd_from_d2(pfa_grid, d2_new)

    roc_pfa_plot = out_dir / "prob5_PD_vs_PFA_original_vs_designed.png"
    save_pd_vs_pfa_plot(
        pfa_grid=pfa_grid,
        pd_orig_curve=pd_orig_curve,
        pd_new_curve=pd_new_curve,
        selected_pfa=PFA,
        pd_orig_sel=pd_orig,
        pd_new_sel=pd_new,
        out_path=roc_pfa_plot,
    )
    if show_inline:
        show_saved_plot(roc_pfa_plot, "prob5_PD_vs_PFA_original_vs_designed")

    # --------------------------------------------------------
    # Problem 5(c): load Set4 through loader module
    # --------------------------------------------------------
    print("\nLoading Set4 ...")
    R4 = load_dataset_mat_only(
        args.set4,
        expected_N=N,
        expected_L=L_expected,
        debug=args.debug_mat,
        fallback_name="UUV_Set4.mat",
    )
    print(f"Loaded Set4 shape: {R4.shape}")

    res4_orig = run_detector_on_dataset(
        R=R4,
        xsig=x,
        a=a,
        sigma2=sigma2,
        PFA=PFA,
        PFA_set=PFA_set,
        out_prefix="prob5_Set4_original_waveform",
        out_dir=out_dir,
        show_inline=show_inline,
    )

    R4_synth = R4 + xhat[:, None]
    res4_synth = run_detector_on_dataset(
        R=R4_synth,
        xsig=xhat,
        a=a,
        sigma2=sigma2,
        PFA=PFA,
        PFA_set=PFA_set,
        out_prefix="prob5_Set4_plus_designed_waveform",
        out_dir=out_dir,
        show_inline=show_inline,
    )

    # --------------------------------------------------------
    # Problem 5(d): PD vs ENR for two PFA values
    # --------------------------------------------------------
    beta_orig = shape_factor_beta(x, a=a, sigma2=sigma2)
    beta_new = shape_factor_beta(xhat, a=a, sigma2=sigma2)

    enr_db_grid = np.linspace(-10.0, 15.0, 600)
    curves = {
        f"Original, PFA={PFA:.6f}": pd_vs_enr_db(enr_db_grid, PFA, beta_orig),
        f"Designed, PFA={PFA:.6f}": pd_vs_enr_db(enr_db_grid, PFA, beta_new),
        f"Original, PFA={args.pfa_alt:.6f}": pd_vs_enr_db(enr_db_grid, args.pfa_alt, beta_orig),
        f"Designed, PFA={args.pfa_alt:.6f}": pd_vs_enr_db(enr_db_grid, args.pfa_alt, beta_new),
    }

    enr_plot = out_dir / "prob5_PD_vs_ENR_two_PFA_original_vs_designed.png"
    save_pd_vs_enr_plot(
        enr_db_grid=enr_db_grid,
        curves=curves,
        out_path=enr_plot,
    )
    if show_inline:
        show_saved_plot(enr_plot, "prob5_PD_vs_ENR_two_PFA_original_vs_designed")

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    print_summary_block("Problem 5(a) waveform design", {
        "N": N,
        "L_expected": L_expected,
        "Selected PFA": PFA,
        "a": a,
        "sigma2": sigma2,
        "Project ENR from original template (dB)": round(enr_db_project, 6),
        "Original energy Ex = x^T x": Ex,
        "Original d2 = x^T Cw^(-1) x": d2_orig,
        "Original gamma": gamma_orig,
        "Original PD_true": pd_orig,
        "lambda_min(Cw)": lam_min_C,
        "Designed d2 = xhat^T Cw^(-1) xhat": d2_new,
        "Designed gamma": gamma_new,
        "Designed PD_true": pd_new,
        "Improvement factor d2_new / d2_orig": improve_factor,
        "Waveform plot clean": str(waveform_plot_clean),
        "Waveform plot overlay": str(waveform_plot_overlay),
    })

    print_summary_block("Problem 5(c) Set4 original-waveform detector result", {
        "mine_present_global": res4_orig["mine_present_global"],
        "fraction_above_gamma": res4_orig["fraction_above_gamma"],
        "Tsum_global": res4_orig["Tsum_global"],
        "gamma_sum_global": res4_orig["gamma_sum_global"],
        "scatter_plot": res4_orig["scatter_plot"],
        "hist_plot": res4_orig["hist_plot"],
    })

    print_summary_block("Problem 5(c) Synthetic Set4 + designed waveform", {
        "mine_present_global": res4_synth["mine_present_global"],
        "fraction_above_gamma = PD_hat": res4_synth["fraction_above_gamma"],
        "PD_hat_per_hydrophone_from_data": res4_synth["PD_hat_per_hydrophone_from_data"],
        "PD_true_per_hydrophone": res4_synth["PD_true_per_hydrophone"],
        "Tsum_global": res4_synth["Tsum_global"],
        "gamma_sum_global": res4_synth["gamma_sum_global"],
        "scatter_plot": res4_synth["scatter_plot"],
        "hist_plot": res4_synth["hist_plot"],
    })

    print_summary_block("Problem 5(d) ENR-curve parameters", {
        "beta_orig": beta_orig,
        "beta_new": beta_new,
        "PFA_selected": PFA,
        "PFA_alt": args.pfa_alt,
        "ROC_PD_vs_PFA_plot": str(roc_pfa_plot),
        "PD_vs_ENR_plot": str(enr_plot),
    })

    print(f"\nSaved all outputs to: {out_dir.resolve()}")

# ------------------------------------------------------------
# Notebook-safe entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    main()



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
from typing import Tuple, Dict, Any, Optional
import importlib

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import matplotlib as mpl
from scipy.linalg import cho_factor, cho_solve

try:
    from IPython.display import display, Image
    HAS_IPYTHON_DISPLAY = True
except Exception:
    HAS_IPYTHON_DISPLAY = False

mpl.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})

# -----------------------------------------------------------------------------
# Global Constants & URLs
# -----------------------------------------------------------------------------
SET3_URL = "data/study1/UUV_Set3.mat"
SET4_URL = "data/study1/UUV_Set4.mat"
LOADER_URL = ""  # Bundled loader is used locally.
LOADER_LOCAL_NAME = "uuv_loader_set34.py"


# -----------------------------------------------------------------------------
# Loader Utilities (Dynamic Download/Import)
# -----------------------------------------------------------------------------
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
fetch_gdrive_mat_bytes = _loader.fetch_gdrive_mat_bytes
load_uuv_mat_from_bytes = _loader.load_uuv_mat_from_bytes
gdown_download = _loader.gdown_download


def _is_url(s: str) -> bool:
    s = s.strip().lower()
    return s.startswith("http://") or s.startswith("https://")


def load_dataset_mat_only(
    source: str,
    expected_N: int,
    expected_L: int,
    debug: bool = False,
    fallback_name: str = "UUV_tmp.mat",
) -> np.ndarray:

    src = source.strip()

    if _is_url(src):
        try:
            # Try in-memory fetch first
            b = fetch_gdrive_mat_bytes(src)
            return load_uuv_mat_from_bytes(
                b,
                expected_N=expected_N,
                expected_L=expected_L,
                debug=debug,
            )
        except Exception as e:
            if debug:
                print("\n[DEBUG] In-memory fetch failed, falling back to gdown.")
                print("Reason:", repr(e))
            p = gdown_download(src, Path(fallback_name))
            b = Path(p).read_bytes()
            return load_uuv_mat_from_bytes(
                b,
                expected_N=expected_N,
                expected_L=expected_L,
                debug=debug,
            )

    p = Path(src)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    b = p.read_bytes()
    return load_uuv_mat_from_bytes(
        b,
        expected_N=expected_N,
        expected_L=expected_L,
        debug=debug,
    )


# -----------------------------------------------------------------------------
# Statistical Helpers
# -----------------------------------------------------------------------------
def Q(x: float) -> float:
    """Standard Q-function: Q(x) = 1 - Phi(x)."""
    return float(st.norm.sf(x))


def Qinv(p: float) -> float:
    """Inverse Q-function: Qinv(p) = Phi^{-1}(1 - p)."""
    return float(st.norm.isf(p))


# -----------------------------------------------------------------------------
# Signal Generation (HFM Template)
# -----------------------------------------------------------------------------
def build_hfm_echo_template(
    fs: float = 5000.0,
    tr: float = 0.01,
    alpha: float = 20.0,
    T: float = 0.1646,
    tau: float = 0.04,
    A: float = 0.95,
    Td: float = 0.2046,
) -> Tuple[np.ndarray, Dict[str, Any]]:

    Ts = 1.0 / fs
    M = int(round(T / Ts)) + 1
    N = int(round(Td / Ts)) + 1
    n0 = int(round(tau / Ts))

    # Generate the active pulse s(t)
    n = np.arange(M)
    t = tr + n * Ts
    s = np.cos(2.0 * math.pi * alpha * np.log(t / tr))

    # Place s[n] into the larger observation window of length N
    x = np.zeros(N, dtype=float)
    start = n0
    end = n0 + M
    if end > N:
        raise ValueError(f"Echo (n0+M={end}) exceeds N={N}.")
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


# -----------------------------------------------------------------------------
# Covariance Matrix Construction
# -----------------------------------------------------------------------------
def build_correlated_covariance(
    N: int,
    a: float = 0.99,
    sigma2: float = 10.0,
) -> np.ndarray:

    idx = np.arange(N)
    absdiff = np.abs(idx[:, None] - idx[None, :])
    Cw = (sigma2 / (1.0 - a**2)) * (a ** absdiff)
    return Cw


# -----------------------------------------------------------------------------
# Generalized Matched Filter (GMF) Preparation
# -----------------------------------------------------------------------------
def prepare_correlated_gmf(
    x: np.ndarray,
    Cw: np.ndarray,
) -> Tuple[np.ndarray, float]:

    cfac = cho_factor(Cw, lower=True, check_finite=False)
    cinv_x = cho_solve(cfac, x, check_finite=False)
    d2 = float(x @ cinv_x)
    return cinv_x, d2


def project_enr_db_from_template(
    x: np.ndarray,
    a: float,
    sigma2: float,
) -> float:

    Es = float(x @ x)
    pw = sigma2 / (1.0 - a**2)
    return 10.0 * math.log10(Es / pw)


# -----------------------------------------------------------------------------
# Plotting Utilities
# -----------------------------------------------------------------------------
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
    #plt.show()
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
    #plt.show()
    plt.close()


# -----------------------------------------------------------------------------
# Detector Execution
# -----------------------------------------------------------------------------
def run_detector_on_dataset(
    R: np.ndarray,
    cinv_x: np.ndarray,
    d2: float,
    PFA: float,
    PFA_set: float,
    out_prefix: str,
    out_dir: Path,
    show_inline: bool = True,
) -> Dict[str, Any]:

    out_dir.mkdir(parents=True, exist_ok=True)

    N, L = R.shape

    # T(r) = r^T (Cw^{-1} x)
    # cinv_x holds (Cw^{-1} x)
    Tvals = R.T @ cinv_x

    # Threshold gamma = sqrt(d^2) * Qinv(PFA)
    gamma = math.sqrt(d2) * Qinv(PFA)

    detections = Tvals > gamma
    K = int(np.sum(detections))
    frac = float(K / L)

    # Global sum statistic for deciding if mine is present in the whole set
    Tsum = float(np.sum(Tvals))
    # Global threshold (Variance scales by L, std dev by sqrt(L))
    gamma_sum = math.sqrt(L * d2) * Qinv(PFA_set)
    mine_present_global = (Tsum > gamma_sum)

    # Theoretical PDs
    PD_true = Q(Qinv(PFA) - math.sqrt(d2))
    PD_set_true = Q(Qinv(PFA_set) - math.sqrt(L * d2))

    # Empirical PD (only valid if H1 is actually true for this set)
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
        "PFA_selected_per_hydrophone": PFA,
        "gamma_per_hydrophone": gamma,
        "d2_correlated_per_hydrophone": d2,
        "K_detections_per_hydrophone": K,
        "fraction_above_gamma": frac,
        "PFA_selected_dataset": PFA_set,
        "Tsum_global": Tsum,
        "gamma_sum_global": gamma_sum,
        "mine_present_global": mine_present_global,
        "PD_true_per_hydrophone": PD_true,
        "PD_hat_per_hydrophone_from_data": PD_hat,
        "PD_true_global_sum_test": PD_set_true,
        "scatter_plot": str(scatter_path),
        "hist_plot": str(hist_path),
    }


def print_dataset_summary(name: str, res: Dict[str, Any]):
    print(f"\n=== Results: {name} ===")
    print(f"mine_present_global: {res['mine_present_global']}")
    print(f"K_detections_per_hydrophone: {res['K_detections_per_hydrophone']}")
    print(f"fraction_above_gamma: {res['fraction_above_gamma']}")
    print(f"gamma_per_hydrophone: {res['gamma_per_hydrophone']}")
    print(f"Tsum_global: {res['Tsum_global']}")
    print(f"gamma_sum_global: {res['gamma_sum_global']}")
    print(f"PD_true_per_hydrophone: {res['PD_true_per_hydrophone']}")
    print(f"PD_true_global_sum_test: {res['PD_true_global_sum_test']}")
    if res["mine_present_global"]:
        print(f"PD_hat_per_hydrophone_from_data: {res['PD_hat_per_hydrophone_from_data']}")
    else:
        print("PD_hat_per_hydrophone_from_data: not reported because mine_present_global=False")
    print(f"scatter_plot: {res['scatter_plot']}")
    print(f"hist_plot: {res['hist_plot']}")



def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="EEE 556 Project 1 - Problem 4 (correlated Gaussian noise)"
    )
    parser.add_argument("--set3", type=str, default=SET3_URL, help="URL or local path to UUV_Set3.mat")
    parser.add_argument("--set4", type=str, default=SET4_URL, help="URL or local path to UUV_Set4.mat")
    parser.add_argument("--pfa", type=float, default=0.016101995, help="Selected per-hydrophone PFA from Problem 1")
    parser.add_argument("--pfa_set", type=float, default=None, help="Dataset-level PFA (defaults to same as --pfa)")
    parser.add_argument("--a", type=float, default=0.99, help="Pole location for correlated noise")
    parser.add_argument("--sigma2", type=float, default=10.0, help="Driving white-noise variance")
    parser.add_argument("--out_dir", type=str, default="outputs_prob4", help="Folder to save output plots")
    parser.add_argument("--debug_mat", action="store_true", help="Print MAT-file debug info")
    parser.add_argument("--no_inline", action="store_true", help="Do not display images inline in notebook")

    args, _unknown = parser.parse_known_args(argv)
    pfa_set = args.pfa if args.pfa_set is None else args.pfa_set

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    show_inline = not args.no_inline

    # 1. Generate Signal Template x[n]
    x, meta = build_hfm_echo_template(fs=5000.0)
    N = meta["N"]
    L_expected = 200

    print("=== Problem 4: Correlated-noise GMF detector ===")
    print("Template meta:", meta)
    print(f"Selected per-hydrophone PFA = {args.pfa}")
    print(f"Selected dataset-level PFA = {pfa_set}")
    print(f"a = {args.a}")
    print(f"sigma2 = {args.sigma2}")

    enr_db_project = project_enr_db_from_template(x, a=args.a, sigma2=args.sigma2)
    print(f"Project-definition ENR from template = {enr_db_project:.6f} dB")


    # 2. Load Data
    print("\nLoading Set3 ...")
    R3 = load_dataset_mat_only(
        args.set3,
        expected_N=N,
        expected_L=L_expected,
        debug=args.debug_mat,
        fallback_name="UUV_Set3.mat",
    )

    print("Loading Set4 ...")
    R4 = load_dataset_mat_only(
        args.set4,
        expected_N=N,
        expected_L=L_expected,
        debug=args.debug_mat,
        fallback_name="UUV_Set4.mat",
    )

    print(f"Loaded Set3 shape: {R3.shape}")
    print(f"Loaded Set4 shape: {R4.shape}")

    # 3. Build Covariance Cw & Pre-calculate Inverse Product
    Cw = build_correlated_covariance(N=N, a=args.a, sigma2=args.sigma2)
    cinv_x, d2 = prepare_correlated_gmf(x, Cw)

    gamma = math.sqrt(d2) * Qinv(args.pfa)
    print(f"\nd2 = x^T Cw^(-1) x = {d2:.6f}")
    print(f"Per-hydrophone gamma = {gamma:.6f}")

    # 4. Run Detector on Set 3
    res3 = run_detector_on_dataset(
        R=R3,
        cinv_x=cinv_x,
        d2=d2,
        PFA=args.pfa,
        PFA_set=pfa_set,
        out_prefix="Set3",
        out_dir=out_dir,
        show_inline=show_inline,
    )

    # 5. Run Detector on Set 4
    res4 = run_detector_on_dataset(
        R=R4,
        cinv_x=cinv_x,
        d2=d2,
        PFA=args.pfa,
        PFA_set=pfa_set,
        out_prefix="Set4",
        out_dir=out_dir,
        show_inline=show_inline,
    )

    # 6. Report Results
    print_dataset_summary("Set3", res3)
    print_dataset_summary("Set4", res4)

    print(f"\nSaved plot PNGs to: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    main()
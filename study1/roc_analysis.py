
"""
Original file is located at
    [source-link-omitted]
"""

import argparse
import math
import sys
import subprocess
import importlib
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import matplotlib as mpl

# --- Optional: Check for IPython display for inline images ---
try:
    from IPython.display import display, Image
    HAS_IPYTHON_DISPLAY = True
except Exception:
    HAS_IPYTHON_DISPLAY = False

# --- Plotting Configuration ---
mpl.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})

# --- Constants and URLs ---
SET1_URL = "data/study1/UUV_Set1.mat"
SET2_URL = "data/study1/UUV_Set2.mat"
LOADER_URL = ""  # Bundled loader is used locally.
LOADER_LOCAL_NAME = "uuv_loader.py"


# ------------------------------------------------------------------
# Dynamic Loader Setup
# ------------------------------------------------------------------
def _ensure_pkg(pkg: str):
    """Ensures a package is installed (e.g., gdown)."""
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
    """Imports the downloaded loader module dynamically."""
    _ensure_loader_module()
    importlib.invalidate_caches()

    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    if "" not in sys.path:
        sys.path.insert(0, "")

    return importlib.import_module("uuv_loader")


# Initialize loader functions
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
            # Try in-memory download first
            b = fetch_gdrive_mat_bytes(src)
            return load_uuv_mat_from_bytes(
                b,
                expected_N=expected_N,
                expected_L=expected_L,
                debug=debug,
            )
        except Exception as e:
            # Fallback to file download
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

    # Local file path
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


# ------------------------------------------------------------------
# Mathematical Helpers
# ------------------------------------------------------------------
def Q(x: float) -> float:
    """Standard Q-function: Q(x) = 1 - CDF(x) for standard normal."""
    return float(st.norm.sf(x))


def Qinv(p: float) -> float:
    """Inverse Q-function."""
    return float(st.norm.isf(p))


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
    # Number of samples in the pulse
    M = int(round(T / Ts)) + 1
    # Total number of samples in observation window
    N = int(round(Td / Ts)) + 1
    # Start index
    n0 = int(round(tau / Ts))

    # Construct the pulse s[n]
    n = np.arange(M)
    t = tr + n * Ts
    s = np.cos(2.0 * math.pi * alpha * np.log(t / tr))

    # Place s[n] into the full window x[n]
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


# ------------------------------------------------------------------
# Detection Statistics
# ------------------------------------------------------------------
def compute_score_vals(R: np.ndarray, x: np.ndarray) -> np.ndarray:

    return R.T @ x


def compute_Tvals(R: np.ndarray, x: np.ndarray, sigma2_used: float) -> np.ndarray:

    return compute_score_vals(R, x) / sigma2_used


def threshold_gamma_T(PFA: float, d2: float) -> float:

    return math.sqrt(d2) * Qinv(PFA)


def threshold_gamma_score(PFA: float, Ex: float, sigma2_used: float) -> float:

    return math.sqrt(Ex * sigma2_used) * Qinv(PFA)


# ------------------------------------------------------------------
# Analysis Functions
# ------------------------------------------------------------------
def run_prob2a_style_summary(
    R: np.ndarray,
    x: np.ndarray,
    sigma2_used: float,
    PFA: float,
    PFA_set: Optional[float] = None,
    out_prefix: str = "Set",
) -> Dict[str, Any]:

    N, L = R.shape
    if x.shape != (N,):
        raise ValueError(f"x shape {x.shape} != (N,) where N={N}")

    if PFA_set is None:
        PFA_set = PFA

    # 1. Compute Test Statistics
    Tvals = compute_Tvals(R, x, sigma2_used=sigma2_used)

    # 2. Deflection Coefficient d^2 = (x^T x) / sigma^2
    d2 = float((x @ x) / sigma2_used)

    # 3. Threshold
    gamma = threshold_gamma_T(PFA, d2)

    # 4. Binary Decisions
    detections = Tvals > gamma
    K = int(np.sum(detections))
    frac = float(K / L)

    # 5. Global Test (Sum of T's)
    Tsum = float(np.sum(Tvals))
    # Global threshold: gamma_sum = sqrt(L * d^2) * Qinv(PFA_set)
    gamma_sum = math.sqrt(L * d2) * Qinv(PFA_set)
    mine_present_global = bool(Tsum > gamma_sum)

    return {
        "out_prefix": out_prefix,
        "N": int(N),
        "L": int(L),
        "sigma2_used": float(sigma2_used),
        "d2": float(d2),
        "gamma": float(gamma),
        "K": K,
        "frac": frac,
        "Tsum": float(Tsum),
        "gamma_sum": float(gamma_sum),
        "mine_present_global": mine_present_global,
        "Tvals": Tvals,
    }


def operating_point_from_selected_pfa(
    R_h0: np.ndarray,
    R_h1: np.ndarray,
    x: np.ndarray,
    sigma2_used: float,
    PFA: float,
) -> Dict[str, float]:

    Ex = float(x @ x)
    gamma_score = threshold_gamma_score(PFA, Ex, sigma2_used)

    S0 = compute_score_vals(R_h0, x)
    S1 = compute_score_vals(R_h1, x)

    PFA_hat = float(np.mean(S0 > gamma_score))
    PD_hat = float(np.mean(S1 > gamma_score))

    d2 = Ex / sigma2_used
    gamma_T = threshold_gamma_T(PFA, d2)

    return {
        "sigma2_used": float(sigma2_used),
        "gamma_score": float(gamma_score),
        "gamma_T": float(gamma_T),
        "PFA_hat": PFA_hat,
        "PD_hat": PD_hat,
    }


def empirical_roc_from_datasets(
    R_h0: np.ndarray,
    R_h1: np.ndarray,
    x: np.ndarray,
    num_thresh: int = 500,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    S0 = compute_score_vals(R_h0, x)
    S1 = compute_score_vals(R_h1, x)

    # Determine range of scores to sweep
    s_min = float(min(S0.min(), S1.min()))
    s_max = float(max(S0.max(), S1.max()))
    eps = 1e-12

    # Create threshold grid
    gammas = np.linspace(s_max + eps, s_min - eps, num_thresh)

    # Compute rates for each threshold
    PFA_hat = np.array([np.mean(S0 > g) for g in gammas], dtype=float)
    PD_hat = np.array([np.mean(S1 > g) for g in gammas], dtype=float)

    # Sort by PFA for cleaner plotting
    idx = np.argsort(PFA_hat)
    return PFA_hat[idx], PD_hat[idx], gammas[idx]


def theoretical_roc(enr_db: float, pfa_grid: np.ndarray) -> np.ndarray:

    enr_lin = 10.0 ** (enr_db / 10.0)
    return st.norm.sf(st.norm.isf(pfa_grid) - math.sqrt(enr_lin))


def pd_vs_enr_curves(
    enr_db_grid: np.ndarray,
    pfa_values: Sequence[float],
) -> Dict[float, np.ndarray]:

    curves: Dict[float, np.ndarray] = {}
    for pfa in pfa_values:
        qinv = float(st.norm.isf(pfa))
        enr_lin = 10.0 ** (enr_db_grid / 10.0)
        curves[pfa] = st.norm.sf(qinv - np.sqrt(enr_lin))
    return curves


# ------------------------------------------------------------------
# Plotting Functions
# ------------------------------------------------------------------
def save_empirical_roc_plot(
    PFA_hat: np.ndarray,
    PD_hat: np.ndarray,
    out_path: Path,
    title: str,
    operating_points: Optional[Sequence[Tuple[str, float, float]]] = None,
):
    plt.figure(figsize=(6.7, 5.2))
    plt.plot(PFA_hat, PD_hat, linewidth=2, label="Empirical ROC")

    if operating_points is not None:
        for label, pfa, pd in operating_points:
            plt.plot(pfa, pd, "o", markersize=8, label=label)

    plt.xlabel(r"$P_{FA}$ (unitless)", fontsize=11)
    plt.ylabel(r"$P_D$ (unitless)", fontsize=11)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(True, linestyle=":")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_theoretical_roc_plot(
    pfa_grid: np.ndarray,
    curves: Dict[str, np.ndarray],
    out_path: Path,
    title: str,
    operating_points: Optional[Sequence[Tuple[str, float, float]]] = None,
):
    plt.figure(figsize=(6.7, 5.2))
    for label, pd in curves.items():
        plt.plot(pfa_grid, pd, linewidth=2, label=label)

    if operating_points is not None:
        for label, pfa, pd in operating_points:
            plt.plot(pfa, pd, "o", markersize=8, label=label)

    plt.xlabel(r"$P_{FA}$ (unitless)", fontsize=11)
    plt.ylabel(r"$P_D$ (unitless)", fontsize=11)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_pd_vs_enr_plot(
    enr_db_grid: np.ndarray,
    curves: Dict[float, np.ndarray],
    out_path: Path,
    title: str,
):
    plt.figure(figsize=(6.7, 5.2))
    for pfa, pd in curves.items():
        plt.plot(enr_db_grid, pd, linewidth=2, label=fr"$P_{{FA}}={pfa:.6f}$")

    plt.xlabel("ENR (dB)", fontsize=11)
    plt.ylabel(r"$P_D$ (unitless)", fontsize=11)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def show_saved_image(path: Path, width: int = 600):
    if HAS_IPYTHON_DISPLAY and path.exists():
        display(Image(filename=str(path), width=width))


# ------------------------------------------------------------------
# Main Execution Block
# ------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="EEE 556 Project 1 - Problem 3 ROC Curves")

    parser.add_argument("--set1", type=str, default=SET1_URL, help="URL or local path to UUV_Set1.mat")
    parser.add_argument("--set2", type=str, default=SET2_URL, help="URL or local path to UUV_Set2.mat")
    parser.add_argument("--pfa_selected", type=float, default=0.016101995, help="Selected PFA from Problems 1 and 2")
    parser.add_argument("--pfa_set", type=float, default=None, help="Dataset-level PFA if needed (defaults to pfa_selected)")
    parser.add_argument("--sigma2_true", type=float, default=30.0, help="White-noise variance used in Problem 2")
    parser.add_argument("--enr_true_db", type=float, default=10.9, help="True ENR in dB")
    parser.add_argument("--enr_alt_db", type=float, default=8.5, help="Alternate ENR in dB")
    parser.add_argument("--out_dir", type=str, default="outputs_prob3", help="Folder to save plots")
    parser.add_argument("--num_thresh", type=int, default=500, help="Number of thresholds for empirical ROC")
    parser.add_argument("--debug_mat", action="store_true", help="Print MAT debug info")
    parser.add_argument("--no_inline", action="store_true", help="Disable inline display of saved PNGs")

    args, _unknown = parser.parse_known_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    show_inline = not args.no_inline
    pfa_set = args.pfa_selected if args.pfa_set is None else args.pfa_set

    # 1. Generate Signal Template
    x, meta = build_hfm_echo_template(fs=5000.0)
    N = meta["N"]
    L_expected = 200

    print("Template meta:", meta)
    print(f"Using sigma2_true = {args.sigma2_true}")
    print(f"Using selected PFA = {args.pfa_selected}")
    print(f"Using dataset-level PFA_set = {pfa_set}")
    print(f"True ENR = {args.enr_true_db} dB")
    print(f"Alternate ENR = {args.enr_alt_db} dB")

    # 2. Load Datasets
    print("\nLoading datasets with existing uuv_loader.py ...")

    # Assume Set1 is Noise-Only (H0) for ROC purposes
    R1 = load_dataset_mat_only(
        args.set1,
        expected_N=N,
        expected_L=L_expected,
        debug=args.debug_mat,
        fallback_name="UUV_Set1.mat",
    )

    # Assume Set2 is Signal+Noise (H1) for ROC purposes
    R2 = load_dataset_mat_only(
        args.set2,
        expected_N=N,
        expected_L=L_expected,
        debug=args.debug_mat,
        fallback_name="UUV_Set2.mat",
    )

    print(f"Loaded Set1 shape: {R1.shape}")
    print(f"Loaded Set2 shape: {R2.shape}")

    # 3. Quick sanity check on scores
    print("\n--- Matched-Filter Scores (first 10 values) ---")
    S0_display = compute_score_vals(R1, x)
    S1_display = compute_score_vals(R2, x)
    print(f"Scores for H0 (R1): {S0_display[:10]}")
    print(f"Scores for H1 (R2): {S1_display[:10]}")
    print("--------------------------------------------------")

    # ------------------------------------------------------------
    # (a) & (b) Empirical ROC from data
    # ------------------------------------------------------------
    PFA_hat_emp, PD_hat_emp, _ = empirical_roc_from_datasets(
        R_h0=R1,
        R_h1=R2,
        x=x,
        num_thresh=args.num_thresh,
    )

    empirical_roc_path = out_dir / "roc_empirical_from_data.png"
    save_empirical_roc_plot(
        PFA_hat_emp,
        PD_hat_emp,
        empirical_roc_path,
        title="Empirical ROC from given datasets (Set1 as $H_0$, Set2 as $H_1$)",
    )

    if show_inline:
        print(f"\nSaved: {empirical_roc_path}")
        show_saved_image(empirical_roc_path)

    # ------------------------------------------------------------
    # (c) Theoretical ROC Curves
    # ------------------------------------------------------------
    # We plot two curves: one for ENR = 10.9 dB, one for ENR = 8.5 dB
    pfa_grid = np.concatenate([
        np.logspace(-5, -2, 250),
        np.linspace(1e-2, 0.999, 750),
    ])

    pd_true_curve = theoretical_roc(args.enr_true_db, pfa_grid)
    pd_alt_curve = theoretical_roc(args.enr_alt_db, pfa_grid)

    theory_roc_path = out_dir / "roc_theoretical_10p9_vs_alt.png"
    save_theoretical_roc_plot(
        pfa_grid,
        {
            fr"ENR = {args.enr_true_db:.1f} dB": pd_true_curve,
            fr"ENR = {args.enr_alt_db:.1f} dB": pd_alt_curve,
        },
        theory_roc_path,
        title="Theoretical ROC curves",
    )

    if show_inline:
        print(f"Saved: {theory_roc_path}")
        show_saved_image(theory_roc_path)

    # ------------------------------------------------------------
    # (d) PD vs ENR for fixed PFA
    # ------------------------------------------------------------
    enr_db_grid = np.linspace(-5.0, 15.0, 500)
    pfa_values = [0.005, args.pfa_selected, 0.02]
    curves_pd_vs_enr = pd_vs_enr_curves(enr_db_grid, pfa_values)

    pd_vs_enr_path = out_dir / "pd_vs_enr_three_pfa.png"
    save_pd_vs_enr_plot(
        enr_db_grid,
        curves_pd_vs_enr,
        pd_vs_enr_path,
        title=r"Theoretical $P_D$ versus ENR for three fixed $P_{FA}$ values",
    )

    if show_inline:
        print(f"Saved: {pd_vs_enr_path}")
        show_saved_image(pd_vs_enr_path)

    # ------------------------------------------------------------
    # (e) Mismatched Variance Analysis
    # ------------------------------------------------------------
    # Compute what sigma^2 would be if ENR was 8.5 dB but x is fixed
    # ENR = Ex / sigma^2  => sigma^2 = Ex / ENR
    Ex = float(x @ x)
    sigma2_est_from_8p5 = Ex / (10.0 ** (args.enr_alt_db / 10.0))

    print("\n=== Mismatch setup for part 3(d) ===")
    print(f"Signal energy Ex = x^T x = {Ex:.6f}")
    print(f"True sigma^2 used = {args.sigma2_true:.6f}")
    print(f"Estimated sigma^2 from ENR = {args.enr_alt_db:.1f} dB is {sigma2_est_from_8p5:.6f}")

    # Run detection on both sets with CORRECT sigma^2
    res_set1_true = run_prob2a_style_summary(
        R1, x, sigma2_used=args.sigma2_true, PFA=args.pfa_selected, PFA_set=pfa_set, out_prefix="Set1_trueSigma"
    )
    res_set2_true = run_prob2a_style_summary(
        R2, x, sigma2_used=args.sigma2_true, PFA=args.pfa_selected, PFA_set=pfa_set, out_prefix="Set2_trueSigma"
    )

    # Run detection on both sets with MISMATCHED sigma^2
    res_set1_mis = run_prob2a_style_summary(
        R1, x, sigma2_used=sigma2_est_from_8p5, PFA=args.pfa_selected, PFA_set=pfa_set, out_prefix="Set1_sigmaFrom8p5dB"
    )
    res_set2_mis = run_prob2a_style_summary(
        R2, x, sigma2_used=sigma2_est_from_8p5, PFA=args.pfa_selected, PFA_set=pfa_set, out_prefix="Set2_sigmaFrom8p5dB"
    )

    print("\n=== Problem 2(a)-style operating point with TRUE sigma^2 ===")
    for res in [res_set1_true, res_set2_true]:
        print(
            f"{res['out_prefix']}: "
            f"d2={res['d2']:.6f}, gamma={res['gamma']:.6f}, "
            f"K={res['K']}, K/L={res['frac']:.6f}, "
            f"mine_present_global={res['mine_present_global']}"
        )

    print("\n=== Problem 2(a)-style operating point with MISMATCHED sigma^2 ===")
    for res in [res_set1_mis, res_set2_mis]:
        print(
            f"{res['out_prefix']}: "
            f"d2={res['d2']:.6f}, gamma={res['gamma']:.6f}, "
            f"K={res['K']}, K/L={res['frac']:.6f}, "
            f"mine_present_global={res['mine_present_global']}"
        )

    # ------------------------------------------------------------
    # Calculate Empirical Operating Points (Hat values)
    # ------------------------------------------------------------
    op_true = operating_point_from_selected_pfa(
        R_h0=R1,
        R_h1=R2,
        x=x,
        sigma2_used=args.sigma2_true,
        PFA=args.pfa_selected,
    )

    op_mis = operating_point_from_selected_pfa(
        R_h0=R1,
        R_h1=R2,
        x=x,
        sigma2_used=sigma2_est_from_8p5,
        PFA=args.pfa_selected,
    )

    print("\n=== Actual operating points on the given datasets ===")
    print(
        f"Correct sigma^2:    PFA_hat={op_true['PFA_hat']:.6f}, "
        f"PD_hat={op_true['PD_hat']:.6f}, "
        f"gamma_T={op_true['gamma_T']:.6f}, gamma_score={op_true['gamma_score']:.6f}"
    )
    print(
        f"Mismatched sigma^2: PFA_hat={op_mis['PFA_hat']:.6f}, "
        f"PD_hat={op_mis['PD_hat']:.6f}, "
        f"gamma_T={op_mis['gamma_T']:.6f}, gamma_score={op_mis['gamma_score']:.6f}"
    )

    # ------------------------------------------------------------
    # (f) Overlay operating points on ROCs
    # ------------------------------------------------------------

    # 1. On Empirical ROC
    empirical_roc_ops_path = out_dir / "roc_empirical_with_operating_points.png"
    save_empirical_roc_plot(
        PFA_hat_emp,
        PD_hat_emp,
        empirical_roc_ops_path,
        title="Empirical ROC with selected operating points",
        operating_points=[
            (r"Correct $\sigma^2$ operating point", op_true["PFA_hat"], op_true["PD_hat"]),
            (r"Mismatched $\sigma^2$ operating point", op_mis["PFA_hat"], op_mis["PD_hat"]),
        ],
    )

    if show_inline:
        print(f"Saved: {empirical_roc_ops_path}")
        show_saved_image(empirical_roc_ops_path)

    # 2. On Theoretical ROC
    theory_true_with_ops_path = out_dir / "roc_true_enr_with_operating_points.png"
    save_theoretical_roc_plot(
        pfa_grid,
        {fr"ENR = {args.enr_true_db:.1f} dB": pd_true_curve},
        theory_true_with_ops_path,
        title=fr"Theoretical ROC at ENR = {args.enr_true_db:.1f} dB with operating points",
        operating_points=[
            (r"Correct $\sigma^2$ operating point", op_true["PFA_hat"], op_true["PD_hat"]),
            (r"Mismatched $\sigma^2$ operating point", op_mis["PFA_hat"], op_mis["PD_hat"]),
        ],
    )

    if show_inline:
        print(f"Saved: {theory_true_with_ops_path}")
        show_saved_image(theory_true_with_ops_path)

    # ------------------------------------------------------------
    # (g) Compare with Theoretical Values
    # ------------------------------------------------------------
    PD_theory_true = float(
        st.norm.sf(st.norm.isf(args.pfa_selected) - math.sqrt(10.0 ** (args.enr_true_db / 10.0)))
    )
    PD_theory_alt = float(
        st.norm.sf(st.norm.isf(args.pfa_selected) - math.sqrt(10.0 ** (args.enr_alt_db / 10.0)))
    )

    print("\n=== Theoretical PD at selected PFA ===")
    print(f"PD_theory(ENR = {args.enr_true_db:.1f} dB, PFA = {args.pfa_selected}) = {PD_theory_true:.6f}")
    print(f"PD_theory(ENR = {args.enr_alt_db:.1f} dB, PFA = {args.pfa_selected}) = {PD_theory_alt:.6f}")

    print("\n=== Output plot files ===")
    print(empirical_roc_path)
    print(theory_roc_path)
    print(pd_vs_enr_path)
    print(empirical_roc_ops_path)
    print(theory_true_with_ops_path)

    return 0


if __name__ == "__main__":
    main()
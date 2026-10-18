
""" Colab link:
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

# Optional inline display for notebooks / Colab
try:
    from IPython.display import display, Image
    HAS_IPYTHON_DISPLAY = True
except Exception:
    HAS_IPYTHON_DISPLAY = False


# ------------------------------------------------------------
# Plot style: match report font size
# ------------------------------------------------------------
mpl.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})


# ------------------------------------------------------------
# Google Drive file URLs
# ------------------------------------------------------------
SET1_URL = "data/study1/UUV_Set1.mat"
SET2_URL = "data/study1/UUV_Set2.mat"
LOADER_URL = ""  # Bundled loader is used locally.
LOADER_LOCAL_NAME = "uuv_loader.py"


# ------------------------------------------------------------
# Install a package only if it is missing
# ------------------------------------------------------------
def _ensure_pkg(pkg: str):
    try:
        __import__(pkg)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


# ------------------------------------------------------------
# Ensure the custom loader file exists locally
# ------------------------------------------------------------
def _ensure_loader_module():
    loader_path = Path(__file__).resolve().parents[1] / LOADER_LOCAL_NAME
    if not loader_path.is_file():
        raise FileNotFoundError(f"Bundled loader not found: {LOADER_LOCAL_NAME}")
    loader_root = str(loader_path.parent)
    if loader_root not in sys.path:
        sys.path.insert(0, loader_root)


# ------------------------------------------------------------
# Import the loader module safely
# ------------------------------------------------------------
def _import_loader():
    _ensure_loader_module()

    if "" not in sys.path:
        sys.path.insert(0, "")

    return importlib.import_module("uuv_loader")


_loader = _import_loader()
fetch_gdrive_mat_bytes = _loader.fetch_gdrive_mat_bytes
load_uuv_mat_from_bytes = _loader.load_uuv_mat_from_bytes
gdown_download = _loader.gdown_download


# ------------------------------------------------------------
# Gaussian tail probability helpers
# ------------------------------------------------------------
def Q(x: float) -> float:
    return float(st.norm.sf(x))


def Qinv(p: float) -> float:
    return float(st.norm.isf(p))


# ------------------------------------------------------------
# Build the delayed HFM echo template x[n]
# ------------------------------------------------------------
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

    n = np.arange(M)
    t = tr + n * Ts
    s = np.cos(2.0 * math.pi * alpha * np.log(t / tr))

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


# ------------------------------------------------------------
# Run detector on one dataset
# ------------------------------------------------------------
def run_detector_on_dataset(
    R: np.ndarray,
    x: np.ndarray,
    sigma2: float,
    PFA: float,
    PFA_set: float,
    enr_db_for_true_pd: float,
    out_prefix: str,
    out_dir: Path,
    show_inline: bool = True,
) -> Dict[str, Any]:

    out_dir.mkdir(parents=True, exist_ok=True)

    N, L = R.shape
    if x.shape != (N,):
        raise ValueError(f"x shape {x.shape} != (N,) where N={N}")

    # --------------------------------------------------------
    # Per-hydrophone test statistic:
    #   T(r_l) = (r_l^T x) / sigma^2
    # --------------------------------------------------------
    Tvals = (R.T @ x) / sigma2

    # --------------------------------------------------------
    # Template energy term:
    #   d^2 = (x^T x) / sigma^2
    # Threshold:
    #   gamma = sqrt(d^2) * Q^{-1}(PFA)
    # --------------------------------------------------------
    d2 = float((x @ x) / sigma2)
    gamma = math.sqrt(d2) * Qinv(PFA)

    # Per-hydrophone decisions
    detections = Tvals > gamma
    K = int(np.sum(detections))
    frac = K / L

    # --------------------------------------------------------
    # Global dataset-level detector using sum of hydrophone statistics
    # --------------------------------------------------------
    Tsum = float(np.sum(Tvals))
    gamma_sum = math.sqrt(L * d2) * Qinv(PFA_set)
    mine_present_global = (Tsum > gamma_sum)

    # --------------------------------------------------------
    # Theoretical PD calculations
    # --------------------------------------------------------
    ENR_lin = 10.0 ** (enr_db_for_true_pd / 10.0)
    PD_true_givenENR = Q(Qinv(PFA) - math.sqrt(ENR_lin))
    PD_true_from_template = Q(Qinv(PFA) - math.sqrt(d2))

    PD_set_givenENR = Q(Qinv(PFA_set) - math.sqrt(L * ENR_lin))
    PD_set_from_template = Q(Qinv(PFA_set) - math.sqrt(L * d2))

    # Empirical PD estimate only when the dataset is declared mine-present
    PD_hat = frac if mine_present_global else None

    # --------------------------------------------------------
    # Plot 1: Per-hydrophone test statistics
    # --------------------------------------------------------
    scatter_path = out_dir / f"{out_prefix}_test_stats_scatter.png"

    plt.figure()
    plt.plot(np.arange(1, L + 1), Tvals, ".", markersize=6)
    plt.axhline(gamma, linewidth=2)
    plt.xlabel("Hydrophone index ℓ (unitless)")
    plt.ylabel("Test statistic T(rℓ) (dimensionless)")
    plt.title(f"{out_prefix}: Per-hydrophone test statistics with threshold")
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=200)
    plt.close()

    # --------------------------------------------------------
    # Plot 2: Histogram of test statistics
    # --------------------------------------------------------
    hist_path = out_dir / f"{out_prefix}_test_stats_hist.png"

    plt.figure()
    plt.hist(Tvals, bins=30)
    plt.axvline(gamma, linewidth=2)
    plt.xlabel("Test statistic T(rℓ) (dimensionless)")
    plt.ylabel("Count (number of hydrophones)")
    plt.title(f"{out_prefix}: Histogram with threshold")
    plt.tight_layout()
    plt.savefig(hist_path, dpi=200)
    plt.close()

    # --------------------------------------------------------
    # Display saved PNGs once in notebook / Colab
    # --------------------------------------------------------
    if show_inline and HAS_IPYTHON_DISPLAY:
        print(f"\n[{out_prefix}] Plots:")
        display(Image(filename=str(scatter_path), width=600))
        display(Image(filename=str(hist_path), width=600))

    return {
        "N": N,
        "L": L,
        "sigma2": sigma2,
        "PFA_selected_per_hydrophone": PFA,
        "gamma_per_hydrophone": gamma,
        "d2_template_per_hydrophone": d2,
        "K_detections_per_hydrophone": K,
        "fraction_above_gamma": frac,
        "PFA_selected_dataset": PFA_set,
        "Tsum_global": Tsum,
        "gamma_sum_global": gamma_sum,
        "mine_present_global": mine_present_global,
        "PD_true_given_ENRdb_per_hydrophone": PD_true_givenENR,
        "PD_true_from_template_per_hydrophone": PD_true_from_template,
        "PD_hat_per_hydrophone_from_data": PD_hat,
        "PD_set_given_ENRdb_global": PD_set_givenENR,
        "PD_set_from_template_global": PD_set_from_template,
        "scatter_plot": str(scatter_path),
        "hist_plot": str(hist_path),
    }


# ------------------------------------------------------------
# Bootstrap estimate of PFA using noise-only samples
# ------------------------------------------------------------
def bootstrap_pfa_estimate_from_noise_only_samples(
    R: np.ndarray,
    x: np.ndarray,
    sigma2: float,
    gamma: float,
    B: int = 20000,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Estimate PFA by resampling from rows where x[n] = 0.

    Steps:
    1. Collect samples from the zero-template region.
    2. Resample synthetic noise vectors.
    3. Evaluate the detector on those synthetic vectors.
    4. Estimate PFA as the exceedance fraction.
    """
    rng = np.random.default_rng(seed)
    N, L = R.shape

    zero_idx = np.where(np.isclose(x, 0.0))[0]
    noise_pool = R[zero_idx, :].reshape(-1)

    W = rng.choice(noise_pool, size=(B, N), replace=True)
    T_noise = (W @ x) / sigma2

    PFA_hat = float(np.mean(T_noise > gamma))
    se = math.sqrt(PFA_hat * (1 - PFA_hat) / B)

    return {
        "B_bootstrap_trials": B,
        "noise_pool_size": int(noise_pool.size),
        "PFA_hat_bootstrap": PFA_hat,
        "PFA_bootstrap_se": se,
    }


# ------------------------------------------------------------
# Part (c): PFA convergence plot
# ------------------------------------------------------------
def save_pfa_convergence_plot(
    R_h0: np.ndarray,
    x: np.ndarray,
    sigma2: float,
    pfa_selected: float,
    gamma: float,
    out_dir: Path,
    show_inline: bool = True,
) -> Path:
    """
    Build and save the Monte Carlo convergence plot:
    estimated PFA versus number of Monte Carlo runs M.
    """
    out_dir.mkdir(exist_ok=True, parents=True)

    zero_idx = np.where(np.isclose(x, 0.0))[0]
    noise_pool = R_h0[zero_idx, :].reshape(-1)
    N = x.shape[0]

    print(f"Noise pool size = {noise_pool.size} samples from {len(zero_idx)} zero-template indices")

    def estimate_pfa_for_M(M: int, batch: int = 5000, seed: int = 0):
        rng = np.random.default_rng(seed)
        exceed = 0
        done = 0

        while done < M:
            b = min(batch, M - done)
            W = rng.choice(noise_pool, size=(b, N), replace=True)
            T_noise = (W @ x) / sigma2
            exceed += int(np.sum(T_noise > gamma))
            done += b

        p_hat = exceed / M
        se_hat = math.sqrt(p_hat * (1 - p_hat) / M)
        return p_hat, se_hat

    M_list = np.array([200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000], dtype=int)

    pfa_hats = []
    pfa_ses = []

    for i, M in enumerate(M_list):
        p_hat, se_hat = estimate_pfa_for_M(M, batch=5000, seed=123 + i)
        pfa_hats.append(p_hat)
        pfa_ses.append(se_hat)

    pfa_hats = np.array(pfa_hats)
    pfa_ses = np.array(pfa_ses)

    out_path = out_dir / "PFA_vs_M.png"

    plt.figure()
    plt.semilogx(M_list, pfa_hats, marker="o", label=r"Estimated $\widehat{P}_{FA}$")
    plt.axhline(pfa_selected, linestyle="--", linewidth=2, label=r"Selected $P_{FA}$")
    plt.xlabel(r"$M$ (Monte Carlo runs) [log scale]")
    plt.ylabel(r"$P_{FA}$ (unitless)")
    plt.title(r"Estimated $P_{FA}$ vs $M$")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    if show_inline and HAS_IPYTHON_DISPLAY:
        display(Image(filename=str(out_path), width=700))

    return out_path


# ------------------------------------------------------------
# Main driver
# ------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set1", type=str, default=SET1_URL)
    parser.add_argument("--set2", type=str, default=SET2_URL)
    parser.add_argument("--pfa", type=float, default=0.016101995)
    parser.add_argument("--pfa_set", type=float, default=None)
    parser.add_argument("--sigma2", type=float, default=30.0)
    parser.add_argument("--enr_db", type=float, default=10.9)
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--debug_mat", action="store_true")
    parser.add_argument("--no_inline", action="store_true")

    # parse_known_args avoids notebook argument issues
    args, _unknown = parser.parse_known_args(argv)

    pfa_set = args.pfa if args.pfa_set is None else args.pfa_set
    out_dir = Path(args.out_dir)
    show_inline = not args.no_inline

    # --------------------------------------------------------
    # Build echo template
    # --------------------------------------------------------
    x, meta = build_hfm_echo_template(fs=5000.0)

    print("Template meta:", meta)
    print(f"Using per-hydrophone: PFA={args.pfa}, sigma2={args.sigma2}, ENR={args.enr_db} dB")
    print(f"Using dataset-level (Option A): PFA_set={pfa_set}")

    # --------------------------------------------------------
    # Load Set1
    # --------------------------------------------------------
    try:
        b1 = (Path(args.set1).read_bytes() if Path(args.set1).is_file()
              else fetch_gdrive_mat_bytes(args.set1))
        R1 = load_uuv_mat_from_bytes(
            b1,
            expected_N=meta["N"],
            expected_L=200,
            debug=args.debug_mat
        )
    except Exception:
        p1 = gdown_download(args.set1, Path("UUV_Set1.mat"))
        b1 = Path(p1).read_bytes()
        R1 = load_uuv_mat_from_bytes(
            b1,
            expected_N=meta["N"],
            expected_L=200,
            debug=args.debug_mat
        )

    # --------------------------------------------------------
    # Load Set2
    # --------------------------------------------------------
    try:
        b2 = (Path(args.set2).read_bytes() if Path(args.set2).is_file()
              else fetch_gdrive_mat_bytes(args.set2))
        R2 = load_uuv_mat_from_bytes(
            b2,
            expected_N=meta["N"],
            expected_L=200,
            debug=args.debug_mat
        )
    except Exception:
        p2 = gdown_download(args.set2, Path("UUV_Set2.mat"))
        b2 = Path(p2).read_bytes()
        R2 = load_uuv_mat_from_bytes(
            b2,
            expected_N=meta["N"],
            expected_L=200,
            debug=args.debug_mat
        )

    # --------------------------------------------------------
    # Run detector on both datasets
    # --------------------------------------------------------
    res1 = run_detector_on_dataset(
        R=R1,
        x=x,
        sigma2=args.sigma2,
        PFA=args.pfa,
        PFA_set=pfa_set,
        enr_db_for_true_pd=args.enr_db,
        out_prefix="Set1",
        out_dir=out_dir,
        show_inline=show_inline,
    )

    res2 = run_detector_on_dataset(
        R=R2,
        x=x,
        sigma2=args.sigma2,
        PFA=args.pfa,
        PFA_set=pfa_set,
        enr_db_for_true_pd=args.enr_db,
        out_prefix="Set2",
        out_dir=out_dir,
        show_inline=show_inline,
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------
    print("\n=== Results: Set1 ===")
    for k, v in res1.items():
        print(f"{k}: {v}")

    print("\n=== Results: Set2 ===")
    for k, v in res2.items():
        print(f"{k}: {v}")

    # --------------------------------------------------------
    # Choose an H0 dataset for empirical / bootstrap PFA analysis
    # --------------------------------------------------------
    if (not res1["mine_present_global"]) or (not res2["mine_present_global"]):
        res_h0 = res1 if (not res1["mine_present_global"]) else res2
        R_h0 = R1 if (not res1["mine_present_global"]) else R2

        print("\n=== Empirical PFA estimate from the H0 dataset (per-hydrophone) ===")
        print(f"PFA_selected_per_hydrophone: {args.pfa}")
        print(f"PFA_hat (fraction above gamma): {res_h0['fraction_above_gamma']}")

        boot = bootstrap_pfa_estimate_from_noise_only_samples(
            R_h0,
            x,
            sigma2=args.sigma2,
            gamma=res_h0["gamma_per_hydrophone"],
            B=20000,
        )

        print("\n=== Bootstrap PFA estimate (noise-only samples where x[n]=0) ===")
        for k, v in boot.items():
            print(f"{k}: {v}")

    else:
        print("\nBoth datasets look like mine-present under the global test.")
        print("Estimating PFA via noise-only bootstrap from Set1 (per-hydrophone threshold).")

        R_h0 = R1
        res_h0 = res1

        boot = bootstrap_pfa_estimate_from_noise_only_samples(
            R1,
            x,
            sigma2=args.sigma2,
            gamma=res1["gamma_per_hydrophone"],
            B=20000,
        )

        for k, v in boot.items():
            print(f"{k}: {v}")

    # --------------------------------------------------------
    # Part (c): PFA convergence plot
    # --------------------------------------------------------
    print("\n=== Part (c): PFA convergence plot ===")
    print("Template N =", meta["N"], " L expected =", 200)
    print("Selected PFA =", args.pfa)
    print("Computed gamma =", res_h0["gamma_per_hydrophone"])

    out_path = save_pfa_convergence_plot(
        R_h0=R_h0,
        x=x,
        sigma2=args.sigma2,
        pfa_selected=args.pfa,
        gamma=res_h0["gamma_per_hydrophone"],
        out_dir=out_dir,
        show_inline=show_inline,
    )

    print(f"Saved: {out_path.resolve()}")
    print(f"\nSaved plot PNGs to: {out_dir.resolve()}")

    return 0


if __name__ == "__main__":
    main()
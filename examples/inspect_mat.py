"""Inspect a local MAT file with the original UUV loader; no downloads."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uuv_loader import load_uuv_mat_from_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mat_file", type=Path)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--channels", type=int, required=True)
    args = parser.parse_args()
    if args.samples <= 1 or args.channels <= 1:
        parser.error("samples and channels must both exceed one")
    matrix = load_uuv_mat_from_bytes(
        args.mat_file.read_bytes(), expected_N=args.samples, expected_L=args.channels
    )
    print(f"Shape (samples, channels): {matrix.shape}")
    print(f"Data type: {matrix.dtype}")
    print(f"All values finite: {bool(np.isfinite(matrix).all())}")
    print(f"Mean: {float(matrix.mean()):.6g}")
    print(f"Standard deviation: {float(matrix.std()):.6g}")


if __name__ == "__main__":
    main()

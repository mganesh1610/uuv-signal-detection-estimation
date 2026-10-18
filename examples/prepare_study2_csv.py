"""Convert local Study 2 MAT arrays to the CSV inputs used by the recovered scripts."""
import argparse
import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mat_files", type=Path, nargs="+")
    parser.add_argument("--out_dir", type=Path, default=Path("data/study2"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for source in args.mat_files:
        match = re.fullmatch(r"Threat_Detection_Set([1-4])\.mat", source.name)
        if not match:
            parser.error(f"Unexpected Study 2 input filename: {source.name}")
        set_number = int(match.group(1))
        expected = (2101 if set_number <= 2 else 2001, 40)
        candidates = []
        for name, value in loadmat(source).items():
            if name.startswith("__") or not isinstance(value, np.ndarray):
                continue
            if value.ndim != 2 or not np.issubdtype(value.dtype, np.number):
                continue
            if value.shape == expected:
                candidates.append(value)
            elif value.T.shape == expected:
                candidates.append(value.T)
        if len(candidates) != 1:
            raise ValueError(f"Expected one numeric array of shape {expected} in {source.name}")
        matrix = candidates[0]
        if np.iscomplexobj(matrix) or not np.isfinite(matrix).all():
            raise ValueError("Expected finite real-valued observations")
        destination = args.out_dir / (source.stem + ".csv")
        np.savetxt(destination, matrix, delimiter=",", fmt="%.18g")
        print(f"{destination}: {matrix.shape}")


if __name__ == "__main__":
    main()

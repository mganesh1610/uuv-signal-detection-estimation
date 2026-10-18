from __future__ import annotations

import re
import sys
import subprocess
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io

# ------------------------------------------------------------
# Default URLs for Problem 4 data
# ------------------------------------------------------------
SET3_URL = "data/study1/UUV_Set3.mat"
SET4_URL = "data/study1/UUV_Set4.mat"


def _extract_gdrive_file_id(url: str) -> str:
    url = url.strip()
    m = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    m = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    raise ValueError(f"Could not extract Google Drive file id from URL: {url}")


def _looks_like_mat(data: bytes) -> bool:
    head = data[:128]
    return (b"MATLAB" in head) and (len(data) > 256)


def _ensure_gdown():
    try:
        import gdown  # noqa: F401
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gdown"])


def fetch_gdrive_mat_bytes(url: str) -> bytes:
    import requests

    file_id = _extract_gdrive_file_id(url)
    session = requests.Session()

    base = "https://drive.google.com/uc"
    params = {"export": "download", "id": file_id}
    r = session.get(base, params=params, allow_redirects=True)

    def _is_html(resp) -> bool:
        ctype = (resp.headers.get("Content-Type") or "").lower()
        return ("text/html" in ctype) or (resp.text.lstrip().startswith("<"))

    if _is_html(r):
        token = None

        for k, v in session.cookies.items():
            if k.startswith("download_warning"):
                token = v
                break

        if token is None:
            html = r.text
            m = re.search(r"confirm=([0-9A-Za-z_]+)", html)
            if m:
                token = m.group(1)

        if token is not None:
            params = {"export": "download", "id": file_id, "confirm": token}
            r = session.get(base, params=params, allow_redirects=True)

    r.raise_for_status()
    data = r.content

    if not _looks_like_mat(data):
        head = data[:120]
        raise ValueError(
            "Google Drive did not return a .mat binary "
            "(likely returned HTML or a consent page).\n"
            f"First bytes: {head!r}\n"
            "Falling back to gdown is recommended."
        )

    return data


def load_uuv_mat_from_bytes(
    data: bytes,
    expected_N: int,
    expected_L: int,
    debug: bool = False
) -> np.ndarray:
    mat = scipy.io.loadmat(BytesIO(data), squeeze_me=True, struct_as_record=False)

    if debug:
        for k, v in mat.items():
            if k.startswith("__"):
                continue
            if isinstance(v, np.ndarray):
                print("[MAT]", k, v.shape, v.dtype)

    candidates = []
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray) and v.ndim == 2 and np.issubdtype(v.dtype, np.number):
            candidates.append((k, v))

    if not candidates:
        raise ValueError("No 2D numeric arrays found inside MAT file.")

    for k, arr in candidates:
        if arr.shape == (expected_N, expected_L):
            return arr.astype(float)
        if arr.shape == (expected_L, expected_N):
            return arr.T.astype(float)

    k, arr = max(candidates, key=lambda kv: kv[1].size)
    arr = arr.astype(float)

    if expected_N in arr.shape and expected_L in arr.shape:
        if arr.shape[0] == expected_N:
            return arr
        if arr.shape[1] == expected_N:
            return arr.T

    raise ValueError(
        f"Found numeric array '{k}' with shape {arr.shape}, "
        f"expected ({expected_N},{expected_L}) or transpose."
    )


def gdown_download(url: str, out_path: Path) -> Path:
    _ensure_gdown()
    import gdown

    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(url, str(out_path), quiet=True, fuzzy=True)

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"gdown download failed: {out_path}")

    return out_path


def _is_url(s: str) -> bool:
    s = s.strip().lower()
    return s.startswith("http://") or s.startswith("https://")


def load_dataset_mat_only(
    source: str,
    expected_N: int,
    expected_L: int,
    debug: bool = False,
    fallback_name: str = "UUV_tmp.mat"
) -> np.ndarray:
    src = source.strip()

    if _is_url(src):
        try:
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


def load_default_set3(expected_N: int, expected_L: int, debug: bool = False) -> np.ndarray:
    return load_dataset_mat_only(
        SET3_URL,
        expected_N=expected_N,
        expected_L=expected_L,
        debug=debug,
        fallback_name="UUV_Set3.mat",
    )


def load_default_set4(expected_N: int, expected_L: int, debug: bool = False) -> np.ndarray:
    return load_dataset_mat_only(
        SET4_URL,
        expected_N=expected_N,
        expected_L=expected_L,
        debug=debug,
        fallback_name="UUV_Set4.mat",
    )
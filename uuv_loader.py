from __future__ import annotations

import re
import sys
import subprocess
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io


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
            "Google Drive did not return a .mat binary (likely returned HTML/consent page).\n"
            f"First bytes: {head!r}\n"
            "Falling back to gdown is recommended."
        )

    return data


def load_uuv_mat_from_bytes(data: bytes, expected_N: int, expected_L: int, debug: bool = False) -> np.ndarray:
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
        raise ValueError("No 2D numeric arrays found inside MAT.")

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

    raise ValueError(f"Found '{k}' with shape {arr.shape}, expected ({expected_N},{expected_L}) or transpose.")


def gdown_download(url: str, out_path: Path) -> Path:
    _ensure_gdown()
    import gdown

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # quiet=True hides the download progress output
    gdown.download(url, str(out_path), quiet=True, fuzzy=True)

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"gdown download failed: {out_path}")
    return out_path
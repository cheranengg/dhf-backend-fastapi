# app/utils/cache_bootstrap.py
from __future__ import annotations
import os, tempfile, pathlib

HF_ENV_KEYS = ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE")

def _hf_suffix(path: str) -> str:
    # Ensure we always use a leaf "hf" directory to avoid writing to a parent
    p = path.rstrip("/")
    return p if p.endswith("/hf") else p + "/hf"

def _writable_dir(path: str) -> bool:
    try:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        test = pathlib.Path(path) / ".touch"
        test.write_text("ok", encoding="utf-8")
        test.unlink(missing_ok=True)
        return True
    except Exception:
        return False

def pick_hf_cache_dir() -> str:
    # 1) respect user-provided envs (but force the /hf leaf)
    candidates = []
    for k in HF_ENV_KEYS:
        v = os.environ.get(k)
        if v:
            candidates.append(_hf_suffix(v))

    # 2) our known-good fallbacks (in priority order)
    candidates += [
        "/workspace/.cache/hf",
        "/home/user/.cache/hf",
        "/data/.cache/hf",  # if persistent storage is enabled
        _hf_suffix(tempfile.gettempdir()),
    ]

    # 3) pick the first that is actually writable
    for c in candidates:
        if _writable_dir(c):
            for k in HF_ENV_KEYS:
                os.environ[k] = c
            return c

    # Last resort: the system temp dir (already ensured above)
    last = _hf_suffix(tempfile.gettempdir())
    for k in HF_ENV_KEYS:
        os.environ[k] = last
    return last

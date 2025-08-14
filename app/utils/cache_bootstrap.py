from __future__ import annotations
import os, tempfile, pathlib

HF_ENV_KEYS = ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE")

def _hf_leaf(path: str) -> str:
    p = path.rstrip("/")
    return p if p.endswith("/hf") else p + "/hf"

def _writable(path: str) -> bool:
    try:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        t = pathlib.Path(path) / ".touch"
        t.write_text("ok", encoding="utf-8")
        t.unlink(missing_ok=True)
        return True
    except Exception:
        return False

def pick_hf_cache_dir() -> str:
    # Normalize any provided envs first
    candidates = []
    for k in HF_ENV_KEYS:
        v = os.environ.get(k)
        if v:
            candidates.append(_hf_leaf(v))

    # Known-good fallbacks (priority order)
    candidates += [
        "/workspace/.cache/hf",
        "/home/user/.cache/hf",
        "/data/.cache/hf",
        _hf_leaf(tempfile.gettempdir()),
    ]

    for c in candidates:
        if _writable(c):
            for k in HF_ENV_KEYS:
                os.environ[k] = c
            return c

    # Last resort: tmp
    last = _hf_leaf(tempfile.gettempdir())
    for k in HF_ENV_KEYS:
        os.environ[k] = last
    return last

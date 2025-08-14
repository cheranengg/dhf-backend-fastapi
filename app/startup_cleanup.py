# startup_cleanup.py
import os, shutil, glob

def _remove(path: str):
    try:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.exists(path):
            os.remove(path)
    except Exception as e:
        print(f"[startup] could not remove {path}: {e}")

def _purge_hf_cache():
    cache_dir = (
        os.environ.get("HF_HOME")
        or os.environ.get("HF_HUB_CACHE")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or os.environ.get("TRANSFORMERS_CACHE")
        or "/tmp/hf"
    )
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except Exception:
        pass

    # Delete any partially downloaded model folders for our repos
    patterns = [
        os.path.join(cache_dir, "models--cheranengg--dhf-ha-merged*"),
        os.path.join(cache_dir, "models--cheranengg--dhf-dvp-merged*"),
        os.path.join(cache_dir, "models--cheranengg--dhf-tm-merged*"),
    ]
    for pat in patterns:
        for p in glob.glob(pat):
            _remove(p)

    # (Optional) nuke HF transfer temp files
    tmp_pat = os.path.join(cache_dir, "tmp*")
    for p in glob.glob(tmp_pat):
        _remove(p)

    print(f"[startup] HF cache cleaned under: {cache_dir}")

# Run immediately on import
_purge_hf_cache()

# startup_cleanup.py
import shutil
import os

def clear_model_cache():
    """
    Deletes the cached Hugging Face model folder for 'cheranengg/dhf-ha-merged'
    from the HF Spaces temporary directory.
    """
    cache_dir = "/tmp/hf/models--cheranengg--dhf-ha-merged"
    if os.path.exists(cache_dir):
        print(f"🧹 Removing old cached model at {cache_dir} ...")
        try:
            shutil.rmtree(cache_dir)
            print("✅ Cache cleared successfully.")
        except Exception as e:
            print(f"⚠️ Failed to clear cache: {e}")
    else:
        print(f"ℹ️ No cache folder found at {cache_dir}")

# This runs automatically on import
clear_model_cache()

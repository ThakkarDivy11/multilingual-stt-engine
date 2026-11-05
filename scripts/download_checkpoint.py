"""
Download and verify the production trilingual STT checkpoint.
Downloads model.safetensors, config.json, vocab.json, and KenLM language model
into multilingual-stt-engine/checkpoints/best_checkpoint/
"""

import os
from pathlib import Path
from huggingface_hub import hf_hub_download

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints" / "best_checkpoint"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

REPO_ID = "deeeeep04/stt-gujarati-hindi-10cer"

FILES = [
    "config.json",
    "vocab.json",
    "model.safetensors",
    "multilingual_3gram.arpa",
    "unigrams.txt"
]

def download_checkpoint():
    print(f"📥 Downloading verified trilingual ASR checkpoint from {REPO_ID} ...")
    print(f"📂 Destination: {CHECKPOINT_DIR}")

    for filename in FILES:
        dest_file = CHECKPOINT_DIR / filename
        if dest_file.exists() and dest_file.stat().st_size > 100:
            print(f"✓ {filename} already exists ({dest_file.stat().st_size / (1024*1024):.2f} MB). Skipping.")
            continue

        print(f"⬇️ Downloading {filename} ...")
        try:
            downloaded_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                local_dir=str(CHECKPOINT_DIR),
                local_dir_use_symlinks=False
            )
            print(f"✓ Downloaded {filename} -> {downloaded_path}")
        except Exception as e:
            print(f"⚠️ Error downloading {filename}: {e}")

    print("\n✓ Checkpoint verification completed.")

if __name__ == "__main__":
    download_checkpoint()


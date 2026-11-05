"""
Automated Deployment Utility for Hugging Face Spaces.
Deploys the complete Trilingual STT Studio container to https://huggingface.co/spaces/divy3108/multilingual-speech-to-text
"""

import os
import sys
import shutil
import tempfile
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo

SPACE_REPO_ID = "divy3108/multilingual-speech-to-text"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

SPACE_CARD = """---
title: Trilingual STT Studio (Gujarati • Hindi • English)
emoji: 🎙️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# 🎙️ Trilingual Speech AI Studio (Gujarati • Hindi • English)

High-Accuracy Automatic Speech Recognition Studio powered by our fine-tuned **Meta MMS-300M** acoustic backbone (`Wav2Vec2ForCTC`) with 250-Token Trilingual Map.

- **Gujarati (`gu`):** High phonetic accuracy on native speech
- **Hindi (`hi`):** Comprehensive Devanagari coverage
- **English (`en`):** Conversational Indian English recognition
- **Real-Time Factor:** ~0.15x (6.7x faster than real-time on CPU)
- **Model Checkpoint:** [`divy3108/multilingual-stt-model`](https://huggingface.co/divy3108/multilingual-stt-model)
"""

def deploy_space(token: str):
    api = HfApi(token=token)

    print(f"🚀 Ensuring Hugging Face Space exists: {SPACE_REPO_ID} (SDK: Docker) ...")
    try:
        create_repo(
            repo_id=SPACE_REPO_ID,
            repo_type="space",
            space_sdk="docker",
            token=token,
            exist_ok=True
        )
        print("✓ Space repository confirmed.")
    except Exception as e:
        print(f"Notice: {e}")

    # Build clean temporary deployment bundle
    with tempfile.TemporaryDirectory() as tmp_dir:
        deploy_dir = Path(tmp_dir)
        print(f"📦 Packaging deployment files into {deploy_dir} ...")

        # 1. Space README with metadata
        with open(deploy_dir / "README.md", "w", encoding="utf-8") as f:
            f.write(SPACE_CARD)

        # 2. Dockerfile & requirements.txt
        shutil.copy2(PROJECT_ROOT / "Dockerfile", deploy_dir / "Dockerfile")
        shutil.copy2(PROJECT_ROOT / "requirements.txt", deploy_dir / "requirements.txt")

        # 3. Source directories
        for folder in ["backend", "frontend", "datasets", "training"]:
            src_folder = PROJECT_ROOT / folder
            if src_folder.exists():
                shutil.copytree(src_folder, deploy_dir / folder, dirs_exist_ok=True)

        # 4. Checkpoints: copy config.json, vocab.json, and model weights
        ckpt_src = PROJECT_ROOT / "checkpoints" / "best_checkpoint"
        ckpt_dest = deploy_dir / "checkpoints" / "best_checkpoint"
        if ckpt_src.exists():
            shutil.copytree(ckpt_src, ckpt_dest, dirs_exist_ok=True)

        print(f"🚀 Uploading Space bundle to https://huggingface.co/spaces/{SPACE_REPO_ID} ...")
        api.upload_folder(
            folder_path=str(deploy_dir),
            repo_id=SPACE_REPO_ID,
            repo_type="space",
            token=token
        )

        print(f"\n🎉 DEPLOYMENT LAUNCHED!")
        print(f"🌐 Live Application URL: https://huggingface.co/spaces/{SPACE_REPO_ID}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, default=os.environ.get("HF_TOKEN"), help="Hugging Face Write Token")
    args = parser.parse_args()

    if not args.token:
        print("❌ Error: Hugging Face token required. Provide via --token or set HF_TOKEN environment variable.")
        sys.exit(1)

    deploy_space(args.token)

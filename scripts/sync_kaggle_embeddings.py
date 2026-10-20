"""Automated synchronization script for Kaggle GPU feature encoding and FAISS stores.

Monitors:
1. `namnguynnnn/encode-coco-karpathy-test-5k-clip-siglip`
2. `namnguynnnn/encode-newsseek-clip-siglip`

Downloads and unzips:
- `storage/coco/clip/` and `storage/coco/siglip/` (FAISS indexes + metadata.json)
- `storage/newsseek/clip/` and `storage/newsseek/siglip/`
- Full benchmark evaluation metrics JSON
"""
from __future__ import annotations

import os
import sys
import time
import json
import zipfile
import argparse
from pathlib import Path

# Force UTF-8 on Windows
os.environ["PYTHONUTF8"] = "1"

import kaggle

ROOT_DIR = Path(__file__).resolve().parent.parent

KERNELS = {
    "coco": {
        "slug": "namnguynnnn/encode-coco-karpathy-test-5k-clip-siglip",
        "zip_name": "coco_vector_stores.zip",
        "bm_name": "coco_benchmark_results.json",
        "dest_dir": ROOT_DIR / "storage" / "coco",
    },
    "newsseek": {
        "slug": "namnguynnnn/encode-newsseek-clip-siglip",
        "zip_name": "newsseek_vector_stores.zip",
        "bm_name": "newsseek_benchmark_results.json",
        "dest_dir": ROOT_DIR / "storage" / "newsseek",
    },
}


def check_status(slug: str) -> tuple[str, Optional[str]]:
    try:
        res = kaggle.api.kernels_status(slug)
        raw_stat = str(getattr(res, "status", ""))
        # Normalize status string
        if "RUNNING" in raw_stat:
            status = "RUNNING"
        elif "COMPLETE" in raw_stat:
            status = "COMPLETE"
        elif "QUEUED" in raw_stat:
            status = "QUEUED"
        elif "ERROR" in raw_stat:
            status = "ERROR"
        else:
            status = raw_stat

        fail_msg = getattr(res, "failure_message", None) or getattr(res, "failureMessage", None)
        return status, fail_msg
    except Exception as e:
        return "ERROR", str(e)


def sync_kernel_artifacts(dataset_key: str, cfg: dict) -> bool:
    slug = cfg["slug"]
    dest_dir = cfg["dest_dir"]
    scratch_dir = ROOT_DIR / "scratch" / f"kaggle_sync_{dataset_key}"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{dataset_key.upper()}] Downloading output artifacts from {slug}...")
    try:
        kaggle.api.kernels_output(slug, path=str(scratch_dir))
    except Exception as e:
        print(f"[{dataset_key.upper()}] Download failed: {e}")
        return False

    # Check for zip file
    zip_path = scratch_dir / cfg["zip_name"]
    if zip_path.exists():
        print(f"[{dataset_key.upper()}] Unzipping {zip_path.name} ({zip_path.stat().st_size / (1024**2):.2f} MB)...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(ROOT_DIR)
        print(f"[{dataset_key.upper()}] Successfully unpacked to {ROOT_DIR / 'storage' / dataset_key}!")
        return True

    # Alternatively unpack individual files
    clip_idx = scratch_dir / "storage" / dataset_key / "clip" / "index.faiss"
    if clip_idx.exists():
        import shutil
        shutil.copytree(scratch_dir / "storage" / dataset_key, dest_dir, dirs_exist_ok=True)
        print(f"[{dataset_key.upper()}] Copied storage directly to {dest_dir}!")
        return True

    print(f"[{dataset_key.upper()}] Note: Expected zip {cfg['zip_name']} not found yet in outputs.")
    return False


def main():
    parser = argparse.ArgumentParser(description="Kaggle Vector Store Synchronization")
    parser.add_argument("--watch", action="store_true", help="Poll continuously until complete")
    parser.add_argument("--interval", type=int, default=30, help="Poll interval in seconds")
    args = parser.parse_args()

    print("=" * 70)
    print("  KAGGLE GPU EMBEDDINGS & FAISS SYNC MONITOR")
    print("=" * 70)

    completed = set()

    while True:
        all_done = True
        for key, cfg in KERNELS.items():
            if key in completed:
                continue

            slug = cfg["slug"]
            status, fail_msg = check_status(slug)

            print(f"[{key.upper()}] Status: {status} ({slug})")
            if fail_msg:
                print(f"         Error details: {fail_msg}")

            if status == "COMPLETE":
                success = sync_kernel_artifacts(key, cfg)
                if success:
                    completed.add(key)
            elif status in ("RUNNING", "QUEUED"):
                all_done = False
            elif status == "ERROR":
                print(f"[{key.upper()}] Kernel failed. Check logs via kaggle.")

        if not args.watch or len(completed) == len(KERNELS) or all_done:
            break

        print(f"\nWaiting {args.interval}s before next check... (Ctrl+C to stop)")
        time.sleep(args.interval)

    print("\n[Done] Active storage status:")
    for key in KERNELS:
        c_dir = ROOT_DIR / "storage" / key / "clip"
        s_dir = ROOT_DIR / "storage" / key / "siglip"
        c_ok = (c_dir / "index.faiss").exists()
        s_ok = (s_dir / "index.faiss").exists()
        print(f"  • {key.upper()}: CLIP Index {'[OK]' if c_ok else '[MISSING]'} | SigLIP Index {'[OK]' if s_ok else '[MISSING]'}")


if __name__ == "__main__":
    main()

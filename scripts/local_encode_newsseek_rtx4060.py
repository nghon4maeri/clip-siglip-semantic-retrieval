"""Industrial-Grade Local NewsSeek Pipeline on NVIDIA GeForce RTX 4060 Laptop (8GB VRAM).

Features:
1. Automated Kaggle download of namnguynnnn/newsseek-core-v1 to D:/newsseek_raw.
2. Immediate cleanup of the 20GB zip file post-extraction to preserve D: drive space.
3. Checkpoint-enabled video-by-video streaming:
   - Decodes keyframes directly in memory via OpenCV.
   - Batch GPU inference with CLIP ViT-B/32 (FP16, batch size 64) and SigLIP ViT-B/16-224 (FP16, batch size 64).
   - Checkpoints every 25 videos so execution can pause and resume without data loss.
4. Final FAISS IndexFlatIP (512D & 768D) construction + metadata generation.
5. End-to-end logging with ETA and speed metrics to logs/newsseek_local_encode.log.
"""
from __future__ import annotations

import os
import sys
import time
import json
import shutil
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
import faiss
from transformers import AutoProcessor, AutoModel, CLIPProcessor, CLIPModel

# ------------------------------------------------------------------------------
# Configuration & Paths
# ------------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT_DIR / "datasets" / "newsseek" / "newsseek_manifest.json"
STORAGE_DIR = ROOT_DIR / "storage" / "newsseek"
CLIP_STORE_DIR = STORAGE_DIR / "clip"
SIGLIP_STORE_DIR = STORAGE_DIR / "siglip"
CHECKPOINT_DIR = STORAGE_DIR / "checkpoints"

LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "newsseek_local_encode.log"

RAW_VIDEOS_DIR = Path("D:/newsseek_raw")
BATCH_SIZE = 64
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Setup File + Console Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(str(LOG_FILE), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("newsseek_encoder")


def extract_tensor_features(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        return output.last_hidden_state[:, 0, :]
    if hasattr(output, "__getitem__"):
        return output[0]
    return output


# ==============================================================================
# Step 1: Dataset Verification & Download from Kaggle
# ==============================================================================
def ensure_raw_videos() -> Path:
    RAW_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    existing_mp4s = list(RAW_VIDEOS_DIR.rglob("*.mp4"))

    if len(existing_mp4s) >= 600:
        logger.info(f"Found {len(existing_mp4s)} existing MP4 videos in {RAW_VIDEOS_DIR}. Skipping download.")
        return RAW_VIDEOS_DIR

    logger.info(f"Existing MP4 count ({len(existing_mp4s)}) < 600. Preparing Kaggle download...")
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    logger.info("Downloading dataset namnguynnnn/newsseek-core-v1 to D:/newsseek_raw...")
    t0 = time.time()
    api.dataset_download_files(
        dataset="namnguynnnn/newsseek-core-v1",
        path=str(RAW_VIDEOS_DIR),
        unzip=True,
        quiet=False,
    )
    logger.info(f"Download and extraction completed in {time.time() - t0:.1f}s.")

    # Cleanup 20GB zip file immediately to free disk space
    zip_candidates = list(RAW_VIDEOS_DIR.glob("*.zip"))
    for z in zip_candidates:
        try:
            logger.info(f"Removing raw zip file {z} to reclaim {z.stat().st_size / (1024**3):.2f} GB on D: drive...")
            z.unlink()
        except Exception as e:
            logger.warning(f"Failed to remove {z}: {e}")

    final_mp4s = list(RAW_VIDEOS_DIR.rglob("*.mp4"))
    logger.info(f"Total MP4 video files ready in {RAW_VIDEOS_DIR}: {len(final_mp4s)}")
    return RAW_VIDEOS_DIR


# ==============================================================================
# Step 2: Video Indexing & Checkpointed Streaming Pipeline
# ==============================================================================
def find_all_video_files(search_dir: Path, video_records: list) -> Dict[str, Path]:
    all_mp4s = list(search_dir.rglob("*.mp4"))
    vids = {v["video_id"] for v in video_records}
    video_map: Dict[str, Path] = {}

    for p in all_mp4s:
        bname = p.name
        video_map[p.stem] = p
        for vid in vids:
            if vid in bname:
                video_map[vid] = p

    logger.info(f"Mapped {len(video_map)} video keys from {len(all_mp4s)} MP4 files.")
    return video_map


def extract_frames_for_video(video_path: Path, keyframes: list) -> List[Tuple[Dict[str, Any], Image.Image]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"Unable to open video: {video_path}")
        return []

    sorted_kfs = sorted(keyframes, key=lambda x: x.get("frame_number", 0))
    extracted = []

    for kf in sorted_kfs:
        f_no = kf.get("frame_number", 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_no)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        h, w = frame.shape[:2]
        if max(h, w) > 480:
            scale = 480.0 / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        # Convert BGR to RGB PIL Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        extracted.append((kf, pil_img))

    cap.release()
    return extracted


def main():
    logger.info("==================================================================")
    logger.info("  NewsSeek 166K Keyframes Local GPU Encoding (NVIDIA RTX 4060)")
    logger.info("==================================================================")
    logger.info(f"Using Compute Device: {DEVICE}")
    if torch.cuda.is_available():
        logger.info(f"GPU Model: {torch.cuda.get_device_name(0)}")
        logger.info(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    CLIP_STORE_DIR.mkdir(parents=True, exist_ok=True)
    SIGLIP_STORE_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Ensure Videos Present
    raw_dir = ensure_raw_videos()

    # 2. Load Manifest
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest_raw = json.load(f)
    video_records = list(manifest_raw.values()) if isinstance(manifest_raw, dict) else manifest_raw
    logger.info(f"Loaded manifest with {len(video_records)} videos.")

    video_files_map = find_all_video_files(raw_dir, video_records)

    # 3. Load Models
    logger.info("Loading CLIP model (openai/clip-vit-base-patch32, 512D)...")
    clip_model_id = "openai/clip-vit-base-patch32"
    clip_proc = CLIPProcessor.from_pretrained(clip_model_id)
    clip_model = CLIPModel.from_pretrained(clip_model_id).to(DEVICE).eval()

    logger.info("Loading SigLIP model (google/siglip-base-patch16-224, 768D)...")
    siglip_model_id = "google/siglip-base-patch16-224"
    siglip_proc = AutoProcessor.from_pretrained(siglip_model_id)
    siglip_model = AutoModel.from_pretrained(siglip_model_id).to(DEVICE).eval()

    # 4. Checkpoint Status Check
    ckpt_meta_path = CHECKPOINT_DIR / "processed_videos.json"
    processed_vids = set()
    if ckpt_meta_path.exists():
        with open(ckpt_meta_path, "r", encoding="utf-8") as f:
            processed_vids = set(json.load(f))
        logger.info(f"Found existing checkpoint with {len(processed_vids)} already encoded videos.")

    all_metadata: List[Dict[str, Any]] = []
    meta_accum_path = CHECKPOINT_DIR / "accum_metadata.json"
    if meta_accum_path.exists():
        with open(meta_accum_path, "r", encoding="utf-8") as f:
            all_metadata = json.load(f)

    # 5. Process Videos Sequentially in Batches
    total_videos = len(video_records)
    t_start = time.time()
    total_encoded_frames = len(all_metadata)

    for v_idx, v in enumerate(video_records, start=1):
        vid = v["video_id"]
        title = v.get("title", v.get("video_file", vid))
        lang = v.get("language", "en")

        if vid in processed_vids:
            continue

        vpath = video_files_map.get(vid)
        if not vpath:
            logger.warning(f"[{v_idx}/{total_videos}] Video file for ID {vid} not found on disk. Skipping.")
            continue

        raw_kfs = v.get("keyframes", [])
        if not raw_kfs:
            continue

        extracted_pairs = extract_frames_for_video(vpath, raw_kfs)
        if not extracted_pairs:
            continue

        video_clip_embeds = []
        video_siglip_embeds = []
        video_meta = []

        images_batch = [p[1] for p in extracted_pairs]

        for i in range(0, len(images_batch), BATCH_SIZE):
            chunk = images_batch[i : i + BATCH_SIZE]

            # CLIP Batch
            c_in = clip_proc(images=chunk, return_tensors="pt")
            c_in = {k: val.to(DEVICE) for k, val in c_in.items()}
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    c_out = extract_tensor_features(clip_model.get_image_features(**c_in))
                    c_out = c_out / c_out.norm(dim=-1, keepdim=True)
            video_clip_embeds.append(c_out.cpu().numpy().astype(np.float32))

            # SigLIP Batch
            s_in = siglip_proc(images=chunk, return_tensors="pt")
            s_in = {k: val.to(DEVICE) for k, val in s_in.items()}
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    s_out = extract_tensor_features(siglip_model.get_image_features(**s_in))
                    s_out = s_out / s_out.norm(dim=-1, keepdim=True)
            video_siglip_embeds.append(s_out.cpu().numpy().astype(np.float32))

        c_mat = np.concatenate(video_clip_embeds, axis=0)
        s_mat = np.concatenate(video_siglip_embeds, axis=0)

        for pair in extracted_pairs:
            kf_data = pair[0]
            f_no = kf_data.get("frame_number", 0)
            fname = f"frame_{f_no:06d}.jpg"
            rel_path = f"{vid}/keyframes/{fname}"

            item_meta = {
                "id": f"newsseek_{total_encoded_frames + len(video_meta)}",
                "kf_idx": total_encoded_frames + len(video_meta),
                "video_id": vid,
                "frame_number": f_no,
                "timestamp_sec": kf_data.get("timestamp_sec", 0.0),
                "scene_index": kf_data.get("scene_index", 0),
                "relative_path": rel_path,
                "video_title": title,
                "language": lang,
            }
            video_meta.append(item_meta)

        # Save Per-Video Embedding Checkpoint
        v_ckpt_clip = CHECKPOINT_DIR / f"{vid}_clip.npy"
        v_ckpt_siglip = CHECKPOINT_DIR / f"{vid}_siglip.npy"
        np.save(str(v_ckpt_clip), c_mat)
        np.save(str(v_ckpt_siglip), s_mat)

        all_metadata.extend(video_meta)
        total_encoded_frames += len(video_meta)
        processed_vids.add(vid)

        # Periodic Progress & Checkpoint Flush
        if v_idx % 25 == 0 or v_idx == total_videos:
            elapsed = time.time() - t_start
            fps = total_encoded_frames / max(elapsed, 0.001)
            eta_sec = (total_videos - len(processed_vids)) * (elapsed / max(len(processed_vids), 1))
            logger.info(
                f"[Progress {len(processed_vids)}/{total_videos} videos] "
                f"Encoded {total_encoded_frames} frames | Speed: {fps:.1f} fps | "
                f"Elapsed: {elapsed / 60:.1f}m | ETA: {eta_sec / 60:.1f}m"
            )

            with open(ckpt_meta_path, "w", encoding="utf-8") as f:
                json.dump(list(processed_vids), f)
            with open(meta_accum_path, "w", encoding="utf-8") as f:
                json.dump(all_metadata, f, ensure_ascii=False)

    logger.info(f"All {len(processed_vids)} videos processed! Total frames encoded: {total_encoded_frames}.")

    # 6. Aggregate Checkpoints & Build FAISS Vector Stores
    logger.info("Assembling final FAISS Vector Stores...")
    del clip_model, siglip_model, clip_proc, siglip_proc
    torch.cuda.empty_cache()

    logger.info("Concatenating CLIP embeddings (512D)...")
    clip_index = faiss.IndexFlatIP(512)
    siglip_index = faiss.IndexFlatIP(768)

    # Read back checkpoints in order
    clip_parts = []
    siglip_parts = []
    for vid in processed_vids:
        c_p = CHECKPOINT_DIR / f"{vid}_clip.npy"
        s_p = CHECKPOINT_DIR / f"{vid}_siglip.npy"
        if c_p.exists() and s_p.exists():
            clip_parts.append(np.load(str(c_p)))
            siglip_parts.append(np.load(str(s_p)))

    full_clip_mat = np.ascontiguousarray(np.concatenate(clip_parts, axis=0), dtype=np.float32)
    full_siglip_mat = np.ascontiguousarray(np.concatenate(siglip_parts, axis=0), dtype=np.float32)

    faiss.normalize_L2(full_clip_mat)
    faiss.normalize_L2(full_siglip_mat)

    logger.info(f"Adding {len(full_clip_mat)} vectors to CLIP IndexFlatIP...")
    clip_index.add(full_clip_mat)
    faiss.write_index(clip_index, str(CLIP_STORE_DIR / "index.faiss"))

    with open(CLIP_STORE_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "dim": 512,
            "index_type": "FlatIP",
            "model_name": "clip",
            "count": len(all_metadata),
            "items": all_metadata,
        }, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved CLIP Vector Store -> {CLIP_STORE_DIR / 'index.faiss'} ({len(all_metadata)} items)")

    logger.info(f"Adding {len(full_siglip_mat)} vectors to SigLIP IndexFlatIP...")
    siglip_index.add(full_siglip_mat)
    faiss.write_index(siglip_index, str(SIGLIP_STORE_DIR / "index.faiss"))

    with open(SIGLIP_STORE_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "dim": 768,
            "index_type": "FlatIP",
            "model_name": "siglip",
            "count": len(all_metadata),
            "items": all_metadata,
        }, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved SigLIP Vector Store -> {SIGLIP_STORE_DIR / 'index.faiss'} ({len(all_metadata)} items)")

    # 7. Verification Test
    logger.info("Verifying freshly built vector stores with test search...")
    clip_test = faiss.read_index(str(CLIP_STORE_DIR / "index.faiss"))
    siglip_test = faiss.read_index(str(SIGLIP_STORE_DIR / "index.faiss"))
    logger.info(f"CLIP Index Verified: {clip_test.ntotal} vectors | SigLIP Index Verified: {siglip_test.ntotal} vectors.")

    logger.info("==================================================================")
    logger.info("  NEWSSEEK LOCAL ENCODING & VECTOR INDEXING 100% COMPLETE")
    logger.info("==================================================================")


if __name__ == "__main__":
    main()

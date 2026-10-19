"""Colab A100 Full End-to-End Execution Script for NewsSeek Dataset.

Steps executed on Colab VM:
1. Setup environment and configure Kaggle credentials.
2. Download namnguynnnn/newsseek-core-v1 at datacenter speed.
3. Multi-threaded video decoding & scene-aware keyframe extraction (Tiered policy).
4. A100 GPU accelerated embedding extraction for CLIP & SigLIP (FP16/TF32 batch 128).
5. Full retrieval benchmark (Video-wise Max-Pooling, Recall@K, MRR, Modality Gap, Multilingual).
6. Artifact compression for instant local pull & Kaggle dataset publishing.
"""

import os
import sys
import time
import math
import json
import shutil
import zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoProcessor, AutoModel

# Directories
WORKSPACE = Path("/content")
DATASET_DIR = WORKSPACE / "newsseek_dataset"
PROCESSED_DIR = WORKSPACE / "newsseek_processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATASET_DIR / "metadata" / "videos.csv"
VIDEO_ROOT = DATASET_DIR / "videos"

ZIP_KEYFRAMES = WORKSPACE / "newsseek_keyframes_processed.zip"
BENCHMARK_JSON = WORKSPACE / "benchmark_results.json"
MANIFEST_JSON = PROCESSED_DIR / "newsseek_manifest.json"


# ==============================================================================
# 1. Step 1: Configure Kaggle Credentials & Download Dataset
# ==============================================================================
def setup_kaggle_credentials():
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    access_token_file = kaggle_dir / "access_token"
    token = os.environ.get("KAGGLE_API_TOKEN", "")
    if token:
        access_token_file.write_text(token.strip())
        os.chmod(str(access_token_file), 0o600)
        print("Configured Kaggle credentials from environment.")
    elif access_token_file.exists():
        print("Using existing Kaggle access token.")
    else:
        print("Note: Set KAGGLE_API_TOKEN environment variable or ~/.kaggle/access_token.")


def download_dataset():
    setup_kaggle_credentials()
    if not CSV_PATH.exists():
        print("[Step 1] Downloading namnguynnnn/newsseek-core-v1 from Kaggle...")
        os.system("kaggle datasets download -d namnguynnnn/newsseek-core-v1 --unzip -p /content/newsseek_dataset")
    else:
        print("[Step 1] Dataset already downloaded at", DATASET_DIR)


# ==============================================================================
# 2. Step 2: Scene Detection & Keyframe Extraction
# ==============================================================================
def keyframe_count_tiered(duration_s: float) -> int:
    if duration_s <= 3.0:
        return 1
    if duration_s <= 10.0:
        return 3
    return 5


def detect_scenes_fast(cap, fps: float, num_frames: int, threshold: float = 0.35):
    if num_frames <= 0:
        return [(0, 0)]
    step = max(1, int(fps * 0.25))
    prev_hist = None
    scene_start = 0
    max_frames = int(25.0 * fps)
    scenes = []

    for f_idx in range(0, num_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        small = cv2.resize(frame, (120, 68), interpolation=cv2.INTER_NEAREST)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [24, 24], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

        if prev_hist is not None:
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            if diff > threshold or (f_idx - scene_start) >= max_frames:
                scenes.append((scene_start, f_idx - 1))
                scene_start = f_idx
        prev_hist = hist

    if scene_start < num_frames:
        scenes.append((scene_start, num_frames - 1))
    return scenes if scenes else [(0, num_frames - 1)]


def select_keyframes(scenes, fps):
    selected = []
    for idx, (s_frame, e_frame) in enumerate(scenes):
        dur_frames = e_frame - s_frame + 1
        if dur_frames <= 0:
            continue
        count = min(dur_frames, keyframe_count_tiered(dur_frames / fps))
        pts = sorted({
            min(e_frame, max(s_frame, s_frame + round((i + 0.5) * dur_frames / count)))
            for i in range(count)
        })
        for p in pts:
            selected.append({"frame_number": int(p), "timestamp_sec": round(p / fps, 3), "scene_index": idx})
    return selected


def process_single_video(row_data):
    row, video_path, out_dir = row_data
    vid = str(row["video_id"]).strip()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 0 or math.isnan(fps):
        fps = 25.0
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = num_frames / fps if fps > 0 else 0.0

    scenes = detect_scenes_fast(cap, fps, num_frames)
    keyframes = select_keyframes(scenes, fps)

    video_folder = out_dir / vid
    kf_dir = video_folder / "keyframes"
    kf_dir.mkdir(parents=True, exist_ok=True)

    extracted = []
    for kf in keyframes:
        f_no = kf["frame_number"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_no)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        h, w = frame.shape[:2]
        if max(h, w) > 720:
            sc = 720 / max(h, w)
            frame = cv2.resize(frame, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)

        fname = f"frame_{f_no:06d}.jpg"
        cv2.imwrite(str(kf_dir / fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        extracted.append({
            "frame_number": f_no,
            "timestamp_sec": kf["timestamp_sec"],
            "scene_index": kf["scene_index"],
            "filename": fname,
            "relative_path": f"{vid}/keyframes/{fname}",
            "full_path": str(kf_dir / fname),
        })
    cap.release()

    meta = {
        "video_id": vid,
        "title": str(row.get("title", "")),
        "language": str(row.get("language", "en")).strip().lower(),
        "source_name": str(row.get("source_name", "")),
        "duration_seconds": dur,
        "num_scenes": len(scenes),
        "num_keyframes": len(extracted),
        "keyframes": extracted,
    }

    with open(video_folder / "keyframes.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return meta


# ==============================================================================
# 3. Step 3: GPU Batch Embeddings on A100
# ==============================================================================
def extract_embeddings_on_a100(model_id, all_keyframes, video_titles, device, batch_size=128):
    print(f"\n--> Loading {model_id} on {device} (A100 optimized)...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()

    from PIL import Image

    # 1. Encode Keyframes
    img_embeds = []
    t_start = time.time()
    with torch.no_grad():
        for i in range(0, len(all_keyframes), batch_size):
            batch_slice = all_keyframes[i : i + batch_size]
            pil_imgs = [Image.open(b["full_path"]).convert("RGB") for b in batch_slice]
            inputs = processor(images=pil_imgs, return_tensors="pt").to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                feats = model.get_image_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            img_embeds.append(feats.cpu().float())

    img_tensor = torch.cat(img_embeds, dim=0)
    img_time = time.time() - t_start
    print(f"    Encoded {len(img_tensor)} keyframes in {img_time:.2f}s ({len(img_tensor)/img_time:.1f} fps)!")

    # 2. Encode Unique Video Titles
    txt_embeds = []
    with torch.no_grad():
        for i in range(0, len(video_titles), batch_size):
            batch_titles = video_titles[i : i + batch_size]
            inputs = processor(text=batch_titles, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                feats = model.get_text_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            txt_embeds.append(feats.cpu().float())
    txt_tensor = torch.cat(txt_embeds, dim=0)

    del model, processor
    torch.cuda.empty_cache()
    return img_tensor, txt_tensor


# ==============================================================================
# 4. Step 4: Video-Wise Retrieval Benchmark Evaluation
# ==============================================================================
def evaluate_video_retrieval(img_embeds, txt_embeds, processed_videos):
    frame_sims = torch.matmul(img_embeds, txt_embeds.T)
    N_videos = len(processed_videos)
    video_sim_matrix = torch.zeros((N_videos, N_videos))

    kf_start = 0
    for v_idx, v in enumerate(processed_videos):
        n_kf = v["num_keyframes"]
        if n_kf > 0:
            v_kf_sims = frame_sims[kf_start : kf_start + n_kf]
            video_sim_matrix[v_idx] = v_kf_sims.max(dim=0).values
        kf_start += n_kf

    k_vals = [1, 5, 10]
    _, topk_v = video_sim_matrix.topk(max(k_vals), dim=0, largest=True)
    targets = torch.arange(N_videos).unsqueeze(0).expand_as(topk_v)
    matches = (topk_v == targets)

    res = {}
    for k in k_vals:
        res[f"R@{k}"] = round((matches[:k].any(dim=0).float().mean().item()) * 100.0, 2)

    # MRR
    _, full_sorted = video_sim_matrix.sort(dim=0, descending=True)
    ranks = (full_sorted == targets).nonzero(as_tuple=True)[0] + 1
    res["MRR"] = round((1.0 / ranks.float()).mean().item(), 4)

    res["mR"] = round(float(np.mean([res[f"R@{k}"] for k in k_vals])), 2)
    gap = torch.norm(img_embeds.mean(dim=0) - txt_embeds.mean(dim=0), p=2).item()
    res["modality_gap"] = round(float(gap), 4)

    return res, video_sim_matrix


# ==============================================================================
# Main Orchestration
# ==============================================================================
def main():
    print("=" * 70)
    print("  COLAB A100 FULL NEWSSEEK PIPELINE & BENCHMARK")
    print("=" * 70)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Step 1: Download
    download_dataset()

    # Step 2: Load CSV & parallel extract keyframes
    df = pd.read_csv(CSV_PATH)
    print(f"Total videos in CSV: {len(df)}")

    tasks = []
    for _, row in df.iterrows():
        vid = str(row["video_id"]).strip()
        local_rel = str(row.get("local_filename", ""))
        candidates = [
            VIDEO_ROOT / local_rel,
            VIDEO_ROOT / "videos" / Path(local_rel).name,
            VIDEO_ROOT / Path(local_rel).name,
            VIDEO_ROOT / f"{vid}.mp4",
        ]
        vpath = next((c for c in candidates if c.exists()), None)
        if vpath is None:
            m = list(VIDEO_ROOT.rglob(f"*{vid}*.mp4"))
            if m:
                vpath = m[0]
        if vpath is not None:
            tasks.append((row, vpath, PROCESSED_DIR))

    print(f"Found {len(tasks)} / {len(df)} video files. Starting multi-threaded extraction...")
    t0 = time.time()
    processed_videos = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for res in tqdm(executor.map(process_single_video, tasks), total=len(tasks), desc="Processing Videos"):
            if res is not None:
                processed_videos.append(res)
    elapsed_extract = time.time() - t0
    total_kf = sum(v["num_keyframes"] for v in processed_videos)
    print(f"Extraction done in {elapsed_extract:.1f}s. Extracted {total_kf} keyframes from {len(processed_videos)} videos.")

    # Save manifest
    manifest_data = {
        "dataset": "newsseek-core-v1",
        "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_videos": len(processed_videos),
        "total_keyframes": total_kf,
        "elapsed_seconds": round(elapsed_extract, 2),
        "videos": processed_videos,
    }
    with open(MANIFEST_JSON, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    # Flatten keyframes
    all_keyframes = []
    for v in processed_videos:
        for kf in v["keyframes"]:
            all_keyframes.append(kf)
    video_titles = [v["title"] for v in processed_videos]

    # Step 3: Embeddings on A100
    clip_img, clip_txt = extract_embeddings_on_a100("openai/clip-vit-base-patch32", all_keyframes, video_titles, device)
    siglip_img, siglip_txt = extract_embeddings_on_a100("google/siglip-base-patch16-224", all_keyframes, video_titles, device)

    # Save embeddings
    np.save(WORKSPACE / "clip_keyframes.npy", clip_img.numpy())
    np.save(WORKSPACE / "clip_titles.npy", clip_txt.numpy())
    np.save(WORKSPACE / "siglip_keyframes.npy", siglip_img.numpy())
    np.save(WORKSPACE / "siglip_titles.npy", siglip_txt.numpy())

    # Step 4: Benchmark
    clip_res, clip_mat = evaluate_video_retrieval(clip_img, clip_txt, processed_videos)
    siglip_res, siglip_mat = evaluate_video_retrieval(siglip_img, siglip_txt, processed_videos)

    # Multilingual Breakdown
    langs = df["language"].value_counts().index.tolist()
    multilingual = {}
    for lang in langs:
        indices = [i for i, v in enumerate(processed_videos) if v["language"] == lang]
        if len(indices) < 2:
            continue
        idx_t = torch.tensor(indices)
        sub_c = clip_mat[idx_t][:, idx_t]
        sub_s = siglip_mat[idx_t][:, idx_t]
        c_r1 = (sub_c.argmax(dim=0) == torch.arange(len(indices))).float().mean().item() * 100.0
        s_r1 = (sub_s.argmax(dim=0) == torch.arange(len(indices))).float().mean().item() * 100.0
        multilingual[lang] = {
            "num_videos": len(indices),
            "clip_R@1": round(c_r1, 2),
            "siglip_R@1": round(s_r1, 2),
        }

    benchmark_summary = {
        "dataset": "NewsSeek (Full 698 Videos)",
        "device": str(device),
        "total_videos": len(processed_videos),
        "total_keyframes": total_kf,
        "clip": clip_res,
        "siglip": siglip_res,
        "multilingual_breakdown": multilingual,
    }

    with open(BENCHMARK_JSON, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 75)
    print("📊 BENCHMARK SUMMARY (A100 RESULTS)")
    print("=" * 75)
    print(f"{'Metric':<25} | {'CLIP (ViT-B/32)':<20} | {'SigLIP (ViT-B/16)':<20}")
    print("-" * 75)
    print(f"{'Text-to-Video R@1':<25} | {clip_res['R@1']:<19.2f}% | {siglip_res['R@1']:<19.2f}%")
    print(f"{'Text-to-Video R@5':<25} | {clip_res['R@5']:<19.2f}% | {siglip_res['R@5']:<19.2f}%")
    print(f"{'Text-to-Video R@10':<25} | {clip_res['R@10']:<19.2f}% | {siglip_res['R@10']:<19.2f}%")
    print(f"{'MRR':<25} | {clip_res['MRR']:<20.4f} | {siglip_res['MRR']:<20.4f}")
    print(f"{'Modality Gap':<25} | {clip_res['modality_gap']:<20.4f} | {siglip_res['modality_gap']:<20.4f}")
    print("=" * 75)

    print("\n🌐 Multilingual R@1:")
    for lang, stats in multilingual.items():
        print(f"  [{lang.upper()}] ({stats['num_videos']} vids): CLIP={stats['clip_R@1']}% | SigLIP={stats['siglip_R@1']}%")

    # Step 5: Archive
    print(f"\nArchiving keyframes to {ZIP_KEYFRAMES} ...")
    shutil.make_archive(str(ZIP_KEYFRAMES.with_suffix("")), "zip", PROCESSED_DIR)
    print(f"Archive ready: {ZIP_KEYFRAMES} ({ZIP_KEYFRAMES.stat().st_size / (1024*1024):.1f} MB)")
    print("All tasks finished successfully on Colab A100!")


if __name__ == "__main__":
    main()

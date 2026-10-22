"""Colab A100 In-Memory Streaming Pipeline for NewsSeek Dataset.

Executes on Colab A100 GPU:
1. Environment configuration & Kaggle API credential setup.
2. Fast datacenter download of namnguynnnn/newsseek-core-v1 to /content/newsseek_raw.
3. In-memory sequential video decoding & keyframe extraction (0 temporary JPEG files written to disk).
4. GPU batch inference for CLIP ViT-B/32 (512D) and SigLIP ViT-B/16-224 (768D) with FP16 autocast.
5. Exact FAISS IndexFlatIP vector store construction + full metadata generation.
6. Video-Wise Max-Pooling Retrieval Benchmark (Recall@1/5/10, MRR, Modality Gap, Multilingual).
7. Compressed packaging into /content/newsseek_vector_stores.zip for instant local download.
"""
from __future__ import annotations

import os
import sys
import time
import math
import json
import gzip
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

# Force unbuffered output and UTF-8 encoding
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

# Colab Workspace Setup
WORKSPACE = Path("/content")
RAW_VIDEOS_DIR = WORKSPACE / "newsseek_raw"
RAW_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

OUT_DIR = WORKSPACE / "output"
STORAGE_DIR = OUT_DIR / "storage" / "newsseek"
CLIP_STORE_DIR = STORAGE_DIR / "clip"
SIGLIP_STORE_DIR = STORAGE_DIR / "siglip"
CLIP_STORE_DIR.mkdir(parents=True, exist_ok=True)
SIGLIP_STORE_DIR.mkdir(parents=True, exist_ok=True)

ZIP_OUTPUT_PATH = WORKSPACE / "newsseek_vector_stores.zip"
MANIFEST_GZ = WORKSPACE / "newsseek_manifest.json.gz"
MANIFEST_JSON = WORKSPACE / "newsseek_manifest.json"

BATCH_SIZE = 256


# ==============================================================================
# Helper: Tensor extraction for newer Transformers versions
# ==============================================================================
def extract_tensor_features(output: Any) -> Any:
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        return output.last_hidden_state[:, 0, :]
    if hasattr(output, "__getitem__") and not hasattr(output, "shape"):
        return output[0]
    return output


# ==============================================================================
# Step 1: Dependencies & Kaggle Download
# ==============================================================================
def setup_environment():
    print("=== [Colab Setup] Verifying System & Installing Packages ===")
    os.system("pip install -q faiss-cpu transformers opencv-python-headless pillow kaggle tqdm")

    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    kaggle_json = kaggle_dir / "kaggle.json"
    if not kaggle_json.exists():
        kaggle_json.write_text('{"username":"namnguynnnn","key":"KGAT_dad33ce94fb8b25d8a9ecd4ca26ce03f"}')
        os.chmod(str(kaggle_json), 0o600)
        print("Configured Kaggle credentials.")

    # Unpack manifest if gzipped
    if MANIFEST_GZ.exists() and not MANIFEST_JSON.exists():
        print(f"Decompressing {MANIFEST_GZ} -> {MANIFEST_JSON}...")
        with gzip.open(MANIFEST_GZ, "rb") as f_in, open(MANIFEST_JSON, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        print("Manifest ready.")


def download_dataset():
    existing_mp4s = list(RAW_VIDEOS_DIR.rglob("*.mp4"))
    if len(existing_mp4s) >= 600:
        print(f"Found {len(existing_mp4s)} MP4 videos already in {RAW_VIDEOS_DIR}. Skipping download.")
        return

    print("--- [Step 1] Downloading namnguynnnn/newsseek-core-v1 from Kaggle at Datacenter Speed ---")
    t0 = time.time()
    cmd = f"kaggle datasets download -d namnguynnnn/newsseek-core-v1 --unzip -p {RAW_VIDEOS_DIR}"
    ret = os.system(cmd)
    if ret != 0:
        print("Kaggle download failed or exited with error code:", ret)
    elapsed = time.time() - t0
    print(f"Download and extraction completed in {elapsed:.1f}s.")

    # Clean up raw zip files to save 20GB of disk space
    for z in RAW_VIDEOS_DIR.glob("*.zip"):
        try:
            print(f"Cleaning up {z.name} to free disk space...")
            z.unlink()
        except Exception:
            pass

    mp4s = list(RAW_VIDEOS_DIR.rglob("*.mp4"))
    print(f"Total MP4 videos ready: {len(mp4s)}")


# ==============================================================================
# Step 2: In-Memory Keyframe Extraction
# ==============================================================================
def find_all_video_files(search_dir: Path, video_records: list) -> Dict[str, Path]:
    all_mp4s = list(search_dir.rglob("*.mp4"))
    vids = {v["video_id"] for v in video_records}
    video_map = {}
    for p in all_mp4s:
        bname = p.name
        video_map[p.stem] = p
        for vid in vids:
            if vid in bname:
                video_map[vid] = p
    return video_map


def extract_video_frames_in_memory(video_path: Path, keyframes: list) -> List[Tuple[Dict[str, Any], Any]]:
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    sorted_kfs = sorted(keyframes, key=lambda x: x.get("frame_number", 0))
    extracted = []
    current_frame = 0

    for kf in sorted_kfs:
        target_f = kf.get("frame_number", 0)
        # Fast sequential seek: if target is within 15 frames, grab forward without full seek
        if 0 <= target_f - current_frame < 15:
            while current_frame < target_f:
                cap.grab()
                current_frame += 1
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_f)
            current_frame = target_f

        ret, frame = cap.read()
        current_frame += 1
        if not ret or frame is None:
            continue

        h, w = frame.shape[:2]
        if max(h, w) > 480:
            scale = 480.0 / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        extracted.append((kf, pil_img))

    cap.release()
    return extracted


# ==============================================================================
# Step 3: GPU Batch Inference & Streaming Embedding Generation
# ==============================================================================
def run_streaming_encoding():
    import numpy as np
    import torch
    import faiss
    from PIL import Image
    from transformers import AutoProcessor, AutoModel, CLIPProcessor, CLIPModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== [Step 2] Initializing Models on Device: {device} ===")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB VRAM)")

    # Load Manifest
    with open(MANIFEST_JSON, "r", encoding="utf-8") as f:
        manifest_raw = json.load(f)
    video_records = list(manifest_raw.values()) if isinstance(manifest_raw, dict) else manifest_raw
    print(f"Loaded manifest: {len(video_records)} videos.")

    video_files_map = find_all_video_files(RAW_VIDEOS_DIR, video_records)
    print(f"Mapped {len(video_files_map)} video files.")

    # Load CLIP
    print("\nLoading CLIP ViT-B/32 (openai/clip-vit-base-patch32, 512D)...")
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()

    # Load SigLIP
    print("Loading SigLIP ViT-B/16-224 (google/siglip-base-patch16-224, 768D)...")
    siglip_proc = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    siglip_model = AutoModel.from_pretrained("google/siglip-base-patch16-224").to(device).eval()

    all_metadata: List[Dict[str, Any]] = []
    clip_embeds_list: List[np.ndarray] = []
    siglip_embeds_list: List[np.ndarray] = []

    unique_titles = []
    unique_vids = []

    print("\n--- [Step 3] Streaming Video Keyframe Extraction & GPU Inference ---")
    t_start = time.time()
    total_videos = len(video_records)
    total_frames_encoded = 0

    for v_idx, v in enumerate(video_records, start=1):
        vid = v["video_id"]
        title = v.get("title", v.get("video_file", vid))
        lang = v.get("language", "en")
        unique_vids.append(vid)
        unique_titles.append(title)

        vpath = video_files_map.get(vid)
        if not vpath:
            continue

        raw_kfs = v.get("keyframes", [])
        if not raw_kfs:
            continue

        extracted_pairs = extract_video_frames_in_memory(vpath, raw_kfs)
        if not extracted_pairs:
            continue

        images = [p[1] for p in extracted_pairs]

        # Batch encode with CLIP & SigLIP
        v_clip_embeds = []
        v_siglip_embeds = []

        for i in range(0, len(images), BATCH_SIZE):
            chunk = images[i : i + BATCH_SIZE]

            # CLIP Inference
            c_in = clip_proc(images=chunk, return_tensors="pt")
            c_in = {k: val.to(device) for k, val in c_in.items()}
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    c_out = extract_tensor_features(clip_model.get_image_features(**c_in))
                    c_out = c_out / c_out.norm(dim=-1, keepdim=True)
            v_clip_embeds.append(c_out.cpu().float().numpy())

            # SigLIP Inference
            s_in = siglip_proc(images=chunk, return_tensors="pt")
            s_in = {k: val.to(device) for k, val in s_in.items()}
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    s_out = extract_tensor_features(siglip_model.get_image_features(**s_in))
                    s_out = s_out / s_out.norm(dim=-1, keepdim=True)
            v_siglip_embeds.append(s_out.cpu().float().numpy())

        v_clip_mat = np.concatenate(v_clip_embeds, axis=0)
        v_siglip_mat = np.concatenate(v_siglip_embeds, axis=0)
        clip_embeds_list.append(v_clip_mat)
        siglip_embeds_list.append(v_siglip_mat)

        for pair in extracted_pairs:
            kf_data = pair[0]
            f_no = kf_data.get("frame_number", 0)
            fname = f"frame_{f_no:06d}.jpg"
            rel_path = f"{vid}/keyframes/{fname}"

            item_meta = {
                "id": f"newsseek_{len(all_metadata)}",
                "kf_idx": len(all_metadata),
                "video_id": vid,
                "frame_number": f_no,
                "timestamp_sec": kf_data.get("timestamp_sec", 0.0),
                "scene_index": kf_data.get("scene_index", 0),
                "relative_path": rel_path,
                "video_title": title,
                "language": lang,
            }
            all_metadata.append(item_meta)

        total_frames_encoded += len(extracted_pairs)

        if v_idx % 25 == 0 or v_idx == total_videos:
            elapsed = time.time() - t_start
            fps = total_frames_encoded / max(elapsed, 0.001)
            eta_sec = (total_videos - v_idx) * (elapsed / v_idx)
            print(
                f"[{v_idx:3d}/{total_videos}] Encoded {total_frames_encoded} frames across {v_idx} videos | "
                f"Speed: {fps:.1f} fps | ETA: {eta_sec/60:.1f} min"
            )

    total_time = time.time() - t_start
    avg_fps = total_frames_encoded / max(total_time, 0.001)
    print(f"\nEncoding Completed in {total_time:.1f}s ({total_time/60:.1f} min). Average Speed: {avg_fps:.1f} fps.")

    # Consolidate Embeddings
    print("Consolidating full embedding matrices...")
    all_clip_matrix = np.concatenate(clip_embeds_list, axis=0).astype(np.float32)
    all_siglip_matrix = np.concatenate(siglip_embeds_list, axis=0).astype(np.float32)

    # Encode Video Query Titles
    print(f"\nEncoding {len(unique_titles)} video query titles with CLIP...")
    clip_txt_list = []
    with torch.inference_mode():
        for i in range(0, len(unique_titles), 128):
            batch_txt = unique_titles[i : i + 128]
            t_in = clip_proc(text=batch_txt, padding=True, truncation=True, return_tensors="pt").to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                t_out = extract_tensor_features(clip_model.get_text_features(**t_in))
                t_out = t_out / t_out.norm(dim=-1, keepdim=True)
            clip_txt_list.append(t_out.cpu().float().numpy())
    all_clip_txt_matrix = np.concatenate(clip_txt_list, axis=0).astype(np.float32)

    print(f"Encoding {len(unique_titles)} video query titles with SigLIP...")
    siglip_txt_list = []
    with torch.inference_mode():
        for i in range(0, len(unique_titles), 128):
            batch_txt = unique_titles[i : i + 128]
            t_in = siglip_proc(text=batch_txt, padding="max_length", truncation=True, return_tensors="pt").to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                t_out = extract_tensor_features(siglip_model.get_text_features(**t_in))
                t_out = t_out / t_out.norm(dim=-1, keepdim=True)
            siglip_txt_list.append(t_out.cpu().float().numpy())
    all_siglip_txt_matrix = np.concatenate(siglip_txt_list, axis=0).astype(np.float32)

    del clip_model, siglip_model
    torch.cuda.empty_cache()

    return all_metadata, video_records, all_clip_matrix, all_clip_txt_matrix, all_siglip_matrix, all_siglip_txt_matrix, total_time, avg_fps


# ==============================================================================
# Step 4: FAISS Vector Stores & Archive Packaging
# ==============================================================================
def build_and_package_stores(
    all_metadata, video_records, clip_mat, clip_txt_mat, siglip_mat, siglip_txt_mat, total_time, avg_fps
):
    import faiss
    import torch
    import numpy as np

    print("\n--- [Step 4] Building FAISS IndexFlatIP Vector Stores ---")

    # 4.1 CLIP Vector Store (512D)
    clip_index = faiss.IndexFlatIP(512)
    faiss.normalize_L2(clip_mat)
    clip_index.add(clip_mat)
    faiss.write_index(clip_index, str(CLIP_STORE_DIR / "index.faiss"))

    with open(CLIP_STORE_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "dim": 512,
                "index_type": "FlatIP",
                "model_name": "clip",
                "count": len(all_metadata),
                "items": all_metadata,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved CLIP Vector Store: {clip_index.ntotal} vectors -> {CLIP_STORE_DIR}")

    # 4.2 SigLIP Vector Store (768D)
    siglip_index = faiss.IndexFlatIP(768)
    faiss.normalize_L2(siglip_mat)
    siglip_index.add(siglip_mat)
    faiss.write_index(siglip_index, str(SIGLIP_STORE_DIR / "index.faiss"))

    with open(SIGLIP_STORE_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "dim": 768,
                "index_type": "FlatIP",
                "model_name": "siglip",
                "count": len(all_metadata),
                "items": all_metadata,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved SigLIP Vector Store: {siglip_index.ntotal} vectors -> {SIGLIP_STORE_DIR}")

    # ==============================================================================
    # Step 5: Benchmark Evaluation (Video-Wise Max-Pooling)
    # ==============================================================================
    print("\n--- [Step 5] Evaluating Video-Wise Max-Pooling Retrieval ---")

    def evaluate_retrieval(kf_mat, txt_mat, records):
        kf_t = torch.from_numpy(kf_mat).cuda()
        txt_t = torch.from_numpy(txt_mat).cuda()
        frame_sims = torch.matmul(kf_t, txt_t.T).cpu()
        N_vid = len(records)
        video_sim_matrix = torch.zeros((N_vid, N_vid))

        kf_start = 0
        for v_idx, v in enumerate(records):
            n_kf = len(v.get("keyframes", []))
            if n_kf > 0 and (kf_start + n_kf) <= len(frame_sims):
                v_kf_sims = frame_sims[kf_start : kf_start + n_kf]
                video_sim_matrix[v_idx] = v_kf_sims.max(dim=0).values
            kf_start += n_kf

        k_vals = [1, 5, 10]
        top_max = max(k_vals)
        _, topk_preds = video_sim_matrix.topk(top_max, dim=0, largest=True)
        targets = torch.arange(N_vid).unsqueeze(0).expand_as(topk_preds)
        matches = topk_preds == targets

        res = {}
        for k in k_vals:
            res[f"R@{k}"] = round(float((matches[:k].any(dim=0).float().mean().item()) * 100.0), 2)

        _, full_sorted = video_sim_matrix.sort(dim=0, descending=True)
        targets_full = torch.arange(N_vid).unsqueeze(0).expand_as(full_sorted)
        match_coords = (full_sorted == targets_full).nonzero(as_tuple=False)
        ranks = match_coords[:, 0] + 1
        res["MRR"] = round(float((1.0 / ranks.float()).mean().item()), 4)
        res["mR"] = round(float(np.mean([res[f"R@{k}"] for k in k_vals])), 2)

        gap = torch.norm(kf_t.mean(dim=0) - txt_t.mean(dim=0), p=2).item()
        res["modality_gap"] = round(float(gap), 4)

        langs = [v.get("language", "en").lower() for v in records]
        en_idx = [i for i, l in enumerate(langs) if l == "en"]
        non_en_idx = [i for i, l in enumerate(langs) if l != "en"]
        if en_idx:
            res["R@1_EN"] = round(float((matches[0, en_idx].float().mean().item()) * 100.0), 2)
        if non_en_idx:
            res["R@1_NonEN"] = round(float((matches[0, non_en_idx].float().mean().item()) * 100.0), 2)

        return res

    clip_bm = evaluate_retrieval(clip_mat, clip_txt_mat, video_records)
    clip_bm["kf_enc_time_sec"] = round(total_time, 2)
    clip_bm["throughput_fps"] = round(avg_fps, 1)

    siglip_bm = evaluate_retrieval(siglip_mat, siglip_txt_mat, video_records)
    siglip_bm["kf_enc_time_sec"] = round(total_time, 2)
    siglip_bm["throughput_fps"] = round(avg_fps, 1)

    bm_payload = {
        "dataset": "newsseek",
        "total_videos": len(video_records),
        "total_keyframes": len(all_metadata),
        "clip_vit_b32": clip_bm,
        "siglip_vit_b16_224": siglip_bm,
    }

    bm_path = OUT_DIR / "newsseek_benchmark_results.json"
    with open(bm_path, "w", encoding="utf-8") as f:
        json.dump(bm_payload, f, indent=2)

    print("\n=== NewsSeek Benchmark Summary ===")
    print(json.dumps(bm_payload, indent=2))

    # Package ZIP archive
    print(f"\n--- [Step 6] Compressing Vector Stores -> {ZIP_OUTPUT_PATH} ---")
    with zipfile.ZipFile(ZIP_OUTPUT_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(CLIP_STORE_DIR / "index.faiss", "storage/newsseek/clip/index.faiss")
        zf.write(CLIP_STORE_DIR / "metadata.json", "storage/newsseek/clip/metadata.json")
        zf.write(SIGLIP_STORE_DIR / "index.faiss", "storage/newsseek/siglip/index.faiss")
        zf.write(SIGLIP_STORE_DIR / "metadata.json", "storage/newsseek/siglip/metadata.json")
        zf.write(bm_path, "newsseek_benchmark_results.json")

    size_mb = ZIP_OUTPUT_PATH.stat().st_size / (1024**2)
    print(f"Archive Created Successfully: {ZIP_OUTPUT_PATH} ({size_mb:.2f} MB)")
    print("=== NEWSSEEK PIPELINE COMPLETE ===")


def main():
    setup_environment()
    download_dataset()
    (
        all_metadata,
        video_records,
        clip_mat,
        clip_txt_mat,
        siglip_mat,
        siglip_txt_mat,
        total_time,
        avg_fps,
    ) = run_streaming_encoding()
    build_and_package_stores(
        all_metadata,
        video_records,
        clip_mat,
        clip_txt_mat,
        siglip_mat,
        siglip_txt_mat,
        total_time,
        avg_fps,
    )


if __name__ == "__main__":
    main()

"""Self-contained Kaggle script for processing the entire NewsSeek dataset (698 videos).
Can be executed directly on Kaggle with `namnguynnnn/newsseek-core-v1` attached.

Workflow:
1. Discovers all 698 multilingual news videos in `/kaggle/input/newsseek-core-v1/videos/`.
2. Employs neural (TransNetV2) or content-aware scene detection with Tiered keyframe selection.
3. Automatically resizes to max 720p for optimal quality vs storage balance.
4. Generates structured `keyframes.json` per video and a consolidated `newsseek_manifest.json`.
5. Packages keyframes into a compressed ZIP archive ready for downloading or Kaggle Dataset creation.
"""

import os
import sys
import math
import json
import time
import shutil
import zipfile
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

import cv2
import pandas as pd
from tqdm import tqdm


# ==============================================================================
# Keyframe Selection Strategy
# ==============================================================================

def keyframe_count_tiered(duration_s: float) -> int:
    """Tiered duration-based selection:
    <= 3s  -> 1 keyframe
    <= 10s -> 3 keyframes
    > 10s  -> 5 keyframes
    """
    if duration_s <= 3.0:
        return 1
    if duration_s <= 10.0:
        return 3
    return 5


def select_keyframes(scenes: List[Tuple[int, int]], fps: float) -> List[Dict[str, Any]]:
    """Selects evenly spaced frame centers inside each scene."""
    if fps <= 0:
        fps = 25.0
    selected = []
    for scene_idx, (start_f, end_f) in enumerate(scenes):
        duration_frames = end_f - start_f + 1
        if duration_frames <= 0:
            continue
        duration_s = duration_frames / fps
        count = min(duration_frames, keyframe_count_tiered(duration_s))

        numbers = sorted({
            min(end_f, max(start_f, start_f + round((i + 0.5) * duration_frames / count)))
            for i in range(count)
        })
        for num in numbers:
            selected.append({
                "frame_number": int(num),
                "timestamp_sec": round(num / fps, 3),
                "scene_index": int(scene_idx)
            })
    return selected


# ==============================================================================
# Scene Detection (TransNetV2 with OpenCV fallback)
# ==============================================================================

def detect_scenes_transnetv2(video_path: Path, threshold: float = 0.5) -> Optional[List[Tuple[int, int]]]:
    """Tries neural shot boundary detection using TransNetV2 if installed."""
    try:
        from transnetv2_pytorch import TransNetV2
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = TransNetV2().to(device).eval()
        _, single_frame_preds, _ = model.predict_video(str(video_path))
        scenes = model.predictions_to_scenes(single_frame_preds, threshold=threshold)
        return [(int(s[0]), int(s[1])) for s in scenes]
    except Exception:
        return None


def detect_scenes_histogram(cap, fps: float, num_frames: int, threshold: float = 0.35) -> List[Tuple[int, int]]:
    """Fast histogram-based shot boundary detection."""
    if num_frames <= 0:
        return [(0, 0)]

    sample_step = max(1, int(fps * 0.2))  # Sample every 0.2s
    prev_hist = None
    current_scene_start = 0
    max_scene_frames = int(30.0 * fps)    # Split if scene > 30s
    scenes = []

    frame_idx = 0
    while frame_idx < num_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_NEAREST)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

        if prev_hist is not None:
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            in_scene = frame_idx - current_scene_start
            if diff > threshold or in_scene >= max_scene_frames:
                scenes.append((current_scene_start, frame_idx - 1))
                current_scene_start = frame_idx

        prev_hist = hist
        frame_idx += sample_step

    if current_scene_start < num_frames:
        scenes.append((current_scene_start, num_frames - 1))

    return scenes if scenes else [(0, num_frames - 1)]


def process_video_entry(
    video_path: Path,
    video_id: str,
    output_dir: Path,
    max_dimension: int = 720,
    jpeg_quality: int = 90,
    use_transnet: bool = True,
) -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 0 or math.isnan(fps):
        fps = 25.0
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = num_frames / fps if fps > 0 else 0.0

    scenes = None
    if use_transnet:
        scenes = detect_scenes_transnetv2(video_path)

    if scenes is None:
        scenes = detect_scenes_histogram(cap, fps, num_frames)

    raw_keyframes = select_keyframes(scenes, fps)

    video_out = output_dir / video_id
    kf_dir = video_out / "keyframes"
    kf_dir.mkdir(parents=True, exist_ok=True)

    extracted = []
    for kf in raw_keyframes:
        f_no = kf["frame_number"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_no)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        h, w = frame.shape[:2]
        if max_dimension > 0 and max(h, w) > max_dimension:
            scale = max_dimension / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        fname = f"frame_{f_no:06d}.jpg"
        cv2.imwrite(str(kf_dir / fname), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

        extracted.append({
            "frame_number": f_no,
            "timestamp_sec": kf["timestamp_sec"],
            "scene_index": kf["scene_index"],
            "filename": fname,
            "relative_path": f"{video_id}/keyframes/{fname}",
        })

    cap.release()

    meta = {
        "video_id": video_id,
        "source_filename": video_path.name,
        "fps": fps,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "duration_seconds": duration_s,
        "num_scenes": len(scenes),
        "strategy": "tiered",
        "num_keyframes": len(extracted),
        "keyframes": extracted,
    }

    with open(video_out / "keyframes.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return meta


# ==============================================================================
# Main Orchestration
# ==============================================================================

def main():
    print("=" * 60)
    print("  Kaggle NewsSeek Video Keyframe Processing")
    print("=" * 60)

    is_kaggle = Path("/kaggle/input/newsseek-core-v1").exists()
    if is_kaggle:
        input_root = Path("/kaggle/input/newsseek-core-v1")
        csv_path = input_root / "metadata" / "videos.csv"
        video_root = input_root / "videos"
        output_dir = Path("/kaggle/working/newsseek_processed")
        zip_output = Path("/kaggle/working/newsseek_keyframes.zip")
    else:
        print("[Notice] Running in local/custom environment.")
        input_root = Path("datasets/newsseek")
        csv_path = input_root / "annotations" / "videos.csv"
        video_root = input_root / "videos"
        output_dir = input_root / "processed"
        zip_output = input_root / "newsseek_keyframes.zip"

    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"[Error] videos.csv not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"Total videos in dataset: {len(df)}")
    print(f"Language distribution:\n{df['language'].value_counts()}\n")

    records = []
    success, missing, failed = 0, 0, 0
    t0 = time.time()

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing NewsSeek"):
        vid = str(row["video_id"]).strip()
        title = str(row.get("title", "")).strip()
        lang = str(row.get("language", "en")).strip()
        local_rel = str(row.get("local_filename", "")).strip() if pd.notna(row.get("local_filename")) else ""

        # Find file
        candidates = [
            video_root / local_rel,
            video_root / "videos" / Path(local_rel).name,
            video_root / Path(local_rel).name,
            video_root / f"{vid}.mp4",
        ]
        video_path = None
        for c in candidates:
            if c.exists():
                video_path = c
                break
        if video_path is None:
            matches = list(video_root.rglob(f"*{vid}*.mp4"))
            if matches:
                video_path = matches[0]

        if video_path is None or not video_path.exists():
            missing += 1
            continue

        try:
            meta = process_video_entry(video_path, vid, output_dir)
            meta["title"] = title
            meta["language"] = lang
            meta["source_name"] = str(row.get("source_name", ""))
            meta["categories"] = str(row.get("categories", ""))
            records.append(meta)
            success += 1
        except Exception as e:
            print(f"Error on {vid}: {e}")
            failed += 1

    elapsed = time.time() - t0

    manifest = {
        "dataset": "newsseek-core-v1",
        "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_videos": len(df),
        "successfully_processed": success,
        "missing_videos": missing,
        "failed_videos": failed,
        "total_keyframes": sum(r["num_keyframes"] for r in records),
        "time_elapsed_seconds": round(elapsed, 2),
        "videos": records,
    }

    manifest_path = output_dir / "newsseek_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"Finished! Processed {success} videos in {elapsed:.1f}s.")
    print(f"Manifest written to: {manifest_path}")

    # Zip results for easy download from Kaggle output
    if is_kaggle and success > 0:
        print(f"Archiving keyframes to {zip_output} ...")
        shutil.make_archive(str(zip_output.with_suffix("")), "zip", output_dir)
        print(f"Archive created: {zip_output} ({zip_output.stat().st_size / (1024*1024):.1f} MB)")
    print("=" * 60)


if __name__ == "__main__":
    main()

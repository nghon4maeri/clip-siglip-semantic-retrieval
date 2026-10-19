"""Batch NewsSeek Video Keyframe Processing Pipeline.

Applies tiered (duration buckets) or linear keyframe selection strategies to NewsSeek multilingual news videos,
supporting neural (TransNetV2) or content-aware scene detection, saving representative keyframes,
and generating structured keyframes.json and a master newsseek_processed_manifest.json for retrieval benchmarking.
"""
from __future__ import annotations

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd
from tqdm import tqdm

# Ensure src is on python path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.video_pipeline.video_processor import VideoProcessor
from src.video_pipeline.keyframe_selection import KEYFRAME_STRATEGIES


def find_video_file(video_root: Path, local_rel: str, vid: str) -> Optional[Path]:
    """Resolves local video path from candidate relative paths."""
    candidates = [
        video_root / local_rel,
        video_root / "videos" / Path(local_rel).name,
        video_root / Path(local_rel).name,
        video_root / f"{vid}.mp4",
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c

    # Fallback glob match
    matches = list(video_root.rglob(f"*{vid}*.mp4"))
    if matches:
        return matches[0]

    return None


def run_pipeline(
    csv_path: str,
    video_root: str,
    output_dir: str,
    strategy: str = "tiered",
    keyframes_per_sec: float = 0.2,
    scene_mode: str = "auto",
    scene_window_sec: float = 10.0,
    languages: str = "all",
    max_videos: int = 0,
    max_dimension: int = 720,
    dry_run: bool = False,
):
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"Annotations CSV not found: {csv_file}")

    video_root_path = Path(video_root)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"============================================================")
    print(f"  NewsSeek Video Keyframe Processing Pipeline")
    print(f"============================================================")
    print(f"  Annotations CSV : {csv_file}")
    print(f"  Video Root      : {video_root_path}")
    print(f"  Output Directory: {out_path}")
    print(f"  Scene Mode      : {scene_mode}")
    print(f"  Strategy        : {strategy} (kps={keyframes_per_sec})")
    print(f"  Languages       : {languages}")
    print(f"  Max Dimension   : {max_dimension}p")
    print(f"  Dry Run         : {dry_run}")
    print(f"============================================================\n")

    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} total videos from CSV.")

    # Filter languages
    if languages.lower() != "all":
        target_langs = [l.strip().lower() for l in languages.split(",")]
        df = df[df["language"].str.lower().isin(target_langs)].copy()
        print(f"Filtered to {len(df)} videos for languages: {target_langs}")

    if max_videos > 0:
        df = df.head(max_videos).copy()
        print(f"Restricted to first {max_videos} videos.")

    lang_counts = df["language"].value_counts().to_dict()
    print(f"Language distribution: {lang_counts}")

    processor = VideoProcessor(
        strategy=strategy,
        keyframes_per_second=keyframes_per_sec,
        scene_mode=scene_mode,
        scene_window_seconds=scene_window_sec,
        max_dimension=max_dimension,
    )

    processed_records: List[Dict[str, Any]] = []
    success_count = 0
    missing_count = 0
    failed_count = 0
    total_keyframes_extracted = 0

    start_time = time.time()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Videos"):
        vid = str(row["video_id"]).strip()
        title = str(row["title"]).strip() if pd.notna(row["title"]) else ""
        lang = str(row["language"]).strip().lower() if pd.notna(row["language"]) else "en"
        source = str(row.get("source_name", "")).strip()
        local_rel = str(row.get("local_filename", "")).strip() if pd.notna(row.get("local_filename")) else ""
        duration_s = float(row.get("duration_seconds", 0.0)) if pd.notna(row.get("duration_seconds")) else 0.0

        video_path = find_video_file(video_root_path, local_rel, vid)

        if video_path is None or not video_path.exists():
            missing_count += 1
            continue

        try:
            if not dry_run:
                result = processor.process_video(
                    video_path=video_path,
                    output_dir=out_path,
                    video_id=vid,
                    save_images=True,
                )
                keyframes = result.get("keyframes", [])
            else:
                meta = processor.probe_video(video_path)
                scenes = processor.detect_scenes(video_path, meta["fps"], meta["num_frames"])
                from src.video_pipeline.keyframe_selection import select_keyframes
                keyframes = select_keyframes(
                    scenes,
                    meta["fps"],
                    strategy=strategy,
                    keyframes_per_second=keyframes_per_sec,
                )

            total_keyframes_extracted += len(keyframes)
            success_count += 1

            processed_records.append({
                "video_id": vid,
                "title": title,
                "language": lang,
                "source_name": source,
                "duration_seconds": duration_s,
                "num_keyframes": len(keyframes),
                "keyframes_json": f"{vid}/keyframes.json",
                "keyframes": keyframes,
            })

        except Exception as e:
            print(f"\n[ERROR] Failed processing video {vid}: {e}")
            failed_count += 1

    elapsed = time.time() - start_time

    # Save master manifest
    manifest_payload = {
        "dataset": "newsseek-core-v1",
        "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "strategy": strategy,
        "keyframes_per_second": keyframes_per_sec,
        "scene_mode": scene_mode,
        "total_videos_in_csv": len(df),
        "total_videos_processed": success_count,
        "total_videos_missing": missing_count,
        "total_videos_failed": failed_count,
        "total_keyframes_extracted": total_keyframes_extracted,
        "elapsed_seconds": round(elapsed, 2),
        "languages_processed": lang_counts,
        "videos": processed_records,
    }

    manifest_file = out_path / "newsseek_processed_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2, ensure_ascii=False)

    print(f"\n============================================================")
    print(f"  NewsSeek Processing Completed!")
    print(f"  - Successfully processed : {success_count} videos")
    print(f"  - Total keyframes saved   : {total_keyframes_extracted} frames")
    print(f"  - Average keyframes/video : {total_keyframes_extracted / max(1, success_count):.1f}")
    print(f"  - Missing video files     : {missing_count} videos")
    print(f"  - Failed during processing: {failed_count} videos")
    print(f"  - Total time elapsed      : {elapsed:.1f}s")
    print(f"  - Master Manifest saved to: {manifest_file}")
    print(f"============================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process NewsSeek videos with scene-aware keyframe pipeline.")
    parser.add_argument("--csv", type=str, default="datasets/newsseek/annotations/videos.csv",
                        help="Path to videos.csv annotations")
    parser.add_argument("--video_root", type=str, default="datasets/newsseek/videos",
                        help="Folder containing raw mp4 video files")
    parser.add_argument("--output_dir", type=str, default="datasets/newsseek/processed",
                        help="Folder to store extracted keyframes and metadata")
    parser.add_argument("--strategy", choices=KEYFRAME_STRATEGIES, default="tiered",
                        help="Keyframe selection strategy: 'tiered' or 'linear'")
    parser.add_argument("--keyframes_per_sec", type=float, default=0.2,
                        help="Sampling rate in frames/sec for 'linear' strategy (default 0.2, i.e. 1 frame every 5s)")
    parser.add_argument("--scene_mode", choices=["auto", "transnetv2", "content_aware", "fixed_window"], default="auto",
                        help="Scene detection engine: 'auto' (content-aware fallback), 'transnetv2', 'content_aware', or 'fixed_window'")
    parser.add_argument("--scene_window_sec", type=float, default=10.0,
                        help="Window length in seconds for scene segmentation fallback (default 10.0s)")
    parser.add_argument("--languages", type=str, default="all",
                        help="Filter languages, e.g. 'all', 'en', or 'en,vi,ja,ko'")
    parser.add_argument("--max_videos", type=int, default=0,
                        help="Max videos to process (0 = all in CSV)")
    parser.add_argument("--max_dim", type=int, default=720,
                        help="Max dimension to resize keyframes (default 720p)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Probe and calculate keyframe plan without saving images to disk")

    args = parser.parse_args()
    run_pipeline(
        csv_path=args.csv,
        video_root=args.video_root,
        output_dir=args.output_dir,
        strategy=args.strategy,
        keyframes_per_sec=args.keyframes_per_sec,
        scene_mode=args.scene_mode,
        scene_window_sec=args.scene_window_sec,
        languages=args.languages,
        max_videos=args.max_videos,
        max_dimension=args.max_dim,
        dry_run=args.dry_run,
    )

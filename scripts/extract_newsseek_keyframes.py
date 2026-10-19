"""
NewsSeek Keyframe Extractor Utility
Trích xuất khung hình đại diện (keyframe) từ các video tiếng Anh trong bộ dữ liệu NewsSeek.
Có thể chạy trực tiếp trên Kaggle hoặc trên máy local.
"""

import os
import sys
import argparse
import json
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def extract_keyframes(metadata_csv, video_root, output_dir, max_videos=None):
    output_path = Path(output_dir)
    keyframes_dir = output_path / "keyframes"
    keyframes_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading metadata: {metadata_csv}")
    df = pd.read_csv(metadata_csv)
    
    # Filter English videos
    en_df = df[df['language'] == 'en'].copy()
    print(f"Found {len(en_df)} English news videos.")
    
    if max_videos and max_videos > 0:
        en_df = en_df.head(max_videos)
        print(f"Limiting to first {max_videos} videos.")
        
    records = []
    success = 0
    fail = 0
    
    video_root_path = Path(video_root)
    
    for _, row in tqdm(en_df.iterrows(), total=len(en_df), desc="Extracting Keyframes"):
        vid = str(row['video_id']).strip()
        title = str(row['title']).strip()
        local_rel = str(row.get('local_filename', '')).strip() if pd.notna(row.get('local_filename')) else ""
        
        # Locate video
        video_path = None
        candidates = [
            video_root_path / local_rel,
            video_root_path / "videos" / Path(local_rel).name,
            video_root_path / Path(local_rel).name
        ]
        for c in candidates:
            if c.exists():
                video_path = c
                break
                
        if video_path is None:
            matches = list(video_root_path.rglob(f"*{vid}*.mp4"))
            if matches:
                video_path = matches[0]
                
        if video_path is None or not video_path.exists():
            fail += 1
            continue
            
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            fail += 1
            continue
            
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Take keyframe at 20-25% or around 10s to skip intros
        target_sec = min(12.0, duration * 0.25) if duration > 20 else duration / 2.0
        target_frame = int(target_sec * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            fail += 1
            continue
            
        # Optional resize to max 720p for fast retrieval
        h, w = frame.shape[:2]
        if max(h, w) > 1080:
            scale = 1080 / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            
        img_name = f"{vid}.jpg"
        img_out = keyframes_dir / img_name
        cv2.imwrite(str(img_out), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        
        records.append({
            "id": vid,
            "filename": img_name,
            "image_path": f"keyframes/{img_name}",
            "caption": title,
            "title": title,
            "source_name": str(row.get('source_name', '')),
            "duration_seconds": float(row.get('duration_seconds', 0.0)) if pd.notna(row.get('duration_seconds')) else 0.0,
            "categories": str(row.get('categories', 'News & Politics'))
        })
        success += 1

    metadata_out = output_path / "metadata.json"
    with open(metadata_out, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
        
    print(f"\nDone! Extracted: {success}, Skipped: {fail}. Output saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract NewsSeek Keyframes")
    parser.add_argument("--csv", type=str, default="datasets/newsseek/annotations/videos.csv",
                        help="Path to videos.csv")
    parser.add_argument("--video_root", type=str, default="datasets/newsseek/videos",
                        help="Root folder containing raw mp4 files")
    parser.add_argument("--output", type=str, default="datasets/newsseek/images",
                        help="Folder to save extracted keyframes and metadata.json")
    parser.add_argument("--max_videos", type=int, default=0,
                        help="Max videos to process (0 = all)")
    args = parser.parse_args()
    
    extract_keyframes(args.csv, args.video_root, args.output, args.max_videos)

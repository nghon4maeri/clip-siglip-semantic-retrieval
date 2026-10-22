import sys
import json
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

with open("storage/newsseek/clip/metadata.json", encoding="utf-8") as f:
    d = json.load(f)

items = d["items"]
videos = {}
for it in items:
    vid = it.get("video_id")
    if not vid:
        continue
    if vid not in videos:
        videos[vid] = {
            "video_id": vid,
            "title": it.get("video_title") or "",
            "frames": [],
        }
    videos[vid]["frames"].append({
        "kf_idx": it.get("kf_idx"),
        "frame_number": it.get("frame_number"),
        "timestamp_sec": it.get("timestamp_sec"),
    })

print(f"Total videos indexed: {len(videos)}")
print(f"Total keyframes indexed: {len(items)}")

# Print 30 sample videos with their titles and duration
print("\n=== SAMPLE 30 VIDEOS FROM NEWSSEEK ===")
for idx, (vid, info) in enumerate(list(videos.items())[:35], start=1):
    max_ts = max(f["timestamp_sec"] for f in info["frames"]) if info["frames"] else 0
    clean_title = info["title"].replace("\n", " ").strip()
    print(f"[{idx:02d}] {vid} | {len(info['frames']):3d} kfs | {max_ts:6.1f}s | {clean_title[:80]}")

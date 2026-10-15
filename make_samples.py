#!/usr/bin/env python3
"""Tạo bộ sample nhỏ (mặc định 5 ảnh test COCO) để smoke test nhanh.

Script này:
    1. Đảm bảo annotations đầy đủ (``dataset_coco.json``) đã có — tải nếu chưa.
    2. Chọn N ảnh đầu tiên thuộc split ``test``.
    3. Ghi ``datasets/coco/annotations/dataset_coco_sample.json``.
    4. Tải các ảnh tương ứng vào ``datasets/coco/images/val2014/``.

Nhờ vậy, bộ sample dùng để kiểm tra code chạy thông (end-to-end) mà không cần
tải toàn bộ ảnh COCO. Muốn benchmark thật thì xem ``datasets/README.md``.

Ví dụ:
    python make_samples.py
    python make_samples.py --num-images 10
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from datasets.download_annotations import DEFAULT_OUTPUT, ensure_annotations  # noqa: E402

SAMPLE_JSON = REPO_ROOT / "datasets" / "coco" / "annotations" / "dataset_coco_sample.json"
IMAGE_DIR = REPO_ROOT / "datasets" / "coco" / "images"
COCO_IMAGE_URL = "https://images.cocodataset.org/{filepath}/{filename}"


_WARNED_HTTP_FALLBACK = False


def download_image(url: str, dest: Path, retries: int = 3) -> bool:
    """Tải 1 ảnh, tự fallback sang http nếu https lỗi chứng chỉ.

    Lưu ý: host ``images.cocodataset.org`` trỏ tới S3 với chứng chỉ không khớp
    hostname, nên https không thể verify — đây là lỗi từ phía COCO. Script thử
    https trước, nếu thất bại sẽ thử http (chỉ để tải ảnh công khai phục vụ
    benchmark).
    """
    global _WARNED_HTTP_FALLBACK
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [bỏ qua] {dest.name} đã tồn tại")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    candidates = [url]
    if url.startswith("https://"):
        candidates.append("http://" + url[len("https://") :])

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        for candidate in candidates:
            try:
                with urllib.request.urlopen(candidate, timeout=60) as response:
                    dest.write_bytes(response.read())
                if candidate != url and not _WARNED_HTTP_FALLBACK:
                    print("  [cảnh báo] https lỗi chứng chỉ với host ảnh COCO, đã tự chuyển sang http")
                    _WARNED_HTTP_FALLBACK = True
                print(f"  [OK] {dest.name}")
                return True
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                dest.unlink(missing_ok=True)
        print(f"  [lỗi lần {attempt}/{retries}] {dest.name}: {last_error}")
    return False


def create_sample_dataset(num_images: int = 5) -> int:
    annotations_path = ensure_annotations(DEFAULT_OUTPUT)

    with open(annotations_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    test_images = [img for img in data["images"] if img["split"] == "test"][:num_images]
    if len(test_images) < num_images:
        print(f"[LỖI] Chỉ tìm được {len(test_images)} ảnh test, cần {num_images}.", file=sys.stderr)
        return 1

    SAMPLE_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(SAMPLE_JSON, "w", encoding="utf-8") as f:
        json.dump({"dataset": "coco", "images": test_images}, f, indent=4, ensure_ascii=False)
    print(f"\n[OK] Đã ghi {SAMPLE_JSON} ({len(test_images)} ảnh)")

    print(f"Đang tải ảnh vào {IMAGE_DIR}/...")
    failed = 0
    for img in test_images:
        filepath = img.get("filepath") or "val2014"
        url = COCO_IMAGE_URL.format(filepath=filepath, filename=img["filename"])
        if not download_image(url, IMAGE_DIR / filepath / img["filename"]):
            failed += 1

    if failed:
        print(f"\n[LỖI] {failed}/{len(test_images)} ảnh tải thất bại.", file=sys.stderr)
        return 1

    print(
        f"\n[OK] Sample sẵn sàng ({len(test_images)} ảnh). Chạy thử:\n"
        "    python scripts/verify_data.py --json-path datasets/coco/annotations/dataset_coco_sample.json --expect-images "
        f"{len(test_images)}\n"
        "    python src/evaluation/benchmark.py --json-path datasets/coco/annotations/dataset_coco_sample.json "
        f"--limit {len(test_images)}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Tạo sample dataset nhỏ từ Karpathy COCO splits.")
    parser.add_argument("--num-images", type=int, default=5, help="Số ảnh test lấy ra (mặc định: 5)")
    args = parser.parse_args()
    return create_sample_dataset(args.num_images)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Kiểm tra dữ liệu trước khi benchmark (không cần torch/GPU).

Việc kiểm tra này bắt buộc trên Kaggle: nếu path ảnh sai hoặc thiếu ảnh,
script sẽ báo lỗi rõ ràng thay vì âm thầm đánh giá trên tập thiếu.

Ví dụ:
    python scripts/verify_data.py                       # yêu cầu đủ 5000 ảnh test
    python scripts/verify_data.py --expect-images 5     # kiểm tra bộ sample
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.download_annotations import sha256_file  # noqa: E402
from src.datasets.paths import resolve_image_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json-path", default=str(REPO_ROOT / "datasets/coco/annotations/dataset_coco.json"))
    parser.add_argument("--image-dir", default=str(REPO_ROOT / "datasets/coco/images"))
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--expect-images",
        type=int,
        default=5000,
        help="Số ảnh kỳ vọng của split (0 = không kiểm tra; mặc định: 5000)",
    )
    parser.add_argument("--show-missing", type=int, default=5, help="In ra tối đa N ảnh bị thiếu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    json_path = Path(args.json_path)

    if not json_path.exists():
        print(f"[LỖI] Không tìm thấy annotations: {json_path}", file=sys.stderr)
        print("  Chạy: python datasets/download_annotations.py", file=sys.stderr)
        return 1

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = [img for img in data["images"] if img["split"] == args.split]
    if not images:
        print(f"[LỖI] Không có ảnh nào với split='{args.split}'", file=sys.stderr)
        return 1

    captions_per_image = [len(img.get("sentences", [])) for img in images]
    raw_captions = sum(captions_per_image)
    min_captions = min(captions_per_image)
    print(f"Annotations : {json_path}")
    print(f"  sha256    : {sha256_file(json_path)}")
    print(f"Split       : {args.split} — {len(images)} ảnh, {raw_captions} captions trong file")

    if args.expect_images and len(images) != args.expect_images:
        print(
            f"[LỖI] Split phải có {args.expect_images} ảnh theo chuẩn, nhưng có {len(images)} ảnh.",
            file=sys.stderr,
        )
        return 1

    if min_captions < 5:
        print(
            f"[LỖI] Có ảnh chỉ chứa {min_captions} captions (< 5). "
            "Protocol COCO cần đủ 5 captions/ảnh — file annotations có vấn đề.",
            file=sys.stderr,
        )
        return 1

    protocol_queries = len(images) * 5
    if raw_captions != protocol_queries:
        print(
            f"  (protocol lấy 5 captions đầu/ảnh => {protocol_queries} text query; "
            f"{raw_captions - protocol_queries} caption thừa trong file gốc sẽ bị bỏ qua)"
        )

    missing: list[str] = []
    for img in images:
        path = resolve_image_path(img, args.image_dir)
        if not Path(path).is_file():
            missing.append(path)

    if missing:
        print(f"[LỖI] Thiếu {len(missing)}/{len(images)} ảnh trong {args.image_dir}", file=sys.stderr)
        for path in missing[: args.show_missing]:
            print(f"   - {path}", file=sys.stderr)
        if len(missing) > args.show_missing:
            print(f"   ... và {len(missing) - args.show_missing} ảnh khác", file=sys.stderr)
        print("  Kiểm tra lại đường dẫn ảnh (xem datasets/README.md).", file=sys.stderr)
        return 1

    print(f"[OK] Đủ {len(images)}/{len(images)} ảnh cho split '{args.split}' — sẵn sàng benchmark.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

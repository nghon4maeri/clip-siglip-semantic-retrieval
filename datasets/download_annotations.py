#!/usr/bin/env python3
"""Tải annotations MS COCO theo Karpathy splits (nguồn chuẩn của đồ án).

Script này là nguồn duy nhất (single source of truth) để lấy file
``dataset_coco.json``. Cả CLIP và SigLIP bắt buộc dùng đúng file này, để kết
quả benchmark có thể so sánh được với nhau và với các công bố khoa học.

Nguồn dữ liệu:
    https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
    (Karpathy & Fei-Fei, "Deep Visual-Semantic Alignments for Generating
    Image Descriptions", CVPR 2015)

Quy trình:
    1. Tải file zip (có retry, có kiểm tra tính toàn vẹn của zip).
    2. Giải nén ``dataset_coco.json``.
    3. Kiểm tra split ``test`` đúng 5.000 ảnh / 25.000 captions.
    4. In SHA256 để ghi lại vào báo cáo (team dùng chung 1 bản).

Ví dụ:
    python datasets/download_annotations.py
    python datasets/download_annotations.py --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_URL = "https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"
DEFAULT_OUTPUT = REPO_ROOT / "datasets" / "coco" / "annotations" / "dataset_coco.json"
ZIP_MEMBER = "dataset_coco.json"

# Con số chuẩn của Karpathy split trên MS COCO (2014).
EXPECTED_TEST_IMAGES = 5000
# Protocol chuẩn: 5 captions/ảnh => 25.000 text query.
# Lưu ý: file gốc Karpathy có 10 ảnh trong test chứa 6 captions (25010 câu);
# theo protocol, chỉ lấy 5 captions đầu của mỗi ảnh.
EXPECTED_TEST_CAPTIONS = 25000
MIN_CAPTIONS_PER_IMAGE = 5

# SHA256 của bản dataset_coco.json chuẩn mà cả nhóm dùng (tải từ nguồn Karpathy ở trên).
# Script chỉ cảnh báo (không chặn) nếu hash lệch — khi đó cần kiểm tra lại nguồn
# để chắc chắn mọi người vẫn benchmark trên cùng một bản annotations.
KNOWN_SHA256 = "2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d"

CHUNK_SIZE = 1 << 20  # 1 MiB


def _progress(done: int, total: int, prefix: str) -> None:
    if total:
        pct = 100.0 * done / total
        sys.stdout.write(f"\r  {prefix}: {done / 1e6:7.1f}/{total / 1e6:.1f} MB ({pct:5.1f}%)")
    else:
        sys.stdout.write(f"\r  {prefix}: {done / 1e6:7.1f} MB")
    sys.stdout.flush()


def download_file(url: str, dest: Path, retries: int = 3) -> None:
    """Tải ``url`` về ``dest``, có retry và không để lại file dở dang."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                total = int(response.headers.get("Content-Length") or 0)
                done = 0
                with open(dest, "wb") as out:
                    while True:
                        chunk = response.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        out.write(chunk)
                        done += len(chunk)
                        _progress(done, total, f"Đang tải (lần {attempt}/{retries})")
            sys.stdout.write("\n")
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            dest.unlink(missing_ok=True)
            sys.stdout.write(f"\n  Lỗi tải (lần {attempt}/{retries}): {exc}\n")

    raise RuntimeError(f"Không tải được {url} sau {retries} lần thử") from last_error


def extract_member(zip_path: Path, member: str, dest: Path) -> None:
    """Giải nén 1 file từ zip, kiểm tra zip hợp lệ trước khi đọc.

    Đây chính là bước còn thiếu ở lần setup trước, dẫn tới việc commit nhầm
    một file zip tải dở (thiếu central directory, không thể unzip).
    """
    try:
        with zipfile.ZipFile(zip_path) as zf:
            bad = zf.testzip()
            if bad is not None:
                raise zipfile.BadZipFile(f"File hỏng bên trong zip: {bad}")
            if member not in zf.namelist():
                raise zipfile.BadZipFile(f"Không tìm thấy '{member}' trong {zip_path.name}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(dest, "wb") as out:
                while True:
                    chunk = src.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    out.write(chunk)
    except zipfile.BadZipFile as exc:
        raise RuntimeError(
            f"File zip không hợp lệ ({exc}). Nhiều khả năng file bị tải dở — "
            f"hãy xoá và chạy lại script này."
        ) from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_annotations(path: Path) -> dict:
    """Đọc và kiểm tra file annotations theo đúng chuẩn Karpathy/COCO.

    Trả về dict thống kê; raise ValueError nếu sai lệch so với con số chuẩn.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data.get("images")
    if not images:
        raise ValueError(f"{path} không có key 'images' hoặc rỗng")

    split_counts = Counter(img["split"] for img in images)
    test_images = [img for img in images if img["split"] == "test"]
    captions_per_image = [len(img.get("sentences", [])) for img in test_images]
    test_captions = sum(captions_per_image)

    if len(test_images) != EXPECTED_TEST_IMAGES:
        raise ValueError(
            f"Split 'test' phải có {EXPECTED_TEST_IMAGES} ảnh, nhưng có {len(test_images)}. "
            "File annotations không đúng chuẩn Karpathy."
        )
    too_few = [
        img["filename"] for img, n in zip(test_images, captions_per_image) if n < MIN_CAPTIONS_PER_IMAGE
    ]
    if too_few:
        raise ValueError(
            f"Có {len(too_few)} ảnh test ít hơn {MIN_CAPTIONS_PER_IMAGE} captions "
            f"(vd: {too_few[:3]}). File annotations bị thiếu hoặc sai nguồn."
        )
    if test_captions < EXPECTED_TEST_CAPTIONS:
        raise ValueError(
            f"Split 'test' phải có tối thiểu {EXPECTED_TEST_CAPTIONS} captions, nhưng có {test_captions}."
        )

    return {
        "total_images": len(images),
        "split_counts": dict(split_counts),
        "test_images": len(test_images),
        "test_captions": test_captions,
        "extra_captions": test_captions - EXPECTED_TEST_CAPTIONS,
        "sha256": sha256_file(path),
    }


def ensure_annotations(output: Path = DEFAULT_OUTPUT, force: bool = False, url: str = DEFAULT_URL) -> Path:
    """Đảm bảo ``output`` tồn tại và hợp lệ; tải về nếu cần. Trả về đường dẫn file."""
    if output.exists() and not force:
        try:
            stats = validate_annotations(output)
            print(f"[OK] Đã có file hợp lệ: {output}")
            print(f"     test: {stats['test_images']} ảnh / {stats['test_captions']} captions")
            print(f"     sha256: {stats['sha256']}")
            return output
        except (ValueError, json.JSONDecodeError) as exc:
            print(f"[!] File hiện có không hợp lệ ({exc}); sẽ tải lại.")

    with tempfile.TemporaryDirectory(prefix="coco_annotations_") as tmp_dir:
        tmp = Path(tmp_dir)
        zip_path = tmp / "caption_datasets.zip"
        json_path = tmp / "dataset_coco.json"

        print(f"Đang tải annotations từ:\n  {url}")
        download_file(url, zip_path)

        size_mb = zip_path.stat().st_size / 1e6
        print(f"Đang giải nén '{ZIP_MEMBER}' (zip {size_mb:.1f} MB)...")
        extract_member(zip_path, ZIP_MEMBER, json_path)

        print("Đang kiểm tra tính đúng đắn của annotations...")
        stats = validate_annotations(json_path)

        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(json_path), str(output))

    print(f"\n[OK] Đã lưu: {output}")
    print(f"     Tổng số ảnh: {stats['total_images']}")
    for split, count in sorted(stats["split_counts"].items()):
        print(f"       - {split:8s}: {count:6d} ảnh")
    print(f"     Split test: {stats['test_images']} ảnh / {stats['test_captions']} captions")
    if stats["extra_captions"]:
        print(
            f"       (protocol dùng 5 captions đầu/ảnh => {EXPECTED_TEST_CAPTIONS} text query; "
            f"{stats['extra_captions']} caption thừa trong file gốc sẽ bị bỏ qua)"
        )
    print(f"     SHA256    : {stats['sha256']}")
    if stats["sha256"] != KNOWN_SHA256:
        print(
            "\n[CẢNH BÁO] SHA256 không khớp bản chuẩn của nhóm "
            f"({KNOWN_SHA256}).\n           Kiểm tra lại nguồn tải trước khi dùng cho báo cáo!"
        )

    sha_path = output.with_suffix(output.suffix + ".sha256")
    sha_path.write_text(stats["sha256"] + "\n", encoding="utf-8")
    print(f"     (đã ghi hash vào {sha_path.name})")

    print(
        "\nBước tiếp theo:\n"
        "  1. Tạo sample nhỏ để smoke test : python make_samples.py\n"
        "  2. Chuẩn bị ảnh cho benchmark     : xem datasets/README.md\n"
        "  3. Kiểm tra dữ liệu đầy đủ       : python scripts/verify_data.py"
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tải annotations MS COCO theo Karpathy splits (dataset_coco.json)."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"Mặc định: {DEFAULT_OUTPUT}")
    parser.add_argument("--url", default=DEFAULT_URL, help="URL file zip annotations")
    parser.add_argument("--force", action="store_true", help="Tải lại kể cả khi file đã tồn tại")
    args = parser.parse_args()

    try:
        ensure_annotations(output=args.output, force=args.force, url=args.url)
    except (RuntimeError, ValueError) as exc:
        print(f"\n[LỖI] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

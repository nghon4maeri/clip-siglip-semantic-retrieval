#!/usr/bin/env python3
"""Tải riêng 5.000 ảnh của split test (Karpathy) — không cần tải cả `val2014.zip`.

COCO cho phép tải từng ảnh qua ``images.cocodataset.org/val2014/<filename>``,
nên chỉ cần đúng 5.000 ảnh của split ``test`` (~800 MB) thay vì 40.504 ảnh
(~6.6 GB). Ảnh tải về là bản gốc (đã đối chiếu SHA256 với ảnh mẫu trong repo).

Dùng được cả ở local và trên Kaggle (cần Internet).

Ví dụ:
    python scripts/download_test_images.py                     # đủ 5000 ảnh test
    python scripts/download_test_images.py --limit 20          # thử nhanh
    python scripts/download_test_images.py --out /kaggle/working/images
    python scripts/download_test_images.py --workers 16
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_JSON = REPO_ROOT / "datasets/coco/annotations/dataset_coco.json"
DEFAULT_OUT = REPO_ROOT / "datasets/coco/images/val2014"
BASE_URL = "http://images.cocodataset.org/val2014"

# Ảnh mẫu trong repo gốc: dùng để xác nhận ảnh tải về là bản gốc, không bị xử lý lại.
CHECKSUMS = {
    "COCO_val2014_000000060623.jpg": "2fcf19f0230097a736040aaa73807e9b",
    "COCO_val2014_000000384213.jpg": "c766951710193774b71200025b27bb8a",
    "COCO_val2014_000000386164.jpg": "71721bb44e14997403895b1d6b730465",
    "COCO_val2014_000000391895.jpg": "d0d62a2eed6433c8d65bae1c8dca849e",
    "COCO_val2014_000000483108.jpg": "5c71251326fb9ece01b5ce6334869861",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--json-path", default=str(DEFAULT_JSON), help="File annotations Karpathy")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Thư mục lưu ảnh")
    parser.add_argument("--split", default="test", help="Split cần tải (mặc định: test)")
    parser.add_argument("--limit", type=int, default=0, help="Chỉ tải N ảnh đầu (0 = tất cả)")
    parser.add_argument("--workers", type=int, default=8, help="Số luồng tải song song")
    parser.add_argument("--force", action="store_true", help="Tải lại cả ảnh đã có")
    parser.add_argument("--base-url", default=BASE_URL, help="URL gốc chứa ảnh")
    return parser.parse_args()


def download_one(url: str, dest: Path, retries: int = 3, timeout: int = 90) -> None:
    """Tải 1 ảnh; ghi ra file tạm rồi rename để không để lại file dở dang."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                data = resp.read()
            if not data:
                raise OSError("file rỗng")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
            tmp.unlink(missing_ok=True)
    raise RuntimeError(f"Tải thất bại sau {retries} lần: {url}") from last


def check_samples(out_dir: Path) -> bool:
    """Đối chiếu SHA256 các ảnh mẫu để xác nhận ảnh tải về là bản gốc."""
    import hashlib

    found = 0
    ok = True
    for name, expected in CHECKSUMS.items():
        path = out_dir / name
        if not path.exists():
            continue
        found += 1
        actual = hashlib.sha256(path.read_bytes()).hexdigest()[:32]
        match = actual == expected
        ok &= match
        print(f"  [{'OK ' if match else 'LỖI'}] {name} — {'khớp' if match else 'KHÁC'} bản gốc")
    if found == 0:
        print("  (không có ảnh mẫu nào trong tập vừa tải để đối chiếu)")
    return ok


def main() -> int:
    args = parse_args()

    json_path = Path(args.json_path)
    if not json_path.exists():
        print(f"[LỖI] Không tìm thấy annotations: {json_path}", file=sys.stderr)
        print("  Chạy trước: python datasets/download_annotations.py", file=sys.stderr)
        return 1

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = [img for img in data["images"] if img.get("split") == args.split]
    if not images:
        print(f"[LỖI] Không có ảnh nào với split='{args.split}'", file=sys.stderr)
        return 1
    if args.limit > 0:
        images = images[: args.limit]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Nếu thư mục đích là symlink (thường trỏ vào /kaggle/input — chỉ đọc), chỉ được
    # phép đọc ảnh đã có, không ghi thêm được. Báo lỗi rõ ràng thay vì fail giữa chừng.
    if out_dir.is_symlink():
        real = out_dir.resolve()
        print(f"Lưu ý: {out_dir} là symlink → {real}")

    todo: list[tuple[str, Path]] = []
    skipped = 0
    for img in images:
        dest = out_dir / img["filename"]
        if dest.exists() and not args.force:
            skipped += 1
            continue
        todo.append((f"{args.base_url}/{img['filename']}", dest))

    total = len(images)

    if todo and out_dir.is_symlink():
        print(
            f"\n[LỖI] Còn {len(todo)} ảnh cần tải nhưng {out_dir} là symlink "
            f"(→ {out_dir.resolve()}), thường là vùng chỉ đọc trên Kaggle.\n"
            "  Cách xử lý: xóa symlink rồi chạy lại, hoặc dùng dataset đã có đủ 5000 ảnh.\n"
            f"  Xóa bằng: rm {out_dir}",
            file=sys.stderr,
        )
        return 1

    print(f"Annotations : {json_path}")
    print(f"Split       : {args.split} — {total} ảnh")
    print(f"Lưu vào     : {out_dir}")
    print(f"Đã có sẵn   : {skipped}")
    print(f"Cần tải     : {len(todo)}")
    if not todo:
        print("\nKhông có gì phải tải.")
    else:
        print()

    done = 0
    failed: list[str] = []
    if todo:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {
                pool.submit(download_one, url, dest): dest.name for url, dest in todo
            }
            for future in as_completed(futures):
                name = futures[future]
                done += 1
                try:
                    future.result()
                except Exception as exc:  # noqa: BLE001 - báo cáo mọi lỗi tải
                    failed.append(f"{name}: {exc}")
                pct = done / len(todo) * 100
                sys.stdout.write(
                    f"\r  Tiến độ: {done}/{len(todo)} ({pct:5.1f}%) — lỗi: {len(failed)}"
                )
                sys.stdout.flush()
        print()

    if failed:
        print(f"\n[LỖI] {len(failed)} ảnh tải thất bại:", file=sys.stderr)
        for item in failed[:10]:
            print(f"   - {item}", file=sys.stderr)
        if len(failed) > 10:
            print(f"   ... và {len(failed) - 10} ảnh khác", file=sys.stderr)
        print("  Chạy lại script để tiếp tục (ảnh đã có sẽ được bỏ qua).", file=sys.stderr)

    present = sum(1 for img in images if (out_dir / img["filename"]).exists())
    print(f"\nKết quả     : {present}/{total} ảnh trong {out_dir}")
    print("Đối chiếu ảnh gốc:")
    samples_ok = check_samples(out_dir)

    if failed or present != total or not samples_ok:
        return 1
    print(f"\n[OK] Đủ {total} ảnh — sẵn sàng benchmark.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

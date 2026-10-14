#!/usr/bin/env python3
"""Benchmark CLIP vs SigLIP trên MS COCO (Karpathy test split).

Protocol chuẩn: 5.000 ảnh test, mỗi ảnh 5 captions => 25.000 captions.
Xem ``src/evaluation/metrics.py`` để biết định nghĩa Recall@K.

Ví dụ:
    # Benchmark đầy đủ (5000 ảnh, cả 2 model)
    python src/evaluation/benchmark.py

    # Chạy trên sample nhỏ (smoke test, không cần GPU)
    python src/evaluation/benchmark.py \
        --json-path datasets/coco/annotations/dataset_coco_sample.json \
        --limit 5 --models clip

Kết quả được lưu vào ``results/benchmark.json`` (đè bằng --output).
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from datasets.download_annotations import sha256_file  # noqa: E402
from src.datasets.dataset import get_dataloader  # noqa: E402
from src.evaluation.metrics import compute_recall_at_k  # noqa: E402
from src.models.clip_wrapper import CLIPWrapper  # noqa: E402
from src.models.siglip_wrapper import SigLIPWrapper  # noqa: E402

WRAPPERS = {"clip": CLIPWrapper, "siglip": SigLIPWrapper}
DEFAULT_CHECKPOINTS = {
    "clip": "openai/clip-vit-base-patch16-224",
    "siglip": "google/siglip-base-patch16-224",
}
PROTOCOL_DESCRIPTION = (
    "MS COCO Karpathy test split — 5 captions/ảnh (5000 ảnh, 25000 captions). "
    "i2t: mỗi ảnh xếp hạng toàn bộ captions, đúng nếu có >=1 caption của ảnh trong top-K. "
    "t2i: mỗi caption xếp hạng toàn bộ ảnh, đúng nếu ảnh gốc trong top-K. "
    "Recall@K = trung bình % trên toàn bộ query."
)


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def get_versions() -> dict:
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def get_device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return platform.processor() or device.type


def extract_embeddings(
    wrapper,
    dataloader,
    n_samples: int,
    text_batch_size: int,
    progress_prefix: str = "",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Trích xuất image/text embeddings cho ``n_samples`` ảnh đầu tiên.

    Trả về (image_embeds [N, D], text_embeds [M, D], image_ids [M], timing).
    ``image_ids[j]`` là chỉ số hàng của ảnh mà caption j thuộc về.
    """
    device = wrapper.device

    all_image_embeds: list[torch.Tensor] = []
    all_captions: list[list[str]] = []
    collected = 0

    t0 = time.perf_counter()
    n_batches = math.ceil(n_samples / dataloader.batch_size)
    for batch in tqdm(dataloader, total=n_batches, desc=f"{progress_prefix}Image embeddings"):
        images = batch["images"]
        captions = batch["captions"]
        if collected + len(images) > n_samples:
            remaining = n_samples - collected
            images = images[:remaining]
            captions = captions[:remaining]
        if not images:
            break

        all_image_embeds.append(wrapper.get_image_embeddings(images))
        all_captions.extend(captions)
        collected += len(images)
        if collected >= n_samples:
            break

    image_embeds = torch.cat(all_image_embeds, dim=0)
    t_images = time.perf_counter() - t0

    texts: list[str] = []
    image_ids: list[int] = []
    for image_idx, captions in enumerate(all_captions):
        for caption in captions:
            texts.append(caption)
            image_ids.append(image_idx)

    all_text_embeds: list[torch.Tensor] = []
    t0 = time.perf_counter()
    for start in tqdm(range(0, len(texts), text_batch_size), desc=f"{progress_prefix}Text embeddings"):
        batch = texts[start : start + text_batch_size]
        all_text_embeds.append(wrapper.get_text_embeddings(batch))
    text_embeds = torch.cat(all_text_embeds, dim=0)
    t_texts = time.perf_counter() - t0

    timing = {
        "image_embedding_seconds": round(t_images, 2),
        "text_embedding_seconds": round(t_texts, 2),
    }
    return image_embeds, text_embeds, torch.tensor(image_ids, device=device), timing


def run_model(
    model_key: str,
    checkpoint: str,
    dataloader,
    n_samples: int,
    text_batch_size: int,
    device: torch.device | None,
) -> dict:
    print(f"\n{'=' * 60}\nĐánh giá {model_key.upper()} — {checkpoint}\n{'=' * 60}")

    if device is not None and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    wrapper = WRAPPERS[model_key](checkpoint, device=str(device) if device else None)
    n_parameters = sum(p.numel() for p in wrapper.model.parameters())

    image_embeds, text_embeds, image_ids, timing = extract_embeddings(
        wrapper, dataloader, n_samples=n_samples, text_batch_size=text_batch_size
    )

    t0 = time.perf_counter()
    similarity = image_embeds @ text_embeds.t()
    metrics = compute_recall_at_k(similarity, image_ids=image_ids)
    timing["similarity_and_metrics_seconds"] = round(time.perf_counter() - t0, 2)
    timing["total_seconds"] = round(sum(timing.values()), 2)

    peak_vram_mb = None
    if wrapper.device.type == "cuda":
        peak_vram_mb = round(torch.cuda.max_memory_allocated(wrapper.device) / 1e6, 1)

    result = {
        "checkpoint": checkpoint,
        "embed_dim": int(image_embeds.shape[1]),
        "n_parameters": n_parameters,
        "timing": timing,
        "peak_vram_mb": peak_vram_mb,
        "metrics": metrics,
    }

    del wrapper, image_embeds, text_embeds, similarity
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def print_summary(all_results: dict) -> None:
    header = f"{'Model':<8} {'i2t R@1':>8} {'i2t R@5':>8} {'i2t R@10':>9} {'t2i R@1':>8} {'t2i R@5':>8} {'t2i R@10':>9} {'Time (s)':>9} {'VRAM (MB)':>10}"
    print("\n" + "=" * len(header))
    print("KẾT QUẢ BENCHMARK")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    def fmt(metrics: dict, key: str, width: int = 8) -> str:
        value = metrics.get(key)
        return f"{value:>{width}.2f}" if value is not None else f"{'-':>{width}}"

    for key, res in all_results.items():
        m = res["metrics"]
        vram = res["peak_vram_mb"] if res["peak_vram_mb"] is not None else "-"
        print(
            f"{key:<8} {fmt(m, 'i2t_R@1')} {fmt(m, 'i2t_R@5')} {fmt(m, 'i2t_R@10', 9)} "
            f"{fmt(m, 't2i_R@1')} {fmt(m, 't2i_R@5')} {fmt(m, 't2i_R@10', 9)} "
            f"{res['timing']['total_seconds']:>9.1f} {str(vram):>10}"
        )
    print("=" * len(header))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default="clip,siglip", help="Danh sách model: clip,siglip (mặc định: cả hai)")
    parser.add_argument("--json-path", default=str(REPO_ROOT / "datasets/coco/annotations/dataset_coco.json"))
    parser.add_argument("--image-dir", default=str(REPO_ROOT / "datasets/coco/images"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size cho ảnh (chỉ ảnh hưởng tốc độ)")
    parser.add_argument("--text-batch-size", type=int, default=256, help="Batch size cho text embeddings")
    parser.add_argument("--num-workers", type=int, default=2, help="Số worker của DataLoader (mặc định: 2)")
    parser.add_argument("--limit", type=int, default=None, help="Chỉ chạy N ảnh đầu (dùng cho smoke test)")
    parser.add_argument("--output", default=str(REPO_ROOT / "results/benchmark.json"), help="File JSON kết quả")
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"], help="Mặc định: tự phát hiện")
    parser.add_argument("--clip-checkpoint", default=DEFAULT_CHECKPOINTS["clip"])
    parser.add_argument("--siglip-checkpoint", default=DEFAULT_CHECKPOINTS["siglip"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    invalid = [m for m in models if m not in WRAPPERS]
    if invalid:
        print(f"[LỖI] Model không hợp lệ: {invalid}. Chọn trong {sorted(WRAPPERS)}", file=sys.stderr)
        return 1

    json_path = Path(args.json_path)
    if not json_path.exists():
        print(
            f"[LỖI] Không tìm thấy annotations: {json_path}\n"
            "  - Benchmark thật: python datasets/download_annotations.py\n"
            "  - Smoke test    : python make_samples.py rồi thêm "
            "--json-path datasets/coco/annotations/dataset_coco_sample.json",
            file=sys.stderr,
        )
        return 1

    dataloader = get_dataloader(
        str(json_path),
        args.image_dir,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        strict=True,
    )
    n_images_available = len(dataloader.dataset)
    n_samples = n_images_available if args.limit is None else min(args.limit, n_images_available)
    device = torch.device(args.device) if args.device else None
    checkpoints = {"clip": args.clip_checkpoint, "siglip": args.siglip_checkpoint}

    if args.image_dir == str(REPO_ROOT / "datasets/coco/images") and n_samples < 10:
        print(
            "[CẢNH BÁO] Đang chạy trên rất ít ảnh — đây là smoke test, "
            "KHÔNG dùng số liệu này cho báo cáo."
        )

    print(f"Annotations : {json_path}")
    print(f"Split       : {args.split} — {n_images_available} ảnh trong annotations")
    print(f"Đánh giá    : {n_samples} ảnh")
    print(f"Models      : {', '.join(models)}")

    all_results = {}
    started = time.perf_counter()
    for model_key in models:
        all_results[model_key] = run_model(
            model_key,
            checkpoints[model_key],
            dataloader,
            n_samples=n_samples,
            text_batch_size=args.text_batch_size,
            device=device,
        )
    total_seconds = round(time.perf_counter() - started, 2)

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "annotation_file": str(json_path),
        "annotation_sha256": sha256_file(json_path),
        "dataset": {
            "split": args.split,
            "n_images": n_samples,
            "n_captions": all_results[models[0]]["metrics"]["n_texts"],
            "limit": args.limit,
        },
        "protocol": PROTOCOL_DESCRIPTION,
        "versions": get_versions(),
        "device": get_device_name(torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))),
        "total_benchmark_seconds": total_seconds,
        "results": all_results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print_summary(all_results)
    print(f"\nKết quả đã lưu: {output_path}")
    print(f"Thời gian tổng: {total_seconds}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

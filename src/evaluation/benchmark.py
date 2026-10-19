"""Comprehensive Evaluation and Benchmarking Suite: CLIP vs SigLIP.

Evaluates semantic retrieval across:
1. MS COCO Karpathy Test Split (1-to-5 image-to-captions mapping, 5k images / 25k captions).
2. NewsSeek Event-Centric News Benchmark (processed keyframes, multilingual breakdowns).

Metrics measured:
- Recall@1, Recall@5, Recall@10 (Text-to-Image & Image-to-Text)
- Mean Recall (mR)
- Mean Reciprocal Rank (MRR)
- Modality Gap distance (||centroid(I) - centroid(T)||_2)
- Inference latency (ms per query / sample)
"""
from __future__ import annotations

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
import numpy as np
from tqdm import tqdm

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.dataset import ImageTextRetrievalDataset, NewsSeekDataset
from src.evaluation.metrics import compute_recall_at_k, compute_modality_gap
from src.models.clip_wrapper import CLIPWrapper
from src.models.siglip_wrapper import SigLIPWrapper


def extract_embeddings_batched(
    model_wrapper,
    images: List[Any],
    texts: List[str],
    batch_size: int = 64,
    device: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Extracts normalized L2 embeddings in batches with timing measurement."""
    model_wrapper.model.eval()
    t0 = time.time()

    all_image_embeds = []
    all_text_embeds = []

    # Batch encode images
    for i in range(0, len(images), batch_size):
        batch_imgs = images[i : i + batch_size]
        with torch.no_grad():
            img_vecs = model_wrapper.get_image_embeddings(batch_imgs)
            # L2 normalize
            img_vecs = img_vecs / img_vecs.norm(dim=-1, keepdim=True)
            all_image_embeds.append(img_vecs.cpu())

    # Batch encode texts
    for i in range(0, len(texts), batch_size):
        batch_txts = texts[i : i + batch_size]
        with torch.no_grad():
            txt_vecs = model_wrapper.get_text_embeddings(batch_txts)
            # L2 normalize
            txt_vecs = txt_vecs / txt_vecs.norm(dim=-1, keepdim=True)
            all_text_embeds.append(txt_vecs.cpu())

    elapsed = time.time() - t0

    image_embeds = torch.cat(all_image_embeds, dim=0) if all_image_embeds else torch.empty((0, 512))
    text_embeds = torch.cat(all_text_embeds, dim=0) if all_text_embeds else torch.empty((0, 512))

    return image_embeds, text_embeds, elapsed


def evaluate_dataset(
    model_wrapper,
    dataset,
    batch_size: int = 64,
    max_samples: int = 0,
    is_coco: bool = False,
) -> Dict[str, Any]:
    """Runs retrieval evaluation on a given dataset instance."""
    total_len = len(dataset)
    n_samples = min(total_len, max_samples) if max_samples > 0 else total_len

    images = []
    texts = []
    languages = []
    valid_count = 0

    print(f"Loading data for {n_samples} items...")
    for idx in tqdm(range(n_samples), desc="Loading items"):
        item = dataset[idx]
        img = item.get("image")
        captions = item.get("captions", [])
        if img is None or not captions:
            continue

        if is_coco:
            # COCO Karpathy has 5 captions per image
            images.append(img)
            for c in captions:
                texts.append(c)
        else:
            # 1-to-1 caption
            images.append(img)
            texts.append(captions[0])
            languages.append(item.get("language", "en"))

        valid_count += 1

    if valid_count == 0:
        print("[Warning] No valid samples found.")
        return {}

    num_captions_per_image = 5 if is_coco else 1
    print(f"Extracted {len(images)} images and {len(texts)} text queries.")

    # Extract embeddings
    img_embeds, txt_embeds, encode_time = extract_embeddings_batched(
        model_wrapper, images, texts, batch_size=batch_size
    )

    # Compute cosine similarity matrix
    sim_matrix = torch.matmul(img_embeds, txt_embeds.T)

    # Compute metrics
    metrics = compute_recall_at_k(
        sim_matrix,
        k_values=[1, 5, 10],
        num_captions_per_image=num_captions_per_image,
    )

    # Modality gap
    gap = compute_modality_gap(img_embeds, txt_embeds)
    metrics["modality_gap"] = gap
    metrics["num_images"] = len(images)
    metrics["num_texts"] = len(texts)
    metrics["total_time_sec"] = round(encode_time, 2)
    metrics["latency_ms_per_item"] = round((encode_time / max(1, len(images))) * 1000, 2)

    return metrics


def print_comparison_table(results_clip: Dict[str, Any], results_siglip: Dict[str, Any], title: str = "Benchmark"):
    print("\n" + "=" * 78)
    print(f"📊 COMPARATIVE BENCHMARK: {title}")
    print("=" * 78)
    print(f"{'Metric':<25} | {'CLIP (ViT-B/32)':<22} | {'SigLIP (ViT-B/16)':<22}")
    print("-" * 78)

    rows = [
        ("Images / Texts Evaluated", f"{results_clip.get('num_images', 0)} / {results_clip.get('num_texts', 0)}",
                                     f"{results_siglip.get('num_images', 0)} / {results_siglip.get('num_texts', 0)}"),
        ("Mean Recall (mR)", f"{results_clip.get('mR', 0):.2f}%", f"{results_siglip.get('mR', 0):.2f}%"),
        ("Text-to-Image R@1", f"{results_clip.get('t2i_R@1', 0):.2f}%", f"{results_siglip.get('t2i_R@1', 0):.2f}%"),
        ("Text-to-Image R@5", f"{results_clip.get('t2i_R@5', 0):.2f}%", f"{results_siglip.get('t2i_R@5', 0):.2f}%"),
        ("Text-to-Image R@10", f"{results_clip.get('t2i_R@10', 0):.2f}%", f"{results_siglip.get('t2i_R@10', 0):.2f}%"),
        ("Text-to-Image MRR", f"{results_clip.get('t2i_MRR', 0):.4f}", f"{results_siglip.get('t2i_MRR', 0):.4f}"),
        ("Image-to-Text R@1", f"{results_clip.get('i2t_R@1', 0):.2f}%", f"{results_siglip.get('i2t_R@1', 0):.2f}%"),
        ("Image-to-Text R@5", f"{results_clip.get('i2t_R@5', 0):.2f}%", f"{results_siglip.get('i2t_R@5', 0):.2f}%"),
        ("Image-to-Text R@10", f"{results_clip.get('i2t_R@10', 0):.2f}%", f"{results_siglip.get('i2t_R@10', 0):.2f}%"),
        ("Image-to-Text MRR", f"{results_clip.get('i2t_MRR', 0):.4f}", f"{results_siglip.get('i2t_MRR', 0):.4f}"),
        ("Modality Gap (||u_I - u_T||)", f"{results_clip.get('modality_gap', 0):.4f}", f"{results_siglip.get('modality_gap', 0):.4f}"),
        ("Inference Latency (ms/img)", f"{results_clip.get('latency_ms_per_item', 0):.2f} ms", f"{results_siglip.get('latency_ms_per_item', 0):.2f} ms"),
        ("Total Extract Time", f"{results_clip.get('total_time_sec', 0):.2f}s", f"{results_siglip.get('total_time_sec', 0):.2f}s"),
    ]

    for label, c_val, s_val in rows:
        print(f"{label:<25} | {c_val:<22} | {s_val:<22}")
    print("=" * 78 + "\n")


def main():
    parser = argparse.ArgumentParser(description="CLIP vs SigLIP Full Semantic Retrieval Benchmark")
    parser.add_argument("--dataset", type=str, choices=["coco", "newsseek", "both"], default="newsseek",
                        help="Dataset to benchmark: 'coco', 'newsseek', or 'both'")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Max samples to evaluate (0 = all available)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for embedding extraction")
    parser.add_argument("--language", type=str, default="all",
                        help="Language filter for NewsSeek ('all', 'en', 'vi', 'ja', 'ko', etc.)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run inference ('cuda' or 'cpu')")
    parser.add_argument("--output_json", type=str, default="evaluation/benchmark_results.json",
                        help="File to save JSON results")
    args = parser.parse_args()

    print("=" * 60)
    print("  CLIP vs SigLIP Semantic Retrieval Benchmark")
    print(f"  Device    : {args.device}")
    print(f"  Dataset   : {args.dataset}")
    print(f"  Batch Size: {args.batch_size}")
    print("=" * 60)

    # Initialize model wrappers
    print("\n[Loading Models]")
    clip_model = CLIPWrapper("openai/clip-vit-base-patch32", device=args.device)
    siglip_model = SigLIPWrapper("google/siglip-base-patch16-224", device=args.device)

    all_results = {}

    # 1. NewsSeek Benchmark
    if args.dataset in ["newsseek", "both"]:
        # Find best available annotations
        ns_candidates = [
            "datasets/newsseek/processed/newsseek_processed_manifest.json",
            "datasets/newsseek/processed/newsseek_manifest.json",
            "datasets/newsseek/annotations/metadata.json",
            "datasets/newsseek/annotations/dataset_newsseek_sample.json",
        ]
        ns_json = None
        for cand in ns_candidates:
            if os.path.exists(cand):
                ns_json = cand
                break

        if ns_json is not None:
            print(f"\n[Benchmarking NewsSeek from: {ns_json}]")
            img_dir = "datasets/newsseek/processed" if "processed" in ns_json else "datasets/newsseek/images"

            # A. Overall / Filtered language
            ds_ns = NewsSeekDataset(json_path=ns_json, image_dir=img_dir, language=args.language, mode="frame")
            print(f"Evaluating NewsSeek (lang={args.language}) with {len(ds_ns)} samples...")

            c_ns_res = evaluate_dataset(clip_model, ds_ns, batch_size=args.batch_size, max_samples=args.max_samples, is_coco=False)
            s_ns_res = evaluate_dataset(siglip_model, ds_ns, batch_size=args.batch_size, max_samples=args.max_samples, is_coco=False)

            title_ns = f"NewsSeek Real-World News (Language: {args.language.upper()})"
            print_comparison_table(c_ns_res, s_ns_res, title=title_ns)
            all_results["newsseek"] = {"clip": c_ns_res, "siglip": s_ns_res}

            # B. If language == 'all', also run English vs Non-English sub-breakdowns if available
            if args.language == "all" and len(ds_ns) > 50:
                print("\n[Running Multilingual Breakdown on NewsSeek...]")
                for sub_lang in ["en", "vi", "ja"]:
                    ds_sub = NewsSeekDataset(json_path=ns_json, image_dir=img_dir, language=sub_lang, mode="frame")
                    if len(ds_sub) >= 5:
                        c_sub = evaluate_dataset(clip_model, ds_sub, batch_size=args.batch_size, max_samples=args.max_samples)
                        s_sub = evaluate_dataset(siglip_model, ds_sub, batch_size=args.batch_size, max_samples=args.max_samples)
                        print_comparison_table(c_sub, s_sub, title=f"NewsSeek - Sub-language: {sub_lang.upper()}")
                        all_results[f"newsseek_{sub_lang}"] = {"clip": c_sub, "siglip": s_sub}
        else:
            print("[Warning] No NewsSeek annotations found. Skipping NewsSeek benchmark.")

    # 2. MS COCO Benchmark
    if args.dataset in ["coco", "both"]:
        coco_candidates = [
            "datasets/coco/annotations/dataset_coco.json",
            "datasets/coco/annotations/dataset_coco_sample.json",
        ]
        coco_json = None
        for cand in coco_candidates:
            if os.path.exists(cand):
                coco_json = cand
                break

        if coco_json is not None:
            print(f"\n[Benchmarking MS COCO Karpathy from: {coco_json}]")
            img_dir = "datasets/coco/images"
            ds_coco = ImageTextRetrievalDataset(json_path=coco_json, image_dir=img_dir, split="test")
            print(f"Loaded {len(ds_coco)} COCO test images (1-to-5 captions).")

            c_coco_res = evaluate_dataset(clip_model, ds_coco, batch_size=args.batch_size, max_samples=args.max_samples, is_coco=True)
            s_coco_res = evaluate_dataset(siglip_model, ds_coco, batch_size=args.batch_size, max_samples=args.max_samples, is_coco=True)

            print_comparison_table(c_coco_res, s_coco_res, title="MS COCO Karpathy Test Split (In-Domain)")
            all_results["coco"] = {"clip": c_coco_res, "siglip": s_coco_res}
        else:
            print("[Warning] No MS COCO annotations found. Skipping COCO benchmark.")

    # Save output JSON
    out_file = Path(args.output_json)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nAll benchmark results successfully exported to: {out_file}")


if __name__ == "__main__":
    main()

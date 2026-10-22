"""Comprehensive Query Taxonomy Benchmark: Textual KIS, Temporal Alignment, and AVS.

Benchmark evaluations across:
1. Textual Known-Item Search (KIS): Unique target retrieval (Rank, Separation Margin).
2. Temporal Retrieval & Alignment: Monotonic DP multi-event chain matching.
3. Ad-hoc Video Search (AVS): Multi-positive broad concept retrieval across 698 videos.
Evaluates both CLIP (512D) and SigLIP (768D) with English and Vietnamese queries.
"""
import os
import sys
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.storage.vector_store import VectorStore
from src.video_pipeline.comparative_search import ComparativeSearchEngine
from src.search_strategies.temporal_dp import match_temporal_sequence


def run_benchmark():
    print("[Taxonomy Benchmark] Initializing models and stores...")
    clip_store = VectorStore.load_db(ROOT_DIR / "storage" / "newsseek" / "clip")
    siglip_store = VectorStore.load_db(ROOT_DIR / "storage" / "newsseek" / "siglip")

    engine = ComparativeSearchEngine()
    engine.set_stores(clip_store, siglip_store)

    # 1. Textual Known-Item Search (KIS) Queries
    kis_queries = [
        {
            "id": "KIS_1",
            "en": "Luigi Mangione state court trial hearing orange jumpsuit",
            "vi": "Phiên tòa xét xử Luigi Mangione trong trang phục áo tù màu cam",
            "expected_theme": "Court / Legal",
        },
        {
            "id": "KIS_2",
            "en": "Space launch rocket engines ignition plume on launchpad",
            "vi": "Động cơ tên lửa vũ trụ bốc khói lửa khi kích hoạt trên bệ phóng",
            "expected_theme": "Space / Rocket",
        },
        {
            "id": "KIS_3",
            "en": "Military soldiers marching in parade with national flag",
            "vi": "Binh lính diễu binh trong lễ duyệt binh với quốc kỳ",
            "expected_theme": "Military / Parade",
        },
        {
            "id": "KIS_4",
            "en": "Firefighter spraying high pressure water hose at house fire",
            "vi": "Lính cứu hỏa phun vòi nước áp lực cao vào đám cháy ngôi nhà",
            "expected_theme": "Disaster / Fire",
        },
        {
            "id": "KIS_5",
            "en": "Gold bullion bars stacked inside bank vault",
            "vi": "Các thỏi vàng miếng xếp chồng bên trong hầm ngân hàng",
            "expected_theme": "Economy / Gold",
        },
    ]

    # 2. Temporal Sequential Alignment Queries (2-step chains)
    temporal_queries = [
        {
            "id": "TEMP_1",
            "name": "Rocket Pre-launch to Liftoff",
            "steps_en": [
                "Rocket standing upright on launch pad before launch",
                "Rocket engine ignition lifting off with smoke and fire",
            ],
            "steps_vi": [
                "Tên lửa đứng thẳng trên bệ phóng trước khi phóng",
                "Tên lửa kích hoạt động cơ cất cánh với khói và lửa",
            ],
        },
        {
            "id": "TEMP_2",
            "name": "Emergency Arrival to Courtroom Escort",
            "steps_en": [
                "Police cars arriving at government building with siren",
                "Suspect handcuffed escorted by police officers inside room",
            ],
            "steps_vi": [
                "Xe cảnh sát tiến đến tòa nhà chính phủ với còi hú",
                "Nghi phạm bị còng tay được cảnh sát áp giải bên trong phòng",
            ],
        },
        {
            "id": "TEMP_3",
            "name": "Forest Wildfire to Aerial Water Drop",
            "steps_en": [
                "Thick wildfire black smoke rising from burning forest",
                "Helicopter dropping water over burning wildfire",
            ],
            "steps_vi": [
                "Khói đen dày đặc bốc lên từ khu rừng đang cháy",
                "Trực thăng thả nước dập tắt đám cháy rừng",
            ],
        },
    ]

    # 3. Ad-hoc Video Search (AVS) Queries (Broad category, multiple matching videos)
    avs_queries = [
        {
            "id": "AVS_1",
            "en": "Wildfire smoke plumes over forest mountain",
            "vi": "Khói cháy rừng bao phủ dãy núi cây cối",
            "category": "Natural Disaster",
        },
        {
            "id": "AVS_2",
            "en": "Presidential press conference at the podium with microphones",
            "vi": "Cuộc họp báo của tổng thống tại bục phát biểu có micro",
            "category": "Politics / Press",
        },
        {
            "id": "AVS_3",
            "en": "Crowded city street traffic with motorcycles and cars",
            "vi": "Đường phố thành phố đông đúc với xe máy và ô tô",
            "category": "Urban Life",
        },
        {
            "id": "AVS_4",
            "en": "Doctors and medical nurses in hospital intensive care unit",
            "vi": "Bác sĩ và y tá trong phòng hồi sức cấp cứu bệnh viện",
            "category": "Healthcare / Medical",
        },
        {
            "id": "AVS_5",
            "en": "Severe storm flash flood water flowing through city roads",
            "vi": "Nước lũ lụt sau bão chảy cuồn cuộn qua các con đường thành phố",
            "category": "Disaster / Weather",
        },
    ]

    results = {
        "metadata": {
            "total_keyframes": 164874,
            "total_videos": 698,
            "clip_model": "ViT-B/32 (512D)",
            "siglip_model": "ViT-B/16-224 (768D)",
            "benchmark_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "kis_results": [],
        "temporal_results": [],
        "avs_results": [],
    }

    print("\n--- [1] Evaluating Textual Known-Item Search (KIS) ---")
    for q in kis_queries:
        for lang, text in [("en", q["en"]), ("vi", q["vi"])]:
            t0 = time.perf_counter()
            res = engine.search_text(text, k=20, filter_duplicates=True)
            elapsed = (time.perf_counter() - t0) * 1000.0

            clip_top1 = res["clip_results"][0]["score"] if res["clip_results"] else 0.0
            clip_top2 = res["clip_results"][1]["score"] if len(res["clip_results"]) > 1 else clip_top1
            clip_margin = clip_top1 - clip_top2

            siglip_top1 = res["siglip_results"][0]["score"] if res["siglip_results"] else 0.0
            siglip_top2 = res["siglip_results"][1]["score"] if len(res["siglip_results"]) > 1 else siglip_top1
            siglip_margin = siglip_top1 - siglip_top2

            agreement = res["benchmark"]["agreement"]

            row = {
                "id": q["id"],
                "theme": q["expected_theme"],
                "lang": lang,
                "query": text,
                "clip_top1": round(float(clip_top1), 4),
                "clip_margin": round(float(clip_margin), 4),
                "siglip_top1": round(float(siglip_top1), 4),
                "siglip_margin": round(float(siglip_margin), 4),
                "jaccard_overlap": agreement.get("jaccard_overlap_pct", 0.0),
                "rank1_match": agreement.get("rank_1_match", False),
                "latency_ms": round(elapsed, 2),
            }
            results["kis_results"].append(row)
            print(f"KIS [{lang.upper()}] '{text[:35]}...' -> CLIP margin: {clip_margin:.4f} | SigLIP margin: {siglip_margin:.4f}")

    print("\n--- [2] Evaluating Temporal Alignment & Key Events ---")
    for tq in temporal_queries:
        for lang, steps in [("en", tq["steps_en"]), ("vi", tq["steps_vi"])]:
            # Compute step rankings for CLIP and SigLIP
            clip_rankings = []
            siglip_rankings = []

            for step_text in steps:
                s_res = engine.search_text(step_text, k=100, filter_duplicates=False)
                clip_rankings.append(s_res["clip_results"])
                siglip_rankings.append(s_res["siglip_results"])

            # Monotonic DP Alignment (min 3 sec offset between step 1 and step 2)
            clip_chains = match_temporal_sequence(clip_rankings, temporal_offsets_sec=[0.0, 3.0])
            siglip_chains = match_temporal_sequence(siglip_rankings, temporal_offsets_sec=[0.0, 3.0])

            def get_chain_span(chain):
                if not chain or "steps" not in chain or len(chain["steps"]) < 2:
                    return 0.0
                return float(chain["steps"][-1].get("timestamp_sec", 0.0)) - float(chain["steps"][0].get("timestamp_sec", 0.0))

            clip_valid = len(clip_chains)
            clip_top_conf = clip_chains[0]["confidence"] if clip_chains else 0.0
            clip_top_span = get_chain_span(clip_chains[0]) if clip_chains else 0.0

            siglip_valid = len(siglip_chains)
            siglip_top_conf = siglip_chains[0]["confidence"] if siglip_chains else 0.0
            siglip_top_span = get_chain_span(siglip_chains[0]) if siglip_chains else 0.0

            row = {
                "id": tq["id"],
                "name": tq["name"],
                "lang": lang,
                "clip_chains_found": clip_valid,
                "clip_top_confidence": round(float(clip_top_conf), 4),
                "clip_span_sec": round(float(clip_top_span), 2),
                "siglip_chains_found": siglip_valid,
                "siglip_top_confidence": round(float(siglip_top_conf), 4),
                "siglip_span_sec": round(float(siglip_top_span), 2),
            }
            results["temporal_results"].append(row)
            print(f"Temporal [{lang.upper()}] '{tq['name']}' -> CLIP chains: {clip_valid} (conf {clip_top_conf:.3f}) | SigLIP chains: {siglip_valid} (conf {siglip_top_conf:.3f})")

    print("\n--- [3] Evaluating Ad-hoc Video Search (AVS) ---")
    for aq in avs_queries:
        for lang, text in [("en", aq["en"]), ("vi", aq["vi"])]:
            res = engine.search_text(text, k=30, filter_duplicates=False)

            clip_hits = res["clip_results"]
            siglip_hits = res["siglip_results"]

            # Video level coverage (how many distinct videos covered in top 30 keyframes)
            clip_vids = len(set(h["video_id"] for h in clip_hits))
            siglip_vids = len(set(h["video_id"] for h in siglip_hits))

            # Mean top-10 confidence score
            clip_mean_top10 = np.mean([h["score"] for h in clip_hits[:10]]) if clip_hits else 0.0
            siglip_mean_top10 = np.mean([h["score"] for h in siglip_hits[:10]]) if siglip_hits else 0.0

            # Video Group top-1 score
            siglip_top_vgroup = res["video_groups"][0]["max_score"] if res["video_groups"] else 0.0

            row = {
                "id": aq["id"],
                "category": aq["category"],
                "lang": lang,
                "query": text,
                "clip_distinct_videos_top30": clip_vids,
                "clip_mean_top10_score": round(float(clip_mean_top10), 4),
                "siglip_distinct_videos_top30": siglip_vids,
                "siglip_mean_top10_score": round(float(siglip_mean_top10), 4),
                "siglip_top_video_score": round(float(siglip_top_vgroup), 4),
            }
            results["avs_results"].append(row)
            print(f"AVS [{lang.upper()}] '{text[:35]}...' -> CLIP vids: {clip_vids} | SigLIP vids: {siglip_vids}")

    # Save JSON results
    out_json = ROOT_DIR / "storage" / "newsseek" / "query_taxonomy_benchmark_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[Success] Results saved to: {out_json}")


if __name__ == "__main__":
    run_benchmark()

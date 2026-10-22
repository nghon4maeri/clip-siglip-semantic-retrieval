"""Generate and evaluate grounded VLM-style benchmark queries from NewsSeek.

Generates 3 query paradigms based on actual NewsSeek video events:
1. Textual Known-Item Search (KIS): Ultra-specific visual targets with fine-grained visual cues.
2. Temporal Retrieval & Alignment: Multi-event chronological sequence queries (E1 -> E2).
3. Ad-hoc Video Search (AVS): Broad thematic concepts matching multiple videos.
"""
import sys
import json
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.storage.vector_store import VectorStore
from src.video_pipeline.comparative_search import ComparativeSearchEngine
from src.search_strategies.temporal_dp import match_temporal_sequence


# Curated VLM-grounded queries based on NewsSeek video analysis
GROUNDED_TAXONOMY_QUERIES = {
    "textual_kis": [
        {
            "id": "KIS_01",
            "theme": "Semiconductor Cleanroom",
            "target_video_id": "O34mNBDyYPE",
            "en": "Laboratory technician in white cleanroom bunny suit inspecting silicon semiconductor wafer under yellow lighting",
            "vi": "Kỹ thuật viên phòng thí nghiệm mặc trang phục phòng sạch màu trắng kiểm tra đĩa bán dẫn silicon dưới ánh đèn vàng",
            "discriminative_cues": "White bunny suit, silicon wafer, yellow cleanroom lighting"
        },
        {
            "id": "KIS_02",
            "theme": "Maritime Ferry Disaster",
            "target_video_id": "K-VSAqEjTHI",
            "en": "Passenger ferry on fire billowing thick black smoke while passengers in orange life jackets board inflatable rescue boat",
            "vi": "Phà chở khách bốc cháy nghi ngút khói đen trong khi các hành khách mặc áo phao màu cam lên thuyền cứu hộ",
            "discriminative_cues": "Burning ferry, black smoke, orange life jackets on rescue boat"
        },
        {
            "id": "KIS_03",
            "theme": "Three Gorges Dam Hydro-solar",
            "target_video_id": "qG97ZIHYSfY",
            "en": "Giant concrete ship lock canal of Three Gorges Dam with massive blue photovoltaic solar panels on surrounding hills",
            "vi": "Âu tàu bê tông khổng lồ của đập Tam Hiệp với các tấm pin năng lượng mặt trời màu xanh trên sườn đồi xung quanh",
            "discriminative_cues": "Three Gorges concrete canal, massive water lock, blue solar panels"
        },
        {
            "id": "KIS_04",
            "theme": "Earthquake Rubble Rescue",
            "target_video_id": "y9BigFu0d7k",
            "en": "Emergency search and rescue workers in yellow helmets digging through cracked concrete rubble with trained search dog",
            "vi": "Lực lượng cứu hộ đội mũ bảo hộ màu vàng đang đào bới đống đổ nát bê tông nứt vỡ cùng chó nghiệp vụ",
            "discriminative_cues": "Yellow helmets, cracked concrete debris, search dog"
        },
        {
            "id": "KIS_05",
            "theme": "Rural Village Basketball Tournament",
            "target_video_id": "7LF0Vva1j54",
            "en": "Outdoor basketball court in rural village surrounded by thousands of cheering spectators holding banners under floodlights",
            "vi": "Sân bóng rổ ngoài trời tại làng quê bao quanh bởi hàng ngàn khán giả cổ vũ cầm biểu ngữ dưới ánh đèn pha",
            "discriminative_cues": "Outdoor basketball game, packed cheering rural crowd, floodlights"
        }
    ],

    "temporal_alignment": [
        {
            "id": "TEMP_01",
            "theme": "Wildfire Progression to Evacuation",
            "video_id_context": "aIA6RE5e6_c",
            "steps_en": [
                "Intense orange fire glow and thick smoke approaching tree line along highway",
                "Fire trucks and police convoy escorting civilian cars evacuating down highway"
            ],
            "steps_vi": [
                "Ánh lửa đỏ rực và khói dày đặc tiến sát rặng cây dọc theo đường cao tốc",
                "Đoàn xe cứu hỏa và xe cảnh sát hộ tống người dân sơ tán trên đường cao tốc"
            ],
            "temporal_logic": "Lửa tiếp cận đường -> Xe cộ sơ tán có cảnh sát hộ tống (t1 < t2)"
        },
        {
            "id": "TEMP_02",
            "theme": "Hospital Emergency Care to Patient Recovery",
            "video_id_context": "8iZSkgqoI",
            "steps_en": [
                "Doctors and intensive care nurses monitoring patient with digital ECG heart rate displays",
                "Patient sitting up talking with doctor holding medical clipboard in hospital ward"
            ],
            "steps_vi": [
                "Bác sĩ và y tá hồi sức cấp cứu theo dõi bệnh nhân với màn hình điện tâm đồ ECG",
                "Bệnh nhân ngồi dậy nói chuyện với bác sĩ cầm tập hồ sơ bệnh án trong phòng bệnh"
            ],
            "temporal_logic": "Theo dõi cấp cứu hồi sức -> Bệnh nhân tỉnh táo phục hồi (t1 < t2)"
        },
        {
            "id": "TEMP_03",
            "theme": "Typhoon Coastal Landfall to Tree Damage",
            "video_id_context": "IX6GDdiKafU",
            "steps_en": [
                "Violent ocean waves and storm surge crashing against concrete sea wall in heavy rain",
                "Uprooted trees and fallen street billboards blocking flooded city street after typhoon"
            ],
            "steps_vi": [
                "Sóng biển dữ dội và triều cường đập mạnh vào bờ kè chắn sóng bê tông trong mưa lớn",
                "Cây cối bật gốc và biển quảng cáo ngã đổ chắn ngang con đường ngập nước sau bão"
            ],
            "temporal_logic": "Bão đổ bộ sóng biển dữ dội -> Cây cối ngã đổ đường ngập sau bão (t1 < t2)"
        }
    ],

    "adhoc_video_search": [
        {
            "id": "AVS_01",
            "concept": "Renewable Clean Energy Infrastructure",
            "en": "Solar photovoltaic power panels and wind turbine generators in rural landscapes",
            "vi": "Các tấm pin năng lượng mặt trời và tuabin gió trên các vùng cảnh quan nông thôn",
            "target_multi_videos": "qG97ZIHYSfY, xBOG9k57w_E, wuQ26k-Lnh4"
        },
        {
            "id": "AVS_02",
            "concept": "Severe Drought and Freshwater Scarcity",
            "en": "Parched cracked dry soil reservoirs and citizens carrying plastic water containers",
            "vi": "Hồ chứa nước khô cạn nứt nẻ và người dân mang can nhựa đi lấy nước",
            "target_multi_videos": "o8L5ZXBBdjs, w, V6XAeLdeI"
        },
        {
            "id": "AVS_03",
            "concept": "Maritime Security and Naval Vessel Incident",
            "en": "Commercial cargo container ships and naval warships operating in open ocean waters",
            "vi": "Tàu chở hàng container thương mại và tàu chiến hải quân hoạt động trên vùng biển mở",
            "target_multi_videos": "xppaKLnZPmA, K-VSAqEjTHI, 47EBP1i1XLc"
        },
        {
            "id": "AVS_04",
            "concept": "Urban Electric and New Energy Vehicles",
            "en": "Modern electric vehicles and automobile assembly line production in manufacturing factory",
            "vi": "Xe điện hiện đại và dây chuyền lắp ráp sản xuất ô tô trong nhà máy",
            "target_multi_videos": "fYgYEoeoYS4, wuQ26k-Lnh4, jK7ueKiAYxU"
        },
        {
            "id": "AVS_05",
            "concept": "Earthquake Post-Disaster Relief Operation",
            "en": "First responders setting up emergency triage tents and distributing relief supplies to victims",
            "vi": "Lực lượng phản ứng nhanh dựng lều cứu trợ khẩn cấp và phân phát nhu yếu phẩm cho nạn nhân",
            "target_multi_videos": "y9BigFu0d7k, Wbm09_gVdqU, TCqAlw1uTQc"
        }
    ]
}


def run_experiment():
    print("[VLM Taxonomy Evaluation] Loading vector stores and models...")
    clip_store = VectorStore.load_db(ROOT_DIR / "storage" / "newsseek" / "clip")
    siglip_store = VectorStore.load_db(ROOT_DIR / "storage" / "newsseek" / "siglip")

    engine = ComparativeSearchEngine()
    engine.set_stores(clip_store, siglip_store)

    output_report = {
        "textual_kis": [],
        "temporal_alignment": [],
        "adhoc_video_search": [],
    }

    print("\n=== [1] Testing Grounded Textual KIS ===")
    for q in GROUNDED_TAXONOMY_QUERIES["textual_kis"]:
        res_en = engine.search_text(q["en"], k=10)
        res_vi = engine.search_text(q["vi"], k=10)

        # Check target video rank in top 10
        def find_target_rank(results, target_vid):
            for rank, r in enumerate(results, start=1):
                if r.get("video_id") == target_vid:
                    return rank, r["score"]
            return None, 0.0

        clip_rank_en, clip_score_en = find_target_rank(res_en["clip_results"], q["target_video_id"])
        siglip_rank_en, siglip_score_en = find_target_rank(res_en["siglip_results"], q["target_video_id"])

        clip_rank_vi, clip_score_vi = find_target_rank(res_vi["clip_results"], q["target_video_id"])
        siglip_rank_vi, siglip_score_vi = find_target_rank(res_vi["siglip_results"], q["target_video_id"])

        item_eval = {
            "id": q["id"],
            "theme": q["theme"],
            "target_video_id": q["target_video_id"],
            "query_en": q["en"],
            "query_vi": q["vi"],
            "discriminative_cues": q["discriminative_cues"],
            "clip_en": {"target_rank": clip_rank_en, "score": round(clip_score_en, 4), "top1_score": round(res_en["clip_results"][0]["score"], 4)},
            "siglip_en": {"target_rank": siglip_rank_en, "score": round(siglip_score_en, 4), "top1_score": round(res_en["siglip_results"][0]["score"], 4)},
            "clip_vi": {"target_rank": clip_rank_vi, "score": round(clip_score_vi, 4), "top1_score": round(res_vi["clip_results"][0]["score"], 4)},
            "siglip_vi": {"target_rank": siglip_rank_vi, "score": round(siglip_score_vi, 4), "top1_score": round(res_vi["siglip_results"][0]["score"], 4)},
        }
        output_report["textual_kis"].append(item_eval)
        print(f"KIS {q['id']} ({q['theme']}): Target {q['target_video_id']} | EN: CLIP rank={clip_rank_en}, SigLIP rank={siglip_rank_en} | VI: CLIP rank={clip_rank_vi}, SigLIP rank={siglip_rank_vi}")

    print("\n=== [2] Testing Grounded Temporal Alignment ===")
    for tq in GROUNDED_TAXONOMY_QUERIES["temporal_alignment"]:
        # EN steps
        clip_rankings_en = [engine.search_text(s, k=60)["clip_results"] for s in tq["steps_en"]]
        siglip_rankings_en = [engine.search_text(s, k=60)["siglip_results"] for s in tq["steps_en"]]

        # VI steps
        clip_rankings_vi = [engine.search_text(s, k=60)["clip_results"] for s in tq["steps_vi"]]
        siglip_rankings_vi = [engine.search_text(s, k=60)["siglip_results"] for s in tq["steps_vi"]]

        clip_chains_en = match_temporal_sequence(clip_rankings_en, temporal_offsets_sec=[0.0, 3.0])
        siglip_chains_en = match_temporal_sequence(siglip_rankings_en, temporal_offsets_sec=[0.0, 3.0])

        clip_chains_vi = match_temporal_sequence(clip_rankings_vi, temporal_offsets_sec=[0.0, 3.0])
        siglip_chains_vi = match_temporal_sequence(siglip_rankings_vi, temporal_offsets_sec=[0.0, 3.0])

        item_eval = {
            "id": tq["id"],
            "theme": tq["theme"],
            "steps_en": tq["steps_en"],
            "steps_vi": tq["steps_vi"],
            "temporal_logic": tq["temporal_logic"],
            "clip_en": {"chains_count": len(clip_chains_en), "top_conf": round(clip_chains_en[0]["confidence"], 4) if clip_chains_en else 0},
            "siglip_en": {"chains_count": len(siglip_chains_en), "top_conf": round(siglip_chains_en[0]["confidence"], 4) if siglip_chains_en else 0},
            "clip_vi": {"chains_count": len(clip_chains_vi), "top_conf": round(clip_chains_vi[0]["confidence"], 4) if clip_chains_vi else 0},
            "siglip_vi": {"chains_count": len(siglip_chains_vi), "top_conf": round(siglip_chains_vi[0]["confidence"], 4) if siglip_chains_vi else 0},
        }
        output_report["temporal_alignment"].append(item_eval)
        print(f"TEMP {tq['id']} ({tq['theme']}): EN chains: CLIP={len(clip_chains_en)} vs SigLIP={len(siglip_chains_en)} | VI chains: CLIP={len(clip_chains_vi)} vs SigLIP={len(siglip_chains_vi)}")

    print("\n=== [3] Testing Grounded Ad-hoc Video Search (AVS) ===")
    for aq in GROUNDED_TAXONOMY_QUERIES["adhoc_video_search"]:
        res_en = engine.search_text(aq["en"], k=30)
        res_vi = engine.search_text(aq["vi"], k=30)

        clip_vids_en = len(set(h["video_id"] for h in res_en["clip_results"]))
        siglip_vids_en = len(set(h["video_id"] for h in res_en["siglip_results"]))

        clip_vids_vi = len(set(h["video_id"] for h in res_vi["clip_results"]))
        siglip_vids_vi = len(set(h["video_id"] for h in res_vi["siglip_results"]))

        item_eval = {
            "id": aq["id"],
            "concept": aq["concept"],
            "query_en": aq["en"],
            "query_vi": aq["vi"],
            "coverage_en": {"clip_distinct_vids": clip_vids_en, "siglip_distinct_vids": siglip_vids_en},
            "coverage_vi": {"clip_distinct_vids": clip_vids_vi, "siglip_distinct_vids": siglip_vids_vi},
        }
        output_report["adhoc_video_search"].append(item_eval)
        print(f"AVS {aq['id']} ({aq['concept']}): Top-30 Distinct Videos -> EN: CLIP={clip_vids_en}, SigLIP={siglip_vids_en} | VI: CLIP={clip_vids_vi}, SigLIP={siglip_vids_vi}")

    # Save to JSON
    out_path = ROOT_DIR / "storage" / "newsseek" / "vlm_grounded_queries_benchmark.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_report, f, indent=2, ensure_ascii=False)
    print(f"\n[Done] Saved grounded benchmark evaluation to: {out_path}")


if __name__ == "__main__":
    run_experiment()

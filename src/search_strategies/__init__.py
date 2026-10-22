"""Search Strategies Package for Multimodal Semantic Retrieval.

Implements state-of-the-art retrieval and ranking algorithms:
- Reciprocal Rank Fusion (RRF) & Linear Score Combination
- Dynamic Programming (DP) Temporal Sequence Search
- Multi-Detail Aspect Decomposition & Fusion
- Greedy Near-Duplicate Suppression & Visual Diversity Filter
"""
from .fusion import reciprocal_rank_fusion, linear_score_fusion
from .similarity_filter import filter_near_duplicate_frames
from .temporal_dp import match_temporal_sequence
from .multi_detail import search_multi_detail

__all__ = [
    "reciprocal_rank_fusion",
    "linear_score_fusion",
    "filter_near_duplicate_frames",
    "match_temporal_sequence",
    "search_multi_detail",
]

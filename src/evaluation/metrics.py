"""Evaluation metrics for Vision-Language Semantic Retrieval.

Supports:
- Recall@K (R@1, R@5, R@10) for 1-to-1 (NewsSeek/custom) and 1-to-5 (MS COCO Karpathy) retrieval
- Mean Recall (mR)
- Mean Reciprocal Rank (MRR)
- Modality Gap distance (L2 norm between centroid embeddings)
"""
from __future__ import annotations

import torch
import numpy as np
from typing import Dict, List, Optional, Union


def compute_recall_at_k(
    similarity_matrix: torch.Tensor,
    k_values: List[int] = [1, 5, 10],
    num_captions_per_image: int = 1,
) -> Dict[str, float]:
    """Computes Recall@K, mR, and MRR for Image-to-Text and Text-to-Image retrieval.
    
    Args:
        similarity_matrix: Tensor of shape [N_images, M_texts] containing cosine similarities.
        k_values: List of K thresholds.
        num_captions_per_image: Number of captions per image (e.g. 1 for NewsSeek, 5 for COCO Karpathy).

    Returns:
        Dict with R@1, R@5, R@10, Mean Recall, and MRR for both directions.
    """
    if isinstance(similarity_matrix, np.ndarray):
        similarity_matrix = torch.from_numpy(similarity_matrix)

    N_images, M_texts = similarity_matrix.shape
    max_k = max(k_values)
    results: Dict[str, float] = {}

    # -------------------------------------------------------------
    # 1. Image-to-Text (I2T) Retrieval
    # Query: Image, Database: Texts
    # -------------------------------------------------------------
    # top_k indices for each image: shape [N_images, max_k]
    _, i2t_top_indices = similarity_matrix.topk(min(max_k, M_texts), dim=1, largest=True)

    i2t_hits = {k: 0 for k in k_values}
    i2t_mrr_sum = 0.0

    if num_captions_per_image == 1:
        # 1-to-1 mapping: image i corresponds to text i
        targets = torch.arange(N_images, device=similarity_matrix.device).unsqueeze(1)
        matches = (i2t_top_indices == targets)  # [N, max_k]

        for k in k_values:
            i2t_hits[k] = matches[:, :k].any(dim=1).float().sum().item()

        # MRR calculation
        # Find rank of ground truth in full sorted row
        _, full_sorted_indices = similarity_matrix.sort(dim=1, descending=True)
        ranks = (full_sorted_indices == targets).nonzero(as_tuple=True)[1] + 1
        i2t_mrr_sum = (1.0 / ranks.float()).sum().item()

    else:
        # 1-to-N mapping (e.g. COCO 1-to-5)
        # Image i corresponds to texts [i*num_captions, (i+1)*num_captions - 1]
        for img_idx in range(N_images):
            valid_text_min = img_idx * num_captions_per_image
            valid_text_max = (img_idx + 1) * num_captions_per_image
            top_ranked = i2t_top_indices[img_idx]

            first_rank = None
            for rank_0, text_idx in enumerate(top_ranked.tolist()):
                if valid_text_min <= text_idx < valid_text_max:
                    if first_rank is None:
                        first_rank = rank_0 + 1
                    for k in k_values:
                        if (rank_0 + 1) <= k:
                            i2t_hits[k] += 1
                    break

            if first_rank is not None:
                i2t_mrr_sum += 1.0 / first_rank
            else:
                # Calculate exact rank from full sort if needed
                row_sim = similarity_matrix[img_idx]
                target_sims = row_sim[valid_text_min:valid_text_max]
                best_target_sim = target_sims.max().item()
                exact_rank = (row_sim > best_target_sim).sum().item() + 1
                i2t_mrr_sum += 1.0 / exact_rank

    for k in k_values:
        results[f"i2t_R@{k}"] = round((i2t_hits[k] / N_images) * 100.0, 2)
    results["i2t_MRR"] = round(i2t_mrr_sum / N_images, 4)

    # -------------------------------------------------------------
    # 2. Text-to-Image (T2I) Retrieval
    # Query: Text, Database: Images
    # -------------------------------------------------------------
    # top_k indices for each text query: shape [M_texts, max_k]
    _, t2i_top_indices = similarity_matrix.T.topk(min(max_k, N_images), dim=1, largest=True)

    t2i_hits = {k: 0 for k in k_values}
    t2i_mrr_sum = 0.0

    if num_captions_per_image == 1:
        # 1-to-1 mapping: text j corresponds to image j
        targets = torch.arange(M_texts, device=similarity_matrix.device).unsqueeze(1)
        matches = (t2i_top_indices == targets)

        for k in k_values:
            t2i_hits[k] = matches[:, :k].any(dim=1).float().sum().item()

        _, full_sorted_indices = similarity_matrix.T.sort(dim=1, descending=True)
        ranks = (full_sorted_indices == targets).nonzero(as_tuple=True)[1] + 1
        t2i_mrr_sum = (1.0 / ranks.float()).sum().item()

    else:
        # 1-to-N mapping: text j corresponds to image (j // num_captions_per_image)
        for txt_idx in range(M_texts):
            target_img_idx = txt_idx // num_captions_per_image
            top_ranked = t2i_top_indices[txt_idx]

            first_rank = None
            for rank_0, img_idx in enumerate(top_ranked.tolist()):
                if img_idx == target_img_idx:
                    first_rank = rank_0 + 1
                    for k in k_values:
                        if (rank_0 + 1) <= k:
                            t2i_hits[k] += 1
                    break

            if first_rank is not None:
                t2i_mrr_sum += 1.0 / first_rank
            else:
                col_sim = similarity_matrix[:, txt_idx]
                target_sim = col_sim[target_img_idx].item()
                exact_rank = (col_sim > target_sim).sum().item() + 1
                t2i_mrr_sum += 1.0 / exact_rank

    for k in k_values:
        results[f"t2i_R@{k}"] = round((t2i_hits[k] / M_texts) * 100.0, 2)
    results["t2i_MRR"] = round(t2i_mrr_sum / M_texts, 4)

    # -------------------------------------------------------------
    # 3. Overall Mean Recall (mR)
    # -------------------------------------------------------------
    all_recalls = [results[f"i2t_R@{k}"] for k in k_values] + [results[f"t2i_R@{k}"] for k in k_values]
    results["mR"] = round(float(np.mean(all_recalls)), 2) if all_recalls else 0.0

    return results


def compute_modality_gap(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
) -> float:
    """Computes Euclidean distance between centroid of image and text embeddings.
    
    Formula:
        Modality Gap = || mean(v_I) - mean(v_T) ||_2
    """
    if isinstance(image_embeds, np.ndarray):
        image_embeds = torch.from_numpy(image_embeds)
    if isinstance(text_embeds, np.ndarray):
        text_embeds = torch.from_numpy(text_embeds)

    mean_img = image_embeds.mean(dim=0)
    mean_txt = text_embeds.mean(dim=0)
    gap = torch.norm(mean_img - mean_txt, p=2).item()
    return round(float(gap), 4)

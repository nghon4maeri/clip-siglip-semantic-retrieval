"""Độ đo Recall@K cho bài toán Text-Image Retrieval (protocol chuẩn MS COCO).

Protocol chuẩn (dùng trong paper CLIP/SigLIP và OpenCLIP):

    - Tập test Karpathy: 5.000 ảnh, mỗi ảnh có 5 captions => 25.000 captions.
    - Ma trận tương đồng kích thước [N_images, M_texts] = [5000, 25000].
    - Text-to-Image (t2i): mỗi caption là 1 query, truy xuất trên toàn bộ 5000
      ảnh; một query đúng nếu ảnh gốc của nó nằm trong top-K.
    - Image-to-Text (i2t): mỗi ảnh là 1 query, truy xuất trên toàn bộ 25000
      captions; một query đúng nếu BẤT KỲ caption nào của ảnh đó nằm trong top-K.
    - Recall@K = trung bình trên toàn bộ query (%), với K = 1, 5, 10.

Hàm cũng hỗ trợ protocol 1:1 (mỗi ảnh 1 caption) nếu ``image_ids=None``.
"""

from __future__ import annotations

import torch

DEFAULT_K_VALUES = (1, 5, 10)


def compute_recall_at_k(
    similarity_matrix: torch.Tensor,
    image_ids: torch.Tensor | list[int] | None = None,
    k_values: tuple[int, ...] | list[int] = DEFAULT_K_VALUES,
) -> dict[str, float]:
    """Tính Recall@K cho cả hai chiều i2t và t2i.

    Args:
        similarity_matrix: Tensor [N_images, M_texts] chứa điểm tương đồng
            (cosine similarity nếu embeddings đã được L2-normalize).
        image_ids: Chỉ số ảnh (theo hàng của ``similarity_matrix``) mà mỗi
            caption thuộc về, độ dài M. Ví dụ [0,0,0,0,0,1,1,1,1,1,...] với
            5 captions/ảnh. Nếu ``None``: giả định protocol 1:1 (M == N và
            caption j thuộc ảnh j).
        k_values: Các giá trị K cần tính.

    Returns:
        dict: {"i2t_R@1": ..., "t2i_R@1": ..., ...} đơn vị phần trăm (0-100).
        K vượt quá số lượng của chiều tương ứng bị bỏ qua: i2t cần K ≤ M_texts,
        t2i cần K ≤ N_images (tránh crash với sample nhỏ).
    """
    if similarity_matrix.dim() != 2:
        raise ValueError(f"similarity_matrix phải là ma trận 2 chiều, nhận được {tuple(similarity_matrix.shape)}")

    n_images, n_texts = similarity_matrix.shape
    device = similarity_matrix.device

    if image_ids is None:
        if n_images != n_texts:
            raise ValueError(
                "Cần truyền image_ids khi số captions khác số ảnh "
                f"(N_images={n_images}, M_texts={n_texts})."
            )
        image_ids_t = torch.arange(n_images, device=device)
    else:
        image_ids_t = torch.as_tensor(image_ids, device=device, dtype=torch.long)
        if image_ids_t.numel() != n_texts:
            raise ValueError(f"image_ids phải có {n_texts} phần tử, nhận được {image_ids_t.numel()}")
        if image_ids_t.numel() and (image_ids_t.min() < 0 or image_ids_t.max() >= n_images):
            raise ValueError("image_ids chứa chỉ số ảnh nằm ngoài khoảng hợp lệ")

    k_values = sorted({int(k) for k in k_values if k > 0})
    if not k_values:
        raise ValueError("k_values phải chứa ít nhất một giá trị dương")

    # Mỗi chiều có giới hạn K khác nhau: i2t xếp hạng trên M captions,
    # t2i xếp hạng trên N ảnh. Các K không khả thi sẽ bị bỏ qua (không crash).
    k_max_i2t = min(max(k_values), n_texts)
    k_max_t2i = min(max(k_values), n_images)
    image_index = torch.arange(n_images, device=device).unsqueeze(1)  # [N, 1]

    # --- Image-to-Text: mỗi ảnh truy xuất trên toàn bộ captions ---
    i2t_topk = similarity_matrix.topk(k_max_i2t, dim=1, largest=True).indices  # [N, k_max_i2t]
    i2t_hit = image_ids_t[i2t_topk] == image_index  # [N, k_max_i2t] bool

    # --- Text-to-Image: mỗi caption truy xuất trên toàn bộ ảnh ---
    t2i_topk = similarity_matrix.t().topk(k_max_t2i, dim=1, largest=True).indices  # [M, k_max_t2i]
    t2i_hit = t2i_topk == image_ids_t.unsqueeze(1)  # [M, k_max_t2i] bool

    results: dict[str, float] = {}
    for k in k_values:
        if k <= k_max_i2t:
            results[f"i2t_R@{k}"] = i2t_hit[:, :k].any(dim=1).float().mean().item() * 100.0
        if k <= k_max_t2i:
            results[f"t2i_R@{k}"] = t2i_hit[:, :k].any(dim=1).float().mean().item() * 100.0

    results["n_images"] = n_images
    results["n_texts"] = n_texts
    return results

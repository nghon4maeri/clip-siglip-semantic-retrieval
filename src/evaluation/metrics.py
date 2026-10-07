import torch

def compute_recall_at_k(similarity_matrix, k_values=[1, 5, 10]):
    """
    Tính toán chỉ số Recall@K cho bài toán Text-to-Image và Image-to-Text retrieval.
    
    Args:
        similarity_matrix (torch.Tensor): Ma trận kích thước [N_images, M_texts] 
                                          chứa cosine similarity giữa image và text embeddings.
        k_values (list): Danh sách các giá trị K để tính Recall@K.
        
    Returns:
        dict: Chứa các giá trị Recall@K cho Text-to-Image (t2i) và Image-to-Text (i2t).
    """
    N_images, M_texts = similarity_matrix.shape
    assert N_images == M_texts, "Ma trận similarity phải là ma trận vuông (chỉ hỗ trợ 1 text per image trong logic đơn giản này)"
    
    # 1. Image-to-Text (i2t)
    # Lấy top K indices lớn nhất trên mỗi hàng (tương ứng với 1 image query ra K texts)
    _, i2t_indices = similarity_matrix.topk(max(k_values), dim=1, largest=True)
    i2t_targets = torch.arange(N_images).view(-1, 1).expand_as(i2t_indices).to(i2t_indices.device)
    i2t_matches = (i2t_indices == i2t_targets)

    # 2. Text-to-Image (t2i)
    # Lấy top K indices lớn nhất trên mỗi cột (tương ứng với 1 text query ra K images)
    _, t2i_indices = similarity_matrix.T.topk(max(k_values), dim=1, largest=True)
    t2i_targets = torch.arange(M_texts).view(-1, 1).expand_as(t2i_indices).to(t2i_indices.device)
    t2i_matches = (t2i_indices == t2i_targets)

    results = {}
    
    for k in k_values:
        # Nếu có match trong top K thì count là 1, tính mean trên toàn bộ query
        i2t_recall = i2t_matches[:, :k].sum(dim=1).bool().float().mean().item() * 100
        t2i_recall = t2i_matches[:, :k].sum(dim=1).bool().float().mean().item() * 100
        
        results[f"i2t_R@{k}"] = i2t_recall
        results[f"t2i_R@{k}"] = t2i_recall
        
    return results

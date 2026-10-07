# Đánh giá model (Evaluation)

Sau khi CLIP/SigLIP retrieval xong, cần biết: "Model nào tốt hơn?"
Folder này nghiên cứu metric.

Ví dụ:
- Recall@1
- Recall@5
- Recall@10
- MRR

Ngoài accuracy còn có thể đánh giá:
- Inference latency
- Embedding extraction time
- GPU memory
- Index size

Vì vậy cuối cùng bạn không chỉ nói: "SigLIP tốt hơn CLIP."
Mà có thể nói: "SigLIP đạt Recall@K cao hơn trong benchmark X, trong khi CLIP có latency thấp hơn..."

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

---

## Hiện trạng cài đặt

- **Recall@1/5/10 (i2t + t2i):** đã cài đặt tại `src/evaluation/metrics.py` theo protocol
  chuẩn 5 captions/ảnh — định nghĩa đầy đủ trong `pipeline/README.md` mục 4.
- **Latency / Embedding time / GPU memory:** benchmark tự log vào `results/benchmark.json`
  (`src/evaluation/benchmark.py`).
- **MRR / Index size:** chưa cài đặt (không bắt buộc theo protocol đã chốt) — có thể bổ sung sau.

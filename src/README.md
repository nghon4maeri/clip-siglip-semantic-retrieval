# Source Code: Semantic Image Retrieval Pipeline

Thư mục `src/` đóng vai trò như một "Sub-repository" (Module mã nguồn độc lập) chứa toàn bộ phần cài đặt (implementation) của đồ án. Khác với các thư mục ở vòng ngoài (dành cho lý thuyết, literature review, báo cáo), thư mục này chỉ tập trung vào **Data Pipeline, Model Architecture, và Evaluation Script**.

## 📂 Cấu trúc mã nguồn

```text
src/
├── README.md               # Mô tả module mã nguồn này
├── demo.py                 # Toy script: Chạy thử inference nhanh trên 1 sample
├── models/
│   ├── clip_wrapper.py     # Lớp bọc (Wrapper) cho OpenAI CLIP model
│   └── siglip_wrapper.py   # Lớp bọc (Wrapper) cho Google SigLIP model
├── datasets/
│   └── dataset.py          # PyTorch Dataloader đọc file JSON (Karpathy splits MS COCO)
└── evaluation/
    ├── metrics.py          # Implement độ đo Recall@1, 5, 10
    └── benchmark.py        # Kịch bản End-to-End benchmark trên 5000 test images
```

## 🚀 Quick Start (Chạy thử Demo)

Để kiểm tra xem môi trường đã cài đặt đúng chưa, cũng như test thử kiến trúc mạng của CLIP và SigLIP trên một tập nhỏ hình ảnh/text, bạn có thể chạy file `demo.py` từ thư mục gốc của project:

```bash
# Đảm bảo bạn đang đứng ở thư mục gốc của repo:
python src/demo.py
```

*Lưu ý: File `demo.py` tự động tải một bức ảnh trên mạng và đưa qua 2 mô hình lớn để trích xuất Feature Embeddings, sau đó xuất ra độ tương đồng (Cosine Similarity / Softmax Probability).*

## 🧪 Benchmark Pipeline

Sau khi đảm bảo hệ thống Core Models chạy ổn định qua bước Quick Start, bạn có thể chạy Benchmark chính thức với bộ dữ liệu MS COCO:

```bash
python src/evaluation/benchmark.py
```

Pipeline sẽ thực hiện:
1. Load tập dữ liệu MS COCO Test (5K ảnh, 25K captions).
2. Trích xuất Text/Image Embeddings.
3. Nhân ma trận (Matrix Multiplication) để tìm Cosine Similarity.
4. Trả về kết quả đánh giá (Recall@K).

---
*Module này được thiết kế theo chuẩn mã nguồn mở, bạn hoàn toàn có thể tái sử dụng (reuse) các file `wrapper` trong thư mục `models/` cho bất kỳ project Image-Text Retrieval nào khác.*

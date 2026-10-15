# Source Code: Semantic Image Retrieval Pipeline

Thư mục `src/` chứa toàn bộ phần cài đặt của đồ án: **Data Pipeline, Model Wrappers, và Evaluation Script**. Các thư mục vòng ngoài (`problem/`, `related-work/`, `pipeline/`) dành cho lý thuyết và báo cáo.

## 📂 Cấu trúc mã nguồn

```text
src/
├── README.md               # Mô tả module mã nguồn này
├── demo.py                 # Toy script: inference nhanh 1 ảnh (không dùng cho benchmark)
├── models/
│   ├── clip_wrapper.py     # Wrapper cho OpenAI CLIP (qua HuggingFace)
│   └── siglip_wrapper.py   # Wrapper cho Google SigLIP (qua HuggingFace)
├── datasets/
│   ├── dataset.py          # PyTorch Dataset/DataLoader (Karpathy splits MS COCO)
│   └── paths.py            # Logic phân giải đường dẫn ảnh (dùng chung với scripts/)
└── evaluation/
    ├── metrics.py          # Recall@1/5/10 theo protocol chuẩn COCO (5 captions/ảnh)
    └── benchmark.py        # End-to-end benchmark trên 5000 ảnh test, xuất JSON
```

## 🚀 Quick Start (Smoke test trên bộ sample)

Không cần tải 5000 ảnh, chỉ cần bộ sample 5 ảnh:

```bash
# Từ thư mục gốc repo
python make_samples.py          # tải 5 ảnh test đầu tiên + sample annotations

python scripts/verify_data.py \
    --json-path datasets/coco/annotations/dataset_coco_sample.json \
    --expect-images 5

python src/evaluation/benchmark.py \
    --json-path datasets/coco/annotations/dataset_coco_sample.json \
    --limit 5
```

## 🧪 Benchmark Pipeline (đầy đủ)

Điều kiện: đã tải annotations (`python datasets/download_annotations.py`) và đủ 5000 ảnh test
(`python scripts/verify_data.py` báo `[OK] Đủ 5000/5000`).

```bash
python src/evaluation/benchmark.py --models clip,siglip --batch-size 64
```

Pipeline thực hiện:
1. Load tập test Karpathy (5000 ảnh, mỗi ảnh lấy tối đa 5 captions ⇒ 25.000 text queries).
2. Trích xuất Image/Text embeddings theo batch (mỗi model dùng đúng processor gốc, L2-normalize).
3. Tính ma trận tương đồng `[5000, 25000]` và Recall@1/5/10 cho cả i2t và t2i.
4. Đo thời gian từng giai đoạn, peak VRAM, số tham số.
5. Lưu `results/benchmark.json` kèm metadata: git commit, SHA256 annotations, phiên bản
   thư viện, device, protocol.

Các tùy chọn hữu ích:

| Cờ | Ý nghĩa |
|---|---|
| `--models clip,siglip` | Chọn model cần chạy |
| `--limit N` | Chỉ chạy N ảnh đầu (smoke test — **không dùng số liệu này cho báo cáo**) |
| `--batch-size`, `--text-batch-size` | Điều chỉnh tốc độ / bộ nhớ |
| `--clip-checkpoint`, `--siglip-checkpoint` | Đổi checkpoint (mặc định: B/16 của cả hai) |
| `--output` | Đường dẫn file JSON kết quả |

## ☁️ Chạy trên Kaggle (khuyến nghị)

Dùng notebook có sẵn: **`notebooks/kaggle_benchmark.ipynb`**. Notebook tự:
clone repo → tải & verify annotations → dò thư mục ảnh `val2014` → symlink → verify 5000 ảnh
→ chạy benchmark → in bảng kết quả.

## 🧷 Test metrics

```bash
python tests/test_metrics.py    # test toán học của Recall@K bằng ma trận tổng hợp
```

---
*Module này được thiết kế theo chuẩn mã nguồn mở, bạn hoàn toàn có thể tái sử dụng (reuse) các file `wrapper` trong thư mục `models/` cho bất kỳ project Image-Text Retrieval nào khác.*

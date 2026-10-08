# Bảng Phân công Công việc (Task Assignment)

Vì phần mã nguồn của mô hình (Model Wrappers) đã hoàn thiện, trọng tâm của nhóm hiện tại là: **Lý thuyết, Đánh giá (Eval), và Chạy Benchmark**. 

Dưới đây là bảng phân công dành cho 3 thành viên để có thể làm việc song song hiệu quả. Các bạn có thể đánh dấu `[x]` vào các task đã hoàn thành.

---

## 🧑‍🏫 Thành viên 1: Chuyên trách Lý thuyết & Khảo sát (Theory & Literature)
*Tập trung vào phần "chữ" và định hình học thuật cho toàn bộ đồ án.*

- [x] Định nghĩa Framework chung và Phát biểu bài toán (`problem/README.md`).
- [ ] Đọc sâu paper gốc của CLIP và SigLIP để rút ra ưu/nhược điểm cốt lõi.
- [ ] Đọc các paper liên quan (ALIGN, BLIP, OpenCLIP) trong thư mục `related-work/papers/`.
- [x] Lên bảng so sánh các SOTA theo framework (`related-work/README.md`).
- [ ] Viết phần **Cơ sở lý thuyết** và **Giải pháp** cho báo cáo Word/Slide (Nguyên lý -> Phương pháp).

## 🧑‍💻 Thành viên 2: Chuyên trách Code Đánh giá (Eval & Metrics)
*Tập trung hoàn thiện hệ thống đo lường và luồng dữ liệu.*

- [ ] Hoàn thiện PyTorch `DataLoader` cho tập MS COCO (`src/datasets/dataset.py`).
- [ ] Code các hàm đo lường độ tương đồng (Cosine Similarity).
- [ ] Code module tính toán thước đo **Recall@1, Recall@5, Recall@10** (`src/evaluation/metrics.py`).
- [ ] Viết script `test_eval.py` để chạy thử (sanity check) luồng tính toán metric trên 1 mini-batch.
- [ ] Viết tài liệu phần **Tiến trình hoạt động (Pipeline)** và **Mô tả Dataset/Metrics** cho báo cáo.

## 🧑‍🔬 Thành viên 3: Chuyên trách Benchmark & Phân tích Kết quả
*Tập trung vào việc vận hành hệ thống, đo lường tốc độ, và xuất ra số liệu cuối cùng.*

- [ ] Lắp ghép Model (đã có) + DataLoader (của TV2) + Metrics (của TV2) thành file `benchmark.py` hoàn chỉnh.
- [ ] Thiết lập môi trường và đẩy code lên Kaggle (hoặc server có GPU) để chạy.
- [ ] Chạy Benchmark toàn bộ tập MS COCO cho **CLIP**. Lưu lại log (thời gian chạy, VRAM, Recall scores).
- [ ] Chạy Benchmark toàn bộ tập MS COCO cho **SigLIP**. Lưu lại log.
- [ ] Vẽ biểu đồ so sánh hiệu năng (Accuracy) và tốc độ (Computational Complexity) giữa 2 mô hình.
- [ ] Viết phần **Bảng kết quả thử nghiệm**, **Đánh giá kết quả** và làm **Slide Demo**.

---

## 🚀 Các Mốc thời gian (Milestones) dự kiến
- **Milestone 1:** Hoàn tất đọc paper (TV1) và Code xong module Eval/Metrics (TV2).
- **Milestone 2:** Chạy xong Benchmark (TV3) và chốt được số liệu cuối cùng.
- **Milestone 3:** Gộp các phần bài viết của 3 thành viên thành 1 bản Báo cáo hoàn chỉnh và Slide thuyết trình.

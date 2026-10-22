# Báo cáo Thực nghiệm Đối chuẩn: Semantic Search CLIP vs SigLIP trên MS COCO và NewsSeek

---

## 1. Giới thiệu & Phạm vi Thử nghiệm

Báo cáo này lưu trữ và tổng hợp kết quả đối chuẩn (benchmark) thực nghiệm chi tiết giữa hai mô hình thị giác - ngôn ngữ nền tảng:
* **OpenAI CLIP (ViT-B/32)**: Không gian nhúng 512D Float32, Patch Size 32x32, Tokenizer BPE (vocab 49,408).
* **Google SigLIP (ViT-B/16-224)**: Không gian nhúng 768D Float32, Patch Size 16x16, Tokenizer WebLI SentencePiece (vocab 250,000).

Thử nghiệm được thực hiện trên hai kho ngữ liệu đại diện cho hai bài toán truy xuất đa phương thức:
1. **MS COCO Karpathy Test Split**: 5,000 ảnh tĩnh chuẩn hóa, 25,010 câu truy vấn chú thích đối xứng.
2. **NewsSeek Multilingual Video Dataset**: 698 video thời sự đa ngôn ngữ, 164,874 khung hình tiêu biểu (keyframes) được trích xuất theo scene boundaries.

---

## 2. Bảng Ma trận Số liệu Định lượng (Quantitative Benchmark Matrix)

### 2.1. Đánh giá trên MS COCO Karpathy Test Split (5K Images / 25,010 Captions)

| Chỉ số Đánh giá | OpenAI CLIP (ViT-B/32) | Google SigLIP (ViT-B/16-224) | Chênh lệch (Delta) | Mô hình Vượt trội |
| :--- | :--- | :--- | :--- | :--- |
| **Recall@1 (Text-to-Image)** | 30.51% | **47.81%** | +17.30% | SigLIP |
| **Recall@5 (Text-to-Image)** | 55.96% | **72.45%** | +16.49% | SigLIP |
| **Recall@10 (Text-to-Image)** | 66.82% | **80.98%** | +14.16% | SigLIP |
| **Mean Recall (mR)** | 51.10% | **67.08%** | +15.98% | SigLIP |
| **Mean Reciprocal Rank (MRR)**| 0.4262 | **0.5908** | +0.1646 | SigLIP |
| **Modality Gap** | 0.8269 | 1.0524 | +0.2255 | CLIP (Khoảng cách hẹp hơn) |
| **Điểm tương đồng trung bình (Positive)** | 0.3046 | 0.1150 | -0.1896 | CLIP (Do nhiệt độ t cao hơn) |
| **Thời gian mã hóa 5,000 ảnh (RTX 4060)** | 888.07 s (5.6 img/s) | **50.46 s (99.1 img/s)** | **17.6x Nhanh hơn** | SigLIP |
| **Thời gian mã hóa 25,010 văn bản** | 7.20 s | 24.26 s | +17.06 s | CLIP |

---

### 2.2. Đánh giá trên NewsSeek Benchmark (698 Videos / 164,874 Keyframes)

| Chỉ số Đánh giá | OpenAI CLIP (ViT-B/32) | Google SigLIP (ViT-B/16-224) | Chênh lệch (Delta) | Mô hình Vượt trội |
| :--- | :--- | :--- | :--- | :--- |
| **Recall@1 (Video Max-Pool)** | 1.86% | **2.29%** | +0.43% (+23.1% tương đối) | SigLIP |
| **Recall@5 (Video Max-Pool)** | 6.88% | **7.74%** | +0.86% | SigLIP |
| **Recall@10 (Video Max-Pool)** | 10.32% | **11.03%** | +0.71% | SigLIP |
| **Mean Recall (mR)** | 6.35% | **7.02%** | +0.67% | SigLIP |
| **Mean Reciprocal Rank (MRR)** | 0.0498 | **0.0548** | +0.0050 | SigLIP |
| **Thông lượng mã hóa (Google Colab A100)**| 42.9 fps | 42.9 fps | Song song FP16 In-Memory | Đồng hạng |
| **Tổng thời gian mã hóa (164,874 frames)**| 3,845.73 s (~64 phút) | 3,845.73 s (~64 phút) | Trích xuất song song | Đồng hạng |
| **Modality Gap** | 0.8664 | 1.0835 | +0.2171 | CLIP |
| **Độ trễ tìm kiếm FAISS IndexFlatIP (Local)**| 25.83 ms | **11.48 ms** | -14.35 ms | SigLIP |

---

## 3. Phân tích Chi tiết Hiệu năng Truy xuất Ngữ nghĩa

### 3.1. Sự vượt trội của SigLIP trên Ảnh Tĩnh (MS COCO)
Trên tập MS COCO, SigLIP đạt mức bứt phá áp đảo với +17.30% R@1 (47.81% so với 30.51%) và MRR tăng từ 0.4262 lên 0.5908:
* **Độ phân giải Patch 16x16**: Với cùng kích thước ảnh đầu vào $224 \times 224$, SigLIP phân chia ảnh thành $14 \times 14 = 196$ patch tokens, trong khi CLIP ViT-B/32 chỉ có $7 \times 7 = 49$ patch tokens. Số lượng visual tokens nhiều gấp 4 lần cho phép SigLIP nhận diện các vật thể nhỏ (ví dụ: quả bóng tennis, chiếc nĩa, biển báo giao thông xa) mà CLIP thường bỏ sót.
* **Không gian biểu diễn 768D**: Chiều không gian nhúng lớn hơn (+256 chiều so với 512D của CLIP) giúp lưu trữ các thuộc tính thuộc về màu sắc, chất liệu và quan hệ không gian mà không bị nén mất thông tin.

### 3.2. Hiệu năng Truy xuất Video Tin tức Đa ngôn ngữ (NewsSeek)
* Với các câu truy vấn tin tức dài hoặc đa ngôn ngữ (tiếng Việt, tiếng Tây Ban Nha, tiếng Trung), tokenizer SentencePiece 250k của SigLIP không bị hiện tượng cắt cụt chuỗi (Token Truncation). 
* Chiến lược **Max-Pooling theo video** (lấy keyframe có điểm tương đồng cao nhất đại diện cho video) giúp cả hai mô hình lọc bỏ các khung hình nhiễu hoặc khung hình chuyển cảnh tối màu, đạt độ chính xác cao nhất ở mốc sự kiện chính.
* SigLIP đạt Recall@1 cao hơn 23.1% về mặt tương đối so với CLIP trên tập 698 video thời sự.

---

## 4. Kết luận Thực nghiệm
Kết quả thực nghiệm trên cả hai tập ngữ liệu chứng minh Google SigLIP vượt trội hơn OpenAI CLIP ở mọi thang đo độ chính xác truy xuất (Recall@K, MRR), đồng thời sở hữu tốc độ mã hóa hình ảnh trên GPU nhanh hơn đáng kể.

# Hướng Dẫn Sử Dụng & Tiền Xử Lý Dữ Liệu NewsSeek

Thư mục này phục vụ cho việc tích hợp và tiền xử lý bộ dữ liệu **NewsSeek: Multilingual Event-Centric News Video Dataset** ([namnguynnnn/newsseek-core-v1](https://www.kaggle.com/datasets/namnguynnnn/newsseek-core-v1)) vào đồ án nghiên cứu benchmark so sánh **CLIP vs SigLIP**, sử dụng pipeline xử lý video dựa trên phân tích phân cảnh và chiến lược trích xuất khung hình đại diện (keyframe extraction).

---

## 1. Tổng Quan Về Tập Dữ Liệu NewsSeek

NewsSeek bao gồm **698 video thời sự truyền hình** với thời lượng trung bình ~8.8 phút (tổng cộng ~103 giờ video) thu thập từ các hãng thông tấn lớn trên thế giới (ABC News, NBC News, CBS News, NHK, VTV, TBS, Arirang, CCTV,...):

| Ngôn ngữ | Mã | Số lượng video | Tỷ lệ (%) | Ý nghĩa Benchmark |
| :--- | :---: | :---: | :---: | :--- |
| **Tiếng Anh (English)** | `en` | 615 | 88.1% | Đánh giá Zero-shot Retrieval & Domain Shift tin tức |
| **Tiếng Nhật (Japanese)** | `ja` | 37 | 5.3% | Đa ngôn ngữ (Multilingual): SigLIP (WebLI 109 ngôn ngữ) vs CLIP (WIT - chủ yếu English) |
| **Tiếng Việt (Vietnamese)**| `vi` | 23 | 3.3% | Đa ngôn ngữ: Đánh giá khả năng hiểu tiếng Việt không cần fine-tune |
| **Tiếng Hàn (Korean)** | `ko` | 12 | 1.7% | Đa ngôn ngữ Châu Á |
| **Tiếng Trung (Chinese)** | `zh` | 9 | 1.3% | Đa ngôn ngữ tượng hình |
| **Tiếng Hindi & Ả Rập** | `hi`, `ar` | 2 | 0.3% | Low-resource / Cross-lingual coverage |
| **Tổng cộng** | - | **698** | **100%** | **Toàn bộ tập NewsSeek Core v1** |

---

## 2. Pipeline Xử Lý Video & Trích Xuất Keyframe

Pipeline nằm trong module `src/video_pipeline/` gồm các bước:

1. **Phát hiện phân cảnh (Scene Boundary Detection):**
   * **TransNetV2 (`transnetv2`):** Mô hình mạng nơ-ron sâu chuyên phát hiện hard cut và transition mượt qua các cửa sổ temporal 100 khung hình. Hỗ trợ tự động khi cài đặt `transnetv2-pytorch`.
   * **OpenCV Color Histogram (`content_aware` / `auto`):** Thuật toán đo khoảng cách tương đồng không gian màu HSV giữa các khung hình liên tiếp. Cực kỳ nhanh (300-500 fps), không cần tải trọng số mô hình.
   * **Phân đoạn cửa sổ cố định (`fixed_window`):** Chia video đều đặn mỗi 10s-30s cho phóng sự dài.
2. **Chiến lược chọn Keyframe (Keyframe Selection Policy):**
   * **Chiến lược Phân tầng (Tiered Policy):**
     * Độ dài phân cảnh $\le 3\text{s}$: Trích xuất **1 keyframe** (khung hình ở tâm).
     * $3\text{s} < \text{Độ dài} \le 10\text{s}$: Trích xuất **3 keyframes** cách đều.
     * $\text{Độ dài} > 10\text{s}$: Trích xuất **5 keyframes** cách đều.
   * **Chiến lược Tuyến tính (Linear Policy):**
     * Số keyframe = $\lceil \text{Thời lượng (s)} \times \text{kps} \rceil$ (ví dụ: `kps = 0.2` tương đương 1 frame mỗi 5 giây).
3. **Chuẩn hóa & Tối ưu kích thước:**
   * Tự động resize ảnh keyframe về chuẩn 720p (chất lượng JPEG 90%) nhằm cân bằng giữa độ nét cho visual encoder và tốc độ I/O.
4. **Định dạng đầu ra (`keyframes.json`):**
   Mỗi video được lưu trữ tại thư mục riêng biệt kèm metadata chi tiết:
   ```text
   datasets/newsseek/processed/<video_id>/
   ├── keyframes/
   │   ├── frame_000075.jpg
   │   ├── frame_000225.jpg
   │   └── ...
   └── keyframes.json
   ```

---

## 3. Cách Thức Xử Lý Dữ Liệu Video

### 🚀 Cách 1: Tiền xử lý trực tiếp trên Kaggle (Khuyên Dùng Nhất)
Toàn bộ 698 file video mp4 (~35GB) nằm sẵn trên Kaggle (`/kaggle/input/newsseek-core-v1/videos/`). Việc chạy trực tiếp trên Kaggle giúp hoàn thành toàn bộ trong vài phút mà không cần tải dữ liệu nặng về máy:

1. Mở Kaggle Notebook, chọn **Add Input** -> `namnguynnnn/newsseek-core-v1`.
2. Tạo 1 code cell và dán nội dung từ file [`scripts/kaggle_process_newsseek.py`](file:///D:/User/nghonamjiwoo/My%20Project/clip-siglip-semantic-retrieval/scripts/kaggle_process_newsseek.py) hoặc chạy lệnh:
   ```bash
   python scripts/kaggle_process_newsseek.py
   ```
3. Script sẽ tự động:
   * Quét và tiền xử lý 698 video.
   * Tạo thư mục `/kaggle/working/newsseek_processed/` và file tổng `newsseek_manifest.json`.
   * Nén toàn bộ keyframes thành `newsseek_keyframes.zip` để tải về hoặc lưu thành Kaggle Output Dataset.

### 💻 Cách 2: Chạy trên máy Local
Nếu bạn đã tải video vào thư mục `datasets/newsseek/videos/`:

```bash
# Xử lý tất cả video bằng Tiered Strategy và tự động dò scene
python scripts/process_newsseek_videos.py \
    --csv datasets/newsseek/annotations/videos.csv \
    --video_root datasets/newsseek/videos \
    --output_dir datasets/newsseek/processed \
    --strategy tiered \
    --scene_mode auto

# Thử nghiệm trên 10 video tiếng Anh
python scripts/process_newsseek_videos.py \
    --languages en \
    --max_videos 10 \
    --strategy tiered
```

---

## 4. Pipeline Truy Vấn & Đánh Giá (Search & Retrieval Pipeline)

Module `src/video_pipeline/video_search.py` hỗ trợ 2 cấp độ truy vấn ngữ nghĩa:

1. **Frame-Level Retrieval:** Truy vấn trực tiếp từng keyframe ảnh riêng lẻ.
2. **Video-Level Retrieval (Max-Pooling & Top-K):**
   * Điểm tương đồng giữa câu truy vấn $q$ và video $V_i$ chứa các keyframes $k \in \mathcal{K}_i$:
     $$\text{Score}(V_i, q) = \max_{k \in \mathcal{K}_i} \left( \frac{\mathbf{v}_k \cdot \mathbf{v}_q}{\|\mathbf{v}_k\| \|\mathbf{v}_q\|} \right)$$
3. **Lọc trùng lặp thông minh (Near-Duplicate BLAS Filtering):**
   * Loại bỏ các khung hình quá tương đồng trong top kết quả trả về bằng ma trận tích vô hướng BLAS.
4. **Hợp nhất kết quả đa mô hình (Reciprocal Rank Fusion - RRF):**
   * Kết hợp xếp hạng giữa CLIP và SigLIP để tạo bộ truy vấn ensemble:
     $$\text{RRF\_Score}(d) = \sum_{m \in \{\text{CLIP}, \text{SigLIP}\}} \frac{w_m}{60 + \text{rank}_m(d)}$$

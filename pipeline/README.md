# Tiến trình Hoạt động & Đánh giá Hiệu suất (Pipeline & Evaluation)

Theo yêu cầu của đồ án, phần này mô tả chi tiết quy trình hoạt động (Pipeline) của hệ thống trong hai giai đoạn: Huấn luyện (Training/Learning) và Kiểm thử (Testing/Inference), cũng như cách thức đánh giá hiệu suất (Performance Metrics).

## 1. Giai đoạn Huấn luyện (Training/Learning Pipeline)

Mục tiêu của giai đoạn huấn luyện là học cách "căn chỉnh" (align) không gian đặc trưng của hình ảnh và văn bản. Trong dự án này, chúng ta xem xét hai luồng Pipeline đại diện cho **CLIP** (Contrastive Learning) và **SigLIP** (Sigmoid Learning).

### 1.1. Input và Groundtruth
* **Input:** Một batch gồm $N$ cặp (hình ảnh, văn bản mô tả). Ví dụ: $(I_1, T_1), (I_2, T_2), ..., (I_N, T_N)$.
* **Output xác thực (Groundtruth):** 
  * Các cặp $(I_i, T_i)$ có cùng chỉ số là các cặp dương tính (Positive pairs - hình ảnh và mô tả khớp nhau).
  * Các cặp $(I_i, T_j)$ với $i \neq j$ là các cặp âm tính (Negative pairs - hình ảnh và mô tả không khớp nhau).
  * Nhãn xác thực (Label) thực chất là ma trận định danh (Identity Matrix) kích thước $N \times N$, trong đó đường chéo chính có giá trị 1 (khớp) và các vị trí khác là 0 (không khớp).

### 1.2. Luồng xử lý (Forward Pass)
1. Truyền $N$ hình ảnh qua Image Encoder để thu được tập vector đặc trưng hình ảnh: $V_I = \{v_{I_1}, v_{I_2}, ..., v_{I_N}\}$.
2. Truyền $N$ câu văn bản qua Text Encoder để thu được tập vector đặc trưng văn bản: $V_T = \{v_{T_1}, v_{T_2}, ..., v_{T_N}\}$.
3. Chuẩn hóa L2 (L2 Normalization) cho tất cả các vector đặc trưng.
4. Tính toán ma trận độ tương đồng (Similarity Matrix) kích thước $N \times N$ bằng cách nhân ma trận: $S = V_I \cdot V_T^T \times e^\tau$ (với $\tau$ là tham số nhiệt độ - temperature).

### 1.3. Tính toán Loss Function và Cập nhật (Backward Pass)
* **Đối với CLIP (InfoNCE Loss):**
  * Áp dụng hàm Softmax lên từng hàng và từng cột của ma trận $S$ để tính xác suất.
  * Tính Cross-Entropy Loss so với Groundtruth.
  * *Hạn chế:* Đòi hỏi phải tính toán toàn cục trên cả dòng và cột, nên phụ thuộc rất nhiều vào Batch size lớn.
* **Đối với SigLIP (Sigmoid Loss):**
  * Áp dụng hàm Sigmoid trực tiếp lên từng phần tử $S_{i,j}$ của ma trận.
  * Tính Binary Cross-Entropy (BCE) Loss độc lập cho từng cặp ảnh-chữ.
  * *Ưu điểm:* Xử lý từng phần tử độc lập, tiết kiệm bộ nhớ, cho phép mở rộng Batch size cực lớn.

5. Sử dụng thuật toán tối ưu (vd: AdamW) để tính đạo hàm ngược (Backpropagation) và cập nhật trọng số $\theta_I, \theta_T$ của hai mạng.

---

## 2. Giai đoạn Kiểm thử (Testing / Inference Pipeline)

Mục tiêu của giai đoạn này là thực hiện chức năng **Text-to-Image Retrieval** trên một tập cơ sở dữ liệu cho trước (ví dụ: tập Validation của MS COCO).

### 2.1. Tiền xử lý Cơ sở dữ liệu (Offline Feature Extraction)
Để tăng tốc độ truy xuất, đặc trưng của hình ảnh được tính toán sẵn:
1. Cho toàn bộ hình ảnh trong tập dữ liệu (ví dụ 5000 ảnh MS COCO) chạy qua Image Encoder đã được huấn luyện.
2. Lưu trữ các vector $v_{I}$ này vào một cơ sở dữ liệu (Database Index). Giai đoạn này chỉ làm 1 lần.

### 2.2. Xử lý Truy vấn (Online Retrieval)
1. **Input:** Người dùng nhập một câu truy vấn văn bản $T_{query}$.
2. Đưa $T_{query}$ qua Text Encoder để lấy vector $v_{T_{query}}$.
3. Lấy vector này tính toán **Cosine Similarity** (tích vô hướng sau khi đã chuẩn hóa) với *toàn bộ* $v_{I}$ trong Database.
4. Sắp xếp mảng độ tương đồng theo thứ tự giảm dần.
5. **Output:** Trả về Top-K (VD: Top-1, Top-5, Top-10) hình ảnh có điểm số cao nhất.

---

## 3. Đánh giá Hiệu suất (Performance Metrics)

Việc "học một đằng, đánh giá một nẻo" là lỗi thường gặp. Tuy quá trình học sử dụng Contrastive/Sigmoid Loss (giảm thiểu sai số dự đoán ma trận), nhưng quá trình đánh giá lại cần đo lường khả năng **truy xuất chính xác**. Do đó, chúng ta sử dụng hai hệ thống thước đo sau:

### 3.1. Độ đo Độ chính xác (Accuracy / Retrieval Metrics)
Trong bài toán Retrieval, thước đo chuẩn mực nhất là **Recall@K**.
* **Định nghĩa:** Recall@K đo lường tỷ lệ phần trăm các truy vấn mà trong đó hình ảnh đích (groundtruth image) xuất hiện trong top K kết quả được hệ thống trả về.
* **Các mốc đánh giá:** R@1, R@5, và R@10.
  * R@1: Hình ảnh đúng nằm ngay ở vị trí top 1 (cực kỳ khắt khe).
  * R@5: Hình ảnh đúng nằm trong top 5.
  * R@10: Hình ảnh đúng nằm trong top 10.
* **Mối liên hệ với Loss:** Hàm Loss càng hội tụ, khoảng cách giữa ảnh và text khớp nhau càng nhỏ, dẫn đến xếp hạng (rank) của ảnh đích càng được đẩy lên cao, từ đó tăng chỉ số R@K.

### 3.2. Độ đo Phức tạp Tính toán (Computational Complexity)
Một mô hình tốt không chỉ chính xác mà còn phải nhanh và nhẹ. Ta so sánh CLIP và SigLIP qua:
1. **Số lượng tham số (Parameter Count):** Kích thước mô hình (vd: ViT-B/16 ~ 150M tham số).
2. **Kích thước Batch Size cực đại (Max Batch Size):** Do giới hạn bộ nhớ VRAM, SigLIP cho phép Batch Size lớn hơn nhiều so với CLIP trên cùng một phần cứng.
3. **Độ trễ Truy xuất (Retrieval Latency):** Thời gian tính toán Cosine Similarity cho 1 query đối với $N$ ảnh. Nhờ sử dụng Dot Product cơ bản, tốc độ này là cực kỳ nhanh ($O(N)$) và có thể tối ưu hóa bằng các thư viện tìm kiếm vector (như FAISS).

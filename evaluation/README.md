# Hệ Thống Thước Đo Đánh Giá & So Sánh (Evaluation Metrics)

Tài liệu này hệ thống hóa toàn bộ các tiêu chí, độ đo toán học và phương pháp đánh giá thực nghiệm nhằm so sánh toàn diện giữa mô hình **CLIP** và **SigLIP** trong bài toán **Semantic Search / Cross-Modal Retrieval** (Truy xuất Ngữ nghĩa Hình ảnh - Văn bản).

Nội dung bám sát tiêu chuẩn nghiên cứu học thuật tại `workflow.md`, làm rõ mối quan hệ giữa **Hàm mất mát lúc huấn luyện** và **Thước đo lúc kiểm thử**, đồng thời mở rộng ra các khía cạnh về **Hiệu năng tính toán** và **Hình học không gian nhúng**.

---

## 1. Mối Quan Hệ Giữa Loss Function và Metrics Đánh Giá

Một lỗi sai kinh điển trong nghiên cứu là *"học một đằng, đánh giá một nẻo"*. Trong bài toán Retrieval, mối liên hệ hữu cơ giữa hai giai đoạn này được giải thích như sau:

* **Trong Giai đoạn Huấn luyện (Training):**
  * **CLIP** dùng *InfoNCE Loss* (tối đa hóa phân phối Softmax của cặp dương tính so với toàn bộ batch).
  * **SigLIP** dùng *Pairwise Sigmoid Loss* (tối đa hóa xác suất nhị phân độc lập $P(y=1)$ của từng cặp dương tính và dìm các cặp âm tính về 0).
  * Cả hai hàm Loss đều có cùng một mục tiêu tối thượng: **Kéo khoảng cách góc giữa vector ảnh $v_I$ và vector văn bản $v_T$ của cùng một sự kiện lại gần 0, đồng thời đẩy các cặp không liên quan ra xa nhau**.

* **Trong Giai đoạn Kiểm thử (Testing / Retrieval):**
  * Khi hàm Loss hội tụ, tích vô hướng $\text{sim}(I_i, T_i) = v_{I_i} \cdot v_{T_i}^T$ (Cosine Similarity sau chuẩn hóa $L_2$) của cặp đúng sẽ đạt giá trị cực đại.
  * Phép so khớp này đẩy thứ hạng (rank) của vật thể đích lên các vị trí đầu tiên trong danh sách kết quả, từ đó trực tiếp làm tăng các chỉ số **Recall@K** và **MRR**.

---

## 2. Nhóm 1: Độ Đo Độ Chính Xác & Chất Lượng Xếp Hạng (Retrieval Accuracy & Ranking Quality)

Đây là nhóm thước đo cốt lõi đo lường năng lực "hiểu đúng và tìm đúng" của hệ thống.

```
                           MA TRẬN ĐỘ TƯƠNG ĐỒNG S (N x M)
                 Văn bản T_1       Văn bản T_2   ...   Văn bản T_M
  Hình ảnh I_1     S(1, 1)           S(1, 2)             S(1, M)   ──> Xếp hạng theo dòng (Image-to-Text)
  Hình ảnh I_2     S(2, 1)           S(2, 2)             S(2, M)
       ...           ...               ...                 ...
  Hình ảnh I_N     S(N, 1)           S(N, 2)             S(N, M)
                      │
                      ▼ Xếp hạng theo cột (Text-to-Image)
```

### 2.1. Recall@K (R@1, R@5, R@10)
**Định nghĩa:** Tỷ lệ phần trăm các câu truy vấn mà kết quả xác thực (Groundtruth) nằm trong danh sách **Top-K** kết quả có điểm tương đồng cao nhất được hệ thống trả về.

#### Công thức Toán học:
Với tập truy vấn $\mathcal{Q} = \{q_1, q_2, \dots, q_{|\mathcal{Q}|}\}$ và hàm kiểm tra $\mathbb{I}(\cdot)$:
$$\text{Recall@K} = \frac{1}{|\mathcal{Q}|} \sum_{i=1}^{|\mathcal{Q}|} \mathbb{I}\left( \text{Rank}(y_i^*) \le K \right) \times 100\%$$
*(Trong đó $\text{Rank}(y_i^*)$ là vị trí xếp hạng của đối tượng đúng $y_i^*$ trong danh sách sắp xếp theo độ tương đồng giảm dần).*

#### Đánh giá đa chiều trên 2 tác vụ:
1. **Text-to-Image (T2I):** Truy vấn bằng văn bản $\to$ Tìm hình ảnh phù hợp trong cơ sở dữ liệu.
   * `T2I R@1`: Thử thách khắt khe nhất (ảnh đúng phải nằm ngay ở vị trí số 1).
   * `T2I R@5`: Phản ánh trải nghiệm người dùng thực tế trên giao diện tìm kiếm (ảnh đúng nằm trong 5 kết quả đầu).
   * `T2I R@10`: Khảo sát vùng bao phủ mở rộng.
2. **Image-to-Text (I2T):** Truy vấn bằng hình ảnh $\to$ Tìm câu mô tả/tin tức khớp nhất (Tìm kiếm ngược / Caption Retrieval).

---

### 2.2. Mean Recall (mR)
**Định nghĩa:** Giá trị trung bình cộng của toàn bộ 6 chỉ số Recall@K trên cả hai chiều T2I và I2T:
$$mR = \frac{1}{6} \left( \text{R@1}_{T2I} + \text{R@5}_{T2I} + \text{R@10}_{T2I} + \text{R@1}_{I2T} + \text{R@5}_{I2T} + \text{R@10}_{I2T} \right)$$
* **Ý nghĩa:** Là chỉ số đơn lẻ tổng quát nhất đại diện cho năng lực biểu diễn đa phương thức của mô hình trên toàn bộ tập benchmark.

---

### 2.3. Mean Reciprocal Rank (MRR)
**Định nghĩa:** Giá trị trung bình của nghịch đảo thứ hạng của kết quả đúng đầu tiên:
$$\text{MRR} = \frac{1}{|\mathcal{Q}|} \sum_{i=1}^{|\mathcal{Q}|} \frac{1}{\text{Rank}(y_i^*)}$$
* **Ý nghĩa:** Nếu ảnh đúng xuất hiện ở vị trí số 1 $\to$ đóng góp $1.0$ điểm; nếu ở vị trí số 2 $\to$ đóng góp $0.5$ điểm; nếu trôi xuống vị trí số 10 $\to$ chỉ còn $0.1$ điểm. Chỉ số này phản ánh rất sát mức độ hài lòng của người dùng khi sử dụng công cụ tìm kiếm.

---

## 3. Nhóm 2: Độ Đo Hiệu Năng Vận Hành & Tài Nguyên Tính Toán (Operational Efficiency & Complexity)

Một mô hình có độ chính xác cao nhưng quá nặng hoặc phản hồi chậm chạp sẽ không thể ứng dụng trong các hệ thống tìm kiếm ngữ nghĩa quy mô lớn (Large-Scale Production).

| Chỉ số Đo lường | Đơn vị | Ý nghĩa Kỹ thuật trong Semantic Search |
| :--- | :---: | :--- |
| **1. Text Encoding Latency** | `ms / query` | Thời gian Text Encoder mã hóa câu query thành vector khi người dùng nhấn "Tìm kiếm" (Yêu cầu thời gian thực: $< 50\text{ ms}$). |
| **2. Image Indexing Speed** | `ms / image` hoặc `images / sec` | Tốc độ Image Encoder quét và đánh chỉ mục thư viện ảnh đưa vào Vector Database. |
| **3. Search Latency** | `ms / search` | Thời gian tính tích vô hướng ma trận giữa vector query với toàn bộ vector cơ sở dữ liệu và sort Top-K. |
| **4. Parameter Count** | `Triệu tham số (M)` | Tổng dung lượng tham số mạng của Image Encoder và Text Encoder (ảnh hưởng tới chi phí phần cứng). |
| **5. Model Weight Size** | `MB` | Dung lượng file checkpoint (`.safetensors` hoặc `.pt`) lưu trữ trên ổ đĩa. |
| **6. Embedding Dimension ($D$)** | `Số chiều` | Chiều dài vector đặc trưng ($D = 512, 768$). Vector càng dài $\to$ tốn RAM/Disk của Vector DB (Milvus/FAISS) và tính toán khoảng cách càng chậm. |
| **7. Peak VRAM Footprint** | `MB / GB` | Lượng bộ nhớ GPU tối đa bị chiếm dụng khi trích xuất theo batch size. |

---

## 4. Nhóm 3: Độ Đo Hình Học Không Gian Nhúng (Embedding Space Geometry)

Nhóm thước đo này giải phẫu trực tiếp bên trong không gian toán học $D$ chiều để kiểm tra chất lượng biểu diễn mà không phụ thuộc vào ngưỡng Top-K.

```
                           KHÔNG GIAN NHÚNG D-CHIỀU
                  Positive Pairs                    Negative Pairs
               (Khớp đúng thực tế)             (Khác nhau về ngữ nghĩa)
           ╭─────────────────────────╮       ╭─────────────────────────╮
           │       Cosine Score      │       │       Cosine Score      │
           │       v_I · v_T^T       │       │       v_I · v_T^T       │
           │       (0.5 ~ 0.9)       │       │      (-0.2 ~ 0.2)       │
           ╰────────────┬────────────╯       ╰────────────┬────────────╯
                        │                                 │
                        └───────────────┬─────────────────┘
                                        ▼
                         KHOẢNG CÁCH BIÊN (SEPARATION MARGIN)
                               Mô hình nào lớn hơn?
```

### 4.1. Modality Gap Distance (Khoảng cách Giữa Hai Phương thức)
Hiện tượng hai đám mây điểm của ảnh và văn bản bị tách rời nhau trong không gian chung:
* Tâm cụm vector hình ảnh: $\mu_I = \frac{1}{N} \sum_{i=1}^N v_{I_i}$
* Tâm cụm vector văn bản: $\mu_T = \frac{1}{M} \sum_{j=1}^M v_{T_j}$
* **Độ đo Modality Gap:**
  $$\Delta_{\text{modality}} = \|\mu_I - \mu_T\|_2$$
* **Ý nghĩa:** Đo lường xem cấu trúc hình học của InfoNCE hay Sigmoid Loss kéo hai thế giới Ảnh - Văn bản lại gần nhau hơn.

### 4.2. Separation Margin (Biên Độ Tách Biệt Điểm Số)
Khoảng cách giữa điểm số tương đồng trung bình của các cặp khớp đúng so với các cặp không khớp:
$$\text{Margin} = \mathbb{E}_{(i, i) \in \text{Pos}} [S_{i, i}] - \mathbb{E}_{(i, j) \in \text{Neg}, i \neq j} [S_{i, j}]$$
* **Ý nghĩa:** $\text{Margin}$ càng lớn thì ranh giới giữa đúng và sai càng rõ rệt, giúp kỹ sư dễ dàng thiết lập **Ngưỡng lọc tương đồng (Similarity Threshold)** để loại bỏ hoàn toàn kết quả rác mà không làm mất kết quả đúng.

---

## 5. Nhóm 4: Độ Đo Độ Bền Vững & Chuyển Dịch Miền (Robustness & Domain Shift)

Nhóm thước đo này đo lường khả năng đối mặt với các thử thách thực tế ngoài đời thực:

### 5.1. Tỷ lệ Suy giảm Dịch chuyển Miền (Domain Gap Drop $\Delta_{\text{domain}}$)
Đo lường mức độ sụt giảm hiệu năng khi đưa mô hình từ tập dữ liệu đời thường chuẩn mực (**MS COCO**) sang tập dữ liệu tin tức truyền hình thực tế (**NewsSeek**):
$$\Delta_{\text{domain}} = \frac{\text{Recall@K}_{\text{COCO}} - \text{Recall@K}_{\text{NewsSeek}}}{\text{Recall@K}_{\text{COCO}}} \times 100\%$$
* **Ý nghĩa:** Mô hình nào có $\Delta_{\text{domain}}$ thấp hơn chứng minh hàm mất mát của mô hình đó mang lại khả năng tổng quát hóa Zero-Shot bền bỉ hơn khi triển khai trong ứng dụng thực tế.

### 5.2. Độ Nhạy Lỗi Chính Tả (Typo / Noise Sensitivity)
Đo lường mức độ suy giảm Recall khi người dùng gõ câu truy vấn bị sai chính tả nhẹ (như hoán đổi ký tự, thiếu nguyên âm):
* Mô hình nào giữ được thứ hạng tốt hơn chứng minh tính linh hoạt của Text Tokenizer và Text Encoder trước dữ liệu người dùng không hoàn hảo.

---

## 6. Ma Trận Bảng Tổng Kết Đề Xuất Cho Báo Cáo / Slide

Dưới đây là khuôn mẫu bảng so sánh chuẩn mực để tổng hợp toàn bộ kết quả benchmark của đồ án:

| Trục Đánh Giá | Độ Đo Cụ Thể | 🔵 CLIP (ViT-B/32) | 🔴 SigLIP (ViT-B/16) | Nhận Xét & Phân Tích Kỹ Thuật |
| :--- | :--- | :---: | :---: | :--- |
| **Độ chính xác (MS COCO)** | T2I R@1 / R@5 / R@10 | *..% / ..% / ..%* | *..% / ..% / ..%* | SigLIP vượt trội nhờ Patch 16 và Sigmoid Loss |
| | I2T R@1 / R@5 / R@10 | *..% / ..% / ..%* | *..% / ..% / ..%* | Đo chiều truy xuất ngược |
| | Mean Recall (mR) | *..%* | *..%* | Năng lực biểu diễn tổng thể |
| **Độ chính xác (NewsSeek)** | T2I R@1 / R@5 / R@10 | *..% / ..% / ..%* | *..% / ..% / ..%* | Thử nghiệm tin tức Zero-shot thực tế |
| **Độ bền miền (Domain Gap)** | $\Delta_{\text{domain}}$ (T2I R@1) | *-..%* | *-..%* | Mô hình nào ít bị rớt điểm hơn khi gặp ảnh tin tức |
| **Hiệu năng Vận hành** | Text Encoding Latency | *.. ms / query* | *.. ms / query* | Thời gian phản hồi truy vấn người dùng |
| | Image Indexing Speed | *.. ms / image* | *.. ms / image* | Tốc độ trích xuất vector ảnh vào database |
| **Tài nguyên Tính toán** | Parameters / Model Size | *151M (~600MB)* | *203M (~800MB)* | CLIP gọn nhẹ hơn về số lượng tham số |
| **Không gian Hình học** | Modality Gap ($\Delta_{\text{modality}}$) | *..* | *..* | Khoảng cách không gian giữa vector ảnh và text |
| | Separation Margin | *..* | *..* | Độ tách bạch giữa Positive và Negative |

---
*Tài liệu này được định dạng theo tiêu chuẩn học thuật để trích dẫn trực tiếp vào Luận văn / Báo cáo đồ án và Slide bảo vệ.*

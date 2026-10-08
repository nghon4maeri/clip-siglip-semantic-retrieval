# Khảo sát Tổng quan (Related Works)

Dựa trên framework chung đã định nghĩa trong `problem/README.md`, phần này tóm tắt và so sánh các giải pháp State-of-the-Art (SOTA) trong lĩnh vực truy xuất hình ảnh theo ngữ nghĩa (Cross-modal Text-to-Image Retrieval). 

Việc so sánh được thực hiện nhất quán theo các tiêu chí (tương ứng với các công đoạn trong framework) nhằm làm nổi bật những đóng góp và hướng giải quyết "ẩn số" của từng giải pháp.

## 1. Bảng So sánh các Mô hình SOTA theo Framework

| Giải pháp (Mô hình) | Công đoạn 1: Image Encoder (Trích xuất ảnh) | Công đoạn 2: Text Encoder (Trích xuất văn bản) | Giai đoạn Huấn luyện: Hàm mục tiêu (Loss Function) | Công đoạn 3: Đo lường độ tương đồng (Similarity) | Điểm đặc trưng & Đóng góp chính |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CLIP** (OpenAI, 2021) | ResNet / ViT (Vision Transformer) | Transformer (Masked Self-Attention) | **InfoNCE Loss** (Contrastive Learning trên toàn bộ mini-batch) | Dot Product / Cosine Similarity | Tiên phong trong học biểu diễn đa phương thức (multimodal) quy mô lớn bằng dữ liệu nhiễu (400M cặp ảnh-text) |
| **ALIGN** (Google, 2021) | EfficientNet | BERT | InfoNCE Loss (Contrastive Learning) | Dot Product / Cosine Similarity | Không cần lọc dữ liệu đầu vào, sử dụng tập dữ liệu cực lớn (1.2 tỷ cặp) có nhiều nhiễu nhưng vẫn đạt hiệu suất cao. |
| **OpenCLIP** (LAION, 2022) | ViT | Transformer | InfoNCE Loss | Dot Product / Cosine Similarity | Phiên bản mã nguồn mở của CLIP được huấn luyện trên tập dữ liệu LAION (lên tới 2B/5B cặp), cung cấp nhiều kiến trúc đa dạng hơn. |
| **BLIP** (Salesforce, 2022) | ViT | Bi-directional Transformer | Kết hợp 3 Loss: Contrastive Loss (ITC), Image-Text Matching (ITM), và Language Modeling (LM) | Cosine Similarity + Cross-Attention Scoring | Sử dụng cơ chế CapFilt (Captioning and Filtering) để làm sạch dữ liệu web nhiễu (bootstrap). Hỗ trợ tốt cả Retrieval và Generation. |
| **BLIP-2** (Salesforce, 2023) | Các mô hình ViT đã pre-train (CLIP/EVA-CLIP) | Các mô hình LLM đã pre-train (OPT/Flan-T5) | Q-Former (Querying Transformer) với các hàm Loss tương tự BLIP | Sinh văn bản và So sánh | "Đóng băng" (Freeze) Image Encoder và LLM, chỉ huấn luyện mạng Q-Former ở giữa. Tiết kiệm chi phí tính toán đáng kể. |
| **SigLIP** (DeepMind, 2023) | ViT | Transformer | **Sigmoid Loss** (Tính toán độc lập từng cặp, không cần góc nhìn toàn cục của batch) | Dot Product / Cosine Similarity | Xóa bỏ nút thắt cổ chai (bottleneck) của hàm Softmax trong Contrastive Loss, cho phép mở rộng kích thước batch size một cách độc lập và tăng tốc độ hội tụ. |

## 2. Chi tiết theo từng công đoạn

### 2.1. Giải pháp cho Công đoạn 1 & 2 (Kiến trúc trích xuất đặc trưng)
* Hầu hết các mô hình SOTA đều dần chuyển dịch từ kiến trúc CNN (như ResNet, EfficientNet) sang **Vision Transformer (ViT)** để mã hóa hình ảnh nhờ khả năng mở rộng (scalability) tốt hơn với dữ liệu lớn.
* Đối với văn bản, kiến trúc **Transformer** vẫn thống trị (như GPT-style trong CLIP hoặc BERT-style trong ALIGN, BLIP).

### 2.2. Giải pháp cho Giai đoạn Huấn luyện (Căn chỉnh không gian nhúng)
Đây là công đoạn có nhiều sự cải tiến nhất:
* **CLIP / ALIGN / OpenCLIP:** Sử dụng Contrastive Learning với hàm mất mát **InfoNCE (dựa trên Softmax)**. Hàm này so sánh 1 cặp positive với $N-1$ cặp negative trong cùng một batch. Điều này đòi hỏi batch size phải rất lớn để có đủ mẫu negative (CLIP dùng batch size = 32,768).
* **BLIP:** Khắc phục nhược điểm học từ dữ liệu web nhiễu của CLIP bằng cách đưa thêm một module sinh văn bản (Captioner) và lọc (Filter) để tạo ra tập dữ liệu sạch hơn trước khi huấn luyện. Bổ sung thêm ITM loss sử dụng Cross-Attention để học các tương tác fine-grained giữa ảnh và chữ.
* **SigLIP:** Cải tiến đột phá về hàm mất mát. Thay vì so sánh chéo bằng Softmax (Contrastive), SigLIP coi bài toán là phân loại nhị phân (Binary Classification) cho từng cặp ảnh-chữ với **Sigmoid Loss**. Điều này giúp tối ưu bộ nhớ, loại bỏ sự phụ thuộc vào kích thước batch, và đặc biệt hiệu quả ở các batch size siêu lớn.

### 2.3. Giải pháp cho Công đoạn Đo lường (Similarity Measurement)
* Ở quy mô truy xuất lớn (Large-scale Retrieval), **Dot Product** hoặc **Cosine Similarity** vẫn là thước đo tối ưu nhất về mặt tốc độ (Computational Complexity). 
* BLIP có khả năng cung cấp độ đo chính xác hơn thông qua cơ chế Image-Text Matching (sử dụng Cross-Attention), nhưng độ phức tạp tính toán rất cao, không phù hợp cho việc đánh giá toàn bộ cơ sở dữ liệu mà chỉ dùng để "Re-ranking" (xếp hạng lại) các kết quả top-K đã được truy xuất sơ bộ bằng Cosine Similarity.

## 3. Khoảng trống Nghiên cứu (Gaps in Related Works)
Mặc dù CLIP và các biến thể đạt hiệu suất SOTA trên nhiều tác vụ, chúng vẫn có một số nhược điểm:
* **Độ phức tạp trong việc Scale-up:** Contrastive Loss của CLIP yêu cầu giao tiếp đa thiết bị (All-gather) cực kỳ lớn để duy trì batch size toàn cục, làm lãng phí tài nguyên mạng (bandwidth).
* **Hiệu năng trên dữ liệu ít:** Việc học biểu diễn bằng Contrastive Loss không hiệu quả nếu kích thước batch nhỏ.
* **Chính vì những khoảng trống này, đồ án tập trung cài đặt và so sánh SigLIP so với CLIP**, để kiểm chứng xem liệu việc thay thế InfoNCE Loss (Contrastive) bằng Sigmoid Loss trong giai đoạn học có thực sự giải quyết được bài toán mở rộng batch size và cải thiện độ chính xác (Performance) trên tập dữ liệu chuẩn MS COCO hay không.

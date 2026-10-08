# Phát biểu bài toán (Problem Statement)

Tài liệu này định nghĩa khuôn khổ (framework) chung cho bài toán Semantic Image Retrieval (Truy xuất hình ảnh theo ngữ nghĩa), bám sát các yêu cầu từ `workflow.md`. Trong phần này, bài toán được phát biểu một cách tổng quát, độc lập với bất kỳ mô hình cụ thể nào (như CLIP hay SigLIP).

## 1. Định nghĩa Bài toán (Input & Output)

Bài toán cốt lõi là **Cross-modal Retrieval** (Truy xuất chéo phương thức), cụ thể ở đây là **Text-to-Image Retrieval**:
- **Input (Đầu vào):** Một câu truy vấn bằng văn bản (Text Query) miêu tả nội dung, ngữ cảnh hoặc đối tượng của hình ảnh mong muốn tìm kiếm.
- **Output (Đầu ra):** Một tập hợp (hoặc danh sách) các hình ảnh từ cơ sở dữ liệu đã cho, được sắp xếp theo mức độ liên quan (độ tương đồng ngữ nghĩa) giảm dần so với văn bản đầu vào.

*(Lưu ý: Hệ thống cũng có thể thực hiện chiều ngược lại - Image-to-Text Retrieval, trong đó Input là hình ảnh và Output là danh sách mô tả văn bản. Tuy nhiên, trọng tâm ứng dụng của đồ án là Text-to-Image.)*

## 2. Framework Chung cho Hệ thống (Generic System Framework)

Một hệ thống truy xuất hình ảnh theo ngữ nghĩa dựa trên phương pháp học biểu diễn (Representation Learning) điển hình sẽ bao gồm 4 công đoạn chính sau đây:

1. **Công đoạn 1: Trích xuất đặc trưng hình ảnh (Image Feature Extraction)**
   - Hệ thống tiếp nhận hình ảnh thô từ cơ sở dữ liệu.
   - Sử dụng một bộ mã hóa (Image Encoder) để biến đổi hình ảnh thô thành một vector toán học đặc trưng trong một không gian đa chiều (Embedding Space).

2. **Công đoạn 2: Trích xuất đặc trưng văn bản (Text Feature Extraction)**
   - Hệ thống tiếp nhận câu truy vấn văn bản thô.
   - Sử dụng một bộ mã hóa (Text Encoder) để biến đổi văn bản thành một vector toán học đặc trưng trong cùng không gian đa chiều với hình ảnh.

3. **Công đoạn 3: Đo lường độ tương đồng (Similarity Measurement)**
   - Trong quá trình truy vấn (Inference), hệ thống so sánh vector đặc trưng của câu truy vấn văn bản với toàn bộ các vector đặc trưng của tập hình ảnh trong cơ sở dữ liệu.
   - Tính toán một điểm số (score) đại diện cho khoảng cách hoặc độ tương đồng giữa hai vector này.

4. **Công đoạn 4: Xếp hạng và Truy xuất (Ranking & Retrieval)**
   - Sắp xếp các hình ảnh theo thứ tự điểm số tương đồng từ cao xuống thấp.
   - Trả về top-K hình ảnh có điểm số cao nhất làm kết quả của quá trình truy xuất.

## 3. Các Ẩn số Cần Xác Định (Unknowns)

Dựa trên framework chung đã định nghĩa, để xây dựng thành công một hệ thống thực tế, chúng ta phải tìm ra giải pháp cho các "ẩn số" tại từng công đoạn:

* **Tại Công đoạn 1 (Image Extraction):**
  * *Ẩn số 1:* Hàm ánh xạ $f_I(I; \theta_I)$ nào phù hợp nhất để bảo toàn các thông tin ngữ nghĩa (vật thể, không gian, màu sắc) của ảnh? Kiến trúc mạng neural nào (CNNs, Vision Transformers) nên được sử dụng và bộ tham số $\theta_I$ được tối ưu ra sao?

* **Tại Công đoạn 2 (Text Extraction):**
  * *Ẩn số 2:* Hàm ánh xạ $f_T(T; \theta_T)$ nào có khả năng hiểu được ngôn ngữ tự nhiên, ngữ pháp và ngữ cảnh? Cấu trúc mạng (RNNs, Transformers) nào nên được dùng cho $\theta_T$?

* **Trong Giai đoạn Huấn luyện (Alignment/Learning):**
  * *Ẩn số 3:* Làm thế nào để "ép" hai không gian đặc trưng của hình ảnh và văn bản (vốn khác biệt về mặt bản chất) về chung một không gian thống nhất (Joint Embedding Space)? 
  * *Ẩn số 4:* Hàm mục tiêu / Hàm mất mát (Loss Function) nào là tối ưu để tính toán sai số giữa dự đoán của mạng và nhãn xác thực (groundtruth), qua đó điều chỉnh $\theta_I$ và $\theta_T$?

* **Tại Công đoạn 3 & 4 (Đo lường & Truy xuất):**
  * *Ẩn số 5:* Thước đo độ tương đồng $S(v_I, v_T)$ nào (vd: Cosine Similarity, Dot Product, Euclidean Distance) vừa đảm bảo tính chính xác cao, vừa tối ưu được độ phức tạp tính toán (Computational Complexity) để đáp ứng việc truy xuất trên dữ liệu quy mô lớn (Large-scale Retrieval)?

## 4. Tập Dữ liệu Thử nghiệm (Standard Dataset)

Để đánh giá công bằng và khách quan hiệu năng giải quyết các "ẩn số" trên của các mô hình khác nhau, dự án sử dụng tập dữ liệu chuẩn:

* **MS COCO (Microsoft Common Objects in Context):**
  * Đây là một trong những tập dữ liệu tiêu chuẩn (benchmark) phổ biến nhất cho các bài toán liên quan đến tầm nhìn và ngôn ngữ.
  * *Đặc điểm:* Chứa các hình ảnh đa dạng về ngữ cảnh hàng ngày. Điểm quan trọng là mỗi hình ảnh đều đi kèm với 5 câu chú thích (captions) do con người gán nhãn, phản ánh chính xác nội dung hình ảnh.
  * *Thách thức:* Yêu cầu mô hình phải hiểu được sự tương tác giữa nhiều đối tượng trong cùng một bức ảnh, thay vì chỉ nhận diện một vật thể đơn lẻ, đòi hỏi khả năng trích xuất đặc trưng sâu và tinh tế.

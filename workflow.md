# Hướng dẫn & Quy trình thực hiện Đồ án (Workflow)

Tài liệu này quy định luồng công việc, các bước thực hiện và những tiêu chuẩn bắt buộc phải tuân theo trong quá trình nghiên cứu, viết báo cáo cho đồ án. 

---

## Phần I. Mục đích & Yêu cầu chung
1. **Mục đích:** Giúp sinh viên có trải nghiệm thực tế trong nghiên cứu ứng dụng và làm việc nhóm.
2. **Yêu cầu:** Chỉ cần đọc, hiểu, và cài đặt lại (reproduce). Chưa đòi hỏi đề xuất giải pháp mới.

---

## Phần II. Các bước cần tiến hành
**2.1. Đọc kỹ tên đề tài**

**2.2. Xác định các từ khóa (keyword) để tìm tài liệu**

**2.3. Tích cực tìm kiếm tài liệu & đọc hiểu:**
- Ưu tiên các tài liệu trong các hội nghị, tạp chí chuyên ngành uy tín (CVPR, ICCV, International Journal of Computer Vision...), trong các luận văn Th.S (Master Thesis), luận án TS (PhD Thesis).
- Dùng các từ khóa: `Survey` | `Overview` | `Literature Review` | `Comprehensive Study` + chủ đề đang nghiên cứu.
- Hoặc tìm kiếm trên "Papers with Code".

**2.4. Phát biểu bài toán (Problem statement):**
- Input, Output là gì?
- Các tác vụ cần thực hiện là gì?
- Tập dữ liệu thử nghiệm (Standard dataset).

**2.5. Khảo sát tổng quan (Survey / Literature Review):**
- Nhằm trả lời 2 câu hỏi: Người ta đã làm gì rồi? Đồ án muốn làm gì tiếp?

**2.6. Giải pháp:**
- Chọn phương pháp tiên tiến đã công bố để trình bày lại.
- Trình tự trình bày theo mạch logic: Nguyên lý -> Phương pháp -> Giải thuật -> Chương trình minh họa.

**2.7. Cài đặt:**
- Môi trường cài đặt: phần cứng, phần mềm.
- Tập dữ liệu thử nghiệm.
- Bảng kết quả thử nghiệm.
- Đánh giá kết quả.

**2.8. Viết báo cáo (Doc, Slide)**

**2.9. Phân công:**
- Lập bảng phân công công việc cho từng thành viên với các cột mốc thời gian cụ thể.

---

## Phần III. Chi tiết các phần báo cáo & Lỗi thường gặp
Dưới đây là 8 tiêu chí quan trọng khi viết báo cáo và các lỗi thường gặp **cần đặc biệt tránh**:

**1. Phát biểu bài toán - Framework chung**
- **Yêu cầu:** Cần xác định framework chung cho hệ thống, gồm các công đoạn chính nào (chưa đi vào phương pháp cụ thể).
- **Lỗi thường gặp:** Xác định luôn phương pháp cụ thể trong framework. Làm như vậy sẽ không thấy được nhiều giải pháp có thể có trong các công đoạn, hạn chế sự sáng tạo.

**2. Phát biểu bài toán - Xác định ẩn số**
- **Yêu cầu:** Cần xác định được các ẩn số phải tìm trong các công đoạn.
- **Lỗi thường gặp:** Không xác định được các ẩn số phải tìm. Làm như vậy sẽ không hiểu được các giải pháp đã được công bố.

**3. Khảo sát tổng quan (Related Works)**
- **Yêu cầu:** Cần nêu được các giải pháp SOTA (State-of-the-Art) và so sánh chúng với cùng các cột tiêu chí theo các công đoạn đã nêu trong phát biểu bài toán.
- **Lỗi thường gặp:** 
  - Mỗi giải pháp được trình bày với dàn bài khác nhau khiến việc so sánh rất khó khăn.
  - Không trình bày các giải pháp ứng với các công đoạn ở mục phát biểu bài toán, làm khó nhận biết các giải pháp đã đóng góp thế nào trong các công đoạn.

**4. Giai đoạn Huấn luyện (Learning/Training)**
- **Yêu cầu:** Cần nêu rõ input, output xác thực (groundtruth) là gì, được đánh nhãn như thế nào và loss function là gì nhằm tối ưu các thông số của mạng.
- **Lỗi thường gặp:** Không đả động gì đến output xác thực, và không rõ nó được dùng như thế nào trong giai đoạn học.

**5. Tiến trình hoạt động (Pipeline)**
- **Yêu cầu:** Trong giai đoạn học và giai đoạn kiểm thử, cần trình bày rõ tiến trình hoạt động của hệ thống để ra được kết quả mong muốn.
- **Lỗi thường gặp:** Không cho thấy được hệ thống hoạt động như thế nào để ra được kết quả.

**6. Đánh giá hiệu suất (Performance/Metrics)**
- **Yêu cầu:** Cần xác định độ đo đánh giá về độ chính xác và cả độ phức tạp tính toán.
- **Lỗi thường gặp:** 
  - Không hiểu về độ đo đánh giá, độ phức tạp tính toán.
  - Không hiểu loss function có liên quan gì đến độ đo đánh giá, dẫn đến tình trạng "học một đằng, đánh giá một nẻo".

**7. Mô tả Dữ liệu (Datasets)**
- **Yêu cầu:** Cần chú ý nhiều hơn trong mô tả tập dữ liệu học, tập dữ liệu kiểm thử. Công tác đánh nhãn như thế nào. Số lượng mẫu. Tính đa dạng của mẫu. Tiêu chí xây dựng tập mẫu.
- **Lỗi thường gặp:** 
  - Không cho thấy tập dữ liệu chứa các thách thức gì. Nếu tập dữ liệu không chứa các thách thức của bài toán thì dù hệ thống có đạt performance rất cao cũng không thể khẳng định đây là giải pháp tốt.
  - Không hiểu được cách đánh nhãn dữ liệu như thế nào.

**8. Tìm khoảng trống nghiên cứu (Gaps in Related Works)**
- **Yêu cầu:** Trong related works, cần nhìn ra các khuyết điểm còn tồn đọng cần giải quyết trong các công đoạn.
- **Lỗi thường gặp:** 
  - Chưa nhìn ra được cần cải tiến nội dung gì trong các công đoạn.
  - Cần cải tiến độ chính xác hay độ phức tạp tính toán chưa được làm rõ.

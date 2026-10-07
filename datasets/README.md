# Hướng Dẫn Tải Dữ Liệu Benchmark (MS COCO)

Folder này dùng để chứa tập dữ liệu MS COCO được sử dụng để benchmark mô hình CLIP và SigLIP trong bài toán Semantic Image Retrieval. Chúng ta sử dụng **Karpathy splits** (phiên bản chia tập train/val/test phổ biến nhất trong research).

---

## 1. Dữ Liệu Annotations (File JSON)
File `dataset_coco.json` chứa thông tin đường dẫn ảnh, caption và việc chia tập train/val/test đã được tải tự động và lưu trong thư mục `datasets/coco/annotations/`.

---

## 2. Giải Pháp Khuyên Dùng: Chạy Benchmark Trực Tiếp Trên Kaggle 🚀
Vì dữ liệu hình ảnh lên tới gần 20GB, giải pháp tối ưu nhất để không tốn ổ cứng và băng thông mạng là **mang code lên Kaggle chạy** thay vì mang data về máy.

**Các bước thực hiện trên Kaggle Notebook:**
1. Tạo một Notebook mới trên Kaggle.
2. Chọn **Add Data**, tìm kiếm và thêm bộ dataset: `nadaibrahim/coco2014` (hoặc `awsaf49/coco-2014-dataset-images`). Dataset này sẽ lập tức được mount vào notebook của bạn mà không tốn 1 giây tải!
3. Clone repo của bạn vào Kaggle cell:
   ```bash
   !git clone https://github.com/nghon4maeri/clip-siglip-semantic-retrieval.git
   %cd clip-siglip-semantic-retrieval
   ```
4. Kéo thả file annotations (`dataset_coco.json`) bằng script:
   ```bash
   !python datasets/download_annotations.py
   ```
5. Symlink (tạo đường dẫn ảo) từ Kaggle dataset vào đúng folder `images` của project:
   ```bash
   !ln -s /kaggle/input/coco2014/val2014 datasets/coco/images/val2014
   ```
6. Chạy Benchmark và tận hưởng sức mạnh GPU T4/P100 miễn phí của Kaggle:
   ```bash
   !python src/evaluation/benchmark.py
   ```

---

## 3. Nếu Bạn Vẫn Muốn Tải Về Máy Local (Local Execution)
Nếu bạn bắt buộc phải chạy ở máy local, thư mục `images/` đã được ignore trong `.gitignore`. Bạn hãy tải tập **val2014** (chứa 5000 test images của Karpathy) bằng một trong hai cách:

**Cách A: Tải từ Kaggle (Tốc độ cao)**
- Link: [COCO 2014 Dataset (Nada Ibrahim)](https://www.kaggle.com/datasets/nadaibrahim/coco2014)
- Giải nén thư mục `val2014` thả vào `datasets/coco/images/`.

**Cách B: Tải từ Server gốc (Tốc độ chậm hơn)**
- Dùng trình duyệt hoặc IDM tải file: `http://images.cocodataset.org/zips/val2014.zip`
- Giải nén thả vào `datasets/coco/images/`.

## Cấu Trúc Thư Mục Chuẩn (Khi chạy Local)
```text
datasets/
├── README.md
└── coco/
    ├── annotations/
    │   └── dataset_coco.json
    └── images/
        ├── val2014/
        │   ├── COCO_val2014_000000000042.jpg
        │   └── ...
```

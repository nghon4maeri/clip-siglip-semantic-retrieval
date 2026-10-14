# Hướng Dẫn Tải Dữ Liệu Benchmark (MS COCO — Karpathy split)

Thư mục này chứa dữ liệu cho benchmark CLIP vs SigLIP. **Cả hai model bắt buộc dùng
cùng một bản annotations và cùng bộ ảnh** — đây là điều kiện để so sánh công bằng.

## Nguồn dữ liệu chuẩn (chốt)

| Thành phần | Nguồn | Ghi chú |
|---|---|---|
| Annotations | `cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip` → `dataset_coco.json` | Karpathy & Fei-Fei (CVPR 2015); script tự tải & verify |
| Ảnh (5.000 ảnh test) | MS COCO 2014 `val2014` | Khuyến nghị lấy qua Kaggle; hoặc tải `val2014.zip` (~6.3 GB) |

Split test theo Karpathy có **5.000 ảnh**. Protocol chuẩn lấy **5 captions đầu của mỗi ảnh
⇒ 25.000 text query** (file gốc có 10 ảnh chứa 6 captions — 10 caption thừa này bị bỏ qua
theo đúng protocol).

> Ảnh `val2014` gồm 40.504 file — nhiều hơn 5.000 ảnh test. Benchmark chỉ dùng đúng
> 5.000 ảnh có trong annotations, các ảnh còn lại không ảnh hưởng kết quả.

## 1. Tải annotations (bắt buộc, làm trước)

```bash
python datasets/download_annotations.py
```

Script sẽ:
1. Tải file zip và **kiểm tra tính toàn vẹn** (trước đây repo từng bị commit nhầm file zip tải dở).
2. Giải nén `dataset_coco.json` vào `datasets/coco/annotations/`.
3. Verify đúng **5.000 ảnh test** và mỗi ảnh đủ ≥ 5 captions.
4. In ra **SHA256** để ghi vào báo cáo (mọi người phải dùng cùng một hash).

File tải về nằm trong `.gitignore`, **không commit**.

## 2. Chuẩn bị ảnh

### Cách khuyên dùng: chạy trên Kaggle 🚀

Không cần tải ~6 GB về máy. Dùng notebook có sẵn trong repo:
**`notebooks/kaggle_benchmark.ipynb`** — notebook tự dò thư mục ảnh, symlink, verify
5.000 ảnh rồi chạy benchmark.

Các bước thủ công (nếu không dùng notebook):
1. Trên Kaggle: **Add Data** → thêm dataset `nadaibrahim/coco2014`
   (hoặc `awsaf49/coco-2014-dataset-images`).
2. Clone repo và tải annotations (bước 1).
3. Tìm thư mục chứa `val2014` trong `/kaggle/input`, rồi symlink:
   ```bash
   rm -rf datasets/coco/images/val2014   # bắt buộc: thư mục đã tồn tại ảnh sample
   ln -s /kaggle/input/<dataset>/val2014 datasets/coco/images/val2014
   ```
4. Verify trước khi chạy:
   ```bash
   python scripts/verify_data.py
   ```

### Cách chạy hoàn toàn ở máy local

- **Từ Kaggle (khuyến nghị):** [COCO 2014 Dataset (Nada Ibrahim)](https://www.kaggle.com/datasets/nadaibrahim/coco2014) —
  giải nén thư mục `val2014` vào `datasets/coco/images/`.
- **Tải chính thức:** `http://images.cocodataset.org/zips/val2014.zip` rồi giải nén vào
  `datasets/coco/images/`.

Sau khi có ảnh:

```bash
python scripts/verify_data.py   # phải báo đủ 5000/5000 ảnh
```

## 3. Bộ sample nhỏ để smoke test (không cần ảnh đầy đủ)

```bash
python make_samples.py
```

Tạo `dataset_coco_sample.json` + tải 5 ảnh test đầu tiên. Dùng để kiểm tra code chạy
thông trước khi lên Kaggle. Lưu ý: script tự fallback từ https sang http khi tải ảnh
(host `images.cocodataset.org` có chứng chỉ không khớp hostname — lỗi từ phía COCO).

## Cấu trúc thư mục sau khi hoàn tất

```text
datasets/
└── coco/
    ├── annotations/
    │   ├── dataset_coco.json          # 123.287 ảnh (đầy đủ), 5000 ảnh test — KHÔNG commit
    │   └── dataset_coco_sample.json   # 5 ảnh test, dùng smoke test (commit)
    └── images/
        └── val2014/
            ├── COCO_val2014_000000391895.jpg   # 5 ảnh sample (commit)
            └── ...                             # ảnh tải thêm (bị .gitignore)
```

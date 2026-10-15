"""Package dữ liệu của đồ án (annotations, ảnh, script tải).

Lưu ý quan trọng: thư mục này PHẢI có ``__init__.py``.

Nếu để nó là namespace package (không có ``__init__.py``), trên môi trường có
cài thư viện ``datasets`` của HuggingFace (Kaggle có sẵn), package thường
(site-packages) sẽ **thắng** namespace package trên ``sys.path`` và
``from datasets.download_annotations import ...`` sẽ thất bại.

Với ``__init__.py``, khi repo root nằm trước site-packages trong ``sys.path``
(điều mà benchmark.py / verify_data.py / make_samples.py đều đảm bảo), thư mục
này được resolve đúng là package của repo. Đổi lại, trong tiến trình đó
``import datasets`` sẽ trỏ về package này thay vì thư viện HuggingFace — dự án
không dùng thư viện HuggingFace ``datasets`` nên điều này là an toàn.
"""

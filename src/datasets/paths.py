"""Tiện ích đường dẫn ảnh dùng chung (không phụ thuộc torch/PIL).

Tách riêng để ``src/datasets/dataset.py`` và ``scripts/verify_data.py`` dùng
cùng một logic phân giải đường dẫn, tránh lệch hành vi.
"""

from __future__ import annotations

import os


def resolve_image_path(img_info: dict, image_dir: str) -> str:
    """Phân giải đường dẫn ảnh từ thông tin Karpathy.

    COCO có thêm ``filepath`` (vd: ``val2014``); Flickr30K thì không. Nếu thư
    mục con không tồn tại, thử tìm ảnh trực tiếp trong ``image_dir``.
    """
    filename = img_info["filename"]
    filepath = img_info.get("filepath") or ""
    if filepath:
        candidate = os.path.join(image_dir, filepath, filename)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(image_dir, filename)

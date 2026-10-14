"""Tiện ích trích xuất embedding, tương thích nhiều phiên bản transformers.

Lý do tồn tại: API của HuggingFace thay đổi giữa các version.

- **transformers v4**: ``model.get_image_features(...)`` / ``get_text_features(...)``
  trả về trực tiếp ``torch.Tensor``.
- **transformers v5**: hai hàm trên trả về ``BaseModelOutputWithPooling``,
  embedding nằm ở thuộc tính ``pooler_output``.

Helper này chuẩn hóa cả hai trường hợp về ``torch.Tensor`` để code benchmark
chạy được trên cả Kaggle (thường là v4.x) và môi trường mới (v5.x).
"""

from __future__ import annotations

import torch


def extract_embedding(output: object) -> torch.Tensor:
    """Chuẩn hóa output của ``get_*_features`` về tensor embedding.

    Args:
        output: ``torch.Tensor`` (v4) hoặc object có ``pooler_output`` (v5).

    Raises:
        TypeError: nếu không trích được embedding.
    """
    if torch.is_tensor(output):
        return output

    pooler_output = getattr(output, "pooler_output", None)
    if pooler_output is not None:
        return pooler_output

    raise TypeError(
        "Không trích được embedding từ output kiểu "
        f"{type(output).__name__}. Kiểm tra lại phiên bản transformers."
    )

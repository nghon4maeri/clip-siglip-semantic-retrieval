"""PyTorch Dataset/DataLoader cho MS COCO & Flickr30K theo Karpathy splits.

Mỗi item trả về đầy đủ 5 captions của ảnh (protocol chuẩn COCO). Việc
preprocessing (resize/normalize/tokenize) do processor của từng model đảm
nhiệm, nên Dataset chỉ trả về PIL Image và text thô.
"""

from __future__ import annotations

import json

from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.datasets.paths import resolve_image_path


class ImageTextRetrievalDataset(Dataset):
    """Đọc annotations Karpathy (``dataset_coco.json`` hoặc ``dataset_flickr30k.json``).

    Args:
        json_path: Đường dẫn file annotations.
        image_dir: Thư mục gốc chứa ảnh (chứa ``val2014/`` hoặc ảnh trực tiếp).
        split: 'train', 'val' (restval), hoặc 'test'.
        strict: Nếu True (mặc định), ảnh lỗi/thiếu sẽ raise lỗi ngay thay vì
            trả về ``None`` im lặng. Benchmark cần biết chính xác có bao nhiêu
            ảnh được đánh giá, nên không được bỏ qua lỗi một cách âm thầm.
        max_captions_per_image: Số captions tối đa lấy mỗi ảnh (mặc định 5 —
            đúng protocol COCO). File gốc Karpathy có 10 ảnh test chứa 6
            captions; lấy 5 đầu để đủ 25.000 text query chuẩn.
    """

    def __init__(
        self,
        json_path: str,
        image_dir: str,
        split: str = "test",
        strict: bool = True,
        max_captions_per_image: int | None = 5,
    ):
        self.image_dir = image_dir
        self.split = split
        self.strict = strict
        self.max_captions_per_image = max_captions_per_image

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.images = [img for img in data["images"] if img["split"] == split]
        if not self.images:
            raise ValueError(f"Không có ảnh nào với split='{split}' trong {json_path}")

    def __len__(self) -> int:
        return len(self.images)

    def _resolve_path(self, img_info: dict) -> str:
        return resolve_image_path(img_info, self.image_dir)

    def __getitem__(self, idx: int) -> dict:
        img_info = self.images[idx]
        img_path = self._resolve_path(img_info)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as exc:
            if self.strict:
                raise FileNotFoundError(
                    f"Không đọc được ảnh [{idx}] '{img_path}': {exc}. "
                    "Kiểm tra lại thư mục ảnh (xem datasets/README.md)."
                ) from exc
            image = None

        captions = [sentence["raw"].strip() for sentence in img_info["sentences"]]
        if self.max_captions_per_image is not None:
            captions = captions[: self.max_captions_per_image]

        return {
            "image_id": img_info["imgid"],
            "filename": img_info["filename"],
            "image": image,
            "captions": captions,
        }


def collate_fn(batch: list[dict]) -> dict:
    """Giữ image là list PIL và captions là list-of-list (không stack tensor)."""
    return {
        "image_ids": [item["image_id"] for item in batch],
        "filenames": [item["filename"] for item in batch],
        "images": [item["image"] for item in batch],
        "captions": [item["captions"] for item in batch],
    }


def get_dataloader(
    json_path: str,
    image_dir: str,
    split: str = "test",
    batch_size: int = 32,
    num_workers: int = 2,
    shuffle: bool = False,
    strict: bool = True,
) -> DataLoader:
    """Tạo DataLoader cho benchmark/inference (không dùng cho training)."""
    dataset = ImageTextRetrievalDataset(json_path, image_dir, split=split, strict=strict)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
    )

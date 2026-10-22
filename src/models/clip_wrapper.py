from pathlib import Path
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

def _extract_features(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        return output.last_hidden_state[:, 0, :]
    if hasattr(output, "__getitem__"):
        return output[0]
    return output

class CLIPWrapper:
    """
    Trình bao bọc (Wrapper) cho mô hình CLIP gốc từ OpenAI.
    Sử dụng bản port của Hugging Face Transformers để dễ dàng triển khai.
    """
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        snapshot_dir = Path.home() / ".cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
        target = str(snapshot_dir) if snapshot_dir.exists() else model_name
        self.model = CLIPModel.from_pretrained(target).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(target)
        self.model.eval()

    def get_image_embeddings(self, images: list) -> torch.Tensor:
        """
        Nhận vào list PIL Images, trả về L2-normalized image embeddings.
        """
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            raw = self.model.get_image_features(**inputs)
            image_features = _extract_features(raw)
        return image_features / image_features.norm(dim=-1, keepdim=True)

    def get_text_embeddings(self, texts: list) -> torch.Tensor:
        """
        Nhận vào list strings, trả về L2-normalized text embeddings.
        """
        inputs = self.processor(text=texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            raw = self.model.get_text_features(**inputs)
            text_features = _extract_features(raw)
        return text_features / text_features.norm(dim=-1, keepdim=True)

    def compute_similarity(self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Tính Cosine similarity
        """
        return image_embeddings @ text_embeddings.T

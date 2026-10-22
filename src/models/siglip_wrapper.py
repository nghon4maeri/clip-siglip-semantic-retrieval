from pathlib import Path
import torch
from transformers import AutoProcessor, AutoModel
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

class SigLIPWrapper:
    """
    Trình bao bọc (Wrapper) cho mô hình SigLIP từ Google.
    Sử dụng bản port chính thức trên Hugging Face Transformers (PyTorch).
    """
    def __init__(self, model_name="google/siglip-base-patch16-224", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        snapshot_dir = Path.home() / ".cache/huggingface/hub/models--google--siglip-base-patch16-224/snapshots/7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"
        target = str(snapshot_dir) if snapshot_dir.exists() else model_name
        self.processor = AutoProcessor.from_pretrained(target)
        self.model = AutoModel.from_pretrained(target).to(self.device)
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
        inputs = self.processor(text=texts, padding="max_length", truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            raw = self.model.get_text_features(**inputs)
            text_features = _extract_features(raw)
        return text_features / text_features.norm(dim=-1, keepdim=True)

    def compute_similarity(self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Đối với SigLIP, mặc dù hàm mất mát lúc huấn luyện là Pairwise Sigmoid, 
        quá trình inference truy xuất vẫn thường sử dụng dot product (cosine similarity).
        """
        return image_embeddings @ text_embeddings.T

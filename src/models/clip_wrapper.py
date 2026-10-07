import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

class CLIPWrapper:
    """
    Trình bao bọc (Wrapper) cho mô hình CLIP gốc từ OpenAI.
    Sử dụng bản port của Hugging Face Transformers để dễ dàng triển khai.
    """
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def get_image_embeddings(self, images: list) -> torch.Tensor:
        """
        Nhận vào list PIL Images, trả về L2-normalized image embeddings.
        """
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        return image_features / image_features.norm(dim=-1, keepdim=True)

    def get_text_embeddings(self, texts: list) -> torch.Tensor:
        """
        Nhận vào list strings, trả về L2-normalized text embeddings.
        """
        inputs = self.processor(text=texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        return text_features / text_features.norm(dim=-1, keepdim=True)

    def compute_similarity(self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Tính Cosine similarity
        """
        return image_embeddings @ text_embeddings.T

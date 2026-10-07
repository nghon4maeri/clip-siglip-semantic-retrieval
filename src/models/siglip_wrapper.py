import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image

class SigLIPWrapper:
    """
    Trình bao bọc (Wrapper) cho mô hình SigLIP từ Google.
    Sử dụng bản port chính thức trên Hugging Face Transformers (PyTorch).
    """
    def __init__(self, model_name="google/siglip-base-patch16-224", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
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
        inputs = self.processor(text=texts, padding="max_length", truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        return text_features / text_features.norm(dim=-1, keepdim=True)

    def compute_similarity(self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Đối với SigLIP, mặc dù hàm mất mát lúc huấn luyện là Pairwise Sigmoid, 
        quá trình inference truy xuất vẫn thường sử dụng dot product (cosine similarity).
        """
        return image_embeddings @ text_embeddings.T

import torch
from PIL import Image
import requests
import os
import sys

# Thêm đường dẫn project vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.clip_wrapper import CLIPWrapper
from src.models.siglip_wrapper import SigLIPWrapper

def run_demo():
    print("=== TẢI ẢNH SAMPLE ===")
    # Dùng 1 ảnh mẫu từ internet (ví dụ: ảnh một chú chó/mèo)
    url = "https://images.unsplash.com/photo-1543852786-1cf6624b9987?q=80&w=400&auto=format&fit=crop"
    try:
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        print("Tải ảnh thành công! (Kích thước:", image.size, ")")
    except Exception as e:
        print("Không thể tải ảnh từ internet. Đang tạo ảnh giả (dummy image)...")
        image = Image.new('RGB', (224, 224), color = 'red')

    # Định nghĩa một số câu query
    texts = [
        "a photo of a cute cat",
        "a photo of a dog playing in the park",
        "a red car driving on the highway",
        "an abstract painting"
    ]
    
    print("\nText Queries:")
    for i, t in enumerate(texts):
        print(f" {i+1}. {t}")

    print("\n" + "="*50)
    print("1. TEST VỚI MÔ HÌNH CLIP (OpenAI)")
    print("="*50)
    clip_model = CLIPWrapper("openai/clip-vit-base-patch32")
    
    # Extract features
    clip_img_embeds = clip_model.get_image_embeddings([image])
    clip_txt_embeds = clip_model.get_text_embeddings(texts)
    
    # Tính similarity (dot product vì đã được L2 normalized)
    clip_sim = (clip_img_embeds @ clip_txt_embeds.T).squeeze(0)
    clip_probs = clip_sim.softmax(dim=-1) # Softmax để chuyển thành xác suất (tuỳ chọn)
    
    for txt, prob in zip(texts, clip_probs):
        print(f" - {txt}: {prob.item() * 100:.2f}%")


    print("\n" + "="*50)
    print("2. TEST VỚI MÔ HÌNH SIGLIP (Google)")
    print("="*50)
    siglip_model = SigLIPWrapper("google/siglip-base-patch16-224")
    
    # Extract features
    siglip_img_embeds = siglip_model.get_image_embeddings([image])
    siglip_txt_embeds = siglip_model.get_text_embeddings(texts)
    
    # Tính similarity (dot product)
    siglip_sim = (siglip_img_embeds @ siglip_txt_embeds.T).squeeze(0)
    
    # SigLIP dùng Sigmoid (độc lập từng cặp) thay vì Softmax
    import math
    siglip_probs = torch.sigmoid(siglip_sim * 10.0 - 10.0) # Công thức sigmoid xấp xỉ temperature
    # Để minh họa đơn giản, ta in luôn Cosine Similarity Score gốc
    
    for txt, score in zip(texts, siglip_sim):
        print(f" - {txt} (Cosine Score): {score.item():.4f}")

    print("\n=== HOÀN TẤT DEMO ===")

if __name__ == "__main__":
    run_demo()

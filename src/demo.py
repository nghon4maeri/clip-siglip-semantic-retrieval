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
    print("=== DOWNLOADING SAMPLE IMAGE ===")
    url = "https://images.unsplash.com/photo-1543852786-1cf6624b9987?q=80&w=400&auto=format&fit=crop"
    try:
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        print("Image downloaded successfully! (Size:", image.size, ")")
    except Exception as e:
        print("Could not download image. Creating a dummy red image...")
        image = Image.new('RGB', (224, 224), color = 'red')

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
    print("1. TESTING WITH CLIP MODEL (OpenAI)")
    print("="*50)
    clip_model = CLIPWrapper("openai/clip-vit-base-patch32")
    
    clip_img_embeds = clip_model.get_image_embeddings([image])
    clip_txt_embeds = clip_model.get_text_embeddings(texts)
    
    clip_sim = (clip_img_embeds @ clip_txt_embeds.T).squeeze(0)
    clip_probs = clip_sim.softmax(dim=-1) 
    
    for txt, prob in zip(texts, clip_probs):
        print(f" - {txt}: {prob.item() * 100:.2f}%")

    print("\n" + "="*50)
    print("2. TESTING WITH SIGLIP MODEL (Google)")
    print("="*50)
    siglip_model = SigLIPWrapper("google/siglip-base-patch16-224")
    
    siglip_img_embeds = siglip_model.get_image_embeddings([image])
    siglip_txt_embeds = siglip_model.get_text_embeddings(texts)
    
    siglip_sim = (siglip_img_embeds @ siglip_txt_embeds.T).squeeze(0)
    
    for txt, score in zip(texts, siglip_sim):
        print(f" - {txt} (Cosine Score): {score.item():.4f}")

    print("\n=== DEMO COMPLETED ===")

if __name__ == "__main__":
    run_demo()

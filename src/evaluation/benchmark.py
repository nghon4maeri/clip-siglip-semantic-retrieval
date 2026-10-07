import torch
from tqdm import tqdm
from src.datasets.dataset import ImageTextRetrievalDataset
from src.evaluation.metrics import compute_recall_at_k
import sys
import os

# Thêm đường dẫn project vào sys.path để import models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.models.clip_wrapper import CLIPWrapper
from src.models.siglip_wrapper import SigLIPWrapper

def run_benchmark(model_wrapper, dataset, batch_size=32):
    """
    Trích xuất embeddings và tính Recall@K
    """
    model_wrapper.model.eval()
    
    all_image_embeds = []
    all_text_embeds = []
    
    print(f"Bắt đầu trích xuất features với batch_size={batch_size}...")
    
    # Lấy sample 500 ảnh đầu tiên từ tập test để đánh giá nhanh
    # Lưu ý: Cần chuyển sang DataLoader và batching thực thụ cho toàn tập 5K ảnh
    n_samples = min(len(dataset), 500)
    
    with torch.no_grad():
        for i in tqdm(range(n_samples), desc="Extracting"):
            item = dataset[i]
            image = item['image']
            
            # Chọn caption đầu tiên làm đại diện cho text query
            caption = item['captions'][0]
            
            if image is None:
                continue # Bỏ qua nếu ảnh bị lỗi hoặc chưa tải được
                
            # Compute Image Embedding
            img_embed = model_wrapper.get_image_embeddings([image])
            all_image_embeds.append(img_embed)
            
            # Compute Text Embedding
            txt_embed = model_wrapper.get_text_embeddings([caption])
            all_text_embeds.append(txt_embed)
            
    if not all_image_embeds:
        print("Không có embeddings nào được trích xuất. Vui lòng kiểm tra dataset images.")
        return None
        
    image_embeds = torch.cat(all_image_embeds, dim=0)
    text_embeds = torch.cat(all_text_embeds, dim=0)
    
    # Tính cosine similarity
    similarity_matrix = torch.matmul(image_embeds, text_embeds.T)
    
    results = compute_recall_at_k(similarity_matrix)
    return results

if __name__ == "__main__":
    # Đã cấu hình lại đường dẫn cho MS COCO
    DATASET_JSON = "datasets/coco/annotations/dataset_coco.json"
    IMAGE_DIR = "datasets/coco/images"
    
    if not os.path.exists(DATASET_JSON):
        print(f"Không tìm thấy file: {DATASET_JSON}")
        print("Vui lòng chạy file `datasets/download_annotations.py` trước!")
        exit(1)
        
    dataset = ImageTextRetrievalDataset(json_path=DATASET_JSON, image_dir=IMAGE_DIR, split="test")
    print(f"Tải thành công MS COCO dataset. Tổng số test samples: {len(dataset)}")
    
    print("\n" + "="*50)
    print("ĐÁNH GIÁ MÔ HÌNH CLIP")
    print("="*50)
    clip_model = CLIPWrapper("openai/clip-vit-base-patch32")
    clip_results = run_benchmark(clip_model, dataset)
    print("Kết quả CLIP:", clip_results)
    
    print("\n" + "="*50)
    print("ĐÁNH GIÁ MÔ HÌNH SIGLIP")
    print("="*50)
    siglip_model = SigLIPWrapper("google/siglip-base-patch16-224")
    siglip_results = run_benchmark(siglip_model, dataset)
    print("Kết quả SigLIP:", siglip_results)

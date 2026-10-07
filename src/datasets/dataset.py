import os
import json
from PIL import Image
from torch.utils.data import Dataset

class ImageTextRetrievalDataset(Dataset):
    """
    Dataset loader for MS COCO and Flickr30K using Karpathy splits.
    """
    def __init__(self, json_path, image_dir, split="test"):
        """
        Args:
            json_path (str): Đường dẫn đến file annotations json (dataset_coco.json hoặc dataset_flickr30k.json)
            image_dir (str): Đường dẫn đến thư mục chứa ảnh
            split (str): 'train', 'val', hoặc 'test' (mặc định)
        """
        self.image_dir = image_dir
        self.split = split
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.images = []
        for img_info in data['images']:
            if img_info['split'] == split:
                self.images.append(img_info)
                
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img_info = self.images[idx]
        
        # COCO có thêm filepath (ví dụ: val2014) trong Karpathy splits, Flickr30K thì rỗng
        if 'filepath' in img_info and img_info['filepath']:
            img_path = os.path.join(self.image_dir, img_info['filepath'], img_info['filename'])
            # Nếu tất cả ảnh được gộp chung 1 folder thì có thể dùng cách này:
            if not os.path.exists(img_path):
                img_path = os.path.join(self.image_dir, img_info['filename'])
        else:
            img_path = os.path.join(self.image_dir, img_info['filename'])
            
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Fallback for missing images during dev
            image = None
            
        # Lấy tất cả các captions raw
        captions = [sentence['raw'].strip() for sentence in img_info['sentences']]
        
        return {
            'image_id': img_info['imgid'],
            'filename': img_info['filename'],
            'image': image,
            'captions': captions
        }

def get_dataloader(json_path, image_dir, split="test", batch_size=32):
    # Dùng custom collate_fn vì captions là một list có độ dài thay đổi và image có thể khác size
    pass

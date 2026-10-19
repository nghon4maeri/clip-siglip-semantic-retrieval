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

class NewsSeekDataset(Dataset):
    """
    Dataset loader for NewsSeek Multilingual News Dataset.
    Supports both multi-keyframe manifest and flattened keyframe metadata,
    with flexible language filtering and frame/video retrieval modes.
    """
    def __init__(self, json_path, image_dir=None, language="all", mode="frame"):
        """
        Args:
            json_path (str): Path to metadata/manifest JSON.
            image_dir (str, optional): Base directory containing keyframe images.
            language (str): 'all' or specific code ('en', 'vi', 'ja', 'ko', 'zh').
            mode (str): 'frame' (each item is 1 keyframe) or 'video' (each item is 1 video with all keyframes).
        """
        self.json_path = json_path
        self.language = language.lower()
        self.mode = mode
        self.image_dir = image_dir or os.path.dirname(json_path)

        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        if isinstance(raw_data, dict):
            if 'videos' in raw_data:
                video_list = raw_data['videos']
            elif 'samples' in raw_data:
                video_list = raw_data['samples']
            else:
                video_list = [raw_data]
        elif isinstance(raw_data, list):
            video_list = raw_data
        else:
            video_list = []

        # Filter language if requested
        if self.language != "all":
            video_list = [
                v for v in video_list
                if str(v.get('language', 'en')).lower() == self.language
            ]

        self.videos = video_list

        if self.mode == "video":
            self.items = self.videos
        else:
            # Flatten into frame-level items
            self.items = []
            for v in self.videos:
                vid = v.get('video_id') or v.get('id', '')
                title = v.get('title') or v.get('caption', '')
                lang = v.get('language', 'en')
                source = v.get('source_name', '')
                categories = v.get('categories', '')

                # Check if it has a list of keyframes
                keyframes = v.get('keyframes', [])
                if keyframes:
                    for kf in keyframes:
                        img_rel = kf.get('image_path') or kf.get('relative_path') or kf.get('filename', '')
                        self.items.append({
                            'id': f"{vid}_{kf.get('frame_number', 0):06d}",
                            'video_id': vid,
                            'frame_number': kf.get('frame_number', 0),
                            'timestamp_sec': kf.get('timestamp_sec', 0.0),
                            'image_path': img_rel,
                            'caption': title,
                            'language': lang,
                            'source_name': source,
                            'categories': categories,
                        })
                else:
                    # Single keyframe entry fallback
                    img_rel = v.get('image_path') or v.get('filename', f"{vid}.jpg")
                    self.items.append({
                        'id': vid,
                        'video_id': vid,
                        'frame_number': 0,
                        'timestamp_sec': 0.0,
                        'image_path': img_rel,
                        'caption': title,
                        'language': lang,
                        'source_name': source,
                        'categories': categories,
                    })

    def __len__(self):
        return len(self.items)

    def _resolve_image_path(self, rel_path: str) -> str:
        candidates = [
            os.path.join(self.image_dir, rel_path),
            os.path.join(self.image_dir, 'keyframes', os.path.basename(rel_path)),
            os.path.join(self.image_dir, os.path.basename(rel_path)),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return candidates[0]

    def __getitem__(self, idx):
        item = self.items[idx]

        if self.mode == "video":
            vid = item.get('video_id') or item.get('id', str(idx))
            title = item.get('title') or item.get('caption', '')
            keyframes = item.get('keyframes', [])
            loaded_images = []
            for kf in keyframes:
                img_rel = kf.get('image_path') or kf.get('relative_path') or kf.get('filename', '')
                p = self._resolve_image_path(img_rel)
                try:
                    img = Image.open(p).convert('RGB')
                except Exception:
                    img = None
                loaded_images.append(img)

            return {
                'video_id': vid,
                'title': title,
                'captions': [title],
                'language': item.get('language', 'en'),
                'source_name': item.get('source_name', ''),
                'duration_seconds': item.get('duration_seconds', 0.0),
                'keyframes': keyframes,
                'images': loaded_images,
            }

        # Frame mode
        img_full_path = self._resolve_image_path(item['image_path'])
        try:
            image = Image.open(img_full_path).convert('RGB')
        except Exception:
            image = None

        caption = item.get('caption', '').strip()
        return {
            'image_id': item['id'],
            'video_id': item.get('video_id', ''),
            'filename': os.path.basename(item['image_path']),
            'image': image,
            'captions': [caption],
            'language': item.get('language', 'en'),
            'source_name': item.get('source_name', ''),
            'categories': item.get('categories', '')
        }



"""Script to generate the Kaggle notebook and metadata for NewsSeek Keyframe encoding."""

import json
from pathlib import Path
import pandas as pd
import nbformat as nbf

def create_newsseek_kernel():
    # Load 698 video records
    csv_path = Path('datasets/newsseek/annotations/videos.csv')
    df = pd.read_csv(csv_path)
    video_info = {}
    for _, row in df.iterrows():
        vid = str(row['video_id']).strip()
        video_info[vid] = {
            'title': str(row.get('title', '')),
            'language': str(row.get('language', 'en')).strip().lower(),
            'source_name': str(row.get('source_name', '')),
        }
    print(f"Loaded {len(video_info)} video metadata records.")

    nb = nbf.v4.new_notebook()
    cells = []

    # Cell 1: Markdown title
    cells.append(nbf.v4.new_markdown_cell(
        "# NewsSeek: 166K Keyframes CLIP vs SigLIP Feature Encoding & Multilingual Benchmark\n\n"
        "This notebook performs:\n"
        "1. GPU batch feature extraction for 166,241 scene-aware keyframes using `openai/clip-vit-base-patch32` and `google/siglip-base-patch16-224`.\n"
        "2. Multilingual query encoding for 698 video events (EN, VI, JA, etc.).\n"
        "3. Exact FAISS `IndexFlatIP` vector store construction and artifact packaging for live search.\n"
        "4. Video-wise Max-Pooling Cross-Modal Retrieval Benchmark (Recall@1, 5, 10, MRR, Modality Gap)."
    ))

    # Cell 2: Imports & Environment
    cell2_code = r"""!pip install -q faiss-cpu transformers

import os
import sys
import time
import json
import zipfile
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import faiss
from transformers import AutoProcessor, AutoModel, CLIPProcessor, CLIPModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
if torch.cuda.is_available():
    print(f'Device Name: {torch.cuda.get_device_name(0)}')
    print(f'VRAM Allocated: {torch.cuda.memory_allocated(0)/(1024**2):.1f} MB')
"""
    cells.append(nbf.v4.new_code_cell(cell2_code))

    # Cell 3: Embed Video Info Dictionary
    video_info_json = json.dumps(video_info, ensure_ascii=False)
    cell3_code = (
        f"# 698 Video Metadata Records (ID, Title, Language, Source)\n"
        f"VIDEO_METADATA = json.loads({repr(video_info_json)})\n"
        f"print(f'Loaded {{len(VIDEO_METADATA)}} video metadata entries.')\n"
    )
    cells.append(nbf.v4.new_code_cell(cell3_code))

    # Cell 4: Locate Manifest & Keyframe Images
    cell4_code = r"""# Locate newsseek_manifest.json
search_roots = [Path('/kaggle/input'), Path('../datasets'), Path('./datasets'), Path('./')]
manifest_path = None
for r in search_roots:
    matches = list(r.glob('**/newsseek_manifest.json'))
    if matches:
        manifest_path = matches[0]
        break

if not manifest_path:
    raise FileNotFoundError('newsseek_manifest.json not found in search paths!')

print(f'Manifest located at: {manifest_path}')
with open(manifest_path, 'r', encoding='utf-8') as f:
    manifest_raw = json.load(f)

video_records = list(manifest_raw.values()) if isinstance(manifest_raw, dict) else manifest_raw
print(f'Total videos in manifest: {len(video_records)}')

# Find keyframes directory
base_dataset_dir = manifest_path.parent
print(f'Base dataset dir: {base_dataset_dir}')

# Flatten all keyframes into a structured list
all_keyframes = []
for v in video_records:
    vid = v['video_id']
    v_info = VIDEO_METADATA.get(vid, {'title': v.get('video_file', vid), 'language': 'en', 'source_name': ''})
    for kf in v.get('keyframes', []):
        kf_rel = kf['file_path']
        cand1 = base_dataset_dir / kf_rel
        cand2 = base_dataset_dir / 'newsseek_keyframes' / kf_rel
        cand3 = base_dataset_dir / Path(kf_rel).name
        
        actual_path = None
        if cand1.exists():
            actual_path = cand1
        elif cand2.exists():
            actual_path = cand2
        elif cand3.exists():
            actual_path = cand3
        else:
            actual_path = cand1 # fallback
            
        all_keyframes.append({
            'kf_idx': len(all_keyframes),
            'video_id': vid,
            'frame_number': kf['frame_number'],
            'timestamp_sec': kf['timestamp_sec'],
            'scene_index': kf['scene_index'],
            'file_path': str(actual_path),
            'relative_path': kf_rel,
            'video_title': v_info['title'],
            'language': v_info['language'],
        })

print(f'Flattened {len(all_keyframes)} total keyframes across {len(video_records)} videos.')
sample_kf = all_keyframes[0]
print(f'Sample Keyframe: {sample_kf["relative_path"]}, exists: {Path(sample_kf["file_path"]).exists()}')
"""
    cells.append(nbf.v4.new_code_cell(cell4_code))

    # Cell 5: PyTorch Dataset for fast batch image loading
    cell5_code = r"""def extract_tensor_features(output):
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, 'pooler_output') and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, 'last_hidden_state') and output.last_hidden_state is not None:
        return output.last_hidden_state[:, 0, :]
    if hasattr(output, '__getitem__'):
        return output[0]
    return output

class KeyframeDataset(Dataset):
    def __init__(self, keyframe_list, processor, is_siglip=False):
        self.items = keyframe_list
        self.processor = processor
        self.is_siglip = is_siglip
        self.fallback_img = Image.new('RGB', (224, 224), color=(128, 128, 128))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p = self.items[idx]['file_path']
        try:
            img = Image.open(p).convert('RGB')
        except Exception:
            img = self.fallback_img
        return img

def collate_fn_clip(batch):
    return clip_processor(images=batch, return_tensors='pt')

def collate_fn_siglip(batch):
    return siglip_processor(images=batch, return_tensors='pt')

print('Defined KeyframeDataset and Collate functions.')
"""
    cells.append(nbf.v4.new_code_cell(cell5_code))

    # Cell 6: CLIP ViT-B/32 Extraction
    cell6_code = r"""# Load CLIP ViT-B/32
print('--> Loading openai/clip-vit-base-patch32...')
clip_model_id = 'openai/clip-vit-base-patch32'
clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
clip_model = CLIPModel.from_pretrained(clip_model_id).to(device).eval()

clip_loader = DataLoader(
    KeyframeDataset(all_keyframes, clip_processor),
    batch_size=256,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn_clip,
    pin_memory=True,
)

clip_kf_embeds = []
t0 = time.time()
with torch.no_grad():
    for batch in tqdm(clip_loader, desc='CLIP Keyframes'):
        inputs = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            feats = extract_tensor_features(clip_model.get_image_features(**inputs))
            feats = feats / feats.norm(dim=-1, keepdim=True)
        clip_kf_embeds.append(feats.cpu().float())

clip_kf_tensor = torch.cat(clip_kf_embeds, dim=0)
clip_kf_time = time.time() - t0
print(f'CLIP Keyframes: {len(clip_kf_tensor)} in {clip_kf_time:.2f}s ({len(clip_kf_tensor)/clip_kf_time:.1f} fps)')

# Encode Video Titles (Queries)
unique_vids = [v['video_id'] for v in video_records]
unique_titles = [VIDEO_METADATA.get(v, {}).get('title', v) for v in unique_vids]
print(f'Encoding {len(unique_titles)} video queries with CLIP...')

clip_txt_embeds = []
with torch.no_grad():
    for i in range(0, len(unique_titles), 128):
        batch_txt = unique_titles[i : i + 128]
        inputs = clip_processor(text=batch_txt, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            feats = extract_tensor_features(clip_model.get_text_features(**inputs))
            feats = feats / feats.norm(dim=-1, keepdim=True)
        clip_txt_embeds.append(feats.cpu().float())

clip_txt_tensor = torch.cat(clip_txt_embeds, dim=0)
del clip_model, clip_processor
torch.cuda.empty_cache()
"""
    cells.append(nbf.v4.new_code_cell(cell6_code))

    # Cell 7: SigLIP ViT-B/16-224 Extraction
    cell7_code = r"""# Load SigLIP ViT-B/16-224
print('--> Loading google/siglip-base-patch16-224...')
siglip_model_id = 'google/siglip-base-patch16-224'
siglip_processor = AutoProcessor.from_pretrained(siglip_model_id)
siglip_model = AutoModel.from_pretrained(siglip_model_id).to(device).eval()

siglip_loader = DataLoader(
    KeyframeDataset(all_keyframes, siglip_processor, is_siglip=True),
    batch_size=256,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn_siglip,
    pin_memory=True,
)

siglip_kf_embeds = []
t0 = time.time()
with torch.no_grad():
    for batch in tqdm(siglip_loader, desc='SigLIP Keyframes'):
        inputs = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            feats = extract_tensor_features(siglip_model.get_image_features(**inputs))
            feats = feats / feats.norm(dim=-1, keepdim=True)
        siglip_kf_embeds.append(feats.cpu().float())

siglip_kf_tensor = torch.cat(siglip_kf_embeds, dim=0)
siglip_kf_time = time.time() - t0
print(f'SigLIP Keyframes: {len(siglip_kf_tensor)} in {siglip_kf_time:.2f}s ({len(siglip_kf_tensor)/siglip_kf_time:.1f} fps)')

# Encode Video Titles
siglip_txt_embeds = []
with torch.no_grad():
    for i in range(0, len(unique_titles), 128):
        batch_txt = unique_titles[i : i + 128]
        inputs = siglip_processor(text=batch_txt, padding='max_length', truncation=True, return_tensors='pt').to(device)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            feats = extract_tensor_features(siglip_model.get_text_features(**inputs))
            feats = feats / feats.norm(dim=-1, keepdim=True)
        siglip_txt_embeds.append(feats.cpu().float())

siglip_txt_tensor = torch.cat(siglip_txt_embeds, dim=0)
del siglip_model, siglip_processor
torch.cuda.empty_cache()
"""
    cells.append(nbf.v4.new_code_cell(cell7_code))

    # Cell 8: FAISS Index Construction & Packaging (Saved FIRST before evaluation)
    cell8_code = r"""# Build FAISS vector stores for NewsSeek
out_dir = Path('/kaggle/working')
storage_dir = out_dir / 'storage' / 'newsseek'
clip_store_dir = storage_dir / 'clip'
siglip_store_dir = storage_dir / 'siglip'
clip_store_dir.mkdir(parents=True, exist_ok=True)
siglip_store_dir.mkdir(parents=True, exist_ok=True)

# 1. Save raw numpy arrays
np.save(out_dir / 'newsseek_clip_keyframes.npy', clip_kf_tensor.numpy())
np.save(out_dir / 'newsseek_clip_titles.npy', clip_txt_tensor.numpy())
np.save(out_dir / 'newsseek_siglip_keyframes.npy', siglip_kf_tensor.numpy())
np.save(out_dir / 'newsseek_siglip_titles.npy', siglip_txt_tensor.numpy())
print('Saved .npy embedding arrays.')

# 2. Build metadata list
metadata_list = []
for kf in all_keyframes:
    metadata_list.append({
        'id': f'newsseek_{kf["kf_idx"]}',
        'kf_idx': kf['kf_idx'],
        'video_id': kf['video_id'],
        'frame_number': kf['frame_number'],
        'timestamp_sec': kf['timestamp_sec'],
        'scene_index': kf['scene_index'],
        'relative_path': kf['relative_path'],
        'video_title': kf['video_title'],
        'language': kf['language'],
    })

# 3. Build & Save CLIP FAISS Store (166k vectors)
clip_index = faiss.IndexFlatIP(512)
clip_vecs = np.ascontiguousarray(clip_kf_tensor.numpy(), dtype=np.float32)
faiss.normalize_L2(clip_vecs)
clip_index.add(clip_vecs)
faiss.write_index(clip_index, str(clip_store_dir / 'index.faiss'))

clip_meta_payload = {
    'dim': 512,
    'index_type': 'FlatIP',
    'model_name': 'clip',
    'count': len(metadata_list),
    'items': metadata_list,
}
with open(clip_store_dir / 'metadata.json', 'w', encoding='utf-8') as f:
    json.dump(clip_meta_payload, f, ensure_ascii=False, indent=2)
print(f'Saved CLIP VectorStore ({clip_index.ntotal} vectors) -> {clip_store_dir}')

# 4. Build & Save SigLIP FAISS Store (166k vectors)
siglip_index = faiss.IndexFlatIP(768)
siglip_vecs = np.ascontiguousarray(siglip_kf_tensor.numpy(), dtype=np.float32)
faiss.normalize_L2(siglip_vecs)
siglip_index.add(siglip_vecs)
faiss.write_index(siglip_index, str(siglip_store_dir / 'index.faiss'))

siglip_meta_payload = {
    'dim': 768,
    'index_type': 'FlatIP',
    'model_name': 'siglip',
    'count': len(metadata_list),
    'items': metadata_list,
}
with open(siglip_store_dir / 'metadata.json', 'w', encoding='utf-8') as f:
    json.dump(siglip_meta_payload, f, ensure_ascii=False, indent=2)
print(f'Saved SigLIP VectorStore ({siglip_index.ntotal} vectors) -> {siglip_store_dir}')

# 5. Pre-package ZIP archive
zip_path = out_dir / 'newsseek_vector_stores.zip'
with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    zf.write(clip_store_dir / 'index.faiss', 'storage/newsseek/clip/index.faiss')
    zf.write(clip_store_dir / 'metadata.json', 'storage/newsseek/clip/metadata.json')
    zf.write(siglip_store_dir / 'index.faiss', 'storage/newsseek/siglip/index.faiss')
    zf.write(siglip_store_dir / 'metadata.json', 'storage/newsseek/siglip/metadata.json')
print(f'Pre-packaged initial vector stores archive ({zip_path.stat().st_size / (1024**2):.2f} MB) -> {zip_path}')
"""
    cells.append(nbf.v4.new_code_cell(cell8_code))

    # Cell 9: Video-wise Max-Pooling Retrieval Benchmark
    cell9_code = r"""def evaluate_newsseek_retrieval(kf_tensor, txt_tensor, video_records, all_keyframes, k_vals=[1, 5, 10]):
    print('Evaluating Video-Wise Max-Pooling Retrieval...')
    # Step 1: Compute similarity between all keyframes and all video queries
    frame_sims = torch.matmul(kf_tensor, txt_tensor.T)
    
    N_vid = len(video_records)
    video_sim_matrix = torch.zeros((N_vid, N_vid))
    
    # Step 2: Max-Pool keyframes for each video
    kf_start = 0
    for v_idx, v in enumerate(video_records):
        n_kf = len(v.get('keyframes', []))
        if n_kf > 0:
            v_kf_sims = frame_sims[kf_start : kf_start + n_kf] # shape: (n_kf, N_vid)
            video_sim_matrix[v_idx] = v_kf_sims.max(dim=0).values
        kf_start += n_kf
        
    # Step 3: Compute Recall@K and MRR
    top_max = max(k_vals)
    _, topk_preds = video_sim_matrix.topk(top_max, dim=0, largest=True)
    targets = torch.arange(N_vid).unsqueeze(0).expand_as(topk_preds)
    matches = (topk_preds == targets)
    
    res = {}
    for k in k_vals:
        res[f'R@{k}'] = round(float((matches[:k].any(dim=0).float().mean().item()) * 100.0), 2)
        
    _, full_sorted = video_sim_matrix.sort(dim=0, descending=True)
    targets_full = torch.arange(N_vid).unsqueeze(0).expand_as(full_sorted)
    match_coords = (full_sorted == targets_full).nonzero(as_tuple=False)
    ranks = match_coords[:, 0] + 1
    res['MRR'] = round(float((1.0 / ranks.float()).mean().item()), 4)
    res['mR'] = round(float(np.mean([res[f'R@{k}'] for k in k_vals])), 2)
    
    # Modality Gap
    gap = torch.norm(kf_tensor.mean(dim=0) - txt_tensor.mean(dim=0), p=2).item()
    res['modality_gap'] = round(float(gap), 4)
    
    # Multilingual breakdown
    langs = [VIDEO_METADATA.get(v['video_id'], {}).get('language', 'en') for v in video_records]
    en_indices = [i for i, l in enumerate(langs) if l == 'en']
    non_en_indices = [i for i, l in enumerate(langs) if l != 'en']
    
    if en_indices:
        res['R@1_EN'] = round(float((matches[0, en_indices].float().mean().item()) * 100.0), 2)
    if non_en_indices:
        res['R@1_NonEN'] = round(float((matches[0, non_en_indices].float().mean().item()) * 100.0), 2)
        
    return res

clip_newsseek_bm = {}
try:
    print('\n=== EVALUATING CLIP ON NEWSSEEK ===')
    clip_newsseek_bm = evaluate_newsseek_retrieval(clip_kf_tensor, clip_txt_tensor, video_records, all_keyframes)
    clip_newsseek_bm['kf_enc_time_sec'] = round(clip_kf_time, 2)
    print(json.dumps(clip_newsseek_bm, indent=2))
except Exception as e:
    print(f'Warning: CLIP evaluation failed: {e}')
    clip_newsseek_bm = {'error': str(e), 'kf_enc_time_sec': round(clip_kf_time, 2)}

siglip_newsseek_bm = {}
try:
    print('\n=== EVALUATING SIGLIP ON NEWSSEEK ===')
    siglip_newsseek_bm = evaluate_newsseek_retrieval(siglip_kf_tensor, siglip_txt_tensor, video_records, all_keyframes)
    siglip_newsseek_bm['kf_enc_time_sec'] = round(siglip_kf_time, 2)
    print(json.dumps(siglip_newsseek_bm, indent=2))
except Exception as e:
    print(f'Warning: SigLIP evaluation failed: {e}')
    siglip_newsseek_bm = {'error': str(e), 'kf_enc_time_sec': round(siglip_kf_time, 2)}

# Save benchmark summary
newsseek_bm = {
    'dataset': 'newsseek-processed-keyframes',
    'total_videos': len(video_records),
    'total_keyframes': len(all_keyframes),
    'clip_vit_b32': clip_newsseek_bm,
    'siglip_vit_b16_224': siglip_newsseek_bm,
}
with open(out_dir / 'newsseek_benchmark_results.json', 'w', encoding='utf-8') as f:
    json.dump(newsseek_bm, f, indent=2)

# Append benchmark json to zip archive
with zipfile.ZipFile(zip_path, 'a') as zf:
    zf.write(out_dir / 'newsseek_benchmark_results.json', 'newsseek_benchmark_results.json')

print(f'Final download archive updated ({zip_path.stat().st_size / (1024**2):.2f} MB) -> {zip_path}')
print('=== NEWSSEEK INGESTION & BENCHMARK COMPLETE ===')
"""
    cells.append(nbf.v4.new_code_cell(cell9_code))

    nb['cells'] = cells
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.12"
        }
    }

    out_file = Path('kaggle/encode_newsseek/encode_newsseek_clip_siglip.ipynb')
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f'Created notebook: {out_file}')

    # Metadata file
    metadata = {
        "id": "namnguynnnn/encode-newsseek-clip-siglip",
        "title": "encode-newsseek-clip-siglip",
        "code_file": "encode_newsseek_clip_siglip.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [
            "namnguynnnn/newsseek-processed-keyframes"
        ],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": []
    }

    meta_path = Path('kaggle/encode_newsseek/kernel-metadata.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f'Created metadata: {meta_path}')

if __name__ == '__main__':
    create_newsseek_kernel()

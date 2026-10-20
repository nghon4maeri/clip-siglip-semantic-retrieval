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
        "3. Video-wise Max-Pooling Cross-Modal Retrieval Benchmark (Recall@1, 5, 10, MRR, Modality Gap).\n"
        "4. Exact FAISS `IndexFlatIP` vector store construction and artifact packaging for live search."
    ))

    # Cell 2: Imports & Environment
    cells.append(nbf.v4.new_code_cell(
        "!pip install -q faiss-cpu transformers\n\n"
        "import os\n"
        "import sys\n"
        "import time\n"
        "import json\n"
        "import zipfile\n"
        "from pathlib import Path\n\n"
        "import torch\n"
        "from torch.utils.data import Dataset, DataLoader\n"
        "import numpy as np\n"
        "from PIL import Image\n"
        "from tqdm.auto import tqdm\n"
        "import faiss\n"
        "from transformers import AutoProcessor, AutoModel, CLIPProcessor, CLIPModel\n\n"
        "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
        "print(f'Using device: {device}')\n"
        "if torch.cuda.is_available():\n"
        "    print(f'Device Name: {torch.cuda.get_device_name(0)}')\n"
        "    print(f'VRAM Allocated: {torch.cuda.memory_allocated(0)/(1024**2):.1f} MB')\n"
    ))

    # Cell 3: Embed Video Info Dictionary
    video_info_json = json.dumps(video_info, ensure_ascii=False)
    cells.append(nbf.v4.new_code_cell(
        f"# 698 Video Metadata Records (ID, Title, Language, Source)\n"
        f"VIDEO_METADATA = json.loads({repr(video_info_json)})\n"
        f"print(f'Loaded {{len(VIDEO_METADATA)}} video metadata entries.')\n"
    ))

    # Cell 4: Locate Manifest & Keyframe Images
    cells.append(nbf.v4.new_code_cell(
        "# Locate newsseek_manifest.json\n"
        "search_roots = [Path('/kaggle/input'), Path('../datasets'), Path('./datasets'), Path('./')]\n"
        "manifest_path = None\n"
        "for r in search_roots:\n"
        "    matches = list(r.glob('**/newsseek_manifest.json'))\n"
        "    if matches:\n"
        "        manifest_path = matches[0]\n"
        "        break\n\n"
        "if not manifest_path:\n"
        "    raise FileNotFoundError('newsseek_manifest.json not found in search paths!')\n\n"
        "print(f'Manifest located at: {manifest_path}')\n"
        "with open(manifest_path, 'r', encoding='utf-8') as f:\n"
        "    manifest_raw = json.load(f)\n\n"
        "video_records = list(manifest_raw.values()) if isinstance(manifest_raw, dict) else manifest_raw\n"
        "print(f'Total videos in manifest: {len(video_records)}')\n\n"
        "# Find keyframes directory\n"
        "base_dataset_dir = manifest_path.parent\n"
        "print(f'Base dataset dir: {base_dataset_dir}')\n\n"
        "# Flatten all keyframes into a structured list\n"
        "all_keyframes = []\n"
        "for v in video_records:\n"
        "    vid = v['video_id']\n"
        "    v_info = VIDEO_METADATA.get(vid, {'title': v.get('video_file', vid), 'language': 'en', 'source_name': ''})\n"
        "    for kf in v.get('keyframes', []):\n"
        "        kf_rel = kf['file_path']\n"
        "        # Candidate paths\n"
        "        cand1 = base_dataset_dir / kf_rel\n"
        "        cand2 = base_dataset_dir / 'newsseek_keyframes' / kf_rel\n"
        "        cand3 = base_dataset_dir / Path(kf_rel).name\n"
        "        \n"
        "        actual_path = None\n"
        "        if cand1.exists():\n"
        "            actual_path = cand1\n"
        "        elif cand2.exists():\n"
        "            actual_path = cand2\n"
        "        elif cand3.exists():\n"
        "            actual_path = cand3\n"
        "        else:\n"
        "            actual_path = cand1 # fallback\n"
        "            \n"
        "        all_keyframes.append({\n"
        "            'kf_idx': len(all_keyframes),\n"
        "            'video_id': vid,\n"
        "            'frame_number': kf['frame_number'],\n"
        "            'timestamp_sec': kf['timestamp_sec'],\n"
        "            'scene_index': kf['scene_index'],\n"
        "            'file_path': str(actual_path),\n"
        "            'relative_path': kf_rel,\n"
        "            'video_title': v_info['title'],\n"
        "            'language': v_info['language'],\n"
        "        })\n\n"
        "print(f'Flattened {len(all_keyframes)} total keyframes across {len(video_records)} videos.')\n"
        "sample_kf = all_keyframes[0]\n"
        "print(f'Sample Keyframe: {sample_kf[\"relative_path\"]}, exists: {Path(sample_kf[\"file_path\"]).exists()}')\n"
    ))

    # Cell 5: PyTorch Dataset for fast batch image loading
    cells.append(nbf.v4.new_code_cell(
        "def extract_tensor_features(output):\n"
        "    if isinstance(output, torch.Tensor):\n"
        "        return output\n"
        "    if hasattr(output, 'pooler_output') and output.pooler_output is not None:\n"
        "        return output.pooler_output\n"
        "    if hasattr(output, 'last_hidden_state') and output.last_hidden_state is not None:\n"
        "        return output.last_hidden_state[:, 0, :]\n"
        "    if hasattr(output, '__getitem__'):\n"
        "        return output[0]\n"
        "    return output\n\n"
        "class KeyframeDataset(Dataset):\n"
        "    def __init__(self, keyframe_list, processor, is_siglip=False):\n"
        "        self.items = keyframe_list\n"
        "        self.processor = processor\n"
        "        self.is_siglip = is_siglip\n"
        "        # Create a default blank image for fallback\n"
        "        self.fallback_img = Image.new('RGB', (224, 224), color=(128, 128, 128))\n\n"
        "    def __len__(self):\n"
        "        return len(self.items)\n\n"
        "    def __getitem__(self, idx):\n"
        "        p = self.items[idx]['file_path']\n"
        "        try:\n"
        "            img = Image.open(p).convert('RGB')\n"
        "        except Exception:\n"
        "            img = self.fallback_img\n"
        "        return img\n\n"
        "def collate_fn_clip(batch):\n"
        "    return clip_processor(images=batch, return_tensors='pt')\n\n"
        "def collate_fn_siglip(batch):\n"
        "    return siglip_processor(images=batch, return_tensors='pt')\n\n"
        "print('Defined KeyframeDataset and Collate functions.')\n"
    ))

    # Cell 6: CLIP ViT-B/32 Extraction
    cells.append(nbf.v4.new_code_cell(
        "# Load CLIP ViT-B/32\n"
        "print('--> Loading openai/clip-vit-base-patch32...')\n"
        "clip_model_id = 'openai/clip-vit-base-patch32'\n"
        "clip_processor = CLIPProcessor.from_pretrained(clip_model_id)\n"
        "clip_model = CLIPModel.from_pretrained(clip_model_id).to(device).eval()\n\n"
        "clip_loader = DataLoader(\n"
        "    KeyframeDataset(all_keyframes, clip_processor),\n"
        "    batch_size=256,\n"
        "    shuffle=False,\n"
        "    num_workers=4,\n"
        "    collate_fn=collate_fn_clip,\n"
        "    pin_memory=True,\n"
        ")\n\n"
        "clip_kf_embeds = []\n"
        "t0 = time.time()\n"
        "with torch.no_grad():\n"
        "    for batch in tqdm(clip_loader, desc='CLIP Keyframes'):\n"
        "        inputs = {k: v.to(device) for k, v in batch.items()}\n"
        "        with torch.autocast(device_type='cuda', dtype=torch.float16):\n"
        "            feats = extract_tensor_features(clip_model.get_image_features(**inputs))\n"
        "            feats = feats / feats.norm(dim=-1, keepdim=True)\n"
        "        clip_kf_embeds.append(feats.cpu().float())\n\n"
        "clip_kf_tensor = torch.cat(clip_kf_embeds, dim=0)\n"
        "clip_kf_time = time.time() - t0\n"
        "print(f'CLIP Keyframes: {len(clip_kf_tensor)} in {clip_kf_time:.2f}s ({len(clip_kf_tensor)/clip_kf_time:.1f} fps)')\n\n"
        "# Encode Video Titles (Queries)\n"
        "unique_vids = [v['video_id'] for v in video_records]\n"
        "unique_titles = [VIDEO_METADATA.get(v, {}).get('title', v) for v in unique_vids]\n"
        "print(f'Encoding {len(unique_titles)} video queries with CLIP...')\n\n"
        "clip_txt_embeds = []\n"
        "with torch.no_grad():\n"
        "    for i in range(0, len(unique_titles), 128):\n"
        "        batch_txt = unique_titles[i : i + 128]\n"
        "        inputs = clip_processor(text=batch_txt, padding=True, truncation=True, return_tensors='pt').to(device)\n"
        "        with torch.autocast(device_type='cuda', dtype=torch.float16):\n"
        "            feats = extract_tensor_features(clip_model.get_text_features(**inputs))\n"
        "            feats = feats / feats.norm(dim=-1, keepdim=True)\n"
        "        clip_txt_embeds.append(feats.cpu().float())\n\n"
        "clip_txt_tensor = torch.cat(clip_txt_embeds, dim=0)\n"
        "del clip_model, clip_processor\n"
        "torch.cuda.empty_cache()\n"
    ))

    # Cell 7: SigLIP ViT-B/16-224 Extraction
    cells.append(nbf.v4.new_code_cell(
        "# Load SigLIP ViT-B/16-224\n"
        "print('--> Loading google/siglip-base-patch16-224...')\n"
        "siglip_model_id = 'google/siglip-base-patch16-224'\n"
        "siglip_processor = AutoProcessor.from_pretrained(siglip_model_id)\n"
        "siglip_model = AutoModel.from_pretrained(siglip_model_id).to(device).eval()\n\n"
        "siglip_loader = DataLoader(\n"
        "    KeyframeDataset(all_keyframes, siglip_processor, is_siglip=True),\n"
        "    batch_size=256,\n"
        "    shuffle=False,\n"
        "    num_workers=4,\n"
        "    collate_fn=collate_fn_siglip,\n"
        "    pin_memory=True,\n"
        ")\n\n"
        "siglip_kf_embeds = []\n"
        "t0 = time.time()\n"
        "with torch.no_grad():\n"
        "    for batch in tqdm(siglip_loader, desc='SigLIP Keyframes'):\n"
        "        inputs = {k: v.to(device) for k, v in batch.items()}\n"
        "        with torch.autocast(device_type='cuda', dtype=torch.float16):\n"
        "            feats = extract_tensor_features(siglip_model.get_image_features(**inputs))\n"
        "            feats = feats / feats.norm(dim=-1, keepdim=True)\n"
        "        siglip_kf_embeds.append(feats.cpu().float())\n\n"
        "siglip_kf_tensor = torch.cat(siglip_kf_embeds, dim=0)\n"
        "siglip_kf_time = time.time() - t0\n"
        "print(f'SigLIP Keyframes: {len(siglip_kf_tensor)} in {siglip_kf_time:.2f}s ({len(siglip_kf_tensor)/siglip_kf_time:.1f} fps)')\n\n"
        "# Encode Video Titles\n"
        "siglip_txt_embeds = []\n"
        "with torch.no_grad():\n"
        "    for i in range(0, len(unique_titles), 128):\n"
        "        batch_txt = unique_titles[i : i + 128]\n"
        "        inputs = siglip_processor(text=batch_txt, padding='max_length', truncation=True, return_tensors='pt').to(device)\n"
        "        with torch.autocast(device_type='cuda', dtype=torch.float16):\n"
        "            feats = extract_tensor_features(siglip_model.get_text_features(**inputs))\n"
        "            feats = feats / feats.norm(dim=-1, keepdim=True)\n"
        "        siglip_txt_embeds.append(feats.cpu().float())\n\n"
        "siglip_txt_tensor = torch.cat(siglip_txt_embeds, dim=0)\n"
        "del siglip_model, siglip_processor\n"
        "torch.cuda.empty_cache()\n"
    ))

    # Cell 8: Video-wise Max-Pooling Retrieval Benchmark
    cells.append(nbf.v4.new_code_cell(
        "def evaluate_newsseek_retrieval(kf_tensor, txt_tensor, video_records, all_keyframes, k_vals=[1, 5, 10]):\n"
        "    print('Evaluating Video-Wise Max-Pooling Retrieval...')\n"
        "    # Step 1: Compute similarity between all keyframes and all video queries\n"
        "    # kf_tensor: (N_kf, D), txt_tensor: (N_vid, D)\n"
        "    # frame_sims: (N_kf, N_vid)\n"
        "    frame_sims = torch.matmul(kf_tensor, txt_tensor.T)\n"
        "    \n"
        "    N_vid = len(video_records)\n"
        "    video_sim_matrix = torch.zeros((N_vid, N_vid))\n"
        "    \n"
        "    # Step 2: Max-Pool keyframes for each video\n"
        "    kf_start = 0\n"
        "    for v_idx, v in enumerate(video_records):\n"
        "        n_kf = len(v.get('keyframes', []))\n"
        "        if n_kf > 0:\n"
        "            v_kf_sims = frame_sims[kf_start : kf_start + n_kf] # shape: (n_kf, N_vid)\n"
        "            video_sim_matrix[v_idx] = v_kf_sims.max(dim=0).values\n"
        "        kf_start += n_kf\n"
        "        \n"
        "    # Step 3: Compute Recall@K and MRR\n"
        "    top_max = max(k_vals)\n"
        "    _, topk_preds = video_sim_matrix.topk(top_max, dim=0, largest=True)\n"
        "    targets = torch.arange(N_vid).unsqueeze(0).expand_as(topk_preds)\n"
        "    matches = (topk_preds == targets)\n"
        "    \n"
        "    res = {}\n"
        "    for k in k_vals:\n"
        "        res[f'R@{k}'] = round(float((matches[:k].any(dim=0).float().mean().item()) * 100.0), 2)\n"
        "        \n"
        "    _, full_sorted = video_sim_matrix.sort(dim=0, descending=True)\n"
        "    ranks = (full_sorted == targets).nonzero(as_tuple=True)[0] + 1\n"
        "    res['MRR'] = round(float((1.0 / ranks.float()).mean().item()), 4)\n"
        "    res['mR'] = round(float(np.mean([res[f'R@{k}'] for k in k_vals])), 2)\n"
        "    \n"
        "    # Modality Gap\n"
        "    gap = torch.norm(kf_tensor.mean(dim=0) - txt_tensor.mean(dim=0), p=2).item()\n"
        "    res['modality_gap'] = round(float(gap), 4)\n"
        "    \n"
        "    # Multilingual breakdown\n"
        "    langs = [VIDEO_METADATA.get(v['video_id'], {}).get('language', 'en') for v in video_records]\n"
        "    en_indices = [i for i, l in enumerate(langs) if l == 'en']\n"
        "    non_en_indices = [i for i, l in enumerate(langs) if l != 'en']\n"
        "    \n"
        "    if en_indices:\n"
        "        res['R@1_EN'] = round(float((matches[0, en_indices].float().mean().item()) * 100.0), 2)\n"
        "    if non_en_indices:\n"
        "        res['R@1_NonEN'] = round(float((matches[0, non_en_indices].float().mean().item()) * 100.0), 2)\n"
        "        \n"
        "    return res\n\n"
        "print('\\n=== EVALUATING CLIP ON NEWSSEEK ===')\n"
        "clip_newsseek_bm = evaluate_newsseek_retrieval(clip_kf_tensor, clip_txt_tensor, video_records, all_keyframes)\n"
        "clip_newsseek_bm['kf_enc_time_sec'] = round(clip_kf_time, 2)\n"
        "print(json.dumps(clip_newsseek_bm, indent=2))\n\n"
        "print('\\n=== EVALUATING SIGLIP ON NEWSSEEK ===')\n"
        "siglip_newsseek_bm = evaluate_newsseek_retrieval(siglip_kf_tensor, siglip_txt_tensor, video_records, all_keyframes)\n"
        "siglip_newsseek_bm['kf_enc_time_sec'] = round(siglip_kf_time, 2)\n"
        "print(json.dumps(siglip_newsseek_bm, indent=2))\n"
    ))

    # Cell 9: FAISS Index Construction & Packaging
    cells.append(nbf.v4.new_code_cell(
        "# Build FAISS vector stores for NewsSeek\n"
        "out_dir = Path('/kaggle/working')\n"
        "storage_dir = out_dir / 'storage' / 'newsseek'\n"
        "clip_store_dir = storage_dir / 'clip'\n"
        "siglip_store_dir = storage_dir / 'siglip'\n"
        "clip_store_dir.mkdir(parents=True, exist_ok=True)\n"
        "siglip_store_dir.mkdir(parents=True, exist_ok=True)\n\n"
        "# 1. Save raw numpy arrays\n"
        "np.save(out_dir / 'newsseek_clip_keyframes.npy', clip_kf_tensor.numpy())\n"
        "np.save(out_dir / 'newsseek_clip_titles.npy', clip_txt_tensor.numpy())\n"
        "np.save(out_dir / 'newsseek_siglip_keyframes.npy', siglip_kf_tensor.numpy())\n"
        "np.save(out_dir / 'newsseek_siglip_titles.npy', siglip_txt_tensor.numpy())\n"
        "print('Saved .npy embedding arrays.')\n\n"
        "# 2. Build metadata list\n"
        "metadata_list = []\n"
        "for kf in all_keyframes:\n"
        "    metadata_list.append({\n"
        "        'id': f'newsseek_{kf[\"kf_idx\"]}',\n"
        "        'kf_idx': kf['kf_idx'],\n"
        "        'video_id': kf['video_id'],\n"
        "        'frame_number': kf['frame_number'],\n"
        "        'timestamp_sec': kf['timestamp_sec'],\n"
        "        'scene_index': kf['scene_index'],\n"
        "        'relative_path': kf['relative_path'],\n"
        "        'video_title': kf['video_title'],\n"
        "        'language': kf['language'],\n"
        "    })\n\n"
        "# 3. Build & Save CLIP FAISS Store (166k vectors)\n"
        "clip_index = faiss.IndexFlatIP(512)\n"
        "clip_vecs = np.ascontiguousarray(clip_kf_tensor.numpy(), dtype=np.float32)\n"
        "faiss.normalize_L2(clip_vecs)\n"
        "clip_index.add(clip_vecs)\n"
        "faiss.write_index(clip_index, str(clip_store_dir / 'index.faiss'))\n\n"
        "clip_meta_payload = {\n"
        "    'dim': 512,\n"
        "    'index_type': 'FlatIP',\n"
        "    'model_name': 'clip',\n"
        "    'count': len(metadata_list),\n"
        "    'items': metadata_list,\n"
        "}\n"
        "with open(clip_store_dir / 'metadata.json', 'w', encoding='utf-8') as f:\n"
        "    json.dump(clip_meta_payload, f, ensure_ascii=False, indent=2)\n"
        "print(f'Saved CLIP VectorStore ({clip_index.ntotal} vectors) -> {clip_store_dir}')\n\n"
        "# 4. Build & Save SigLIP FAISS Store (166k vectors)\n"
        "siglip_index = faiss.IndexFlatIP(768)\n"
        "siglip_vecs = np.ascontiguousarray(siglip_kf_tensor.numpy(), dtype=np.float32)\n"
        "faiss.normalize_L2(siglip_vecs)\n"
        "siglip_index.add(siglip_vecs)\n"
        "faiss.write_index(siglip_index, str(siglip_store_dir / 'index.faiss'))\n\n"
        "siglip_meta_payload = {\n"
        "    'dim': 768,\n"
        "    'index_type': 'FlatIP',\n"
        "    'model_name': 'siglip',\n"
        "    'count': len(metadata_list),\n"
        "    'items': metadata_list,\n"
        "}\n"
        "with open(siglip_store_dir / 'metadata.json', 'w', encoding='utf-8') as f:\n"
        "    json.dump(siglip_meta_payload, f, ensure_ascii=False, indent=2)\n"
        "print(f'Saved SigLIP VectorStore ({siglip_index.ntotal} vectors) -> {siglip_store_dir}')\n\n"
        "# 5. Save benchmark summary\n"
        "newsseek_bm = {\n"
        "    'dataset': 'newsseek-processed-keyframes',\n"
        "    'total_videos': len(video_records),\n"
        "    'total_keyframes': len(all_keyframes),\n"
        "    'clip_vit_b32': clip_newsseek_bm,\n"
        "    'siglip_vit_b16_224': siglip_newsseek_bm,\n"
        "}\n"
        "with open(out_dir / 'newsseek_benchmark_results.json', 'w', encoding='utf-8') as f:\n"
        "    json.dump(newsseek_bm, f, indent=2)\n\n"
        "# 6. Package compact ZIP for local download\n"
        "zip_path = out_dir / 'newsseek_vector_stores.zip'\n"
        "with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:\n"
        "    zf.write(clip_store_dir / 'index.faiss', 'storage/newsseek/clip/index.faiss')\n"
        "    zf.write(clip_store_dir / 'metadata.json', 'storage/newsseek/clip/metadata.json')\n"
        "    zf.write(siglip_store_dir / 'index.faiss', 'storage/newsseek/siglip/index.faiss')\n"
        "    zf.write(siglip_store_dir / 'metadata.json', 'storage/newsseek/siglip/metadata.json')\n"
        "    zf.write(out_dir / 'newsseek_benchmark_results.json', 'newsseek_benchmark_results.json')\n\n"
        "print(f'Created compact download archive ({zip_path.stat().st_size / (1024**2):.2f} MB) -> {zip_path}')\n"
        "print('=== NEWSSEEK INGESTION & BENCHMARK COMPLETE ===')\n"
    ))

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

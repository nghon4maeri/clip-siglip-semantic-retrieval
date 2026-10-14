# Semantic Image Retrieval: A Comparative Study of CLIP and SigLIP

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Framework-EE4C2C.svg)

## Overview

This repository contains the official codebase, theoretical framework, and literature review for a comprehensive comparative study between **CLIP** (Contrastive Language-Image Pretraining) and **SigLIP** (Sigmoid Loss for Language Image Pre-Training) on the task of **Semantic Image Retrieval**.

The primary objective of this project is to reproduce, benchmark, and critically evaluate the performance of these state-of-the-art Vision-Language Models (VLMs) using standardized academic metrics. Instead of proposing a novel architecture, this project emphasizes rigorous empirical comparison, deep theoretical understanding of multi-modal embedding spaces, and the impact of contrastive vs. sigmoid loss formulations on retrieval accuracy.

Note: this project runs **inference/benchmark on pretrained models only** — no training is performed. The benchmark is designed to run on Kaggle (free GPU) or any machine with PyTorch.

## Problem Statement

**Semantic Image Retrieval** entails retrieving the most semantically relevant images from a large-scale database given a natural language query (Text-to-Image Retrieval), and vice versa (Image-to-Text Retrieval).
*   **Input:** A query modality (e.g., text description) and a search pool of the target modality (e.g., image database).
*   **Output:** A ranked list of items from the search pool based on multi-modal semantic similarity.
*   **Core Challenge:** Bridging the semantic gap between visual and textual representations by mapping both modalities into a shared, dense embedding space.

See `problem/README.md` for the generic framework and its unknowns.

## Repository Structure

```text
clip-siglip-semantic-retrieval/
├── problem/                # Formal problem definition, unknowns, and generic retrieval framework
├── related-work/           # Literature review, SOTA comparisons, and gap analysis
├── pipeline/               # Training/Testing pipelines, evaluation metrics & protocol
├── clip/                   # Theoretical research notes and official repo clone for CLIP
├── siglip/                 # Theoretical research notes and official repo clone for SigLIP
├── datasets/               # Dataset download scripts, instructions and sample data
├── notebooks/              # kaggle_benchmark.ipynb — run the full benchmark on Kaggle
├── scripts/                # verify_data.py — check dataset before running the benchmark
├── src/                    # Source code for reproduction and benchmarking
│   ├── models/             # PyTorch inference wrappers (CLIP & SigLIP)
│   ├── datasets/           # PyTorch Dataset/DataLoader implementations
│   └── evaluation/         # Metrics (Recall@K) and benchmark pipeline
├── tests/                  # Synthetic smoke tests for the evaluation metrics
├── workflow.md             # Project guidelines, workflow phases, and common pitfalls to avoid
└── README.md               # Project overview
```

## Evaluated Models

1.  **CLIP (OpenAI, 2021):** Utilizes a standard InfoNCE (Contrastive) Loss requiring computation of global pairwise similarities across batches. Checkpoint: `openai/clip-vit-base-patch16-224`.
2.  **SigLIP (Google DeepMind, 2023):** Introduces a pairwise Sigmoid Loss that operates solely on image-text pairs, removing the need for global batch normalization and allowing for significant scaling. Checkpoint: `google/siglip-base-patch16-224`.

Both checkpoints use ViT-B/16 and 224×224 inputs, so the accuracy/complexity comparison is fair in terms of patch size and resolution.

## Datasets and Evaluation

### Dataset
The benchmark relies on the standard **MS COCO (2014)** dataset with the **Karpathy splits**. The test set is 5,000 images with 5 reference captions per image. All download/verification steps are documented in `datasets/README.md`.

*Note: Large files (annotations JSON, COCO images) are excluded from version control. All data is downloaded to `datasets/coco/` and ignored by git.*

### Protocol and Metrics
*   **Protocol:** 5,000 test images × 5 captions = **25,000 text queries** (the standard CLIP/SigLIP retrieval protocol; 10 images in the raw Karpathy file that carry a 6th caption are truncated to 5).
*   **Image-to-Text (i2t):** each image ranks all 25,000 captions; a query is correct if **any** of its captions is in the top-K.
*   **Text-to-Image (t2i):** each caption ranks all 5,000 images; a query is correct if its source image is in the top-K.
*   **Metrics:** Recall@1, Recall@5, Recall@10 for both directions, averaged over all queries.
*   **Complexity:** wall-clock time per stage (image embedding, text embedding, similarity), peak VRAM, and parameter count are logged for each model.

The full protocol is documented in `pipeline/README.md`.

## Installation & Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/nghon4maeri/clip-siglip-semantic-retrieval.git
   cd clip-siglip-semantic-retrieval
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   (On Kaggle, PyTorch and Transformers are already installed.)

3. **Download annotations** (verifies 5,000 test images / 25,000 captions and prints a SHA256):
   ```bash
   python datasets/download_annotations.py
   ```

4. **Get COCO val2014 images** — the recommended way is to run the Kaggle notebook `notebooks/kaggle_benchmark.ipynb`, which mounts the dataset and verifies all 5,000 test images automatically. For local runs, follow `datasets/README.md`. Then verify:
   ```bash
   python scripts/verify_data.py
   ```

5. **Run the benchmark** (both models, full 5,000 images):
   ```bash
   python src/evaluation/benchmark.py --models clip,siglip --batch-size 64
   ```
   Results (metrics + timings + VRAM) are written to `results/benchmark.json`.

6. **Smoke test without the full dataset** (5 sample images):
   ```bash
   python make_samples.py
   python src/evaluation/benchmark.py --json-path datasets/coco/annotations/dataset_coco_sample.json --limit 5
   ```

## References

*   Radford, A., et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision.* ICML.
*   Zhai, X., et al. (2023). *Sigmoid Loss for Language Image Pre-Training.* ICCV.
*   Karpathy, A., & Fei-Fei, L. (2015). *Deep Visual-Semantic Alignments for Generating Image Descriptions.* CVPR.

---
*This repository is developed for academic research, reproduction, and educational purposes.*

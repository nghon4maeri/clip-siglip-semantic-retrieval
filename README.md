# Semantic Image Retrieval: A Comparative Study of CLIP and SigLIP

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Framework-EE4C2C.svg)

## Overview

This repository contains the official codebase, theoretical framework, and literature review for a comprehensive comparative study between **CLIP** (Contrastive Language-Image Pretraining) and **SigLIP** (Sigmoid Loss for Language Image Pre-Training) on the task of **Semantic Image Retrieval**. 

The primary objective of this project is to reproduce, benchmark, and critically evaluate the performance of these state-of-the-art Vision-Language Models (VLMs) using standardized academic metrics. Instead of proposing a novel architecture, this project emphasizes rigorous empirical comparison, deep theoretical understanding of multi-modal embedding spaces, and the impact of contrastive vs. sigmoid loss formulations on retrieval accuracy.

## Problem Statement

**Semantic Image Retrieval** entails retrieving the most semantically relevant images from a large-scale database given a natural language query (Text-to-Image Retrieval), and vice versa (Image-to-Text Retrieval). 
*   **Input:** A query modality (e.g., text description) and a search pool of the target modality (e.g., image database).
*   **Output:** A ranked list of items from the search pool based on multi-modal semantic similarity.
*   **Core Challenge:** Bridging the semantic gap between visual and textual representations by mapping both modalities into a shared, dense embedding space.

## Repository Structure

The project is strictly organized to reflect a systematic research workflow, separating theoretical formulation, literature review, and source code implementation.

```text
clip-siglip-semantic-retrieval/
├── problem/                # Formal problem definition, unknowns, and generic retrieval framework
├── related-work/           # Literature review, SOTA comparisons, and gap analysis
├── pipeline/               # Training/Testing pipelines and Evaluation Metrics definitions
├── clip/                   # Theoretical research notes and official repo clone for CLIP
├── siglip/                 # Theoretical research notes and official repo clone for SigLIP
├── datasets/               # MS COCO datasets, Karpathy splits, and download scripts
├── src/                    # Source code for reproduction and benchmarking
│   ├── models/             # PyTorch inference wrappers (CLIP & SigLIP)
│   ├── datasets/           # PyTorch Dataset/DataLoader implementations
│   └── evaluation/         # Metrics (Recall@K) and benchmark pipeline
├── workflow.md             # Project guidelines, workflow phases, and common pitfalls to avoid
└── README.md               # Project overview
```

## Evaluated Models

1.  **CLIP (OpenAI, 2021):** Utilizes a standard InfoNCE (Contrastive) Loss requiring computation of global pairwise similarities across batches.
2.  **SigLIP (Google DeepMind, 2023):** Introduces a pairwise Sigmoid Loss that operates solely on image-text pairs, removing the need for global batch normalization and allowing for significant scaling.

## Datasets and Evaluation

### Dataset
The benchmark relies on the standard **MS COCO (2014)** dataset, specifically utilizing the highly adopted **Karpathy splits**. The test set comprises 5,000 images with 5 reference captions per image.

*Note: The actual multi-gigabyte image files are excluded from version control. Please refer to `datasets/README.md` for dataset download instructions.*

### Metrics
System performance is evaluated using standard retrieval metrics:
*   **Recall@1, Recall@5, Recall@10** for both Text-to-Image (T2I) and Image-to-Text (I2T) retrieval tasks.

## Installation & Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/clip-siglip-semantic-retrieval.git
   cd clip-siglip-semantic-retrieval
   ```

2. **Install dependencies:**
   Ensure you have PyTorch installed, then install the HuggingFace `transformers` and `tqdm`:
   ```bash
   pip install torch torchvision transformers tqdm
   ```

3. **Prepare the Data:**
   Navigate to the `datasets/` directory and run the annotation download script. Then, follow the instructions in `datasets/README.md` to download the COCO `val2014` images.
   ```bash
   cd datasets
   python download_annotations.py
   ```

4. **Run the Benchmark:**
   Execute the evaluation pipeline to extract features and compute similarity matrices.
   ```bash
   python src/evaluation/benchmark.py
   ```

## References

*   Radford, A., et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision.* ICML.
*   Zhai, X., et al. (2023). *Sigmoid Loss for Language Image Pre-Training.* ICCV.
*   Karpathy, A., & Fei-Fei, L. (2015). *Deep Visual-Semantic Alignments for Generating Image Descriptions.* CVPR.

---
*This repository is developed for academic research, reproduction, and educational purposes.*

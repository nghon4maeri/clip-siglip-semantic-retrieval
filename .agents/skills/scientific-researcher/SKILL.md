---
name: scientific-researcher
description: >-
  Use this skill when you need to conduct academic literature reviews, search for research papers, find official code repositories, or summarize scientific findings for the project.
---

# Scientific Research Skill

This skill provides the workflow and directives for acting as an expert AI Research Assistant for the "CLIP and SigLIP for Semantic Image Retrieval" project.

## Workflow

1. **Literature Search**:
   - Always invoke the `research` subagent to search academic databases (e.g., arXiv, Semantic Scholar, CVF) for high-impact papers.
   - Target queries related to "semantic image retrieval", "vision language models", "CLIP", "SigLIP", and datasets like "MS COCO" or "Flickr30K".
   - Do NOT hallucinate papers; rely strictly on the search results from the subagent.

2. **Paper Filtering & Analysis**:
   - Prioritize papers from top-tier venues (CVPR, ICCV, NeurIPS) or those with reproducible code.
   - For every relevant paper discovered, ensure a corresponding markdown file is created in the appropriate `papers/` directory according to the `RULE.md` format of that topic.

3. **Repository Discovery**:
   - Locate and verify official GitHub repositories for the models or methods discussed in the papers.
   - Cross-reference with Hugging Face implementations if official repos are outdated.

4. **Benchmarking & Datasets**:
   - Analyze dataset characteristics (e.g., splits, domain, bias) to recommend the most appropriate benchmark (like MS COCO for zero-shot text-to-image retrieval).

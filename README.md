<h1 align="center">Seeing Without Looking: Visual Dependency in Vision-Language Models</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2605.22903">
    <img src="https://img.shields.io/badge/arXiv-2605.22903-b31b1b.svg" alt="arXiv">
  </a>
  <a href="https://openaccess.thecvf.com/content/CVPR2026W/GRAIL-V/papers/Lan_Seeing_without_Looking_Do_Vision-Language_Benchmarks_Really_Test_Vision_CVPRW_2026_paper.pdf">
    <img src="https://img.shields.io/static/v1?label=Paper&message=CVPR%202026&color=blue" alt="Paper">
  </a>
  <img src="https://img.shields.io/static/v1?label=Task&message=VLM%20Evaluation&color=green" alt="Task">
  <img src="https://img.shields.io/static/v1?label=Focus&message=Visual%20Grounding&color=orange" alt="Focus">
</p>

<p align="center">
  <b>Do vision-language benchmarks really require vision?</b><br>
  We find that several widely used VLM benchmarks can be partially solved even when visual input is removed or degraded, raising questions about how well they measure visual grounding.
</p>

<p align="center">
  <img src="assets/teaser.png" alt="Seeing without Looking teaser" width="60%">
</p>

## Overview

This repository contains the official implementation for **Seeing without Looking: Do Vision-Language Benchmarks Really Test Vision?**

We investigate whether widely used vision-language benchmarks truly require visual grounding, or whether strong performance can be achieved even when models do not meaningfully rely on the image. Our experiments evaluate vision-language models under different visual input conditions, including standard image-based evaluation and image-removed or visually perturbed settings, to measure how much benchmark performance depends on actual visual understanding.

This repository provides evaluation scripts for multiple vision-language benchmarks and open vision-language models.



## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

For AMBER evaluation, download the required spaCy model:

```bash
python -m spacy download en_core_web_lg
```


## Models

We evaluate the following vision-language models:

| Model | HuggingFace ID |
|-------|---------------|
| Qwen3-VL-32B | `Qwen/Qwen3-VL-32B-Instruct` |
| InternVL3-8B | `OpenGVLab/InternVL3-8B` |
| Qwen3-VL-8B | `Qwen/Qwen3-VL-7B-Instruct` |
| Gemma3-12B | `google/gemma-3-12b-it` |
| Qwen3-VL-4B | `Qwen/Qwen3-VL-4B-Instruct` |
| LLaVA-1.5-7B | `llava-hf/llava-1.5-7b-hf` |
| Molmo-7B-D-0924 | `allenai/Molmo-7B-D-0924` |

## Quick Start


```bash
# POPE (auto-downloaded from HuggingFace)
python Pope_Eval/eval_llava.py
python Pope_Eval/eval_internvl.py

# MME
python MME_Eval/eval_mme_qwen.py --metadata_path /path/to/metadata.json --root_dir /path/to/mme

# A-OKVQA
python Aokvqa_Eval/eval_aokvqa_qwen.py --pack_root /path/to/aokvqa_pack

# AMBER
python Amber_Eval/eval_amber_generative_qwen.py --amber_dir /path/to/AMBER
```

## Citation

If you find this repository useful, please cite our paper:

```bibtex

@InProceedings{Lan_2026_CVPR,
    author    = {Lan, Zixuan and Sun, Luzhe and Walter, Matthew R and Zhou, Jiawei},
    title     = {Seeing without Looking: Do Vision-Language Benchmarks Really Test Vision?},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2026},
    pages     = {11260-11273}
}
```

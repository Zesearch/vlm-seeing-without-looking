#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ViT Spatial Structure Analysis
================================
Metrics:
  1. K-Means spatial compactness (data-driven, no manual block assumption)
  2. Effective rank (token representation diversity)

Model: LLaVA-1.5-7B (CLIP ViT-L/14-336)
576 patch tokens per image, 25 layers
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPImageProcessor, LlavaForConditionalGeneration

# ========================
# Constants
# ========================
GRID_SIZE = 24
N_TOKENS  = GRID_SIZE ** 2  # 576
K         = 16              # number of clusters
N_ITER    = 20              # kmeans iterations


# ========================
# K-Means (pure torch GPU)
# ========================
def torch_kmeans(X_t, k=K, n_iter=N_ITER):
    """
    X_t: (576, D) float32 GPU tensor
    Returns cluster_ids: (576,) numpy array
    """
    idx = torch.randperm(X_t.shape[0], device=X_t.device)[:k]
    centroids = X_t[idx].clone()
    for _ in range(n_iter):
        dists = torch.cdist(X_t, centroids)
        cluster_ids = dists.argmin(dim=1)
        for c in range(k):
            mask = cluster_ids == c
            if mask.sum() > 0:
                centroids[c] = X_t[mask].mean(0)
    return cluster_ids.cpu().numpy()


# ========================
# Per-layer metrics
# ========================
def compute_metrics(patch_tokens: torch.Tensor, device: torch.device):
    """
    patch_tokens: (576, D) float32 CPU tensor
    Returns dict of scalar metrics.
    """
    X_t = patch_tokens.to(device)

    # 1. Spatial compactness
    cluster_ids = torch_kmeans(X_t)
    rows = np.arange(N_TOKENS) // GRID_SIZE
    cols = np.arange(N_TOKENS) % GRID_SIZE
    variances = []
    for c in range(K):
        mask = cluster_ids == c
        if mask.sum() < 2:
            continue
        var = np.var(rows[mask]) + np.var(cols[mask])
        variances.append(var)
    spatial_var = float(np.mean(variances))

    # 2. Effective rank
    X_centered = X_t - X_t.mean(0, keepdim=True)
    _, sv, _ = torch.linalg.svd(X_centered, full_matrices=False)
    sv2 = sv ** 2
    p = sv2 / (sv2.sum() + 1e-8)
    eff_rank = float(torch.exp(-torch.sum(p * torch.log(p + 1e-12))).item())

    return {"spatial_compactness": spatial_var, "effective_rank": eff_rank}


# ========================
# Per-image analysis
# ========================
def analyze_image(vision_tower, image_processor, image_path, device):
    image  = Image.open(image_path).convert("RGB")
    inputs = image_processor(images=image, return_tensors="pt")
    pv     = inputs["pixel_values"].to(device=vision_tower.device, dtype=vision_tower.dtype)

    with torch.no_grad():
        out = vision_tower(pv, output_hidden_states=True)

    results = []
    for hs in out.hidden_states:
        patch_tokens = hs[0, 1:, :].cpu().float()  # (576, D)
        results.append(compute_metrics(patch_tokens, device))
    return results


# ========================
# Aggregate
# ========================
def aggregate(all_results):
    num_layers = len(all_results[0])
    keys = list(all_results[0][0].keys())
    means, stds = {}, {}
    for k in keys:
        mat = np.array([[img[l][k] for l in range(num_layers)] for img in all_results])
        means[k] = np.nanmean(mat, axis=0)
        stds[k]  = np.nanstd(mat, axis=0) / (np.sqrt(mat.shape[0]) + 1e-8)
    return means, stds


# ========================
# Plot style
# ========================
def _apply_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'mathtext.fontset': 'dejavusans',
        'font.size': 12,
        'axes.labelsize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10.5,
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'axes.facecolor': '#F5F5F5',
    })


def _style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#AAAAAA')
    ax.spines['left'].set_linewidth(0.7)
    ax.spines['bottom'].set_color('#AAAAAA')
    ax.spines['bottom'].set_linewidth(0.7)
    ax.tick_params(axis='x', length=0, pad=8, colors='#444444')
    ax.tick_params(axis='y', length=0, pad=6, colors='#555555')
    ax.grid(True, axis='y', which='major', linestyle='-', linewidth=0.45,
            color='#D0D0D0', zorder=0)
    ax.grid(False, axis='x')


# ========================
# Plot
# ========================
def plot(means, stds, num_layers, n_images, output_dir):
    _apply_style()
    layers = np.arange(num_layers)
    llava_layer = num_layers - 2

    def shade(ax, y, e, color, label):
        ax.plot(layers, y, color=color, linewidth=5.5, alpha=0.10, solid_capstyle='round')
        ax.plot(layers, y, '-o', color=color, lw=2.2, ms=6,
                markerfacecolor=color, markeredgecolor='white',
                markeredgewidth=1.5, label=label, alpha=0.92, zorder=5)
        ax.fill_between(layers, y - e, y + e, alpha=0.12, color=color, zorder=4)
        ax.axvline(llava_layer, color='#888888', ls='--', lw=1.2, alpha=0.7,
                   label=f'LLaVA feature layer ({llava_layer})', zorder=3)
        ax.set_xlabel('Layer Index', fontsize=13, fontweight='medium', labelpad=8, color='#333333')
        ax.set_xticks(layers)
        ax.set_xticklabels(['E' if i == 0 else str(i) for i in layers], fontsize=8)
        leg = ax.legend(frameon=True, framealpha=0.95, edgecolor='#DDDDDD',
                        fancybox=True, borderpad=0.6)
        leg.get_frame().set_linewidth(0.6)

    # Figure 1: K-Means Spatial Compactness
    fig1, ax1 = plt.subplots(figsize=(7, 4.2), dpi=300)
    shade(ax1, means['spatial_compactness'], stds['spatial_compactness'],
          '#4E79A7', 'Spatial Variance')
    ax1.set_title('K-Means Spatial Compactness\n(lower = clusters are spatially tighter)',
                  fontsize=13, fontweight='medium', color='#333333', pad=12)
    ax1.set_ylabel('Mean Spatial Variance per Cluster', fontsize=13,
                   fontweight='medium', labelpad=8, color='#333333')
    _style_ax(ax1)
    fig1.tight_layout()
    p1 = os.path.join(output_dir, 'spatial_compactness.pdf')
    fig1.savefig(p1, format='pdf', bbox_inches='tight', pad_inches=0.12)
    print(f'[Saved] {p1}')
    plt.close(fig1)

    # Figure 2: Effective Rank
    fig2, ax2 = plt.subplots(figsize=(7, 4.2), dpi=300)
    shade(ax2, means['effective_rank'], stds['effective_rank'], '#E15759', 'Effective Rank')
    ax2.set_title('Effective Rank\n(lower = tokens collapse to fewer dimensions)',
                  fontsize=13, fontweight='medium', color='#333333', pad=12)
    ax2.set_ylabel('Effective Rank', fontsize=13, fontweight='medium', labelpad=8, color='#333333')
    _style_ax(ax2)
    fig2.tight_layout()
    p2 = os.path.join(output_dir, 'effective_rank.pdf')
    fig2.savefig(p2, format='pdf', bbox_inches='tight', pad_inches=0.12)
    print(f'[Saved] {p2}')
    plt.close(fig2)


# ========================
# Data loading
# ========================
def load_image_paths(json_path, image_dir, max_images=None):
    if json_path and os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
        paths = [os.path.join(image_dir, item['image_id']) for item in data
                 if os.path.exists(os.path.join(image_dir, item['image_id']))]
    else:
        paths = [os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir))
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if max_images:
        paths = paths[:max_images]
    return paths


# ========================
# Main
# ========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default="llava-hf/llava-1.5-7b-hf",
                    help="Path to LLaVA-1.5 model (local or HuggingFace hub)")
    ap.add_argument("--image_dir", type=str, required=True,
                    help="Directory containing images")
    ap.add_argument("--json_path", type=str, default=None,
                    help="Optional JSON file with image_id list")
    ap.add_argument("--output_dir", type=str, default="./results_kmeans_analysis",
                    help="Output directory for plots and data")
    ap.add_argument("--max_images", type=int, default=None,
                    help="Limit number of images to process")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('=' * 60)
    print('ViT Spatial Structure Analysis (K-Means + Effective Rank)')
    print('=' * 60)

    image_paths = load_image_paths(args.json_path, args.image_dir, args.max_images)
    print(f'Images: {len(image_paths)}')

    print('Loading model...')
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map='auto'
    ).eval()
    vision_tower    = model.model.vision_tower
    image_processor = CLIPImageProcessor.from_pretrained(args.model_path)
    print('Model loaded.')

    all_results = []
    for path in tqdm(image_paths, desc='Analyzing images'):
        try:
            res = analyze_image(vision_tower, image_processor, path, device)
            all_results.append(res)
        except Exception as e:
            print(f'[WARN] {path}: {e}')

    if not all_results:
        print('No results.')
        return

    means, stds = aggregate(all_results)
    num_layers  = len(all_results[0])

    np.save(os.path.join(args.output_dir, 'means.npy'), means)
    np.save(os.path.join(args.output_dir, 'stds.npy'),  stds)

    print(f'\n{"Layer":<8} {"spatial_var":>12} {"eff_rank":>10}')
    for l in range(num_layers):
        name   = 'Embed' if l == 0 else str(l)
        marker = ' <--' if l == num_layers - 2 else ''
        print(f'{name:<8} {means["spatial_compactness"][l]:>12.4f} {means["effective_rank"][l]:>10.2f}{marker}')

    plot(means, stds, num_layers, len(all_results), args.output_dir)
    print('\nDone!')


if __name__ == '__main__':
    main()

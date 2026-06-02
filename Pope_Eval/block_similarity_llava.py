#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fine-Grained Block-Level Visual Token Similarity Analysis for LLaVA-1.5

Split the 24x24 patch grid into 4x4 = 16 blocks (each block is 6x6 = 36 tokens).
Compute:
  1. Intra-block similarity: tokens within the same block becoming uniform?
  2. Inter-block similarity: tokens across different blocks becoming uniform?
  3. Per-layer block-level similarity heatmap

This reveals whether local detail (intra) and regional identity (inter)
are being diluted at deeper layers.
"""

import argparse
import json
import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPImageProcessor, LlavaForConditionalGeneration

# ========================
# Constants
# ========================
GRID_SIZE = 24
BLOCK_SPLIT = 4
BLOCK_SIZE = GRID_SIZE // BLOCK_SPLIT  # 6
NUM_BLOCKS = BLOCK_SPLIT * BLOCK_SPLIT  # 16


# ========================
# Data loading
# ========================
def load_image_paths(json_path, image_dir, max_images=None):
    if json_path and os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        paths = []
        for item in data:
            img_path = os.path.join(image_dir, item["image_id"])
            if os.path.exists(img_path):
                paths.append(img_path)
    else:
        paths = []
        for f in sorted(os.listdir(image_dir)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(image_dir, f))

    if max_images is not None:
        paths = paths[:max_images]
    return paths


# ========================
# Block indices
# ========================
def get_block_indices():
    """
    Precompute which of the 576 patch tokens belong to which block.
    Patch tokens are in row-major order: token k corresponds to
    row = k // 24, col = k % 24 in the 24x24 grid.
    Returns list of 16 lists, each containing token indices for that block.
    """
    blocks = [[] for _ in range(NUM_BLOCKS)]
    for token_idx in range(GRID_SIZE * GRID_SIZE):
        row = token_idx // GRID_SIZE
        col = token_idx % GRID_SIZE
        block_row = row // BLOCK_SIZE
        block_col = col // BLOCK_SIZE
        block_idx = block_row * BLOCK_SPLIT + block_col
        blocks[block_idx].append(token_idx)
    return blocks


# ========================
# Similarity computation
# ========================
def compute_mean_cosine_sim(tokens_a, tokens_b=None):
    """
    Compute mean cosine similarity.
    If tokens_b is None: pairwise within tokens_a (upper triangle).
    If tokens_b is given: all pairs between tokens_a and tokens_b.
    """
    normed_a = F.normalize(tokens_a, p=2, dim=-1)
    if tokens_b is None:
        sim_matrix = torch.mm(normed_a, normed_a.t())
        n = sim_matrix.size(0)
        if n < 2:
            return 0.0
        mask = torch.triu(torch.ones(n, n, device=sim_matrix.device), diagonal=1).bool()
        return sim_matrix[mask].mean().item()
    else:
        normed_b = F.normalize(tokens_b, p=2, dim=-1)
        return torch.mm(normed_a, normed_b.t()).mean().item()


def analyze_single_image(vision_tower, image_processor, image_path, block_indices):
    """
    Returns per-layer:
      - mean intra-block similarity (averaged over 16 blocks)
      - mean inter-block similarity (averaged over all block pairs)
      - 16x16 block-level similarity matrix
    """
    image = Image.open(image_path).convert("RGB")
    image_inputs = image_processor(images=image, return_tensors="pt")
    pixel_values = image_inputs["pixel_values"].to(
        device=vision_tower.device, dtype=vision_tower.dtype
    )

    with torch.no_grad():
        outputs = vision_tower(pixel_values, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    num_layers = len(hidden_states)

    layer_intra, layer_inter, layer_block_matrices = [], [], []

    for layer_idx in range(num_layers):
        patch_tokens = hidden_states[layer_idx][0, 1:, :]  # (576, hidden_dim)

        intra_sims = [compute_mean_cosine_sim(patch_tokens[block]) for block in block_indices]
        mean_intra = np.mean(intra_sims)

        block_centroids = torch.stack([patch_tokens[block].mean(dim=0) for block in block_indices])
        normed_centroids = F.normalize(block_centroids, p=2, dim=-1)
        block_sim_matrix = torch.mm(normed_centroids, normed_centroids.t())

        n = block_sim_matrix.size(0)
        mask = torch.triu(torch.ones(n, n, device=block_sim_matrix.device), diagonal=1).bool()
        mean_inter = block_sim_matrix[mask].mean().item()

        layer_intra.append(mean_intra)
        layer_inter.append(mean_inter)
        layer_block_matrices.append(block_sim_matrix.cpu().numpy())

    return layer_intra, layer_inter, layer_block_matrices


# ========================
# Plotting
# ========================
def plot_curves(mean_intra, mean_inter, ci_intra, ci_inter, num_layers, llava_layer, n_images, output_dir):
    color_intra = '#2E86AB'
    color_inter = '#A23B72'
    layers = np.arange(num_layers)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(layers, mean_intra, color=color_intra, linewidth=2.5,
            marker='o', markersize=5, label='Intra-block similarity', markevery=2, zorder=3)
    ax.fill_between(layers, mean_intra - ci_intra, mean_intra + ci_intra, alpha=0.2, color=color_intra, zorder=2)
    ax.plot(layers, mean_inter, color=color_inter, linewidth=2.5,
            marker='s', markersize=5, label='Inter-block similarity', markevery=2, zorder=3)
    ax.fill_between(layers, mean_inter - ci_inter, mean_inter + ci_inter, alpha=0.2, color=color_inter, zorder=2)
    ax.axvline(x=llava_layer, color='#555555', linestyle='--', linewidth=2,
               label=f'LLaVA extracted layer (L{llava_layer})', zorder=1)

    ax.annotate(f'{mean_intra[llava_layer]:.3f}',
                xy=(llava_layer, mean_intra[llava_layer]),
                xytext=(llava_layer-2, mean_intra[llava_layer]+0.03),
                fontsize=10, color=color_intra,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color_intra, linewidth=1.5))
    ax.annotate(f'{mean_inter[llava_layer]:.3f}',
                xy=(llava_layer, mean_inter[llava_layer]),
                xytext=(llava_layer+1.5, mean_inter[llava_layer]+0.03),
                fontsize=10, color=color_inter,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color_inter, linewidth=1.5))

    ax.set_xlabel("Layer Index", fontsize=14, fontweight='bold')
    ax.set_ylabel("Mean Cosine Similarity", fontsize=14, fontweight='bold')
    ax.set_title("Regional Distinctiveness Loss Across Vision Transformer Layers",
                 fontsize=15, fontweight='bold', pad=15)
    ax.legend(fontsize=12, frameon=True, shadow=True, loc='upper left')
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
    ax.set_xlim(-0.5, num_layers-0.5)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(range(0, num_layers, 4))
    ax.tick_params(axis='both', labelsize=11)
    ax.text(0.02, 0.98, f'N = {n_images} images\n4x4 blocks (6x6 patches each)',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.15))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "intra_inter_block_curve.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "intra_inter_block_curve.pdf"), bbox_inches="tight")
    plt.close()
    print("Curve plot saved (PNG + PDF).")


def plot_heatmaps(avg_block_matrices, num_layers, llava_layer, n_images, output_dir):
    selected_layers = [l for l in [1, 7, 11, 15, 19, llava_layer, num_layers-1] if l < num_layers]
    n_plots = len(selected_layers)

    fig = plt.figure(figsize=(18, 5))
    gs = gridspec.GridSpec(1, n_plots+1, width_ratios=[1]*n_plots + [0.05])
    vmin, vmax = 0.3, 1.0

    for idx, layer_idx in enumerate(selected_layers):
        ax = fig.add_subplot(gs[0, idx])
        im = ax.imshow(avg_block_matrices[layer_idx], cmap='RdYlBu_r',
                       vmin=vmin, vmax=vmax, aspect='auto', interpolation='nearest')
        layer_name = "Embed" if layer_idx == 0 else f"Layer {layer_idx}"
        ax.set_title(layer_name, fontsize=13,
                     fontweight='bold' if layer_idx == llava_layer else 'normal',
                     color='darkred' if layer_idx == llava_layer else 'black', pad=8)
        if layer_idx == llava_layer:
            for spine in ax.spines.values():
                spine.set_edgecolor('darkred')
                spine.set_linewidth(3)
        ax.set_xticks(range(0, NUM_BLOCKS, 4))
        ax.set_yticks(range(0, NUM_BLOCKS, 4))
        ax.set_xticklabels([str(i) for i in range(0, NUM_BLOCKS, 4)], fontsize=9)
        ax.set_yticklabels([str(i) for i in range(0, NUM_BLOCKS, 4)], fontsize=9)
        if idx == 0:
            ax.set_ylabel('Block Index', fontsize=11, fontweight='bold')
        if idx == n_plots // 2:
            ax.set_xlabel('Block Index', fontsize=11, fontweight='bold')

    cbar_ax = fig.add_subplot(gs[0, -1])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Cosine Similarity\n(Block Centroids)', fontsize=12,
                   fontweight='bold', rotation=270, labelpad=25)
    cbar.ax.tick_params(labelsize=10)
    fig.suptitle(f"Block-Level Similarity Evolution Across ViT Layers (N={n_images} images)",
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "block_heatmaps.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "block_heatmaps.pdf"), bbox_inches="tight")
    plt.close()
    print("Heatmap plot saved (PNG + PDF).")


def plot_gap(mean_intra, mean_inter, num_layers, llava_layer, output_dir):
    color_intra = '#2E86AB'
    color_inter = '#A23B72'
    layers = np.arange(num_layers)
    gap = mean_inter - mean_intra

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    colors = [color_inter if g > 0 else color_intra for g in gap]
    bars = ax.bar(layers, gap, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    bars[llava_layer].set_edgecolor('darkred')
    bars[llava_layer].set_linewidth(2.5)
    ax.axhline(y=0, color='black', linewidth=1.5, zorder=1)
    ax.axvline(x=llava_layer, color='#555555', linestyle='--', linewidth=2,
               label=f'LLaVA layer (L{llava_layer})', zorder=1)
    ax.annotate(f'{gap[llava_layer]:.3f}',
                xy=(llava_layer, gap[llava_layer]),
                xytext=(llava_layer, gap[llava_layer]+0.05),
                fontsize=11, fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow',
                         alpha=0.3, edgecolor='darkred', linewidth=2))
    ax.set_xlabel("Layer Index", fontsize=14, fontweight='bold')
    ax.set_ylabel("Similarity Gap\n(Inter-block - Intra-block)", fontsize=14, fontweight='bold')
    ax.set_title("Regional Distinctiveness Gap Across Layers", fontsize=15, fontweight='bold', pad=15)
    ax.legend(fontsize=12, frameon=True, shadow=True)
    ax.grid(True, alpha=0.25, axis='y', linestyle='--', linewidth=0.5)
    ax.set_xlim(-0.5, num_layers-0.5)
    ax.set_xticks(range(0, num_layers, 4))
    ax.tick_params(axis='both', labelsize=11)
    ax.text(0.02, 0.98,
            'Positive gap: regions becoming similar\nNegative gap: local features prominent',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.15))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "similarity_gap.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "similarity_gap.pdf"), bbox_inches="tight")
    plt.close()
    print("Gap plot saved (PNG + PDF).")


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
                    help="Optional JSON file with image_id list (if not set, scans image_dir)")
    ap.add_argument("--output_dir", type=str, default="./results_block_similarity",
                    help="Output directory for plots and data")
    ap.add_argument("--max_images", type=int, default=None,
                    help="Limit number of images to process")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Block-Level Visual Token Similarity Analysis")
    print(f"Grid: {GRID_SIZE}x{GRID_SIZE}, Blocks: {BLOCK_SPLIT}x{BLOCK_SPLIT}")
    print(f"Each block: {BLOCK_SIZE}x{BLOCK_SIZE} = {BLOCK_SIZE**2} tokens")
    print("=" * 60)

    image_paths = load_image_paths(args.json_path, args.image_dir, args.max_images)
    print(f"Total images: {len(image_paths)}")

    block_indices = get_block_indices()

    print(f"\nLoading model from: {args.model_path}")
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True,
    )
    vision_tower = model.vision_tower
    try:
        image_processor = CLIPImageProcessor.from_pretrained(args.model_path)
    except Exception:
        image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

    all_intra, all_inter, all_block_matrices = [], [], []
    failed = 0

    for img_path in tqdm(image_paths, desc="Processing"):
        try:
            intra, inter, matrices = analyze_single_image(
                vision_tower, image_processor, img_path, block_indices
            )
            all_intra.append(intra)
            all_inter.append(inter)
            all_block_matrices.append(matrices)
        except Exception as e:
            print(f"\nError: {img_path}: {e}")
            failed += 1

    print(f"\nProcessed: {len(all_intra)}, Failed: {failed}")

    all_intra = np.array(all_intra)
    all_inter = np.array(all_inter)
    num_layers = all_intra.shape[1]
    llava_layer = num_layers - 2

    mean_intra = np.mean(all_intra, axis=0)
    mean_inter = np.mean(all_inter, axis=0)
    ci_intra = 1.96 * np.std(all_intra, axis=0) / np.sqrt(len(all_intra))
    ci_inter = 1.96 * np.std(all_inter, axis=0) / np.sqrt(len(all_inter))

    print("\n" + "=" * 70)
    print(f"{'Layer':<10} {'Intra-Block':<14} {'Inter-Block':<14} {'Gap':<14}")
    print("-" * 52)
    for i in range(num_layers):
        layer_name = "Embed" if i == 0 else f"Layer {i}"
        if i == llava_layer:
            layer_name += " *"
        print(f"{layer_name:<10} {mean_intra[i]:<14.4f} {mean_inter[i]:<14.4f} {mean_inter[i]-mean_intra[i]:<14.4f}")

    n_images = len(all_intra)
    plot_curves(mean_intra, mean_inter, ci_intra, ci_inter, num_layers, llava_layer, n_images, args.output_dir)

    avg_block_matrices = [
        np.mean([all_block_matrices[img][layer] for img in range(n_images)], axis=0)
        for layer in range(num_layers)
    ]
    plot_heatmaps(avg_block_matrices, num_layers, llava_layer, n_images, args.output_dir)
    plot_gap(mean_intra, mean_inter, num_layers, llava_layer, args.output_dir)

    np.savez(
        os.path.join(args.output_dir, "block_similarity_data.npz"),
        all_intra=all_intra, all_inter=all_inter,
        mean_intra=mean_intra, mean_inter=mean_inter,
        ci_intra=ci_intra, ci_inter=ci_inter,
        avg_block_matrices=np.array(avg_block_matrices),
    )
    print("Data saved.")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    min_intra_layer = np.argmin(mean_intra)
    min_inter_layer = np.argmin(mean_inter)
    print(f"Intra-block similarity lowest at Layer {min_intra_layer}: {mean_intra[min_intra_layer]:.4f}")
    print(f"Inter-block similarity lowest at Layer {min_inter_layer}: {mean_inter[min_inter_layer]:.4f}")
    print(f"LLaVA layer ({llava_layer}) intra: {mean_intra[llava_layer]:.4f}, inter: {mean_inter[llava_layer]:.4f}")
    print(f"Final layer ({num_layers-1}) intra: {mean_intra[-1]:.4f}, inter: {mean_inter[-1]:.4f}")
    print(f"\nIntra increase (min->LLaVA): {((mean_intra[llava_layer] - mean_intra[min_intra_layer]) / mean_intra[min_intra_layer]) * 100:.1f}%")
    print(f"Inter increase (min->LLaVA): {((mean_inter[llava_layer] - mean_inter[min_inter_layer]) / mean_inter[min_inter_layer]) * 100:.1f}%")


if __name__ == "__main__":
    main()

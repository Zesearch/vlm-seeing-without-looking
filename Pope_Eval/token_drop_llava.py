#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Token Drop Experiment: LLaVA-1.5 on POPE
Randomly drops 0/25/50/75% of image tokens after projector, before LLM.
Correct implementation: inputs_embeds only, no pixel_values.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq

_YN_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)

def normalize_yn(text):
    if text is None:
        return None
    m = _YN_RE.search(text.strip())
    return m.group(1).lower() if m else None

def load_dataset(json_path, image_root):
    with open(json_path) as f:
        items = json.load(f)
    dataset, missing = [], 0
    for ex in items:
        img_path = image_root / ex["image_id"]
        if not img_path.exists():
            missing += 1
            continue
        dataset.append({
            "image": Image.open(img_path).convert("RGB"),
            "question": ex["question"],
            "answer": ex["answer"],
        })
    if missing:
        print(f"[WARN] Missing images: {missing}")
    return dataset


@torch.inference_mode()
def get_image_features(model, pixel_values):
    """Run ViT + projector to get image token embeddings."""
    image_outputs = model.vision_tower(pixel_values, output_hidden_states=True)
    image_features = model.multi_modal_projector(
        image_outputs.hidden_states[-2][:, 1:]  # drop CLS
    )
    return image_features  # (B, 576, D)


@torch.inference_mode()
def infer_with_drop(model, processor, images, questions, drop_ratio=0.0,
                    max_new_tokens=8, seed=0):
    prompts = [f"USER: <image>\n{q}\nASSISTANT:" for q in questions]
    inputs = processor(text=prompts, images=images, padding=True,
                       return_tensors="pt").to(model.device)

    input_ids    = inputs["input_ids"]
    pixel_values = inputs["pixel_values"]
    image_token_id = model.config.image_token_index

    if drop_ratio == 0.0:
        generated = model.generate(
            **inputs, do_sample=False, num_beams=1,
            max_new_tokens=max_new_tokens, use_cache=True,
        )
        trimmed = [out[len(inp):] for inp, out in zip(input_ids, generated)]

    else:
        image_features = get_image_features(model, pixel_values)  # (B, 576, D)
        n_img  = image_features.shape[1]
        n_keep = int(n_img * (1.0 - drop_ratio))

        rng = torch.Generator()
        rng.manual_seed(seed)
        keep = torch.randperm(n_img, generator=rng)[:n_keep].sort().values
        image_features = image_features[:, keep, :]  # (B, n_keep, D)

        text_embeds = model.get_input_embeddings()(input_ids)  # (B, L, D)
        final_embeds, final_masks = [], []

        for b in range(input_ids.shape[0]):
            ids = input_ids[b]
            img_pos   = (ids == image_token_id).nonzero(as_tuple=True)[0]
            img_start = img_pos[0].item()
            img_end   = img_pos[-1].item() + 1

            before = text_embeds[b, :img_start, :]
            after  = text_embeds[b, img_end:, :]
            new_emb = torch.cat([before, image_features[b], after], dim=0)
            new_msk = torch.ones(new_emb.shape[0], device=new_emb.device, dtype=torch.long)

            final_embeds.append(new_emb)
            final_masks.append(new_msk)

        max_len = max(e.shape[0] for e in final_embeds)
        D = final_embeds[0].shape[1]
        padded_embeds = torch.zeros(len(final_embeds), max_len, D,
                                    dtype=final_embeds[0].dtype,
                                    device=final_embeds[0].device)
        padded_masks  = torch.zeros(len(final_masks), max_len,
                                    dtype=final_masks[0].dtype,
                                    device=final_masks[0].device)
        for b, (emb, msk) in enumerate(zip(final_embeds, final_masks)):
            L = emb.shape[0]
            padded_embeds[b, :L] = emb
            padded_masks[b, :L]  = msk

        generated = model.generate(
            inputs_embeds=padded_embeds,
            attention_mask=padded_masks,
            do_sample=False, num_beams=1,
            max_new_tokens=max_new_tokens, use_cache=True,
        )
        trimmed = generated  # inputs_embeds mode returns new tokens only

    return processor.batch_decode(trimmed, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)


def evaluate(dataset, model, processor, drop_ratio, n_eval=300,
             batch_size=8, max_new_tokens=8, seed=0):
    n_eval = min(n_eval, len(dataset))
    tp = tn = fp = fn = total = 0

    for start in tqdm(range(0, n_eval, batch_size), desc=f"drop={int(drop_ratio*100)}%", leave=False):
        end   = min(start + batch_size, n_eval)
        batch = dataset[start:end]
        imgs  = [ex["image"] for ex in batch]
        qs    = [ex["question"].strip() + " Answer yes or no only." for ex in batch]
        gts   = [normalize_yn(ex["answer"]) for ex in batch]

        preds = [normalize_yn(t) for t in infer_with_drop(
            model, processor, imgs, qs,
            drop_ratio=drop_ratio, max_new_tokens=max_new_tokens, seed=seed,
        )]

        for gt, pred in zip(gts, preds):
            if gt not in ("yes","no") or pred not in ("yes","no"):
                continue
            total += 1
            if   gt=="yes" and pred=="yes": tp += 1
            elif gt=="no"  and pred=="no":  tn += 1
            elif gt=="no"  and pred=="yes": fp += 1
            else:                           fn += 1

        if start % 64 == 0:
            torch.cuda.empty_cache()

    acc = (tp + tn) / max(total, 1)
    return {"acc": acc, "n": total, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def plot_results(results, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    drop_ratios = [0, 25, 50, 75]
    accs = [results[r]["acc"] * 100 for r in drop_ratios]

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.set_facecolor("#fafafa")
    fig.patch.set_facecolor("white")
    ax.grid(axis="y", color="#d0d0d0", linewidth=0.7, linestyle="--", zorder=0)
    ax.grid(axis="x", color="#e8e8e8", linewidth=0.5, linestyle=":", zorder=0)

    color = "#2563EB"
    ax.plot(drop_ratios, accs, color=color, linewidth=2.0, marker="o",
            markersize=7, markerfacecolor="white",
            markeredgecolor=color, markeredgewidth=2.0, zorder=3)

    for x, y in zip(drop_ratios, accs):
        ax.annotate(f"{y:.1f}%", xy=(x, y), xytext=(0, 10),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=9, color="#1e3a8a", fontweight="semibold")

    ax.fill_between(drop_ratios, accs[0], accs, alpha=0.08, color=color)

    ax.set_xlabel("Image Token Drop Ratio (%)", fontsize=11, labelpad=6)
    ax.set_ylabel("Accuracy (%)", fontsize=11, labelpad=6)
    ax.set_title("Effect of Random Image Token Dropping\non POPE Accuracy (LLaVA-1.5-7B)",
                 fontsize=11, fontweight="bold", pad=10)

    ax.set_xticks(drop_ratios)
    ax.set_xticklabels([f"{r}%" for r in drop_ratios], fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.tick_params(axis="both", labelsize=10)

    y_min = min(accs) - 3
    y_max = max(accs) + 4
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(-5, 80)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved: {output_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="llava-hf/llava-1.5-7b-hf",
                    help="Path to LLaVA-1.5 model (local or HuggingFace hub)")
    ap.add_argument("--json_path",  required=True,
                    help="Path to POPE dataset JSON file")
    ap.add_argument("--image_root", required=True,
                    help="Root directory containing POPE images")
    ap.add_argument("--n_eval",     type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=6)
    ap.add_argument("--seed",       type=int, default=0)
    ap.add_argument("--output_json", default="token_drop_results.json")
    ap.add_argument("--output_fig",  default="token_drop_figure.pdf")
    args = ap.parse_args()

    print("Loading model...")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto",
    ).eval()

    dataset = load_dataset(Path(args.json_path), Path(args.image_root))
    print(f"Dataset: {len(dataset)} samples, evaluating {args.n_eval}\n")

    drop_ratios = [0, 25, 50, 75]
    all_results = {}

    for dr in drop_ratios:
        res = evaluate(dataset, model, processor,
                       drop_ratio=dr/100.0,
                       n_eval=args.n_eval,
                       batch_size=args.batch_size,
                       max_new_tokens=args.max_new_tokens,
                       seed=args.seed)
        all_results[dr] = res
        print(f"Drop {dr:3d}%  ->  acc={res['acc']:.4f}  (n={res['n']})")

    with open(args.output_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {args.output_json}")

    plot_results(all_results, args.output_fig)


if __name__ == "__main__":
    main()

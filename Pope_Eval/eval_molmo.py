#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Molmo-7B evaluation script for H200 cluster
Note: Molmo does NOT support batch inference, process one by one
"""

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig


# ============================================================
# 1) Parse yes/no
# ============================================================
_YN_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)

def normalize_yn(text: str):
    if text is None:
        return None
    m = _YN_RE.search(text.strip())
    return m.group(1).lower() if m else None


# ============================================================
# 2) Image utils & perturbations
# ============================================================
def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")

def structured_black_occlusion(img: Image.Image, p: float) -> Image.Image:
    """
    Structured occlusion with black regions
    p=0.5: occlude top half (50%)
    p=0.75: occlude top half + bottom-left quarter (75%)
    p=1.0: occlude entire image (100%)
    """
    arr = np.asarray(img.convert("RGB")).copy()
    H, W, _ = arr.shape
    h2 = H // 2
    w2 = W // 2

    if p >= 0.5:
        arr[0:h2, :, :] = 0
    if p >= 0.75:
        arr[h2:H, 0:w2, :] = 0
    if p >= 1.0:
        arr[:, :, :] = 0

    return Image.fromarray(arr)

def mix_with_noise(img: Image.Image, p: float, seed: int = 0) -> Image.Image:
    """Mix image with random noise"""
    rng = np.random.default_rng(seed)
    arr = np.asarray(img.convert("RGB")).astype(np.float32)
    h, w, _ = arr.shape
    noise = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8).astype(np.float32)
    mixed = (1.0 - p) * arr + p * noise
    mixed = np.clip(mixed, 0, 255).astype(np.uint8)
    return Image.fromarray(mixed).convert("RGB")


# ============================================================
# 3) Random swap indices
# ============================================================
def build_random_swap_indices(n, seed=0):
    rng = random.Random(seed)
    perm = list(range(n))
    rng.shuffle(perm)
    for i in range(n):
        if perm[i] == i:
            j = (i + 1) % n
            perm[i], perm[j] = perm[j], perm[i]
    return perm


# ============================================================
# 4) Molmo inference (single sample only)
# ============================================================
@torch.inference_mode()
def infer_molmo_single(model, processor, image_pil, question, max_new_tokens=8):
    """
    Single sample inference for Molmo (no batch support)
    """
    inputs = processor.process(
        images=[image_pil],
        text=question
    )
    
    # Move to device and add batch dimension
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
    
    # Convert images to bfloat16
    inputs["images"] = inputs["images"].to(torch.bfloat16)
    
    output = model.generate_from_batch(
        inputs,
        GenerationConfig(max_new_tokens=max_new_tokens, stop_strings="<|endoftext|>"),
        tokenizer=processor.tokenizer
    )
    
    # Decode generated tokens
    generated_tokens = output[0, inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return generated_text


def infer_with_image_loop(model, processor, images, questions, max_new_tokens=8):
    """Loop through samples one by one"""
    results = []
    for img, q in zip(images, questions):
        text = infer_molmo_single(model, processor, img, q, max_new_tokens=max_new_tokens)
        results.append(text)
    return results


def infer_no_image_loop(model, processor, questions, max_new_tokens=8):
    """Use dummy black image for no_image baseline"""
    dummy = Image.new("RGB", (224, 224), (0, 0, 0))
    results = []
    for q in questions:
        text = infer_molmo_single(model, processor, dummy, q, max_new_tokens=max_new_tokens)
        results.append(text)
    return results


# ============================================================
# 5) Confusion & metrics
# ============================================================
def init_confusion():
    return {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "n": 0, "n_pos": 0, "n_neg": 0}

def update_confusion(cm, gt, pred):
    cm["n"] += 1
    if gt == "yes":
        cm["n_pos"] += 1
        if pred == "yes":
            cm["tp"] += 1
        else:
            cm["fn"] += 1
    else:
        cm["n_neg"] += 1
        if pred == "no":
            cm["tn"] += 1
        else:
            cm["fp"] += 1

def metrics_from_confusion(cm):
    tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]
    n = max(cm["n"], 1)
    n_pos = max(cm["n_pos"], 1)
    n_neg = max(cm["n_neg"], 1)

    acc = (tp + tn) / n
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    neg_acc = tn / n_neg
    fp_rate = fp / n_neg
    fn_rate = fn / n_pos

    f1 = 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "n": cm["n"], "n_pos": cm["n_pos"], "n_neg": cm["n_neg"],
        "acc": acc, "precision": precision, "recall": recall, "f1": f1,
        "neg_acc": neg_acc, "fp_rate": fp_rate, "fn_rate": fn_rate,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }

def print_metrics(name, m):
    print(f"\n[{name}] n={m['n']} (pos={m['n_pos']}, neg={m['n_neg']})")
    print(f"  acc       : {m['acc']:.4f}")
    print(f"  precision : {m['precision']:.4f}")
    print(f"  recall    : {m['recall']:.4f}")
    print(f"  f1        : {m['f1']:.4f}")
    print(f"  neg_acc   : {m['neg_acc']:.4f}")
    print(f"  fp_rate   : {m['fp_rate']:.4f}")
    print(f"  fn_rate   : {m['fn_rate']:.4f}")


# ============================================================
# 6) Dataset loading
# ============================================================
def load_dataset(dataset_name: str = "chadlzx/pope-subset-300"):
    from datasets import load_dataset as hf_load_dataset
    hf_ds = hf_load_dataset(dataset_name, split="train")
    dataset = []
    for ex in hf_ds:
        dataset.append({
            "image": ex["image"].convert("RGB"),
            "question": ex["question"],
            "answer": ex["answer"],
        })
    return dataset


# ============================================================
# 7) Main evaluation (NO BATCHING - process one by one)
# ============================================================
def eval_all_loop(
    dataset, model, processor,
    n_eval=300,
    ps=(0.5, 0.75, 1.0),
    seed=0,
    batch_size=8,  # Still useful for grouping in progress bar
    max_new_tokens=8,
):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_eval = min(n_eval, len(dataset))
    perm = build_random_swap_indices(n_eval, seed=seed)

    cms = {
        "normal": init_confusion(),
        "swap": init_confusion(),
        "no_image": init_confusion(),
        **{f"black_p{p}": init_confusion() for p in ps},
        **{f"noise_p{p}": init_confusion() for p in ps},
    }

    # Process in "chunks" for progress bar, but actually process one by one
    for start in tqdm(range(0, n_eval, batch_size), desc="Evaluating"):
        end = min(start + batch_size, n_eval)
        batch = [dataset[i] for i in range(start, end)]

        qs = [ex["question"].strip() + " Answer yes or no only." for ex in batch]
        gts = [normalize_yn(ex["answer"]) for ex in batch]

        imgs = [ex["image"] for ex in batch]
        swap_imgs = [dataset[perm[i]]["image"] for i in range(start, end)]

        # Process one by one (no real batching)
        pred_normal = [normalize_yn(t) for t in infer_with_image_loop(
            model, processor, imgs, qs, max_new_tokens=max_new_tokens
        )]
        pred_swap = [normalize_yn(t) for t in infer_with_image_loop(
            model, processor, swap_imgs, qs, max_new_tokens=max_new_tokens
        )]
        pred_noimg = [normalize_yn(t) for t in infer_no_image_loop(
            model, processor, qs, max_new_tokens=max_new_tokens
        )]

        pred_black, pred_noise = {}, {}
        for p in ps:
            bimgs = [structured_black_occlusion(img, p) for img in imgs]
            pred_black[p] = [normalize_yn(t) for t in infer_with_image_loop(
                model, processor, bimgs, qs, max_new_tokens=max_new_tokens
            )]

            nimgs = [mix_with_noise(img, p, seed=seed + start * 100 + k) for k, img in enumerate(imgs)]
            pred_noise[p] = [normalize_yn(t) for t in infer_with_image_loop(
                model, processor, nimgs, qs, max_new_tokens=max_new_tokens
            )]

        for k in range(len(batch)):
            gt = gts[k]
            if gt not in ("yes", "no"):
                continue

            if pred_normal[k] in ("yes", "no"):
                update_confusion(cms["normal"], gt, pred_normal[k])
            if pred_swap[k] in ("yes", "no"):
                update_confusion(cms["swap"], gt, pred_swap[k])
            if pred_noimg[k] in ("yes", "no"):
                update_confusion(cms["no_image"], gt, pred_noimg[k])

            for p in ps:
                if pred_black[p][k] in ("yes", "no"):
                    update_confusion(cms[f"black_p{p}"], gt, pred_black[p][k])
                if pred_noise[p][k] in ("yes", "no"):
                    update_confusion(cms[f"noise_p{p}"], gt, pred_noise[p][k])

        # Clean cache periodically
        if start % 10 == 0:
            torch.cuda.empty_cache()

    results = {k: metrics_from_confusion(v) for k, v in cms.items()}
    return results


# ============================================================
# 8) Model loading
# ============================================================
def load_molmo_local(model_path: str):
    print("="*80)
    print("Loading Molmo-7B Model")
    print("="*80)
    
    print("\n[1/2] Loading processor...")
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto',
    )
    
    print("[2/2] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto',
    ).eval()
    
    # Convert to bfloat16
    model.to(dtype=torch.bfloat16)
    
    print("Model loaded successfully!")
    print("="*80)
    
    return model, processor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default="allenai/Molmo-7B-D-0924",
                    help="Path to Molmo-7B model (local or HuggingFace hub)")
    ap.add_argument("--dataset", type=str, default="chadlzx/pope-subset-300",
                    help="HuggingFace dataset name")
    ap.add_argument("--n_eval", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=8)  # For progress bar grouping only
    ap.add_argument("--max_new_tokens", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ps", type=float, nargs="+", default=[0.5, 0.75, 1.0])
    ap.add_argument("--output_json", type=str, default="perturbation_results_molmo.json",
                    help="Output JSON file for results")
    args = ap.parse_args()

    model, processor = load_molmo_local(args.model_path)
    dataset = load_dataset(args.dataset)

    print(f"\nLoaded {len(dataset)} samples from dataset")
    print(f"Evaluating {args.n_eval} samples (NO BATCHING - one by one)\n")

    results = eval_all_loop(
        dataset, model, processor,
        n_eval=args.n_eval,
        ps=tuple(args.ps),
        seed=args.seed,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    results['config'] = {
        'n_eval': args.n_eval,
        'ps': args.ps,
        'seed': args.seed,
    }

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    for k in ["normal", "swap", "no_image",
              "black_p0.5", "black_p0.75", "black_p1.0",
              "noise_p0.5", "noise_p0.75", "noise_p1.0"]:
        if k in results:
            print_metrics(k, results[k])

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print(f"Results saved to: {args.output_json}")
    print("="*80)


if __name__ == "__main__":
    main()

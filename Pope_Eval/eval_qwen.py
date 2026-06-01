#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Qwen3-VL evaluation script for H200 cluster
Fixed: Using structured_black_occlusion (not brightness dimming)
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
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


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
        # Occlude top half
        arr[0:h2, :, :] = 0

    if p >= 0.75:
        # Additionally occlude bottom-left quarter
        arr[h2:H, 0:w2, :] = 0
    
    if p >= 1.0:
        # Occlude everything
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
# 4) Qwen3-VL inference (true batch)
# ============================================================
@torch.inference_mode()
def infer_qwen3vl_batch(model, processor, images_pil, questions, max_new_tokens=8):
    """
    Batch inference for Qwen3-VL using apply_chat_template -> generate -> trim prompt.
    images_pil: list[PIL.Image]
    questions: list[str]
    returns: list[str]
    """
    assert len(images_pil) == len(questions)

    conversations = []
    for img, q in zip(images_pil, questions):
        conversations.append([
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": q},
                ],
            }
        ])

    inputs = processor.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        padding=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs.pop("token_type_ids", None)
    inputs = inputs.to(model.device)

    generated = model.generate(
        **inputs,
        do_sample=False,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        use_cache=True,
    )

    # Trim prompt tokens per-sample
    input_ids = inputs["input_ids"]
    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated)]
    texts = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return texts


@torch.inference_mode()
def infer_with_image_batch(model, processor, images, questions, max_new_tokens=8):
    return infer_qwen3vl_batch(model, processor, images, questions, max_new_tokens=max_new_tokens)


@torch.inference_mode()
def infer_no_image_batch(model, processor, questions, max_new_tokens=8):
    """
    Use a dummy black image as "no_image" proxy.
    """
    dummy = Image.new("RGB", (224, 224), (0, 0, 0))
    images = [dummy] * len(questions)
    return infer_qwen3vl_batch(model, processor, images, questions, max_new_tokens=max_new_tokens)


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
# 7) Main evaluation
# ============================================================
def eval_all_batch(
    dataset, model, processor,
    n_eval=300,
    ps=(0.5, 0.75, 1.0),
    seed=0,
    batch_size=8,
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

    for start in tqdm(range(0, n_eval, batch_size), desc="Evaluating"):
        end = min(start + batch_size, n_eval)
        batch = [dataset[i] for i in range(start, end)]

        qs = [ex["question"].strip() + " Answer yes or no only." for ex in batch]
        gts = [normalize_yn(ex["answer"]) for ex in batch]

        imgs = [ex["image"] for ex in batch]
        swap_imgs = [dataset[perm[i]]["image"] for i in range(start, end)]

        pred_normal = [normalize_yn(t) for t in infer_with_image_batch(
            model, processor, imgs, qs, max_new_tokens=max_new_tokens
        )]
        pred_swap = [normalize_yn(t) for t in infer_with_image_batch(
            model, processor, swap_imgs, qs, max_new_tokens=max_new_tokens
        )]
        pred_noimg = [normalize_yn(t) for t in infer_no_image_batch(
            model, processor, qs, max_new_tokens=max_new_tokens
        )]

        pred_black, pred_noise = {}, {}
        for p in ps:
            # Use structured_black_occlusion (not brightness dimming)
            bimgs = [structured_black_occlusion(img, p) for img in imgs]
            pred_black[p] = [normalize_yn(t) for t in infer_with_image_batch(
                model, processor, bimgs, qs, max_new_tokens=max_new_tokens
            )]

            nimgs = [mix_with_noise(img, p, seed=seed + start * 100 + k) for k, img in enumerate(imgs)]
            pred_noise[p] = [normalize_yn(t) for t in infer_with_image_batch(
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
        if start % 50 == 0:
            torch.cuda.empty_cache()

    results = {k: metrics_from_confusion(v) for k, v in cms.items()}
    return results


# ============================================================
# 8) Model loading
# ============================================================
def load_qwen3vl_local(model_path: str):
    print("="*80)
    print("Loading Qwen3-VL Model")
    print("="*80)
    
    print("\n[1/2] Loading processor...")
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    
    # Left padding for decoder-only batch generation
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    
    print("[2/2] Loading model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    
    print("Model loaded successfully!")
    print("="*80)
    
    return model, processor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default="Qwen/Qwen3-8B",
                    help="Path to Qwen3-VL model (local or HuggingFace hub)")
    ap.add_argument("--dataset", type=str, default="chadlzx/pope-subset-300",
                    help="HuggingFace dataset name")
    ap.add_argument("--n_eval", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ps", type=float, nargs="+", default=[0.5, 0.75, 1.0])
    ap.add_argument("--output_json", type=str, default="perturbation_results_qwen.json",
                    help="Output JSON file for results")
    args = ap.parse_args()

    model, processor = load_qwen3vl_local(args.model_path)
    dataset = load_dataset(args.dataset)

    print(f"\nLoaded {len(dataset)} samples from dataset")
    print(f"Evaluating {args.n_eval} samples\n")

    results = eval_all_batch(
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

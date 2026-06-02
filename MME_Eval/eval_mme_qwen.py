#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MME evaluation script for Qwen3-VL.
Conditions: normal, no_image, black_p0.5, black_p0.75, blur_p0.5, blur_p0.75
Supports per-category breakdown.
"""

import argparse
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image, ImageFilter
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


# ============================================================
# 1) Parse yes/no
# ============================================================
_YN_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)

def parse_yes_no(text: str) -> Optional[str]:
    if text is None:
        return None
    t = str(text).strip().lower()
    has_yes = "yes" in t
    has_no = "no" in t
    if has_yes and has_no:
        return None
    m = _YN_RE.search(t)
    return m.group(1).lower() if m else None


# ============================================================
# 2) Prompt
# ============================================================
def build_binary_prompt(question: str) -> str:
    return f"{question}\nAnswer with Yes or No only."


# ============================================================
# 3) Dataset
# ============================================================
def load_mme_dataset(metadata_path: str, root_dir: str) -> List[Dict[str, Any]]:
    metadata_path = Path(metadata_path)
    root_dir = Path(root_dir)

    with metadata_path.open("r", encoding="utf-8") as f:
        items = json.load(f)

    for item in items:
        item["answer_yn"] = item["answer"].lower()
        img_rel = item["image_path"]
        item["_abs_imagefile"] = str((root_dir / img_rel).resolve())

    if len(items) == 0:
        raise ValueError(f"No usable items loaded from {metadata_path}")

    return items

def load_image_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


# ============================================================
# 4) Perturbations
# ============================================================
def structured_black_occlusion(img: Image.Image, p: float) -> Image.Image:
    p = float(p)
    if p not in (0.5, 0.75):
        raise ValueError(f"structured_black_occlusion only supports p in {{0.5, 0.75}}, got {p}")
    arr = np.asarray(img.convert("RGB")).copy()
    H, W, _ = arr.shape
    h2 = H // 2
    w2 = W // 2
    arr[0:h2, :, :] = 0
    if p == 0.75:
        arr[h2:H, 0:w2, :] = 0
    return Image.fromarray(arr)

def blur_global(img: Image.Image, p: float) -> Image.Image:
    radius = float(p) * 10.0
    return img.convert("RGB").filter(ImageFilter.GaussianBlur(radius=radius))


# ============================================================
# 5) Qwen3-VL inference
# ============================================================
@torch.inference_mode()
def infer_with_image_batch_qwen(model, processor, image_paths, prompts, max_new_tokens=5):
    messages_batch = []
    for img_path, prompt in zip(image_paths, prompts):
        messages_batch.append([{
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": prompt}
            ]
        }])

    inputs = processor.apply_chat_template(
        messages_batch, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt", padding=True,
    ).to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)


@torch.inference_mode()
def infer_no_image_batch_qwen(model, processor, prompts, max_new_tokens=5):
    messages_batch = [[{"role": "user", "content": [{"type": "text", "text": p}]}] for p in prompts]

    inputs = processor.apply_chat_template(
        messages_batch, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt", padding=True,
    ).to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)


# ============================================================
# 6) Confusion matrix
# ============================================================
def init_confusion():
    return {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "n_valid": 0, "n_parse_fail": 0, "n_pos": 0, "n_neg": 0}

def update_confusion(cm, gt, pred):
    if pred not in ("yes", "no"):
        cm["n_parse_fail"] += 1
        return
    cm["n_valid"] += 1
    if gt == "yes":
        cm["n_pos"] += 1
        if pred == "yes": cm["tp"] += 1
        else: cm["fn"] += 1
    else:
        cm["n_neg"] += 1
        if pred == "no": cm["tn"] += 1
        else: cm["fp"] += 1

def metrics_from_confusion(cm):
    tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]
    n_valid = max(cm["n_valid"], 1)
    n_total = cm["n_valid"] + cm["n_parse_fail"]
    n_pos = max(cm["n_pos"], 1)
    n_neg = max(cm["n_neg"], 1)

    acc = (tp + tn) / n_valid
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    return {
        "n_total": n_total, "n_valid": cm["n_valid"],
        "n_pos": cm["n_pos"], "n_neg": cm["n_neg"],
        "parse_fail_rate": cm["n_parse_fail"] / max(n_total, 1),
        "acc": acc, "precision": precision, "recall": recall, "f1": f1,
        "neg_acc": tn / n_neg, "fp_rate": fp / n_neg, "fn_rate": fn / n_pos,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }

def print_metrics(name, m):
    print(f"\n[{name}] n_total={m['n_total']} (valid={m['n_valid']}, pos={m['n_pos']}, neg={m['n_neg']})")
    print(f"  parse_fail : {m['parse_fail_rate']:.4f}")
    print(f"  acc        : {m['acc']:.4f}")
    print(f"  precision  : {m['precision']:.4f}")
    print(f"  recall     : {m['recall']:.4f}")
    print(f"  f1         : {m['f1']:.4f}")
    print(f"  neg_acc    : {m['neg_acc']:.4f}")
    print(f"  fp_rate    : {m['fp_rate']:.4f}")
    print(f"  fn_rate    : {m['fn_rate']:.4f}")


# ============================================================
# 7) Main evaluation
# ============================================================
def eval_all_qwen(dataset, model, processor, ps=(0.5, 0.75), batch_size=50, max_new_tokens=5):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    n_eval = len(dataset)
    categories = sorted(set(item["category"] for item in dataset))

    def init_cond():
        return {
            "normal": init_confusion(),
            "no_image": init_confusion(),
            **{f"black_p{p}": init_confusion() for p in ps if p in (0.5, 0.75)},
            **{f"blur_p{p}": init_confusion() for p in ps},
        }

    cms = {"overall": init_cond()}
    for cat in categories:
        cms[cat] = init_cond()

    robust_samples = []
    tmp_dir = Path(tempfile.mkdtemp())

    try:
        for start in tqdm(range(0, n_eval, batch_size), desc="Evaluating"):
            end = min(start + batch_size, n_eval)
            batch = dataset[start:end]

            prompts = [build_binary_prompt(ex["question"]) for ex in batch]
            gts = [ex["answer_yn"] for ex in batch]
            cats = [ex["category"] for ex in batch]
            img_paths_normal = [ex["_abs_imagefile"] for ex in batch]

            out_normal = infer_with_image_batch_qwen(model, processor, img_paths_normal, prompts, max_new_tokens)
            pred_normal = [parse_yes_no(t) for t in out_normal]

            out_noimg = infer_no_image_batch_qwen(model, processor, prompts, max_new_tokens)
            pred_noimg = [parse_yes_no(t) for t in out_noimg]

            pred_black, pred_blur = {}, {}
            out_black, out_blur = {}, {}

            for p in ps:
                if p in (0.5, 0.75):
                    black_paths = []
                    for idx, img_path in enumerate(img_paths_normal):
                        img = load_image_rgb(img_path)
                        tmp_path = tmp_dir / f"black_{p}_{start+idx}.png"
                        structured_black_occlusion(img, p).save(tmp_path)
                        black_paths.append(str(tmp_path))
                    out_black[p] = infer_with_image_batch_qwen(model, processor, black_paths, prompts, max_new_tokens)
                    pred_black[p] = [parse_yes_no(t) for t in out_black[p]]

                blur_paths = []
                for idx, img_path in enumerate(img_paths_normal):
                    img = load_image_rgb(img_path)
                    tmp_path = tmp_dir / f"blur_{p}_{start+idx}.png"
                    blur_global(img, p).save(tmp_path)
                    blur_paths.append(str(tmp_path))
                out_blur[p] = infer_with_image_batch_qwen(model, processor, blur_paths, prompts, max_new_tokens)
                pred_blur[p] = [parse_yes_no(t) for t in out_blur[p]]

            for k in range(len(batch)):
                gt = gts[k]
                cat = cats[k]

                for scope in ["overall", cat]:
                    update_confusion(cms[scope]["normal"], gt, pred_normal[k])
                    update_confusion(cms[scope]["no_image"], gt, pred_noimg[k])
                    for p in ps:
                        if p in (0.5, 0.75):
                            update_confusion(cms[scope][f"black_p{p}"], gt, pred_black[p][k])
                        update_confusion(cms[scope][f"blur_p{p}"], gt, pred_blur[p][k])

                is_robust = (
                    pred_normal[k] == gt and pred_noimg[k] == gt and
                    all(pred_black.get(p, [None])[k] == gt for p in ps if p in (0.5, 0.75)) and
                    all(pred_blur.get(p, [None])[k] == gt for p in ps)
                )
                if is_robust:
                    robust_samples.append({
                        "idx": start + k, "category": cat,
                        "question": batch[k]["question"], "gt": gt,
                        "pred_normal": pred_normal[k], "pred_no_image": pred_noimg[k],
                        "pred_black_0.75": pred_black.get(0.75, [None])[k],
                        "pred_blur_0.75": pred_blur.get(0.75, [None])[k],
                    })

            torch.cuda.empty_cache()

    finally:
        shutil.rmtree(tmp_dir)

    results = {key: {k: metrics_from_confusion(v) for k, v in conds.items()} for key, conds in cms.items()}
    return results, robust_samples, categories


# ============================================================
# 8) Model loading
# ============================================================
def load_model(model_path):
    print("="*80)
    print("Loading Qwen3-VL Model")
    print("="*80)

    print("[1/2] Loading processor...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
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


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default="Qwen/Qwen3-8B",
                    help="Path to Qwen3-VL model (local or HuggingFace hub)")
    ap.add_argument("--metadata_path", type=str, required=True,
                    help="Path to MME metadata.json")
    ap.add_argument("--root_dir", type=str, required=True,
                    help="Root directory of MME dataset")
    ap.add_argument("--output_dir", type=str, default="mme_results_qwen",
                    help="Directory to save results")
    ap.add_argument("--batch_size", type=int, default=50)
    ap.add_argument("--max_new_tokens", type=int, default=5)
    ap.add_argument("--ps", type=float, nargs="+", default=[0.5, 0.75])
    args = ap.parse_args()

    model, processor = load_model(args.model_path)

    print("\nLoading MME dataset...")
    dataset = load_mme_dataset(args.metadata_path, args.root_dir)
    print(f"Loaded {len(dataset)} samples")

    print("\nRunning evaluation...")
    results, robust_samples, categories = eval_all_qwen(
        dataset, model, processor,
        ps=tuple(args.ps),
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    cond_order = ["normal", "no_image"] + \
                 [f"black_p{p}" for p in args.ps if p in (0.5, 0.75)] + \
                 [f"blur_p{p}" for p in args.ps]

    print("\n" + "="*80)
    print("OVERALL RESULTS")
    print("="*80)
    for k in cond_order:
        print_metrics(k, results["overall"][k])

    for cat in categories:
        print(f"\n{'='*80}")
        print(f"CATEGORY: {cat}")
        print(f"{'='*80}")
        for k in cond_order:
            print_metrics(k, results[cat][k])

    print(f"\nFound {len(robust_samples)} robust samples")

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "mme_evaluation_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "overall": results["overall"],
            "per_category": {cat: results[cat] for cat in categories},
            "n_robust_samples": len(robust_samples),
            "n_total_samples": len(dataset),
            "categories": categories,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

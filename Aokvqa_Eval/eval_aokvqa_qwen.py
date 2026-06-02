#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A-OKVQA evaluation script for Qwen3-VL.
Task: multi-choice (A/B/C/D)
Conditions: normal, no_image, swap, black_p0.5, black_p0.75, blur_p0.5, blur_p0.75
"""

import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageFilter
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


# ============================================================
# Parsing (ABCD)
# ============================================================
_ABCD_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)

def normalize_abcd_idx(text: str) -> Optional[int]:
    if text is None:
        return None
    t = str(text).strip()
    m = _ABCD_RE.search(t)
    if not m:
        m2 = re.search(r"[\(\[]?\s*([ABCD])\s*[\)\].,:]?", t, re.IGNORECASE)
        if not m2:
            return None
        ch = m2.group(1).upper()
    else:
        ch = m.group(1).upper()
    return ord(ch) - ord("A")


# ============================================================
# Prompt (MCQ)
# ============================================================
def build_mc_prompt(question: str, choices: List[str]) -> str:
    assert len(choices) == 4
    return (
        f"Question: {question}\n"
        f"Choices:\n"
        f"(A) {choices[0]}\n"
        f"(B) {choices[1]}\n"
        f"(C) {choices[2]}\n"
        f"(D) {choices[3]}\n"
        "Answer with A, B, C, or D only."
    )


# ============================================================
# Dataset (JSONL)
# ============================================================
def load_jsonl_dataset(jsonl_path: str, root_dir: str) -> List[Dict[str, Any]]:
    items = []
    jsonl_path = Path(jsonl_path)
    root_dir = Path(root_dir)

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            if "choices" not in ex or len(ex["choices"]) != 4:
                continue
            if "answer_idx" not in ex:
                continue
            img_rel = ex.get("imagefile")
            if not img_rel:
                continue
            ex["_abs_imagefile"] = str((root_dir / img_rel).resolve())
            items.append(ex)

    if not items:
        raise ValueError(f"No usable items loaded from {jsonl_path}")
    return items

def load_image_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


# ============================================================
# Perturbations
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
# Swap indices
# ============================================================
def build_random_swap_indices(n: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    perm = list(range(n))
    rng.shuffle(perm)
    for i in range(n):
        if perm[i] == i:
            j = (i + 1) % n
            perm[i], perm[j] = perm[j], perm[i]
    return perm


# ============================================================
# Qwen3-VL inference
# ============================================================
@torch.inference_mode()
def qwen3vl_generate_one(model, processor, image: Optional[Image.Image], text: str,
                         max_new_tokens: int = 6) -> str:
    if image is None:
        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    else:
        messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                  {"type": "text", "text": text}]}]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    in_ids = inputs["input_ids"]
    trimmed = [out_row[len(in_row):] for in_row, out_row in zip(in_ids, generated_ids)]
    out_text = processor.batch_decode(trimmed, skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)[0]
    return out_text


def infer_with_image(model, processor, images, prompts, max_new_tokens=6):
    return [qwen3vl_generate_one(model, processor, img, p, max_new_tokens) for img, p in zip(images, prompts)]

def infer_no_image(model, processor, prompts, max_new_tokens=6):
    return [qwen3vl_generate_one(model, processor, None, p, max_new_tokens) for p in prompts]


# ============================================================
# Eval
# ============================================================
def _update_bucket(bucket, gt, pred):
    bucket["n"] += 1
    if pred is None:
        bucket["n_parse_fail"] += 1
    elif pred == gt:
        bucket["n_ok"] += 1

def eval_all(
    dataset, model, processor,
    n_eval=300,
    ps=(0.5, 0.75),
    seed=0,
    max_new_tokens=6,
    do_swap=True,
    do_noimg=True,
    debug_print_k=3,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_eval = min(n_eval, len(dataset))
    perm = build_random_swap_indices(n_eval, seed=seed) if do_swap else None

    def init_bucket():
        return {"n": 0, "n_ok": 0, "n_parse_fail": 0}

    buckets = {"normal": init_bucket()}
    if do_noimg: buckets["no_image"] = init_bucket()
    if do_swap:  buckets["swap"] = init_bucket()
    for p in ps:
        if p in (0.5, 0.75):
            buckets[f"black_p{p}"] = init_bucket()
        buckets[f"blur_p{p}"] = init_bucket()

    per_sample = []
    printed = 0

    for i in tqdm(range(n_eval), desc="eval"):
        ex = dataset[i]
        prompt = build_mc_prompt(ex["question"], ex["choices"])
        gt = int(ex["answer_idx"])
        img = load_image_rgb(ex["_abs_imagefile"])

        out_normal = infer_with_image(model, processor, [img], [prompt], max_new_tokens)[0]
        pred_normal = normalize_abcd_idx(out_normal)

        out_noimg, pred_noimg = None, None
        if do_noimg:
            out_noimg = infer_no_image(model, processor, [prompt], max_new_tokens)[0]
            pred_noimg = normalize_abcd_idx(out_noimg)

        out_swap, pred_swap = None, None
        if do_swap:
            swap_img = load_image_rgb(dataset[perm[i]]["_abs_imagefile"])
            out_swap = infer_with_image(model, processor, [swap_img], [prompt], max_new_tokens)[0]
            pred_swap = normalize_abcd_idx(out_swap)

        outs_black, preds_black = {}, {}
        outs_blur, preds_blur = {}, {}

        for p in ps:
            if p in (0.5, 0.75):
                ob = infer_with_image(model, processor, [structured_black_occlusion(img, p)], [prompt], max_new_tokens)[0]
                outs_black[p] = ob
                preds_black[p] = normalize_abcd_idx(ob)
            obl = infer_with_image(model, processor, [blur_global(img, p)], [prompt], max_new_tokens)[0]
            outs_blur[p] = obl
            preds_blur[p] = normalize_abcd_idx(obl)

        _update_bucket(buckets["normal"], gt, pred_normal)
        if do_noimg: _update_bucket(buckets["no_image"], gt, pred_noimg)
        if do_swap:  _update_bucket(buckets["swap"], gt, pred_swap)
        for p in ps:
            if p in (0.5, 0.75):
                _update_bucket(buckets[f"black_p{p}"], gt, preds_black[p])
            _update_bucket(buckets[f"blur_p{p}"], gt, preds_blur[p])

        per_sample.append({
            "idx": i, "id": ex.get("id"), "source_id": ex.get("source_id"),
            "abs_imagefile": ex["_abs_imagefile"],
            "question": ex["question"], "choices": ex["choices"], "gt_idx": gt,
            "out_normal": out_normal, "pred_normal": pred_normal,
            "out_noimg": out_noimg, "pred_noimg": pred_noimg,
            "out_swap": out_swap, "pred_swap": pred_swap,
            "outs_black": outs_black, "preds_black": preds_black,
            "outs_blur": outs_blur, "preds_blur": preds_blur,
        })

        if printed < debug_print_k:
            printed += 1
            gt_letter = chr(ord("A") + gt)
            fmt = lambda pred: None if pred is None else chr(ord("A") + pred)
            print(f"\n--- DEBUG [{i}] id={ex.get('id')} GT={gt_letter} ---")
            print("normal:", out_normal, "->", fmt(pred_normal))
            if do_noimg: print("noimg:", out_noimg, "->", fmt(pred_noimg))
            if do_swap:  print("swap:", out_swap, "->", fmt(pred_swap))
            for p in ps:
                if p in (0.5, 0.75): print(f"black p={p}:", outs_black[p], "->", fmt(preds_black[p]))
                print(f"blur  p={p}:", outs_blur[p], "->", fmt(preds_blur[p]))

    results = {}
    for name, b in buckets.items():
        n = max(int(b["n"]), 1)
        results[name] = {
            "n": int(b["n"]),
            "acc": float(b["n_ok"]) / n,
            "parse_fail": float(b["n_parse_fail"]) / n,
        }
    return results, per_sample


def print_results(results, order=None):
    if order is None:
        order = list(results.keys())
    for k in order:
        m = results[k]
        print(f"\n[{k}] n={m['n']}")
        print(f"  acc        : {m['acc']:.4f}")
        print(f"  parse_fail : {m['parse_fail']:.4f}")


# ============================================================
# Model loading
# ============================================================
def load_model(model_path: str):
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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default="Qwen/Qwen3-8B",
                    help="Path to Qwen3-VL model (local or HuggingFace hub)")
    ap.add_argument("--pack_root", type=str, required=True,
                    help="Root directory of aokvqa_pack (contains validation.jsonl and images)")
    ap.add_argument("--split", type=str, default="validation",
                    help="Dataset split jsonl filename without extension")
    ap.add_argument("--n_eval", type=int, default=300)
    ap.add_argument("--max_new_tokens", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ps", type=float, nargs="+", default=[0.5, 0.75])
    ap.add_argument("--no_swap", action="store_true")
    ap.add_argument("--no_noimg", action="store_true")
    ap.add_argument("--debug_print_k", type=int, default=3)
    ap.add_argument("--output_json", type=str, default="aokvqa_results_qwen.json")
    args = ap.parse_args()

    model, processor = load_model(args.model_path)

    jsonl_path = os.path.join(args.pack_root, f"{args.split}.jsonl")
    dataset = load_jsonl_dataset(jsonl_path, root_dir=args.pack_root)
    print(f"\nLoaded {len(dataset)} samples from {jsonl_path}")

    results, per_sample = eval_all(
        dataset, model, processor,
        n_eval=args.n_eval,
        ps=tuple(args.ps),
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        do_swap=not args.no_swap,
        do_noimg=not args.no_noimg,
        debug_print_k=args.debug_print_k,
    )

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print_results(results, order=[
        "normal", "no_image", "swap",
        "black_p0.5", "black_p0.75",
        "blur_p0.5", "blur_p0.75",
    ])

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {args.output_json}")


if __name__ == "__main__":
    main()

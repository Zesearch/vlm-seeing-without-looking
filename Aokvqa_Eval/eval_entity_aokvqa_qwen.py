#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A-OKVQA entity masking evaluation script for Qwen3-VL.
Compares accuracy across: original, blackbox (bbox), blackmask (SAM2).
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
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
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    in_ids = inputs["input_ids"]
    trimmed = [out_row[len(in_row):] for in_row, out_row in zip(in_ids, generated_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


# ============================================================
# Eval: original vs blackbox vs blackmask
# ============================================================
def eval_entity_masking(
    dataset_original, dataset_blackbox, dataset_blackmask,
    model, processor,
    n_eval=None,
    max_new_tokens=6,
    debug_print_k=3,
):
    max_possible = min(len(dataset_original), len(dataset_blackbox), len(dataset_blackmask))
    n_eval = max_possible if n_eval is None else min(n_eval, max_possible)
    print(f"Evaluating {n_eval} samples (max available: {max_possible})")

    results = {
        "original":  {"n": 0, "n_ok": 0, "n_parse_fail": 0},
        "blackbox":  {"n": 0, "n_ok": 0, "n_parse_fail": 0},
        "blackmask": {"n": 0, "n_ok": 0, "n_parse_fail": 0},
    }

    per_sample = []
    printed = 0

    for i in tqdm(range(n_eval), desc="Evaluating entity masking"):
        ex_orig = dataset_original[i]
        ex_box  = dataset_blackbox[i]
        ex_mask = dataset_blackmask[i]

        assert ex_orig["id"] == ex_box["id"] == ex_mask["id"], \
            f"ID mismatch at index {i}: {ex_orig['id']} vs {ex_box['id']} vs {ex_mask['id']}"

        prompt = build_mc_prompt(ex_orig["question"], ex_orig["choices"])
        gt = int(ex_orig["answer_idx"])

        img_orig = load_image_rgb(ex_orig["_abs_imagefile"])
        img_box  = load_image_rgb(ex_box["_abs_imagefile"])
        img_mask = load_image_rgb(ex_mask["_abs_imagefile"])

        out_orig = qwen3vl_generate_one(model, processor, img_orig, prompt, max_new_tokens)
        out_box  = qwen3vl_generate_one(model, processor, img_box,  prompt, max_new_tokens)
        out_mask = qwen3vl_generate_one(model, processor, img_mask, prompt, max_new_tokens)

        pred_orig = normalize_abcd_idx(out_orig)
        pred_box  = normalize_abcd_idx(out_box)
        pred_mask = normalize_abcd_idx(out_mask)

        for name, pred in [("original", pred_orig), ("blackbox", pred_box), ("blackmask", pred_mask)]:
            results[name]["n"] += 1
            if pred is None:
                results[name]["n_parse_fail"] += 1
            elif pred == gt:
                results[name]["n_ok"] += 1

        per_sample.append({
            "idx": i, "id": ex_orig["id"], "entity": ex_orig.get("entity"),
            "question": ex_orig["question"], "choices": ex_orig["choices"], "gt_idx": gt,
            "out_orig": out_orig, "pred_orig": pred_orig,
            "out_box": out_box,   "pred_box": pred_box,
            "out_mask": out_mask, "pred_mask": pred_mask,
        })

        if printed < debug_print_k:
            printed += 1
            gt_letter = chr(ord("A") + gt)
            fmt = lambda p: None if p is None else chr(ord("A") + p)
            print(f"\n--- DEBUG [{i}] id={ex_orig['id']} entity={ex_orig.get('entity')} GT={gt_letter} ---")
            print("original:", out_orig, "->", fmt(pred_orig))
            print("blackbox:", out_box,  "->", fmt(pred_box))
            print("blackmask:", out_mask, "->", fmt(pred_mask))

    final_results = {}
    for name, stats in results.items():
        n = max(stats["n"], 1)
        final_results[name] = {
            "n": stats["n"],
            "acc": stats["n_ok"] / n,
            "parse_fail": stats["n_parse_fail"] / n,
        }

    return final_results, per_sample


def print_results(results):
    print("\n" + "="*60)
    print("ENTITY MASKING RESULTS")
    print("="*60)

    for name in ["original", "blackbox", "blackmask"]:
        m = results[name]
        print(f"\n[{name}] n={m['n']}")
        print(f"  acc        : {m['acc']:.4f}")
        print(f"  parse_fail : {m['parse_fail']:.4f}")

    if results["original"]["acc"] > 0:
        drop_box  = results["original"]["acc"] - results["blackbox"]["acc"]
        drop_mask = results["original"]["acc"] - results["blackmask"]["acc"]
        print(f"\n[Performance Drop]")
        print(f"  BlackBox:  {drop_box:.4f} ({drop_box / results['original']['acc'] * 100:.1f}%)")
        print(f"  BlackMask: {drop_mask:.4f} ({drop_mask / results['original']['acc'] * 100:.1f}%)")


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default="Qwen/Qwen3-8B",
                    help="Path to Qwen3-VL model (local or HuggingFace hub)")
    ap.add_argument("--pack_root", type=str, required=True,
                    help="Root directory of aokvqa_pack (contains jsonl files and images)")
    ap.add_argument("--original_jsonl", type=str, default="validation_with_entity.jsonl")
    ap.add_argument("--blackbox_jsonl", type=str, default="validation_blackbox.jsonl")
    ap.add_argument("--blackmask_jsonl", type=str, default="validation_blackmask.jsonl")
    ap.add_argument("--n_eval", type=int, default=None,
                    help="Number of samples to evaluate (default: all)")
    ap.add_argument("--max_new_tokens", type=int, default=6)
    ap.add_argument("--debug_print_k", type=int, default=3)
    ap.add_argument("--output_json", type=str, default="entity_eval_aokvqa_qwen.json")
    args = ap.parse_args()

    model, processor = load_model(args.model_path)

    print("\nLoading datasets...")
    dataset_original = load_jsonl_dataset(os.path.join(args.pack_root, args.original_jsonl), args.pack_root)
    dataset_blackbox  = load_jsonl_dataset(os.path.join(args.pack_root, args.blackbox_jsonl),  args.pack_root)
    dataset_blackmask = load_jsonl_dataset(os.path.join(args.pack_root, args.blackmask_jsonl), args.pack_root)
    print(f"Loaded {len(dataset_original)} original, {len(dataset_blackbox)} blackbox, {len(dataset_blackmask)} blackmask")

    results, per_sample = eval_entity_masking(
        dataset_original, dataset_blackbox, dataset_blackmask,
        model, processor,
        n_eval=args.n_eval,
        max_new_tokens=args.max_new_tokens,
        debug_print_k=args.debug_print_k,
    )

    print_results(results)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {args.output_json}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MME entity masking evaluation script for Qwen3-VL.
Compares yes-rate across: original, black_mask (SAM2), and black_box (bbox).
"""

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


# ============================================================
# yes/no parsing
# ============================================================
_YN_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)

def parse_yes_no(text: str) -> Optional[str]:
    """Strict parsing: if both yes and no appear, return None."""
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
# Qwen3-VL batch inference
# ============================================================
@torch.inference_mode()
def infer_qwen3vl_batch(model, processor, image_paths, questions, max_new_tokens=8):
    assert len(image_paths) == len(questions)

    conversations = []
    for img_p, q in zip(image_paths, questions):
        conversations.append([{
            "role": "user",
            "content": [
                {"type": "image", "image": img_p},
                {"type": "text", "text": q},
            ],
        }])

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

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # Trim prompt tokens
    input_ids = inputs["input_ids"]
    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)


# ============================================================
# Eval: compare yes-rate across conditions
# ============================================================
def eval_yesrate_on_condition(model, processor, items, base_dir: Path, path_key: str,
                               batch_size=16, max_new_tokens=8):
    """
    path_key: 'original_path' | 'black_box_path' | 'black_mask_path'
    All GT labels are 'yes', so yes_rate == accuracy == recall.
    """
    ran = 0
    missing_img = 0
    parsed = 0
    yes_cnt = 0

    for s in tqdm(range(0, len(items), batch_size), desc=f"Eval {path_key}"):
        batch = items[s:s+batch_size]

        img_paths = []
        qs = []
        for ex in batch:
            img_full_path = base_dir / ex[path_key]
            if not img_full_path.exists():
                missing_img += 1
                continue
            img_paths.append(str(img_full_path))
            qs.append(ex["question"])

        if not img_paths:
            continue

        outs = infer_qwen3vl_batch(model, processor, img_paths, qs, max_new_tokens=max_new_tokens)
        preds = [parse_yes_no(t) for t in outs]

        for pred in preds:
            ran += 1
            if pred in ("yes", "no"):
                parsed += 1
                if pred == "yes":
                    yes_cnt += 1

    yes_rate = yes_cnt / max(parsed, 1)
    return {
        "ran": ran,
        "missing_img": missing_img,
        "parsed": parsed,
        "parse_fail": ran - parsed,
        "parse_fail_rate": (ran - parsed) / max(ran, 1),
        "yes_cnt": yes_cnt,
        "yes_rate": yes_rate,
    }

def pretty_print(tag, r):
    print(f"\n[{tag}]")
    print(f"  ran        : {r['ran']} (missing_img={r['missing_img']})")
    print(f"  parsed     : {r['parsed']} (parse_fail={r['parse_fail']}, rate={r['parse_fail_rate']:.4f})")
    if r["parsed"] > 0:
        print(f"  YES count  : {r['yes_cnt']} / {r['parsed']}")
        print(f"  YES rate   : {r['yes_rate']:.4f}   (== Acc == Recall, since GT all yes)")


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
    ap.add_argument("--data_json", type=str, required=True,
                    help="Path to metadata_entity_masked.json")
    ap.add_argument("--base_dir", type=str, required=True,
                    help="Base directory for resolving image paths in the JSON")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--output_json", type=str, default="entity_eval_mme_qwen.json")
    args = ap.parse_args()

    model, processor = load_model(args.model_path)

    print("\n" + "="*80)
    print("Loading test data...")
    print("="*80)

    with open(args.data_json, "r", encoding="utf-8") as f:
        items = json.load(f)

    print(f"Loaded {len(items)} samples")

    bad = [ex for ex in items if str(ex.get("answer", "")).strip().lower() != "yes"]
    if bad:
        print(f"WARNING: found {len(bad)} non-yes labels. First: {bad[0]}")

    print("\n" + "="*80)
    print("Starting evaluation...")
    print("="*80)

    base_dir = Path(args.base_dir)
    r_orig = eval_yesrate_on_condition(model, processor, items, base_dir, "original_path",
                                       batch_size=args.batch_size, max_new_tokens=args.max_new_tokens)
    r_mask = eval_yesrate_on_condition(model, processor, items, base_dir, "black_mask_path",
                                       batch_size=args.batch_size, max_new_tokens=args.max_new_tokens)
    r_box  = eval_yesrate_on_condition(model, processor, items, base_dir, "black_box_path",
                                       batch_size=args.batch_size, max_new_tokens=args.max_new_tokens)

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    pretty_print("Original", r_orig)
    pretty_print("Black Mask", r_mask)
    pretty_print("Black Box", r_box)

    print("\n=== Delta YES rate (occluded - original) ===")
    print(f"  black_mask - original: {r_mask['yes_rate'] - r_orig['yes_rate']:.4f}")
    print(f"  black_box  - original: {r_box['yes_rate'] - r_orig['yes_rate']:.4f}")

    results = {
        "original": r_orig,
        "black_mask": r_mask,
        "black_box": r_box,
        "delta": {
            "black_mask_minus_original": r_mask['yes_rate'] - r_orig['yes_rate'],
            "black_box_minus_original": r_box['yes_rate'] - r_orig['yes_rate'],
        }
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {args.output_json}")


if __name__ == "__main__":
    main()

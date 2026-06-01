#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
InternVL3 evaluation for entity masking experiments.
Compares yes-rate across: original images, blackmask (SAM2), and blackbox (bbox).
"""

import argparse
import json
import re
from pathlib import Path

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm


# ============================================================
# InternVL3 图像预处理
# ============================================================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image_from_path(image_file: str, input_size=448, max_num=6):
    from PIL import Image
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


# ============================================================
# yes/no parsing
# ============================================================
_YN_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)

def normalize_yn(text: str):
    if text is None:
        return None
    m = _YN_RE.search(text.strip())
    return m.group(1).lower() if m else None


# ============================================================
# InternVL3 batch inference
# ============================================================
@torch.inference_mode()
def infer_internvl_batch(model, tokenizer, image_paths, questions, max_new_tokens=8, max_num=6):
    assert len(image_paths) == len(questions)

    pixel_values_list = []
    num_patches_list = []

    for img_path in image_paths:
        pv = load_image_from_path(img_path, max_num=max_num).to(torch.bfloat16).cuda()
        pixel_values_list.append(pv)
        num_patches_list.append(pv.shape[0])

    pixel_values = torch.cat(pixel_values_list, dim=0)

    responses = model.batch_chat(
        tokenizer,
        pixel_values,
        questions,
        generation_config=dict(max_new_tokens=max_new_tokens, do_sample=False),
        num_patches_list=num_patches_list
    )

    return responses


# ============================================================
# Eval: all GT are "yes" => YesRate = Acc = Recall
# ============================================================
def eval_yesrate_on_folder(model, tokenizer, items, img_dir: Path, batch_size=16, max_new_tokens=8):
    ran = 0
    missing_img = 0
    parsed = 0
    yes_cnt = 0

    for s in tqdm(range(0, len(items), batch_size), desc=f"Evaluating {img_dir.name}"):
        batch = items[s:s+batch_size]

        img_paths = []
        qs = []
        for ex in batch:
            p = img_dir / ex["image_id"]
            if not p.exists():
                missing_img += 1
                continue
            img_paths.append(str(p))
            qs.append(ex["question"])

        if not img_paths:
            continue

        try:
            outs = infer_internvl_batch(model, tokenizer, img_paths, qs, max_new_tokens=max_new_tokens)
            preds = [normalize_yn(t) for t in outs]

            for pred in preds:
                ran += 1
                if pred in ("yes", "no"):
                    parsed += 1
                    if pred == "yes":
                        yes_cnt += 1

        except Exception as e:
            print(f"\n⚠️ Batch error: {e}")
            for img_path, q in zip(img_paths, qs):
                try:
                    out = infer_internvl_batch(model, tokenizer, [img_path], [q], max_new_tokens=max_new_tokens)[0]
                    pred = normalize_yn(out)
                    ran += 1
                    if pred in ("yes", "no"):
                        parsed += 1
                        if pred == "yes":
                            yes_cnt += 1
                except:
                    ran += 1

        if s % 100 == 0:
            torch.cuda.empty_cache()

    yes_rate = yes_cnt / max(parsed, 1)
    return {
        "ran": ran,
        "missing_img": missing_img,
        "parsed": parsed,
        "parse_fail": ran - parsed,
        "yes_cnt": yes_cnt,
        "yes_rate": yes_rate,
    }

def pretty_print(tag, r):
    print(f"\n[{tag}]")
    print(f"  ran        : {r['ran']} (missing_img={r['missing_img']})")
    print(f"  parsed     : {r['parsed']} (parse_fail={r['parse_fail']})")
    if r["parsed"] > 0:
        print(f"  YES count  : {r['yes_cnt']} / {r['parsed']}")
        print(f"  YES rate   : {r['yes_rate']:.4f}   (== Acc == Recall, since GT all yes)")


# ============================================================
# Model loading
# ============================================================
def load_model(model_path: str):
    print("="*80)
    print("Loading InternVL3 for Black Box/Mask Testing")
    print("="*80)

    print("\n[1/2] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False
    )

    print("[2/2] Loading model...")
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map='auto'
    ).eval()

    print("Model loaded successfully!")
    print("="*80)

    return model, tokenizer


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default="OpenGVLab/InternVL3-8B",
                    help="Path to InternVL3 model (local or HuggingFace hub)")
    ap.add_argument("--data_json", type=str, required=True,
                    help="Path to data2.json (output of extract_entity.py, all GT=yes)")
    ap.add_argument("--orig_dir", type=str, required=True,
                    help="Directory with original images")
    ap.add_argument("--blackmask_dir", type=str, required=True,
                    help="Directory with SAM2 masked images")
    ap.add_argument("--blackbox_dir", type=str, required=True,
                    help="Directory with bbox blacked-out images")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--output_json", type=str, default="entity_eval_results.json",
                    help="Output JSON file for results")
    args = ap.parse_args()

    model, tokenizer = load_model(args.model_path)

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

    r_orig = eval_yesrate_on_folder(model, tokenizer, items, Path(args.orig_dir),
                                    batch_size=args.batch_size, max_new_tokens=args.max_new_tokens)
    r_mask = eval_yesrate_on_folder(model, tokenizer, items, Path(args.blackmask_dir),
                                    batch_size=args.batch_size, max_new_tokens=args.max_new_tokens)
    r_box  = eval_yesrate_on_folder(model, tokenizer, items, Path(args.blackbox_dir),
                                    batch_size=args.batch_size, max_new_tokens=args.max_new_tokens)

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    pretty_print("original", r_orig)
    pretty_print("blackmask", r_mask)
    pretty_print("blackbox", r_box)

    print("\n=== Delta YES rate (occluded - original) ===")
    print(f"  blackmask - original: {r_mask['yes_rate'] - r_orig['yes_rate']:.4f}")
    print(f"  blackbox  - original: {r_box['yes_rate'] - r_orig['yes_rate']:.4f}")

    results = {
        "original": r_orig,
        "blackmask": r_mask,
        "blackbox": r_box,
        "delta": {
            "blackmask_minus_original": r_mask['yes_rate'] - r_orig['yes_rate'],
            "blackbox_minus_original": r_box['yes_rate'] - r_orig['yes_rate'],
        }
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {args.output_json}")


if __name__ == "__main__":
    main()

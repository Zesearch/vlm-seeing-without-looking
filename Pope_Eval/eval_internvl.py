#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
InternVL3 evaluation script for H200 cluster
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
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


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

def load_image_from_pil(image: Image.Image, input_size=448, max_num=6):
    """从PIL Image加载（用于batch处理）"""
    image = image.convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


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
# 4) InternVL3 Batch Inference
# ============================================================
@torch.inference_mode()
def infer_internvl_batch(model, tokenizer, images_pil, questions, max_new_tokens=8, max_num=6):
    """
    Batch inference for InternVL3
    images_pil: list[PIL.Image]
    questions: list[str]
    returns: list[str]
    """
    pixel_values_list = []
    num_patches_list = []
    
    for img in images_pil:
        pv = load_image_from_pil(img, max_num=max_num).to(torch.bfloat16).cuda()
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


@torch.inference_mode()
def infer_with_image_batch(model, tokenizer, images, questions, max_new_tokens=8):
    return infer_internvl_batch(model, tokenizer, images, questions, max_new_tokens=max_new_tokens)


@torch.inference_mode()
def infer_no_image_batch(model, tokenizer, questions, max_new_tokens=8):
    """
    Use a dummy black image as "no_image" proxy.
    """
    dummy = Image.new("RGB", (224, 224), (0, 0, 0))
    images = [dummy] * len(questions)
    return infer_internvl_batch(model, tokenizer, images, questions, max_new_tokens=max_new_tokens)


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
    dataset, model, tokenizer,
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

        try:
            pred_normal = [normalize_yn(t) for t in infer_with_image_batch(
                model, tokenizer, imgs, qs, max_new_tokens=max_new_tokens
            )]
            pred_swap = [normalize_yn(t) for t in infer_with_image_batch(
                model, tokenizer, swap_imgs, qs, max_new_tokens=max_new_tokens
            )]
            pred_noimg = [normalize_yn(t) for t in infer_no_image_batch(
                model, tokenizer, qs, max_new_tokens=max_new_tokens
            )]

            pred_black, pred_noise = {}, {}
            for p in ps:
                bimgs = [structured_black_occlusion(img, p) for img in imgs]
                pred_black[p] = [normalize_yn(t) for t in infer_with_image_batch(
                    model, tokenizer, bimgs, qs, max_new_tokens=max_new_tokens
                )]

                nimgs = [mix_with_noise(img, p, seed=seed + start * 100 + k) for k, img in enumerate(imgs)]
                pred_noise[p] = [normalize_yn(t) for t in infer_with_image_batch(
                    model, tokenizer, nimgs, qs, max_new_tokens=max_new_tokens
                )]

        except Exception as e:
            print(f"\n⚠️ Batch error at {start}: {e}")
            pred_normal = []
            pred_swap = []
            pred_noimg = []
            pred_black = {p: [] for p in ps}
            pred_noise = {p: [] for p in ps}
            
            for k in range(len(batch)):
                q = [qs[k]]
                try:
                    pred_normal.append(normalize_yn(infer_with_image_batch(model, tokenizer, [imgs[k]], q, max_new_tokens)[0]))
                    pred_swap.append(normalize_yn(infer_with_image_batch(model, tokenizer, [swap_imgs[k]], q, max_new_tokens)[0]))
                    pred_noimg.append(normalize_yn(infer_no_image_batch(model, tokenizer, q, max_new_tokens)[0]))
                    
                    for p in ps:
                        bimg = structured_black_occlusion(imgs[k], p)
                        pred_black[p].append(normalize_yn(infer_with_image_batch(model, tokenizer, [bimg], q, max_new_tokens)[0]))
                        
                        nimg = mix_with_noise(imgs[k], p, seed=seed + start * 100 + k)
                        pred_noise[p].append(normalize_yn(infer_with_image_batch(model, tokenizer, [nimg], q, max_new_tokens)[0]))
                except:
                    pred_normal.append(None)
                    pred_swap.append(None)
                    pred_noimg.append(None)
                    for p in ps:
                        pred_black[p].append(None)
                        pred_noise[p].append(None)

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

        if start % 50 == 0:
            torch.cuda.empty_cache()

    results = {k: metrics_from_confusion(v) for k, v in cms.items()}
    return results


# ============================================================
# 8) Model loading
# ============================================================
def load_internvl_local(model_path: str):
    print("="*80)
    print("Loading InternVL3 Model")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default="OpenGVLab/InternVL3-8B",
                    help="Path to InternVL3 model (local or HuggingFace hub)")
    ap.add_argument("--dataset", type=str, default="chadlzx/pope-subset-300",
                    help="HuggingFace dataset name")
    ap.add_argument("--n_eval", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ps", type=float, nargs="+", default=[0.5, 0.75, 1.0])
    ap.add_argument("--output_json", type=str, default="perturbation_results.json",
                    help="Output JSON file for results")
    args = ap.parse_args()

    model, tokenizer = load_internvl_local(args.model_path)
    dataset = load_dataset(args.dataset)
    
    print(f"\nLoaded {len(dataset)} samples from dataset")
    print(f"Evaluating {args.n_eval} samples\n")

    results = eval_all_batch(
        dataset, model, tokenizer,
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

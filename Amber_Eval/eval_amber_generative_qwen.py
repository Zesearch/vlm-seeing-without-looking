#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AMBER generative task evaluation script for Qwen3-VL.
Runs inference across conditions: original, black_0.5, black_0.75, blur_0.5, blur_0.75, no_image.
Saves per-condition output JSONs.
"""

import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


# ============================================================
# Inference
# ============================================================
def infer_batch(model, processor, image_paths, prompts, max_new_tokens=128):
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

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def infer_no_image_batch(model, processor, prompts, max_new_tokens=128):
    messages_batch = [[{"role": "user", "content": [{"type": "text", "text": p}]}] for p in prompts]

    inputs = processor.apply_chat_template(
        messages_batch, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt", padding=True,
    ).to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)


# ============================================================
# Run one condition
# ============================================================
def run_inference(condition, model, processor, queries, image_base, batch_size=25):
    results = []
    img_dir = None if condition == "no_image" else Path(image_base) / condition

    for start in tqdm(range(0, len(queries), batch_size), desc=f"Inferring {condition}"):
        end = min(start + batch_size, len(queries))
        batch = queries[start:end]
        prompts = [item["query"] for item in batch]

        if condition == "no_image":
            outputs = infer_no_image_batch(model, processor, prompts, max_new_tokens=128)
        else:
            img_paths = [str(img_dir / item["image"]) for item in batch]
            outputs = infer_batch(model, processor, img_paths, prompts, max_new_tokens=128)

        for item, output in zip(batch, outputs):
            results.append({"id": item["id"], "response": output})

        torch.cuda.empty_cache()

    return results


# ============================================================
# Model loading
# ============================================================
def load_model(model_path):
    print("="*80)
    print("Loading Qwen3-VL Model")
    print("="*80)

    print("[1/2] Loading processor...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

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
    ap.add_argument("--amber_dir", type=str, required=True,
                    help="Root AMBER dataset directory (contains data/ and images_generative/)")
    ap.add_argument("--output_dir", type=str, default=None,
                    help="Output directory for results (default: <amber_dir>/outputs_generative)")
    ap.add_argument("--batch_size", type=int, default=5)
    ap.add_argument("--conditions", type=str, nargs="+",
                    default=["original", "black_0.5", "black_0.75", "blur_0.5", "blur_0.75", "no_image"])
    args = ap.parse_args()

    amber_dir = Path(args.amber_dir)
    image_base = amber_dir / "images_generative"
    query_path = amber_dir / "data/query/query_generative.json"
    output_dir = Path(args.output_dir) if args.output_dir else amber_dir / "outputs_generative"
    output_dir.mkdir(exist_ok=True)

    model, processor = load_model(args.model_path)

    queries = json.load(open(query_path))
    print(f"\nLoaded {len(queries)} queries")

    for condition in args.conditions:
        print(f"\n{'='*60}")
        print(f"Processing: {condition}")
        print(f"{'='*60}")

        results = run_inference(condition, model, processor, queries, image_base, batch_size=args.batch_size)

        output_file = output_dir / f"outputs_{condition}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()

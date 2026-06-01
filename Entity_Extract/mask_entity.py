#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 2: Use Grounding DINO + SAM2 to detect and mask entities in images.
Saves masked images (entity region blacked out) to output directory.
"""

import os
import json
import argparse
import torch
import numpy as np
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    Sam2Processor,
    Sam2Model,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="data2.json",
                    help="Input JSON file (output of extract_entity.py)")
    ap.add_argument("--image_dir", type=str, default="images",
                    help="Directory containing source images")
    ap.add_argument("--output_dir", type=str, default="blackbox",
                    help="Directory to save masked images")
    ap.add_argument("--gdino_model", type=str, default="IDEA-Research/grounding-dino-base",
                    help="Grounding DINO model (local path or HuggingFace hub)")
    ap.add_argument("--sam2_model", type=str, default="facebook/sam2.1-hiera-tiny",
                    help="SAM2 model (local path or HuggingFace hub)")
    ap.add_argument("--det_threshold", type=float, default=0.3,
                    help="Grounding DINO detection threshold")
    ap.add_argument("--text_threshold", type=float, default=0.25,
                    help="Grounding DINO text threshold")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print("加载 Grounding DINO...")
    processor_gdino = AutoProcessor.from_pretrained(args.gdino_model)
    model_gdino = AutoModelForZeroShotObjectDetection.from_pretrained(
        args.gdino_model
    ).to(device)

    print("加载 SAM2...")
    processor_sam2 = Sam2Processor.from_pretrained(args.sam2_model)
    model_sam2 = Sam2Model.from_pretrained(args.sam2_model).to(device)

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        image_id = item['image_id']
        entity = item['entity']

        print(f"\n处理: {image_id}, Entity: {entity}")

        image_path = os.path.join(args.image_dir, image_id)
        image = Image.open(image_path).convert("RGB")

        # Step 1: Grounding DINO 检测entity
        text = f"a {entity.lower()}."
        inputs_gdino = processor_gdino(images=image, text=text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs_gdino = model_gdino(**inputs_gdino)

        results = processor_gdino.post_process_grounded_object_detection(
            outputs_gdino,
            inputs_gdino.input_ids,
            threshold=args.det_threshold,
            text_threshold=args.text_threshold,
            target_sizes=[image.size[::-1]]
        )[0]

        output_path = os.path.join(args.output_dir, image_id)

        if len(results['boxes']) == 0:
            print(f"  ⚠ 未检测到 {entity}，保存原图")
            image.save(output_path)
            continue

        best_idx = results['scores'].argmax()
        bbox = results['boxes'][best_idx].cpu().numpy()
        print(f"  ✓ 检测到bbox: {bbox}, score: {results['scores'][best_idx]:.3f}")

        # Step 2: SAM2 生成mask
        input_boxes = [[bbox.tolist()]]
        inputs_sam2 = processor_sam2(
            images=image,
            input_boxes=input_boxes,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs_sam2 = model_sam2(**inputs_sam2)

        masks = processor_sam2.post_process_masks(
            outputs_sam2.pred_masks.cpu(),
            inputs_sam2["original_sizes"]
        )[0]

        print(f"  ✓ 生成mask, shape: {masks.shape}")

        mask = masks[0, 0].numpy() > 0

        # Step 3: 涂黑
        image_array = np.array(image)
        image_array[mask] = 0
        masked_image = Image.fromarray(image_array)
        masked_image.save(output_path)
        print(f"  ✓ 保存到: {output_path}")

    print(f"\n✓ 完成！检查 {args.output_dir}/ 文件夹")


if __name__ == "__main__":
    main()

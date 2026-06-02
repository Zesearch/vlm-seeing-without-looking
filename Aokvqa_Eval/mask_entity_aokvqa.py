#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A-OKVQA entity masking script using Grounding DINO + SAM2.
Reads validation_with_entity.jsonl and generates:
  - validation_blackbox.jsonl  (bbox blacked out)
  - validation_blackmask.jsonl (SAM2 mask blacked out)
"""

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    Sam2Model,
    Sam2Processor,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", type=str, required=True,
                    help="Input JSONL file (output of extract_entity_aokvqa.py)")
    ap.add_argument("--pack_root", type=str, required=True,
                    help="Root directory of aokvqa_pack (images and jsonl files live here)")
    ap.add_argument("--output_jsonl_box", type=str, default="validation_blackbox.jsonl",
                    help="Output JSONL for blackbox images (relative to pack_root)")
    ap.add_argument("--output_jsonl_mask", type=str, default="validation_blackmask.jsonl",
                    help="Output JSONL for blackmask images (relative to pack_root)")
    ap.add_argument("--gdino_model", type=str, default="IDEA-Research/grounding-dino-base",
                    help="Grounding DINO model (local path or HuggingFace hub)")
    ap.add_argument("--sam2_model", type=str, default="facebook/sam2.1-hiera-tiny",
                    help="SAM2 model (local path or HuggingFace hub)")
    ap.add_argument("--det_threshold", type=float, default=0.3)
    ap.add_argument("--text_threshold", type=float, default=0.25)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pack_root = args.pack_root
    blackbox_dir = os.path.join(pack_root, "images", "blackbox")
    blackmask_dir = os.path.join(pack_root, "images", "blackmask")
    os.makedirs(blackbox_dir, exist_ok=True)
    os.makedirs(blackmask_dir, exist_ok=True)

    output_jsonl_box  = os.path.join(pack_root, args.output_jsonl_box)
    output_jsonl_mask = os.path.join(pack_root, args.output_jsonl_mask)

    print("Loading Grounding DINO...")
    processor_gdino = AutoProcessor.from_pretrained(args.gdino_model)
    model_gdino = AutoModelForZeroShotObjectDetection.from_pretrained(args.gdino_model).to(device)

    print("Loading SAM2...")
    processor_sam2 = Sam2Processor.from_pretrained(args.sam2_model)
    model_sam2 = Sam2Model.from_pretrained(args.sam2_model).to(device)

    data = []
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    total = len(data)
    processed_box = 0
    processed_mask = 0
    skipped = 0

    for item in tqdm(data, desc="Processing"):
        entity = item.get('entity')

        # No entity: write original path to both output jsonls
        if not entity:
            skipped += 1
            with open(output_jsonl_box, "a", encoding="utf-8") as f:
                f.write(json.dumps(item.copy(), ensure_ascii=False) + '\n')
            with open(output_jsonl_mask, "a", encoding="utf-8") as f:
                f.write(json.dumps(item.copy(), ensure_ascii=False) + '\n')
            continue

        img_id = item['id']
        image = Image.open(item['imagefile']).convert("RGB")

        # Grounding DINO detection
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

        # No detection: write original path to both output jsonls
        if len(results['boxes']) == 0:
            with open(output_jsonl_box, "a", encoding="utf-8") as f:
                f.write(json.dumps(item.copy(), ensure_ascii=False) + '\n')
            with open(output_jsonl_mask, "a", encoding="utf-8") as f:
                f.write(json.dumps(item.copy(), ensure_ascii=False) + '\n')
            continue

        best_idx = results['scores'].argmax()
        bbox = results['boxes'][best_idx].cpu().numpy()

        # BlackBox: draw filled black rectangle over bbox
        image_box = image.copy()
        draw = ImageDraw.Draw(image_box)
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], fill="black")

        box_filename = os.path.join("images", "blackbox", f"{img_id}.jpg")
        image_box.save(os.path.join(pack_root, box_filename))
        processed_box += 1

        item_box = item.copy()
        item_box['imagefile'] = box_filename
        with open(output_jsonl_box, "a", encoding="utf-8") as f:
            f.write(json.dumps(item_box, ensure_ascii=False) + '\n')

        # BlackMask: use SAM2 to generate precise mask and black it out
        inputs_sam2 = processor_sam2(
            images=image,
            input_boxes=[[bbox.tolist()]],
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs_sam2 = model_sam2(**inputs_sam2)

        masks = processor_sam2.post_process_masks(
            outputs_sam2.pred_masks.cpu(),
            inputs_sam2["original_sizes"]
        )[0]

        mask = masks[0, 0].numpy() > 0
        image_array = np.array(image)
        image_array[mask] = 0

        mask_filename = os.path.join("images", "blackmask", f"{img_id}.jpg")
        Image.fromarray(image_array).save(os.path.join(pack_root, mask_filename))
        processed_mask += 1

        item_mask = item.copy()
        item_mask['imagefile'] = mask_filename
        with open(output_jsonl_mask, "a", encoding="utf-8") as f:
            f.write(json.dumps(item_mask, ensure_ascii=False) + '\n')

    print(f"\nDone!")
    print(f"  Total          : {total}")
    print(f"  BlackBox saved : {processed_box}")
    print(f"  BlackMask saved: {processed_mask}")
    print(f"  Skipped        : {skipped}")
    print(f"\nOutput files:")
    print(f"  {output_jsonl_box}")
    print(f"  {output_jsonl_mask}")


if __name__ == "__main__":
    main()

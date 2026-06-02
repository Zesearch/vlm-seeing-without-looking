#!/usr/bin/env python3
"""
Split AMBER generative outputs into 5 random subsets and evaluate each.
Computes mean/std across splits to assess stability.
"""
import json
import random
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict, Any
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--amber_dir", type=str, required=True,
                        help="Root AMBER dataset directory (contains data/, images_generative/)")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Input JSON file path relative to amber_dir (e.g., outputs_generative_molmo/outputs_original.json)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results relative to amber_dir (e.g., evaluation_results_molmo/split_analysis)")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., molmo)")
    parser.add_argument("--condition", type=str, required=True,
                        help="Condition name (e.g., original, black_0.5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of splits")
    parser.add_argument("--samples_per_split", type=int, default=200, help="Samples per split")
    return parser.parse_args()


def load_outputs(filepath: str) -> List[Dict[str, Any]]:
    with open(filepath, 'r') as f:
        return json.load(f)


def split_dataset(data: List[Dict], n_splits: int, samples_per_split: int, seed: int) -> List[List[Dict]]:
    random.seed(seed)
    indices = list(range(len(data)))
    random.shuffle(indices)

    splits = []
    for i in range(n_splits):
        start_idx = i * samples_per_split
        end_idx = start_idx + samples_per_split
        split_indices = indices[start_idx:end_idx]
        split_data = [data[idx] for idx in sorted(split_indices)]
        splits.append(split_data)

    return splits


def save_split(split_data: List[Dict], output_path: str):
    with open(output_path, 'w') as f:
        json.dump(split_data, f, indent=2)


def run_evaluation(inference_file: str, base_dir: Path) -> Dict[str, float]:
    cmd = [
        "python", "inference.py",
        "--inference_data", inference_file,
        "--annotation", str(base_dir / "data/annotations.json"),
        "--word_association", str(base_dir / "data/relation.json"),
        "--safe_words", str(base_dir / "data/safe_words.txt"),
        "--metrics", str(base_dir / "data/metrics.txt"),
        "--evaluation_type", "g"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(base_dir))

    metrics = {}
    for line in result.stdout.split('\n'):
        if 'CHAIR:' in line:
            metrics['CHAIR'] = float(line.split()[-1])
        elif 'Cover:' in line:
            metrics['Cover'] = float(line.split()[-1])
        elif 'Hal:' in line:
            metrics['Hal'] = float(line.split()[-1])
        elif 'Cog:' in line:
            metrics['Cog'] = float(line.split()[-1])

    return metrics


def compute_statistics(all_metrics: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    stats = {}
    for metric in ['CHAIR', 'Cover', 'Hal', 'Cog']:
        values = [m[metric] for m in all_metrics]
        stats[metric] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'values': values
        }
    return stats


def main():
    args = get_args()

    base_dir = Path(args.amber_dir)
    input_file = base_dir / args.input_file
    output_dir = base_dir / args.output_dir
    split_dir = base_dir / f"splits_{args.model}" / args.condition

    split_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"========================================")
    print(f"Model: {args.model}")
    print(f"Condition: {args.condition}")
    print(f"Random seed: {args.seed}")
    print(f"========================================\n")

    print(f"Loading: {input_file}")
    data = load_outputs(str(input_file))
    print(f"Total samples: {len(data)}\n")

    print(f"Creating {args.n_splits} splits of {args.samples_per_split} samples each...")
    splits = split_dataset(data, args.n_splits, args.samples_per_split, args.seed)

    all_metrics = []
    for i, split_data in enumerate(splits, 1):
        split_file = split_dir / f"split_{i}.json"
        save_split(split_data, str(split_file))
        print(f"\n[Split {i}/{args.n_splits}] Evaluating {len(split_data)} samples...")

        metrics = run_evaluation(str(split_file), base_dir)
        all_metrics.append(metrics)

        print(f"  CHAIR: {metrics['CHAIR']:.1f}")
        print(f"  Cover: {metrics['Cover']:.1f}")
        print(f"  Hal:   {metrics['Hal']:.1f}")
        print(f"  Cog:   {metrics['Cog']:.1f}")

    print(f"\n========================================")
    print(f"Statistics across {args.n_splits} splits:")
    print(f"========================================")

    stats = compute_statistics(all_metrics)

    for metric in ['CHAIR', 'Cover', 'Hal', 'Cog']:
        s = stats[metric]
        print(f"\n{metric}:")
        print(f"  Mean: {s['mean']:.2f}")
        print(f"  Std:  {s['std']:.2f}")
        print(f"  Min:  {s['min']:.1f}")
        print(f"  Max:  {s['max']:.1f}")
        print(f"  Values: {[f'{v:.1f}' for v in s['values']]}")

    output_file = output_dir / f"{args.condition}_split_stats.json"
    results = {
        'model': args.model,
        'condition': args.condition,
        'n_splits': args.n_splits,
        'samples_per_split': args.samples_per_split,
        'seed': args.seed,
        'statistics': stats,
        'split_metrics': all_metrics
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: {output_file}")


if __name__ == "__main__":
    main()

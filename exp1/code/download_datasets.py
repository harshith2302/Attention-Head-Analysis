#!/usr/bin/env python3
"""
Download ALL 14 evaluation datasets used in the MoH paper (Table 4).
Run on LOGIN NODE (internet required). Total download: ~5-8 GB.

Usage:
    python download_datasets.py --output_dir ./datasets
    python download_datasets.py --output_dir ./datasets --only sciq piqa mmlu
"""

import argparse
import os
import sys
import time


DATASETS = {
    # 0-shot benchmarks
    "sciq": {
        "hf_id": "allenai/sciq",
        "subset": None,
        "description": "1K test science MCQ",
    },
    "piqa": {
        "hf_id": "ybisk/piqa",
        "subset": None,
        "description": "1.8K val physical commonsense",
    },
    "winogrande": {
        "hf_id": "allenai/winogrande",
        "subset": "winogrande_xl",
        "description": "1.3K val coreference",
    },
    "logiqa": {
        "hf_id": "lucasmccabe/logiqa",
        "subset": None,
        "description": "651 test logical reasoning",
    },
    "truthfulqa": {
        "hf_id": "truthfulqa/truthful_qa",
        "subset": "multiple_choice",
        "description": "817 val truthfulness",
    },
    "lambada": {
        "hf_id": "cimec/lambada",
        "subset": None,
        "description": "5.2K test word prediction",
    },
    # Few-shot benchmarks
    "hellaswag": {
        "hf_id": "Rowan/hellaswag",
        "subset": None,
        "description": "10K val commonsense NLI",
    },
    "mmlu": {
        "hf_id": "cais/mmlu",
        "subset": "all",
        "description": "14K test multitask knowledge (57 subjects)",
    },
    "arc_challenge": {
        "hf_id": "allenai/ai2_arc",
        "subset": "ARC-Challenge",
        "description": "1.2K test science reasoning",
    },
    "boolq": {
        "hf_id": "google/boolq",
        "subset": None,
        "description": "3.3K val yes/no QA",
    },
    "gsm8k": {
        "hf_id": "openai/gsm8k",
        "subset": "main",
        "description": "1.3K test grade-school math",
    },
    "natural_questions": {
        "hf_id": "google-research-datasets/nq_open",
        "subset": None,
        "description": "3.6K val open-domain QA",
    },
    # Chinese benchmarks
    "ceval": {
        "hf_id": "ceval/ceval-exam",
        "subset": "all",
        "description": "1.3K val Chinese knowledge (52 subjects)",
    },
    "cmmlu": {
        "hf_id": "haonan-li/cmmlu",
        "subset": "all",
        "description": "11.5K test Chinese multitask",
    },
}


def download_one(name, info, output_dir):
    from datasets import load_dataset

    save_path = os.path.join(output_dir, name)
    if os.path.exists(save_path) and len(os.listdir(save_path)) > 0:
        print(f"  [SKIP] {name} — already at {save_path}")
        return True

    print(f"  [{name}] Downloading {info['hf_id']} ... ", end="", flush=True)
    t0 = time.time()
    try:
        if info["subset"]:
            ds = load_dataset(info["hf_id"], info["subset"], trust_remote_code=True)
        else:
            ds = load_dataset(info["hf_id"], trust_remote_code=True)

        ds.save_to_disk(save_path)
        elapsed = time.time() - t0

        # Count total samples
        total = sum(len(split) for split in ds.values()) if hasattr(ds, "values") else len(ds)
        print(f"OK ({total} samples, {elapsed:.1f}s)")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download MoH evaluation datasets")
    parser.add_argument("--output_dir", type=str, default="./datasets")
    parser.add_argument("--only", type=str, nargs="+", default=None,
                        help="Download only these datasets (default: all)")
    parser.add_argument("--list", action="store_true", help="List datasets and exit")
    args = parser.parse_args()

    if args.list:
        print(f"\n{'Dataset':<25} {'Description'}")
        print("-" * 65)
        for name, info in DATASETS.items():
            print(f"  {name:<23} {info['description']}")
        print(f"\nTotal: {len(DATASETS)} datasets")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    targets = args.only if args.only else list(DATASETS.keys())

    print(f"\nDownloading {len(targets)} datasets to {args.output_dir}/")
    print("=" * 60)

    ok, fail = 0, []
    for name in targets:
        if name not in DATASETS:
            print(f"  [UNKNOWN] {name}")
            continue
        if download_one(name, DATASETS[name], args.output_dir):
            ok += 1
        else:
            fail.append(name)

    print("\n" + "=" * 60)
    print(f"Success: {ok}/{ok + len(fail)}")
    if fail:
        print(f"Failed:  {fail}")
        print("\nFor failed downloads, try:")
        for f in fail:
            info = DATASETS[f]
            cmd = f"  python -c \"from datasets import load_dataset; "
            if info['subset']:
                cmd += f"load_dataset('{info['hf_id']}', '{info['subset']}', trust_remote_code=True)"
            else:
                cmd += f"load_dataset('{info['hf_id']}', trust_remote_code=True)"
            cmd += f".save_to_disk('{args.output_dir}/{f}')\""
            print(cmd)

    # Verify disk usage
    total_size = 0
    for name in targets:
        path = os.path.join(args.output_dir, name)
        if os.path.exists(path):
            for root, dirs, files in os.walk(path):
                for f in files:
                    total_size += os.path.getsize(os.path.join(root, f))
    print(f"\nTotal disk usage: {total_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
    
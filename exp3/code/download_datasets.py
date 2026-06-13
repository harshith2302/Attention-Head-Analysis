#!/usr/bin/env python3
"""
Robust dataset downloader for HF benchmarks.
Handles both snapshot + load_dataset cases.
"""

import os
import argparse
from huggingface_hub import snapshot_download
from datasets import load_dataset


DATASETS = {
    "mmlu": {"type": "load", "path": "cais/mmlu", "subset": "all"},
    "hellaswag": {"type": "load", "path": "Rowan/hellaswag"},
    "arc_challenge": {"type": "load", "path": "allenai/ai2_arc", "subset": "ARC-Challenge"},
    "boolq": {"type": "load", "path": "google/boolq"},
    "piqa": {"type": "load", "path": "ybisk/piqa"},
    "winogrande": {"type": "load", "path": "allenai/winogrande", "subset": "winogrande_xl"},
    "lambada": {"type": "load", "path": "EleutherAI/lambada_openai"},
    "sciq": {"type": "load", "path": "allenai/sciq"},
    "truthfulqa": {"type": "load", "path": "truthfulqa/truthful_qa", "subset": "multiple_choice"},
    "logiqa": {"type": "load", "path": "lucasmccabe/logiqa"},
    "gsm8k": {"type": "load", "path": "openai/gsm8k", "subset": "main"},
    "natural_questions": {"type": "stream", "path": "google-research-datasets/natural_questions"},
}


def download_with_load(name, config, save_path):
    try:
        # Add trust_remote_code=True here
        kwargs = {"path": config["path"], "trust_remote_code": True} 
        if "subset" in config:
            kwargs["name"] = config["subset"]

        print(f"  Loading via datasets API...")
        ds = load_dataset(**kwargs)

        ds.save_to_disk(save_path)
        print(f"  Saved to {save_path}")

    except Exception as e:
        print(f"  Failed (load_dataset): {e}")


def download_with_stream(name, config, save_path):
    try:
        print(f"  Using streaming mode (partial save)...")
        ds = load_dataset(config["path"], streaming=True)

        # Save only small subset to disk
        samples = []
        for i, item in enumerate(ds["train"]):
            samples.append(item)
            if i >= 1000:  # limit
                break

        import json
        with open(os.path.join(save_path, "sample.json"), "w") as f:
            json.dump(samples, f)

        print(f"  Saved partial data to {save_path}")

    except Exception as e:
        print(f"  Failed (stream): {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./datasets")
    parser.add_argument("--datasets", nargs="*", default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    targets = args.datasets if args.datasets else list(DATASETS.keys())

    for name in targets:
        if name not in DATASETS:
            print(f"\nUnknown dataset: {name}, skipping")
            continue

        config = DATASETS[name]
        save_path = os.path.join(args.output_dir, name)
        os.makedirs(save_path, exist_ok=True)

        print(f"\nDownloading {name} ({config['path']})...")

        if config["type"] == "load":
            download_with_load(name, config, save_path)

        elif config["type"] == "stream":
            download_with_stream(name, config, save_path)

        else:
            try:
                snapshot_download(
                    repo_id=config["path"],
                    repo_type="dataset",
                    local_dir=save_path,
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )
                print(f"  Saved to {save_path}")
            except Exception as e:
                print(f"  Failed (snapshot): {e}")

    print("\n=== Download Complete ===")


if __name__ == "__main__":
    main()
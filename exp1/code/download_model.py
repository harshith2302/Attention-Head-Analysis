#!/usr/bin/env python3
"""
Download LLaMA3-8B model from HuggingFace.
Run this on the LOGIN NODE (with internet access).

Usage:
    python download_model.py --model_name meta-llama/Meta-Llama-3-8B --output_dir ./llama3-8b
    
Note: You need HuggingFace access to LLaMA3. Request at:
    https://huggingface.co/meta-llama/Meta-Llama-3-8B
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Download LLaMA3-8B from HuggingFace")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B",
                        help="HuggingFace model ID")
    parser.add_argument("--output_dir", type=str, default="./llama3-8b",
                        help="Local directory to save the model")
    parser.add_argument("--token", type=str, default=None,
                        help="HuggingFace access token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN", None)

    print(f"Downloading {args.model_name} to {args.output_dir}...")
    print("This may take 15-30 minutes depending on connection speed (~16GB).")

    # Method 1: Using huggingface_hub (preferred for large models)
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=args.model_name,
            local_dir=args.output_dir,
            token=token,
            ignore_patterns=["*.msgpack", "*.h5"],  # skip non-PyTorch formats
        )
        print(f"Model downloaded successfully to {args.output_dir}")
        return
    except Exception as e:
        print(f"snapshot_download failed: {e}")
        print("Falling back to transformers AutoModel download...")

    # Method 2: Using transformers (downloads to cache, then we save)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=token)
    tokenizer.save_pretrained(args.output_dir)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        token=token,
        torch_dtype="auto",
    )
    model.save_pretrained(args.output_dir)
    print(f"Model downloaded successfully to {args.output_dir}")


if __name__ == "__main__":
    main()
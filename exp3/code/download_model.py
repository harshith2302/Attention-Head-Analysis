#!/usr/bin/env python3
import os
from huggingface_hub import snapshot_download

# 🔐 PASTE YOUR TOKEN HERE
HF_TOKEN = "HF_TOKEN"

MODEL_ID = "meta-llama/Meta-Llama-3-8B"
OUTPUT_DIR = "./model"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Downloading {MODEL_ID}...")

snapshot_download(
    repo_id=MODEL_ID,
    local_dir=OUTPUT_DIR,
    local_dir_use_symlinks=False,
    resume_download=True,
    token=HF_TOKEN,   # 🔥 THIS IS THE KEY LINE
)

print("✅ Done. Model downloaded.")
"""
Model and tokenizer loading.
"""
import os
import torch
import logging
from typing import Tuple
from .config import ModelConfig

logger = logging.getLogger(__name__)

# ╔═══════════════════════════════════════════════════════════╗
# ║  PASTE YOUR HUGGING FACE TOKEN BELOW                     ║
# ╚═══════════════════════════════════════════════════════════╝
HF_TOKEN = ""
# ═══════════════════════════════════════════════════════════


def load_model_and_tokenizer(config: ModelConfig, device="cuda", offline=False) -> Tuple:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = config.full_model_name
    cache_dir = config.cache_dir
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map.get(config.torch_dtype, torch.bfloat16)

    kwargs = {}
    if offline:
        kwargs["local_files_only"] = True
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
    else:
        kwargs["token"] = HF_TOKEN

    logger.info(f"Loading: {model_name} | dtype={config.torch_dtype} | attn={config.attn_implementation}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True, **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name, cache_dir=cache_dir, torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else device,
        attn_implementation=config.attn_implementation, trust_remote_code=True, **kwargs,
    )
    model.eval()
    logger.info(f"Loaded: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")
    return model, tokenizer

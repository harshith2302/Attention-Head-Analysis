#!/usr/bin/env python3
"""
Main pipeline for attention head importance analysis on LLaMA3-8B.

Extracts query vector L2 norms (as in MoH paper) and gradient-based
importance scores via forward hooks, then aggregates using advanced methods.

Usage:
    python head_importance_analysis.py \
        --model_path ./llama3-8b \
        --datasets_dir ./datasets \
        --output_dir ./results \
        --samples_per_task 0 \
        --batch_size 1 \
        --device cuda
"""

import argparse
import gc
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from aggregation import HeadImportanceAggregator
from dataset_loader import MoHFullEvalDataset, collate_fn, print_dataset_summary


# =============================================================================
# Hook-based Extraction
# =============================================================================

class AttentionHeadHooker:
    """
    Registers forward hooks on all attention layers to capture:
    1. Query states (for L2 norm computation, as in MoH)
    2. Attention output per head (for gradient-based importance)
    3. Attention weight matrices (for attention rollout)
    """

    def __init__(self, model, num_layers: int, num_heads: int, head_dim: int):
        self.model = model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hooks = []

        # Storage for captured tensors (cleared after each prompt)
        self.q_states: Dict[int, torch.Tensor] = {}       # layer -> (batch, seq, heads, dim)
        self.attn_weights: Dict[int, torch.Tensor] = {}   # layer -> (batch, heads, seq, seq)
        self.head_outputs: Dict[int, torch.Tensor] = {}   # layer -> (batch, seq, heads, dim)

    def register_hooks(self):
        """Register forward hooks on all attention layers."""
        for layer_idx in range(self.num_layers):
            layer = self._get_attention_layer(layer_idx)
            if layer is None:
                print(f"  [WARN] Could not find attention layer {layer_idx}")
                continue

            # Hook on q_proj to capture query states
            hook = layer.q_proj.register_forward_hook(
                self._make_q_hook(layer_idx)
            )
            self.hooks.append(hook)

            # Hook on the full attention forward to capture attention weights
            # We use a forward hook on the attention module itself
            hook2 = layer.register_forward_hook(
                self._make_attn_output_hook(layer_idx)
            )
            self.hooks.append(hook2)

    def _get_attention_layer(self, layer_idx: int):
        """Get the attention module for a given layer index."""
        # Standard HuggingFace LLaMA structure
        try:
            return self.model.model.layers[layer_idx].self_attn
        except (AttributeError, IndexError):
            pass
        # Try alternative structures
        try:
            return self.model.layers[layer_idx].self_attn
        except (AttributeError, IndexError):
            pass
        return None

    def _make_q_hook(self, layer_idx: int):
        """Create a hook that captures query projection output."""
        def hook_fn(module, input, output):
            # output shape: (batch, seq_len, num_heads * head_dim)
            self.q_states[layer_idx] = output.detach()
        return hook_fn

    def _make_attn_output_hook(self, layer_idx: int):
        """Create a hook that captures full attention layer output."""
        def hook_fn(module, input, output):
            # output is a tuple: (attn_output, attn_weights, past_kv, ...)
            if isinstance(output, tuple):
                attn_out = output[0]  # (batch, seq, hidden)
                # Reshape to per-head: (batch, seq, num_heads, head_dim)
                bsz, seq_len, _ = attn_out.shape
                self.head_outputs[layer_idx] = attn_out.view(
                    bsz, seq_len, self.num_heads, self.head_dim
                ).detach()

                # Capture attention weights if available
                if len(output) > 1 and output[1] is not None:
                    self.attn_weights[layer_idx] = output[1].detach()
        return hook_fn

    def clear(self):
        """Clear captured tensors."""
        self.q_states.clear()
        self.attn_weights.clear()
        self.head_outputs.clear()

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def compute_q_norms(self) -> Dict[int, np.ndarray]:
        """
        Compute L2 norms of query vectors per head per token (MoH method).

        Returns:
            {layer_idx: array of shape (num_tokens, num_heads)}
        """
        result = {}
        for layer_idx, q_state in self.q_states.items():
            # q_state: (batch, seq_len, num_heads * head_dim)
            bsz, seq_len, _ = q_state.shape
            # Reshape to (batch * seq_len, num_heads, head_dim)
            q = q_state.view(-1, self.num_heads, self.head_dim)
            # L2 norm per head per token
            norms = torch.norm(q.float(), p=2, dim=-1)  # (batch * seq_len, num_heads)
            result[layer_idx] = norms.cpu().numpy()

        return result


# =============================================================================
# Gradient-based importance
# =============================================================================

def compute_gradient_importance(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    hooker: AttentionHeadHooker,
) -> Tuple[Dict[int, np.ndarray], float, float]:
    """
    Compute gradient-based head importance scores.

    Runs a forward+backward pass and measures the gradient norm
    flowing through each head's output projection.

    Returns:
        grad_norms: {layer_idx: array of shape (num_heads,)}
        loss_value: scalar loss
        total_grad_norm: scalar total gradient norm
    """
    model.zero_grad()

    # Enable gradients on head outputs
    head_output_grads = {}

    def make_grad_hook(layer_idx):
        def hook_fn(grad):
            # grad shape: (batch, seq, num_heads, head_dim)
            # Compute per-head gradient norm
            per_head_norm = torch.norm(
                grad.float(), p=2, dim=(0, 1, 3)  # reduce batch, seq, dim
            )  # shape: (num_heads,)
            head_output_grads[layer_idx] = per_head_norm.detach().cpu().numpy()
        return hook_fn

    # Forward pass with gradient tracking
    model.train()  # enable grad computation
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        output_attentions=True,  # try to get attention weights
    )
    loss = outputs.loss

    # Register gradient hooks on head outputs
    grad_hooks = []
    for layer_idx, head_out in hooker.head_outputs.items():
        if head_out.requires_grad:
            h = head_out.register_hook(make_grad_hook(layer_idx))
            grad_hooks.append(h)

    # Backward pass
    if loss is not None and loss.requires_grad:
        loss.backward()

        # Total gradient norm
        total_grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.data.norm(2).item() ** 2
        total_grad_norm = total_grad_norm ** 0.5

        loss_value = loss.item()
    else:
        total_grad_norm = 1.0
        loss_value = 0.0

    # Clean up grad hooks
    for h in grad_hooks:
        h.remove()

    model.eval()
    model.zero_grad()

    return head_output_grads, loss_value, total_grad_norm


# =============================================================================
# Alternative gradient approach (more robust)
# =============================================================================

def compute_gradient_importance_via_params(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    num_layers: int,
    num_heads: int,
    head_dim: int,
) -> Tuple[Dict[int, np.ndarray], float, float]:
    """
    Compute gradient-based head importance by measuring gradient norms
    of the output projection weight matrix, decomposed per head.

    This is more robust than hooking tensor gradients directly.

    The output projection W_O can be decomposed by rows into per-head
    sub-matrices W_O^i, each of shape (head_dim, hidden_size).
    The gradient norm of W_O^i indicates how important head i is.
    """
    model.zero_grad()

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    loss = outputs.loss

    if loss is None or not loss.requires_grad:
        return {}, 0.0, 1.0

    loss.backward()

    loss_value = loss.item()

    grad_norms = {}
    total_grad_norm = 0.0

    for layer_idx in range(num_layers):
        try:
            # Get o_proj gradient
            o_proj = model.model.layers[layer_idx].self_attn.o_proj
            if o_proj.weight.grad is not None:
                # o_proj.weight shape: (hidden_size, hidden_size)
                # = (hidden_size, num_heads * head_dim)
                grad = o_proj.weight.grad.float()
                # Reshape to (hidden_size, num_heads, head_dim)
                grad_reshaped = grad.view(-1, num_heads, head_dim)
                # Per-head gradient norm: norm over (hidden_size, head_dim)
                per_head = torch.norm(grad_reshaped, p=2, dim=(0, 2))
                grad_norms[layer_idx] = per_head.float().cpu().numpy()
                total_grad_norm += per_head.sum().item()

            # Also check q_proj gradients as additional signal
            q_proj = model.model.layers[layer_idx].self_attn.q_proj
            if q_proj.weight.grad is not None:
                q_grad = q_proj.weight.grad.float()
                q_grad_reshaped = q_grad.view(num_heads, head_dim, -1)
                q_per_head = torch.norm(q_grad_reshaped, p=2, dim=(1, 2))
                if layer_idx in grad_norms:
                    # Combine o_proj and q_proj gradient importance
                    grad_norms[layer_idx] = (
                        grad_norms[layer_idx] + q_per_head.float().cpu().numpy()
                    ) / 2.0

        except (AttributeError, IndexError):
            continue

    model.zero_grad()

    return grad_norms, loss_value, total_grad_norm


# =============================================================================
# Main Analysis Pipeline
# =============================================================================

def run_analysis(
    model_path: str,
    datasets_dir: str,
    output_dir: str,
    samples_per_task: int = 50,
    batch_size: int = 1,
    max_seq_len: int = 512,
    device: str = "cuda",
    topk_tokens: int = 10,
    token_pool_method: str = "topk",
    use_gradient: bool = True,
    dtype: str = "float16",
):
    """Run the full head importance analysis pipeline."""

    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()

    print("=" * 70)
    print("MoH Attention Head Importance Analysis")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load model and tokenizer
    # ------------------------------------------------------------------
    print(f"\n[1/5] Loading model from {model_path}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(dtype, torch.float16)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device if device == "auto" else None,
        trust_remote_code=True,
        attn_implementation="eager",  # Need full attention weights, not flash
    )

    if device != "auto":
        model = model.to(device)

    model.eval()

    # Extract model dimensions
    config = model.config
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // num_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)

    print(f"  Model: {config._name_or_path}")
    print(f"  Layers: {num_layers}, Heads: {num_heads}, Head dim: {head_dim}")
    print(f"  KV heads: {num_kv_heads}, Hidden size: {config.hidden_size}")
    print(f"  dtype: {torch_dtype}")

    # ------------------------------------------------------------------
    # 2. Load datasets
    # ------------------------------------------------------------------
    print(f"\n[2/5] Loading evaluation datasets from {datasets_dir}...")
    max_per_task = samples_per_task if samples_per_task > 0 else None
    dataset = MoHFullEvalDataset(
        tokenizer=tokenizer,
        datasets_dir=datasets_dir,
        max_seq_len=max_seq_len,
        max_samples_per_task=max_per_task,  # None = load ALL samples
    )
    print_dataset_summary(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=0,
    )

    # ------------------------------------------------------------------
    # 3. Setup hooks and aggregator
    # ------------------------------------------------------------------
    print(f"\n[3/5] Setting up hooks and aggregator...")
    hooker = AttentionHeadHooker(model, num_layers, num_heads, head_dim)
    hooker.register_hooks()

    aggregator = HeadImportanceAggregator(
        num_layers=num_layers,
        num_heads=num_heads,
        topk_tokens=topk_tokens,
        token_pool_method=token_pool_method,
    )

    # ------------------------------------------------------------------
    # 4. Run inference and collect importance scores
    # ------------------------------------------------------------------
    print(f"\n[4/5] Running inference on {len(dataset)} prompts...")
    print(f"  Gradient-based weighting: {use_gradient}")
    print(f"  Token pooling: {token_pool_method} (k={topk_tokens})")

    num_processed = 0
    task_counts = defaultdict(int)

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing")):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        task_names = batch["task_names"]

        try:
            # Clear previous captures
            hooker.clear()

            # ---- Forward pass (captures Q states via hooks) ----
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                )

            # Extract Q norms (MoH method)
            q_norms = hooker.compute_q_norms()

            # Extract attention weights for rollout (if captured)
            attn_weights_dict = {}
            if hasattr(outputs, "attentions") and outputs.attentions is not None:
                for i, aw in enumerate(outputs.attentions):
                    if aw is not None:
                        attn_weights_dict[i] = aw[0].float().cpu().numpy()  # first in batch

            # ---- Gradient-based importance ----
            grad_norms = {}
            loss_value = 0.0
            total_grad_norm = 1.0

            if use_gradient:
                try:
                    grad_norms, loss_value, total_grad_norm = (
                        compute_gradient_importance_via_params(
                            model, input_ids, attention_mask, labels,
                            num_layers, num_heads, head_dim,
                        )
                    )
                except Exception as e:
                    print(f"  [WARN] Gradient computation failed: {e}")

            # ---- Add to aggregator ----
            for sample_idx in range(input_ids.size(0)):
                task = task_names[sample_idx] if sample_idx < len(task_names) else "unknown"

                # Per-sample Q norms (if batch_size > 1, we'd need to split)
                # For batch_size=1 this is direct
                aggregator.add_prompt_result(
                    q_norms_per_layer=q_norms,
                    grad_norms_per_layer=grad_norms if grad_norms else None,
                    attn_weights_per_layer=attn_weights_dict if attn_weights_dict else None,
                    prompt_loss=loss_value,
                    prompt_grad_norm=total_grad_norm,
                    task_name=task,
                )
                task_counts[task] += 1
                num_processed += 1

        except torch.cuda.OutOfMemoryError:
            print(f"  [OOM] Skipping batch {batch_idx}, clearing cache...")
            torch.cuda.empty_cache()
            gc.collect()
            continue
        except Exception as e:
            print(f"  [ERROR] Batch {batch_idx}: {e}")
            continue

        # Periodic memory cleanup
        if batch_idx % 50 == 0 and device != "cpu":
            torch.cuda.empty_cache()

    hooker.remove_hooks()

    print(f"\n  Processed {num_processed} prompts across tasks:")
    for task, count in sorted(task_counts.items()):
        print(f"    {task}: {count}")

    # ------------------------------------------------------------------
    # 5. Compute results and save
    # ------------------------------------------------------------------
    print(f"\n[5/5] Computing final results...")

    # Final importance matrix
    importance = aggregator.compute_final_importance()
    normalized = aggregator.compute_normalized_heatmap()
    ranking = aggregator.compute_global_ranking()
    stats = aggregator.compute_analysis_stats()

    # Shared heads at different thresholds
    shared_75 = aggregator.get_top_k_shared_heads(75.0)
    shared_50 = aggregator.get_top_k_shared_heads(50.0)
    shared_25 = aggregator.get_top_k_shared_heads(25.0)

    # Save raw importance matrix
    np.save(os.path.join(output_dir, "importance_raw.npy"), importance)
    np.save(os.path.join(output_dir, "importance_normalized.npy"), normalized)

    # Save global ranking as CSV
    import csv
    with open(os.path.join(output_dir, "global_ranking.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "layer_idx", "head_idx", "importance_score"])
        for rank, (layer, head, score) in enumerate(ranking, 1):
            writer.writerow([rank, layer, head, f"{score:.6f}"])

    # Save shared heads
    for name, heads, pct in [
        ("shared_heads_top75.csv", shared_75, 75),
        ("shared_heads_top50.csv", shared_50, 50),
        ("shared_heads_top25.csv", shared_25, 25),
    ]:
        with open(os.path.join(output_dir, name), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([f"# Top {pct}% shared heads ({len(heads)} heads)"])
            writer.writerow(["layer_idx", "head_idx"])
            for layer, head in heads:
                writer.writerow([layer, head])

    # Save analysis report
    elapsed = time.time() - start_time
    report = generate_analysis_report(
        stats, ranking, shared_50, normalized,
        num_layers, num_heads, num_processed, elapsed, task_counts,
    )
    report_path = os.path.join(output_dir, "analysis_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(report)

    # Save config
    config_dict = {
        "model_path": model_path,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "samples_per_task": samples_per_task,
        "max_seq_len": max_seq_len,
        "topk_tokens": topk_tokens,
        "token_pool_method": token_pool_method,
        "use_gradient": use_gradient,
        "num_prompts_processed": num_processed,
        "elapsed_seconds": elapsed,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    # Generate visualization
    print("\nGenerating heatmap visualization...")
    from visualization import generate_all_visualizations
    generate_all_visualizations(
        normalized_heatmap=normalized,
        raw_importance=importance,
        ranking=ranking,
        stats=stats,
        shared_heads_50=shared_50,
        per_task_importance=aggregator.get_per_task_importance(),
        output_dir=output_dir,
        num_layers=num_layers,
        num_heads=num_heads,
    )

    print(f"\nAll results saved to {output_dir}/")
    print(f"Total time: {elapsed:.1f}s")


def generate_analysis_report(
    stats, ranking, shared_50, normalized,
    num_layers, num_heads, num_processed, elapsed, task_counts,
):
    """Generate a text analysis report."""
    lines = [
        "=" * 70,
        "ATTENTION HEAD IMPORTANCE ANALYSIS REPORT",
        "=" * 70,
        "",
        f"Model: LLaMA3-8B ({num_layers} layers, {num_heads} heads/layer)",
        f"Total heads: {num_layers * num_heads}",
        f"Prompts processed: {num_processed}",
        f"Tasks covered: {', '.join(stats['tasks_covered'])}",
        f"Elapsed time: {elapsed:.1f}s",
        "",
        "--- Global Distribution ---",
        f"Mean importance: {stats['global_importance_mean']:.4f}",
        f"Std importance:  {stats['global_importance_std']:.4f}",
        f"Entropy:         {stats['importance_entropy']:.4f}",
        f"Mean Gini coeff: {stats['mean_gini']:.4f} (0=uniform, 1=concentrated)",
        "",
        "--- Layer Analysis ---",
    ]

    # Layer-by-layer stats
    for i in range(num_layers):
        lines.append(
            f"  Layer {i:2d}: mean={stats['layer_mean_importance'][i]:.4f}, "
            f"std={stats['layer_std_importance'][i]:.4f}, "
            f"max={stats['layer_max_importance'][i]:.4f}, "
            f"gini={stats['gini_per_layer'][i]:.4f}, "
            f"shared_50_count={int(stats['top50_heads_per_layer'][i])}"
        )

    lines.extend([
        "",
        "--- Top 20 Most Important Heads ---",
    ])
    for rank, (layer, head, score) in enumerate(ranking[:20], 1):
        lines.append(f"  #{rank:3d}: Layer {layer:2d}, Head {head:2d} (score={score:.6f})")

    lines.extend([
        "",
        "--- Bottom 20 Least Important Heads ---",
    ])
    for rank_offset, (layer, head, score) in enumerate(ranking[-20:]):
        rank = num_layers * num_heads - 19 + rank_offset
        lines.append(f"  #{rank:3d}: Layer {layer:2d}, Head {head:2d} (score={score:.6f})")

    # Shared head distribution
    lines.extend([
        "",
        "--- Top 50% Shared Heads Distribution ---",
        f"Total shared heads: {len(shared_50)}",
    ])

    # Count per layer
    layer_counts = defaultdict(int)
    for l, h in shared_50:
        layer_counts[l] += 1

    lines.append("  Heads selected per layer:")
    for l in range(num_layers):
        bar = "█" * layer_counts.get(l, 0) + "░" * (num_heads - layer_counts.get(l, 0))
        lines.append(f"    Layer {l:2d}: {layer_counts.get(l, 0):2d}/{num_heads} [{bar}]")

    # Concentration analysis
    lines.extend([
        "",
        "--- Concentration Analysis ---",
    ])
    gini_vals = stats["gini_per_layer"]
    concentrated_layers = np.where(gini_vals > np.mean(gini_vals) + np.std(gini_vals))[0]
    uniform_layers = np.where(gini_vals < np.mean(gini_vals) - np.std(gini_vals))[0]

    lines.append(f"  Highly concentrated layers (high Gini): {list(concentrated_layers)}")
    lines.append(f"  Relatively uniform layers (low Gini):   {list(uniform_layers)}")
    lines.append("")

    if np.mean(gini_vals) > 0.3:
        lines.append("  CONCLUSION: Head importance is CONCENTRATED — a subset of heads")
        lines.append("  dominates across layers, supporting the MoH approach of selective activation.")
    else:
        lines.append("  CONCLUSION: Head importance is relatively UNIFORM — most heads")
        lines.append("  contribute similarly, suggesting high redundancy amenable to pruning.")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MoH Attention Head Importance Analysis for LLaMA3-8B"
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to local LLaMA3-8B model directory")
    parser.add_argument("--datasets_dir", type=str, default="./datasets",
                        help="Path to downloaded datasets")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Output directory for results")
    parser.add_argument("--samples_per_task", type=int, default=0,
                        help="Max samples per task (0 = ALL samples, full eval splits)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for inference (1 recommended for gradient analysis)")
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu", "auto"],
                        help="Device for inference")
    parser.add_argument("--topk_tokens", type=int, default=10,
                        help="K for top-k token pooling")
    parser.add_argument("--token_pool_method", type=str, default="topk",
                        choices=["topk", "max"],
                        help="Token-level aggregation method")
    parser.add_argument("--no_gradient", action="store_true",
                        help="Disable gradient-based importance (faster but less accurate)")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Model precision")

    args = parser.parse_args()

    run_analysis(
        model_path=args.model_path,
        datasets_dir=args.datasets_dir,
        output_dir=args.output_dir,
        samples_per_task=args.samples_per_task,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        device=args.device,
        topk_tokens=args.topk_tokens,
        token_pool_method=args.token_pool_method,
        use_gradient=not args.no_gradient,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
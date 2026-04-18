#!/usr/bin/env python3
"""
Advanced aggregation methods for attention head importance scores.

Implements:
1. Token-level: Top-k pooling / max pooling
2. Cross-layer: Attention Rollout (layer-wise propagation weighting)
3. Cross-head: Gradient-based weighting
4. Cross-prompt: Confidence-weighted averaging

References:
- Abnar & Zuidema (2020): "Quantifying Attention Flow in Transformers" (Attention Rollout)
- MoH paper Section 4.4: L2-norm based routing scores
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats


def topk_pool_tokens(
    head_norms: np.ndarray,
    k: int = 10,
) -> np.ndarray:
    num_tokens, num_heads = head_norms.shape
    k = min(k, num_tokens)

    # For each head, take mean of top-k token norms
    result = np.zeros(num_heads)
    for h in range(num_heads):
        top_indices = np.argpartition(head_norms[:, h], -k)[-k:]
        result[h] = head_norms[top_indices, h].mean()

    return result


def max_pool_tokens(head_norms: np.ndarray) -> np.ndarray:
    return head_norms.max(axis=0)


def attention_rollout_weights(
    attn_matrices: List[np.ndarray],
    num_layers: int,
) -> np.ndarray:
    if attn_matrices is not None and len(attn_matrices) == num_layers:
        rollout = None
        layer_weights = np.zeros(num_layers)

        for layer_idx in range(num_layers):
            attn = attn_matrices[layer_idx]  # (num_heads, seq_len, seq_len)
            attn_avg = attn.mean(axis=0)  # (seq_len, seq_len)
            attn_avg = 0.5 * attn_avg + 0.5 * np.eye(attn_avg.shape[0])
            attn_avg = attn_avg / (attn_avg.sum(axis=-1, keepdims=True) + 1e-12)

            if rollout is None:
                rollout = attn_avg
            else:
                rollout = rollout @ attn_avg

            layer_weights[layer_idx] = np.linalg.norm(attn_avg, "fro")

        layer_weights = layer_weights / (layer_weights.sum() + 1e-12)
        return layer_weights

    else:
        weights = np.exp(np.linspace(-1, 1, num_layers))
        return weights / weights.sum()


def gradient_head_weights(
    grad_norms: np.ndarray,
) -> np.ndarray:
    grad_norms = grad_norms + 1e-12
    layer_sums = grad_norms.sum(axis=1, keepdims=True)
    weights = grad_norms / layer_sums
    return weights


def confidence_weighted_average(
    per_prompt_scores: List[np.ndarray],
    per_prompt_weights: List[float],
) -> np.ndarray:
    weights = np.array(per_prompt_weights)
    weights = weights / (weights.sum() + 1e-12)

    result = np.zeros_like(per_prompt_scores[0])
    for score, w in zip(per_prompt_scores, weights):
        result += w * score

    return result


class HeadImportanceAggregator:

    def __init__(
        self,
        num_layers: int = 32,
        num_heads: int = 32,
        topk_tokens: int = 10,
        token_pool_method: str = "topk", 
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.topk_tokens = topk_tokens
        self.token_pool_method = token_pool_method

        # Accumulate per-prompt results
        self.prompt_scores: List[np.ndarray] = []       
        self.prompt_weights: List[float] = []            
        self.prompt_task_names: List[str] = []
        
        # [FIXED]: Store computed float weights instead of huge attention matrices
        self.rollout_layer_weights: List[np.ndarray] = []

    def add_prompt_result(
        self,
        q_norms_per_layer: Dict[int, np.ndarray],
        grad_norms_per_layer: Optional[Dict[int, np.ndarray]] = None,
        attn_weights_per_layer: Optional[Dict[int, np.ndarray]] = None,
        prompt_loss: Optional[float] = None,
        prompt_grad_norm: Optional[float] = None,
        task_name: str = "unknown",
    ):
        # Step 1: Token-level aggregation per layer
        pooled = np.zeros((self.num_layers, self.num_heads))
        for layer_idx in range(self.num_layers):
            if layer_idx in q_norms_per_layer:
                norms = q_norms_per_layer[layer_idx] 
                if self.token_pool_method == "topk":
                    pooled[layer_idx] = topk_pool_tokens(norms, self.topk_tokens)
                else:
                    pooled[layer_idx] = max_pool_tokens(norms)

        # Step 2: Apply gradient-based head weighting
        if grad_norms_per_layer is not None:
            grad_matrix = np.zeros((self.num_layers, self.num_heads))
            for layer_idx in range(self.num_layers):
                if layer_idx in grad_norms_per_layer:
                    grad_matrix[layer_idx] = grad_norms_per_layer[layer_idx]

            grad_weights = gradient_head_weights(grad_matrix)
            pooled = pooled * (1.0 + grad_weights) 

        # Step 3: Store for cross-prompt aggregation
        self.prompt_scores.append(pooled)
        self.prompt_task_names.append(task_name)

        if prompt_grad_norm is not None:
            self.prompt_weights.append(prompt_grad_norm)
        elif prompt_loss is not None:
            self.prompt_weights.append(abs(prompt_loss))
        else:
            self.prompt_weights.append(1.0)

        # [FIXED]: Compute and store layer rollout weights immediately.
        # This allows Python's garbage collector to destroy the heavy raw attention matrices.
        if attn_weights_per_layer is not None:
            attn_list = [
                attn_weights_per_layer.get(i, None)
                for i in range(self.num_layers)
            ]
            if all(a is not None for a in attn_list):
                lw = attention_rollout_weights(attn_list, self.num_layers)
                self.rollout_layer_weights.append(lw)

    def compute_final_importance(self) -> np.ndarray:
        if not self.prompt_scores:
            return np.zeros((self.num_layers, self.num_heads))

        # Step 4: Cross-prompt weighted average
        importance = confidence_weighted_average(
            self.prompt_scores,
            self.prompt_weights,
        )

        # Step 5: Apply attention rollout layer weighting
        if hasattr(self, "rollout_layer_weights") and self.rollout_layer_weights:
            avg_layer_weights = np.mean(self.rollout_layer_weights, axis=0)
            importance = importance * avg_layer_weights[:, None]
        else:
            layer_weights = attention_rollout_weights(None, self.num_layers)
            importance = importance * layer_weights[:, None]

        return importance

    def compute_normalized_heatmap(self) -> np.ndarray:
        importance = self.compute_final_importance()
        normalized = np.zeros_like(importance)
        for layer_idx in range(self.num_layers):
            row = importance[layer_idx]
            rmin, rmax = row.min(), row.max()
            if rmax - rmin > 1e-12:
                normalized[layer_idx] = (row - rmin) / (rmax - rmin)
            else:
                normalized[layer_idx] = 0.5  

        return normalized

    def compute_global_ranking(self) -> List[Tuple[int, int, float]]:
        importance = self.compute_final_importance()
        ranking = []
        for layer_idx in range(self.num_layers):
            for head_idx in range(self.num_heads):
                ranking.append((layer_idx, head_idx, importance[layer_idx, head_idx]))

        ranking.sort(key=lambda x: x[2], reverse=True)
        return ranking

    def get_top_k_shared_heads(self, k_percent: float = 50.0) -> List[Tuple[int, int]]:
        ranking = self.compute_global_ranking()
        total_heads = self.num_layers * self.num_heads
        k = int(total_heads * k_percent / 100.0)
        return [(r[0], r[1]) for r in ranking[:k]]

    def get_per_task_importance(self) -> Dict[str, np.ndarray]:
        task_scores: Dict[str, List[np.ndarray]] = {}
        for score, task in zip(self.prompt_scores, self.prompt_task_names):
            if task not in task_scores:
                task_scores[task] = []
            task_scores[task].append(score)

        result = {}
        for task, scores in task_scores.items():
            result[task] = np.mean(scores, axis=0)

        return result

    def compute_analysis_stats(self) -> Dict:
        importance = self.compute_final_importance()
        normalized = self.compute_normalized_heatmap()

        layer_means = importance.mean(axis=1)
        layer_stds = importance.std(axis=1)
        layer_max = importance.max(axis=1)

        gini_per_layer = []
        for layer_idx in range(self.num_layers):
            row = importance[layer_idx]
            row_sorted = np.sort(row)
            n = len(row_sorted)
            index = np.arange(1, n + 1)
            gini = (2 * np.sum(index * row_sorted) / (n * np.sum(row_sorted) + 1e-12)) - (n + 1) / n
            gini_per_layer.append(gini)

        flat = importance.flatten()
        top10_threshold = np.percentile(flat, 90)
        top10_heads = [(l, h) for l in range(self.num_layers) for h in range(self.num_heads)
                       if importance[l, h] >= top10_threshold]

        top50_heads = self.get_top_k_shared_heads(50.0)
        heads_per_layer = np.zeros(self.num_layers)
        for l, h in top50_heads:
            heads_per_layer[l] += 1

        return {
            "layer_mean_importance": layer_means,
            "layer_std_importance": layer_stds,
            "layer_max_importance": layer_max,
            "gini_per_layer": np.array(gini_per_layer),
            "mean_gini": np.mean(gini_per_layer),
            "global_importance_mean": flat.mean(),
            "global_importance_std": flat.std(),
            "top10_percent_heads": top10_heads,
            "top50_heads_per_layer": heads_per_layer,
            "importance_entropy": stats.entropy(flat / (flat.sum() + 1e-12)),
            "num_prompts_processed": len(self.prompt_scores),
            "tasks_covered": list(set(self.prompt_task_names)),
        }
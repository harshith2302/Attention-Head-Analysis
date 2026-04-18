"""
Head Scoring Methods for Dynamic Inference-Time Attention Head Pruning.

All methods compute per-head importance scores without any training.
Each scorer returns a tensor of shape [num_heads] with importance scores.
"""
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from abc import ABC, abstractmethod


class HeadScorer(ABC):
    """Base class for head importance scoring methods."""

    def __init__(self, num_heads: int = 32, head_dim: int = 128):
        self.num_heads = num_heads
        self.head_dim = head_dim

    @abstractmethod
    def score(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        hidden_states: torch.Tensor,
        layer_idx: int,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute importance scores for each head.

        Args:
            query_states: [batch, seq_len, num_heads * head_dim] (pre-reshape)
            key_states: [batch, seq_len, num_kv_heads * head_dim] (pre-reshape)
            hidden_states: [batch, seq_len, hidden_size] - input to attention
            layer_idx: current layer index

        Returns:
            scores: [batch, num_heads] importance scores per head
        """
        pass


class QueryNormTopKScorer(HeadScorer):
    """
    Method 1: Query Norm Top-K
    s_h = mean_t ||Q_h(t)||_2

    Scores each head by the average L2 norm of its query vectors across tokens.
    Optionally uses only the last token.
    """

    def __init__(self, num_heads: int = 32, head_dim: int = 128,
                 last_token_only: bool = False):
        super().__init__(num_heads, head_dim)
        self.last_token_only = last_token_only

    def score(self, query_states, key_states, hidden_states, layer_idx, **kwargs):
        bsz, seq_len, _ = query_states.shape
        # Reshape to [batch, seq_len, num_heads, head_dim]
        q = query_states.view(bsz, seq_len, self.num_heads, self.head_dim)

        if self.last_token_only:
            # Use only last token: [batch, 1, num_heads, head_dim]
            q = q[:, -1:, :, :]

        # Compute L2 norm per head per token: [batch, seq_len, num_heads]
        q_norms = torch.norm(q, p=2, dim=-1)
        # Average across tokens: [batch, num_heads]
        scores = q_norms.mean(dim=1)
        return scores


class LastTokenQueryNormScorer(HeadScorer):
    """
    Method 2: Last Token Query Norm
    s_h = ||Q_h(last)||_2

    Scores each head by the L2 norm of the query at the last token position.
    """

    def score(self, query_states, key_states, hidden_states, layer_idx, **kwargs):
        bsz, seq_len, _ = query_states.shape
        q = query_states.view(bsz, seq_len, self.num_heads, self.head_dim)
        # Last token only: [batch, num_heads, head_dim]
        q_last = q[:, -1, :, :]
        # L2 norm: [batch, num_heads]
        scores = torch.norm(q_last, p=2, dim=-1)
        return scores


class QKNormProductScorer(HeadScorer):
    """
    Method 3: Query-Key Norm Product
    s_h = mean_t [ ||Q_h(t)|| * ||K_h(t)|| ]

    For GQA (LLaMA3-8B: 32 Q heads, 8 KV heads), each KV head
    is shared across num_heads // num_kv_heads query heads.
    """

    def __init__(self, num_heads: int = 32, head_dim: int = 128,
                 num_kv_heads: int = 8, last_token_only: bool = False):
        super().__init__(num_heads, head_dim)
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.last_token_only = last_token_only

    def score(self, query_states, key_states, hidden_states, layer_idx, **kwargs):
        bsz, seq_len, _ = query_states.shape
        q = query_states.view(bsz, seq_len, self.num_heads, self.head_dim)
        k = key_states.view(bsz, seq_len, self.num_kv_heads, self.head_dim)

        if self.last_token_only:
            q = q[:, -1:, :, :]
            k = k[:, -1:, :, :]

        # Q norms: [batch, seq_len, num_heads]
        q_norms = torch.norm(q, p=2, dim=-1)

        # K norms: [batch, seq_len, num_kv_heads]
        k_norms = torch.norm(k, p=2, dim=-1)

        # Expand K norms to match Q heads (GQA)
        # Each KV head maps to num_kv_groups Q heads
        k_norms_expanded = k_norms.repeat_interleave(self.num_kv_groups, dim=-1)

        # Product: [batch, seq_len, num_heads]
        product = q_norms * k_norms_expanded

        # Average across tokens: [batch, num_heads]
        scores = product.mean(dim=1)
        return scores


class TokenSaliencyQueryNormScorer(HeadScorer):
    """
    Method 4: Token Saliency + Query Norm

    Step 1: Compute token importance I_t = ||hidden_state_t||
    Step 2: Take top salient tokens
    Step 3: s_h = average over top salient tokens of ||Q_h(t)||
    """

    def __init__(self, num_heads: int = 32, head_dim: int = 128,
                 top_salient_ratio: float = 0.3):
        super().__init__(num_heads, head_dim)
        self.top_salient_ratio = top_salient_ratio

    def score(self, query_states, key_states, hidden_states, layer_idx, **kwargs):
        bsz, seq_len, _ = query_states.shape
        q = query_states.view(bsz, seq_len, self.num_heads, self.head_dim)

        # Step 1: Token importance via hidden state norm
        # hidden_states: [batch, seq_len, hidden_size]
        token_importance = torch.norm(hidden_states, p=2, dim=-1)  # [batch, seq_len]

        # Step 2: Select top salient tokens
        k_tokens = max(1, int(seq_len * self.top_salient_ratio))
        _, top_indices = torch.topk(token_importance, k=k_tokens, dim=1)  # [batch, k_tokens]

        # Step 3: Gather Q at salient positions and compute norms
        # Expand indices for gathering: [batch, k_tokens, num_heads, head_dim]
        top_indices_expanded = top_indices.unsqueeze(-1).unsqueeze(-1).expand(
            bsz, k_tokens, self.num_heads, self.head_dim
        )
        q_salient = torch.gather(q, dim=1, index=top_indices_expanded)

        # L2 norm per head: [batch, k_tokens, num_heads]
        q_norms = torch.norm(q_salient, p=2, dim=-1)

        # Average over salient tokens: [batch, num_heads]
        scores = q_norms.mean(dim=1)
        return scores


class RollingLayerGatingScorer(HeadScorer):
    """
    Method 5: Rolling Layer Gating

    Chunk-based routing: one layer computes head importance,
    and the selected heads are reused for the next N layers.

    E.g., layer 8 computes importance -> reuse for layers 9..(8+N-1)
    """

    def __init__(self, num_heads: int = 32, head_dim: int = 128,
                 chunk_size: int = 4):
        super().__init__(num_heads, head_dim)
        self.chunk_size = chunk_size
        self._cached_scores: Dict[int, torch.Tensor] = {}
        self._base_scorer = QueryNormTopKScorer(num_heads, head_dim)

    def reset_cache(self):
        """Clear cached scores between samples."""
        self._cached_scores.clear()

    def score(self, query_states, key_states, hidden_states, layer_idx, **kwargs):
        # Determine if this layer is a "scoring" layer (chunk boundary)
        warmup = kwargs.get('warmup_layers', 0)
        effective_idx = layer_idx - warmup

        if effective_idx < 0:
            # Within warmup, return uniform high scores (all heads active)
            bsz = query_states.shape[0]
            return torch.ones(bsz, self.num_heads, device=query_states.device)

        chunk_start = (effective_idx // self.chunk_size) * self.chunk_size + warmup

        if chunk_start not in self._cached_scores or layer_idx == chunk_start:
            # Compute fresh scores at chunk boundary
            scores = self._base_scorer.score(
                query_states, key_states, hidden_states, layer_idx
            )
            self._cached_scores[chunk_start] = scores
            return scores
        else:
            # Reuse cached scores from chunk start
            return self._cached_scores[chunk_start]


class HybridDynamicRoutingScorer(HeadScorer):
    """
    Method 6: Hybrid Dynamic Token-aware Head Routing

    s_h = λ * ||Q_h|| * ||K_h|| + (1-λ) * avg_top_tokens ||Q_h(t)||

    Combines QK norm product with token-saliency-weighted query norms.
    """

    def __init__(self, num_heads: int = 32, head_dim: int = 128,
                 num_kv_heads: int = 8, lambda_val: float = 0.5,
                 top_salient_ratio: float = 0.3):
        super().__init__(num_heads, head_dim)
        self.lambda_val = lambda_val
        self.qk_scorer = QKNormProductScorer(num_heads, head_dim, num_kv_heads)
        self.saliency_scorer = TokenSaliencyQueryNormScorer(
            num_heads, head_dim, top_salient_ratio
        )

    def score(self, query_states, key_states, hidden_states, layer_idx, **kwargs):
        # Component 1: QK norm product
        qk_scores = self.qk_scorer.score(
            query_states, key_states, hidden_states, layer_idx
        )
        # Component 2: Saliency-weighted query norm
        sal_scores = self.saliency_scorer.score(
            query_states, key_states, hidden_states, layer_idx
        )

        # Normalize each component to [0, 1] range for fair combination
        qk_norm = qk_scores / (qk_scores.max(dim=-1, keepdim=True).values + 1e-8)
        sal_norm = sal_scores / (sal_scores.max(dim=-1, keepdim=True).values + 1e-8)

        # Hybrid combination
        scores = self.lambda_val * qk_norm + (1 - self.lambda_val) * sal_norm
        return scores


class NoOpScorer(HeadScorer):
    """Baseline: no pruning, all heads get equal scores."""

    def score(self, query_states, key_states, hidden_states, layer_idx, **kwargs):
        bsz = query_states.shape[0]
        return torch.ones(bsz, self.num_heads, device=query_states.device)


# ─── Factory ───

def create_scorer(method: str, num_heads: int = 32, head_dim: int = 128,
                  num_kv_heads: int = 8, **kwargs) -> HeadScorer:
    """Create a head scorer by method name."""
    scorers = {
        "query_norm_topk": lambda: QueryNormTopKScorer(
            num_heads, head_dim,
            last_token_only=kwargs.get('last_token_only', False)
        ),
        "last_token_query_norm": lambda: LastTokenQueryNormScorer(
            num_heads, head_dim
        ),
        "qk_norm_product": lambda: QKNormProductScorer(
            num_heads, head_dim, num_kv_heads,
            last_token_only=kwargs.get('last_token_only', False)
        ),
        "token_saliency_query_norm": lambda: TokenSaliencyQueryNormScorer(
            num_heads, head_dim,
            top_salient_ratio=kwargs.get('top_salient_ratio', 0.3)
        ),
        "rolling_layer_gating": lambda: RollingLayerGatingScorer(
            num_heads, head_dim,
            chunk_size=kwargs.get('rolling_chunk_size', 4)
        ),
        "hybrid_dynamic_routing": lambda: HybridDynamicRoutingScorer(
            num_heads, head_dim, num_kv_heads,
            lambda_val=kwargs.get('lambda_val', 0.5),
            top_salient_ratio=kwargs.get('top_salient_ratio', 0.3)
        ),
        "none": lambda: NoOpScorer(num_heads, head_dim),
    }

    if method not in scorers:
        raise ValueError(f"Unknown scoring method: {method}. Available: {list(scorers.keys())}")
    return scorers[method]()

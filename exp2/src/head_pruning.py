"""
Dynamic Head Pruning — monkey-patches attention forward.

KEY FIX: Masking happens BEFORE o_proj (like MoH paper Eq.3-4),
not after. Post-o_proj the head dimensions are mixed and can't be separated.
"""
import torch
import torch.nn.functional as F
from typing import Dict, Optional
from .head_scoring import HeadScorer, create_scorer, RollingLayerGatingScorer
from .config import PruningConfig
import time
import math
import logging

logger = logging.getLogger(__name__)


class HeadPruningManager:
    def __init__(self, model, config: PruningConfig):
        self.model = model
        self.config = config
        self._original_forwards = {}
        self.stats = {'head_selections': {}, 'scores': {}, 'timings': {}}
        self._detect_architecture()
        self.scorer = create_scorer(
            method=config.method, num_heads=self.num_heads,
            head_dim=self.head_dim, num_kv_heads=self.num_kv_heads,
            last_token_only=config.last_token_only,
            top_salient_ratio=config.top_salient_ratio,
            lambda_val=config.lambda_val,
            rolling_chunk_size=config.rolling_chunk_size,
        )

    def _detect_architecture(self):
        base = self.model.model if hasattr(self.model, 'model') else self.model
        self.layers = base.layers
        self.num_layers = len(self.layers)
        a = self.layers[0].self_attn
        self.num_heads = a.num_heads
        self.head_dim = a.head_dim
        self.num_kv_heads = a.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.hidden_size = self.num_heads * self.head_dim
        logger.info(f"Detected: {self.num_layers} layers, {self.num_heads} heads, "
                     f"{self.num_kv_heads} KV heads, head_dim={self.head_dim}")

    def install_hooks(self):
        self.remove_hooks()
        for idx, layer in enumerate(self.layers):
            attn = layer.self_attn
            self._original_forwards[idx] = attn.forward
            attn.forward = self._make_pruned_forward(idx, attn)
        logger.info(f"Installed {len(self._original_forwards)} pruning patches")

    def _make_pruned_forward(self, layer_idx, attn):
        mgr = self
        cfg = self.config
        scorer = self.scorer

        def pruned_forward(hidden_states, attention_mask=None, position_ids=None,
                           past_key_value=None, output_attentions=False,
                           use_cache=False, **kwargs):
            ek = cfg.get_effective_k(layer_idx, mgr.num_layers)

            # Warmup layers or no pruning => original forward
            if ek >= mgr.num_heads or cfg.method == "none":
                return mgr._original_forwards[layer_idx](
                    hidden_states, attention_mask=attention_mask,
                    position_ids=position_ids, past_key_value=past_key_value,
                    output_attentions=output_attentions, use_cache=use_cache, **kwargs
                )

            bsz, seq_len, _ = hidden_states.shape

            # 1) Q, K, V projections
            Q = attn.q_proj(hidden_states)
            K = attn.k_proj(hidden_states)
            V = attn.v_proj(hidden_states)

            # 2) Score heads
            with torch.no_grad():
                t0 = time.perf_counter()
                scores = scorer.score(Q, K, hidden_states, layer_idx=layer_idx,
                                      warmup_layers=cfg.warmup_layers)
                _, top_idx = torch.topk(scores, k=ek, dim=-1)
                mask = torch.zeros(bsz, mgr.num_heads, device=hidden_states.device,
                                   dtype=hidden_states.dtype)
                mask.scatter_(1, top_idx, 1.0)
                t1 = time.perf_counter()
            mgr.stats['head_selections'][layer_idx] = top_idx.detach().cpu()
            mgr.stats['scores'][layer_idx] = scores.detach().cpu()
            mgr.stats['timings'][layer_idx] = t1 - t0

            # 3) Reshape to head format
            Q = Q.view(bsz, seq_len, mgr.num_heads, mgr.head_dim).transpose(1, 2)
            K = K.view(bsz, seq_len, mgr.num_kv_heads, mgr.head_dim).transpose(1, 2)
            V = V.view(bsz, seq_len, mgr.num_kv_heads, mgr.head_dim).transpose(1, 2)

            # 4) RoPE
            kv_len = K.shape[-2]
            if past_key_value is not None:
                if hasattr(past_key_value, 'get_usable_length'):
                    kv_len += past_key_value.get_usable_length(kv_len, attn.layer_idx)
                elif isinstance(past_key_value, tuple):
                    kv_len += past_key_value[0].shape[-2]

            cos, sin = attn.rotary_emb(V, position_ids)
            try:
                from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
            except ImportError:
                from transformers.models.llama.modeling_llama import rotate_half
                def apply_rotary_pos_emb(q, k, cos, sin, pos_ids, unsqueeze_dim=1):
                    c = cos[pos_ids].unsqueeze(unsqueeze_dim)
                    s = sin[pos_ids].unsqueeze(unsqueeze_dim)
                    return (q*c)+(rotate_half(q)*s), (k*c)+(rotate_half(k)*s)
                def repeat_kv(x, n):
                    if n == 1: return x
                    b, h, s, d = x.shape
                    return x[:,:,None,:,:].expand(b,h,n,s,d).reshape(b,h*n,s,d)

            Q, K = apply_rotary_pos_emb(Q, K, cos, sin, position_ids)

            # 5) KV cache
            if past_key_value is not None and hasattr(past_key_value, 'update'):
                K, V = past_key_value.update(K, V, attn.layer_idx, {"sin": sin, "cos": cos})

            # 6) GQA expand
            K = repeat_kv(K, mgr.num_kv_groups)
            V = repeat_kv(V, mgr.num_kv_groups)

            # 7) Attention
            aw = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(mgr.head_dim)
            if attention_mask is not None:
                cm = attention_mask
                if cm.dim() == 2: cm = cm.unsqueeze(0).unsqueeze(0)
                elif cm.dim() == 3: cm = cm.unsqueeze(1)
                if cm.shape[-1] >= aw.shape[-1]:
                    cm = cm[:, :, :aw.shape[-2], :aw.shape[-1]]
                aw = aw + cm
            aw = F.softmax(aw, dim=-1, dtype=torch.float32).to(Q.dtype)
            ao = torch.matmul(aw, V)

            # 8) MASK HEADS **BEFORE** o_proj (the critical fix)
            ao = ao.transpose(1, 2).contiguous()  # [bsz, seq, heads, dim]
            head_mask = mask.unsqueeze(1).unsqueeze(-1)  # [bsz, 1, heads, 1]
            ao = ao * head_mask * (mgr.num_heads / ek)

            # 9) o_proj on masked output
            ao = ao.reshape(bsz, seq_len, mgr.hidden_size)
            ao = attn.o_proj(ao)

            return ao, None, past_key_value

        return pruned_forward

    def remove_hooks(self):
        for idx, fwd in self._original_forwards.items():
            self.layers[idx].self_attn.forward = fwd
        self._original_forwards.clear()

    def reset_stats(self):
        self.stats = {'head_selections': {}, 'scores': {}, 'timings': {}}
        if isinstance(self.scorer, RollingLayerGatingScorer):
            self.scorer.reset_cache()

    def get_active_head_percentage(self):
        t = sum(self.config.get_effective_k(i, self.num_layers) for i in range(self.num_layers))
        return (t / (self.num_layers * self.num_heads)) * 100

    def estimate_flops_reduction(self):
        return (1 - self.get_active_head_percentage() / 100.0) * 35.0

    def get_scoring_overhead(self):
        return sum(self.stats['timings'].values())

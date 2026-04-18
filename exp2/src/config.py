"""
Configuration for Dynamic Inference-Time Head Pruning experiments.
"""
import os
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

PROJECT_ROOT = "/home/cccp/25m0834/RND5"


@dataclass
class ModelConfig:
    model_name: str = "meta-llama/Meta-Llama-3-8B"
    use_instruct: bool = False
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    hidden_size: int = 4096
    cache_dir: str = os.path.join(PROJECT_ROOT, "cache", "models")
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "eager"

    @property
    def full_model_name(self):
        if self.use_instruct:
            return self.model_name.replace("Meta-Llama-3-8B", "Meta-Llama-3-8B-Instruct")
        return self.model_name


@dataclass
class PruningConfig:
    method: str = "query_norm_topk"
    warmup_layers: int = 8
    top_k: int = 24
    progressive_k: bool = False
    mid_layer_k: int = 24
    late_layer_k: int = 16
    lambda_val: float = 0.5
    rolling_chunk_size: int = 4
    last_token_only: bool = False
    top_salient_ratio: float = 0.3

    def get_effective_k(self, layer_idx: int, total_layers: int = 32) -> int:
        if layer_idx < self.warmup_layers:
            return 32
        if not self.progressive_k:
            return self.top_k
        two_thirds = int(total_layers * 2 / 3)
        return self.mid_layer_k if layer_idx < two_thirds else self.late_layer_k


@dataclass
class BenchmarkConfig:
    benchmarks: List[str] = field(default_factory=lambda: [
        "mmlu", "gsm8k", "truthfulqa", "hellaswag", "piqa",
        "winogrande", "arc_challenge", "boolq", "lambada", "sciq", "logiqa"
    ])
    num_fewshot: Dict[str, int] = field(default_factory=lambda: {
        "mmlu": 5, "gsm8k": 8, "truthfulqa": 0, "hellaswag": 10,
        "piqa": 0, "winogrande": 0, "arc_challenge": 25, "boolq": 32,
        "lambada": 0, "sciq": 0, "logiqa": 0,
    })
    batch_size: int = 4
    max_samples: Optional[int] = 200
    cache_dir: str = os.path.join(PROJECT_ROOT, "cache", "datasets")


@dataclass
class OutputConfig:
    base_dir: str = os.path.join(PROJECT_ROOT, "results")

    def get_experiment_dir(self, method, warmup, top_k, lambda_val=None, chunk=None):
        name = f"{method}_w{warmup}_k{top_k}"
        if lambda_val is not None:
            name += f"_l{lambda_val}"
        if chunk is not None:
            name += f"_c{chunk}"
        path = os.path.join(self.base_dir, name)
        os.makedirs(path, exist_ok=True)
        return path


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    seed: int = 42
    device: str = "cuda"

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


METHODS = [
    "query_norm_topk", "last_token_query_norm", "qk_norm_product",
    "token_saliency_query_norm", "rolling_layer_gating", "hybrid_dynamic_routing",
]
LAMBDA_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]
WARMUP_CONFIGS = [6, 8, 10]
TOPK_CONFIGS = [24, 20, 16]


def generate_all_experiment_configs(max_samples=200):
    """Generate all 100 experiment configs."""
    configs = []
    for method in METHODS:
        for warmup in WARMUP_CONFIGS:
            for top_k in TOPK_CONFIGS:
                if method == "hybrid_dynamic_routing":
                    for lam in LAMBDA_VALUES:
                        configs.append(ExperimentConfig(
                            pruning=PruningConfig(method=method, warmup_layers=warmup,
                                                  top_k=top_k, lambda_val=lam),
                            benchmark=BenchmarkConfig(max_samples=max_samples),
                        ))
                elif method == "rolling_layer_gating":
                    for chunk in [2, 4]:
                        configs.append(ExperimentConfig(
                            pruning=PruningConfig(method=method, warmup_layers=warmup,
                                                  top_k=top_k, rolling_chunk_size=chunk),
                            benchmark=BenchmarkConfig(max_samples=max_samples),
                        ))
                else:
                    configs.append(ExperimentConfig(
                        pruning=PruningConfig(method=method, warmup_layers=warmup, top_k=top_k),
                        benchmark=BenchmarkConfig(max_samples=max_samples),
                    ))
    # Baseline
    configs.append(ExperimentConfig(
        pruning=PruningConfig(method="none", warmup_layers=32, top_k=32),
        benchmark=BenchmarkConfig(max_samples=max_samples),
    ))
    return configs


if __name__ == "__main__":
    configs = generate_all_experiment_configs()
    print(f"Total: {len(configs)} experiments")
    from collections import Counter
    c = Counter(cfg.pruning.method for cfg in configs)
    for m, n in sorted(c.items()):
        print(f"  {m}: {n}")

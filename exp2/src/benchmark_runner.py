"""
Benchmark evaluation runner.
"""
import os
import json
import time
import logging
import torch
import numpy as np
from typing import Dict, List, Any
from dataclasses import asdict
from .config import ExperimentConfig, PruningConfig
from .head_pruning import HeadPruningManager
from .head_scoring import RollingLayerGatingScorer

logger = logging.getLogger(__name__)


def run_single_experiment(model, tokenizer, config: ExperimentConfig) -> Dict[str, Any]:
    """Run one experiment config: pruning + all benchmarks + latency."""
    p = config.pruning
    extra = None
    chunk = None
    if p.method == "hybrid_dynamic_routing":
        extra = p.lambda_val
    if p.method == "rolling_layer_gating":
        chunk = p.rolling_chunk_size

    output_dir = config.output.get_experiment_dir(
        p.method, p.warmup_layers, p.top_k, lambda_val=extra, chunk=chunk
    )
    results_file = os.path.join(output_dir, "results.json")

    # Resume support
    if os.path.exists(results_file):
        logger.info(f"  SKIP (already done): {os.path.basename(output_dir)}")
        with open(results_file) as f:
            return json.load(f)

    logger.info(f"  RUN: method={p.method} warmup={p.warmup_layers} k={p.top_k}"
                + (f" lambda={p.lambda_val}" if extra else "")
                + (f" chunk={p.rolling_chunk_size}" if chunk else ""))

    manager = HeadPruningManager(model, p)
    manager.install_hooks()

    try:
        # Latency measurement
        latency = _measure_latency(model, tokenizer, manager)

        # Benchmark evaluation
        scores = _run_benchmarks(model, tokenizer, config)

        avg_score = float(np.mean(list(scores.values()))) if scores else 0.0

        results = {
            "config": asdict(p),
            "scores": scores,
            "average_score": avg_score,
            "latency": latency,
            "active_head_pct": manager.get_active_head_percentage(),
            "estimated_flops_reduction_pct": manager.estimate_flops_reduction(),
        }

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"  DONE: avg={avg_score*100:.1f}%  speed={latency.get('avg_tokens_per_sec',0):.1f} tok/s")
        return results

    except Exception as e:
        logger.error(f"  FAILED: {e}", exc_info=True)
        err_result = {"config": asdict(p), "error": str(e), "scores": {}, "average_score": 0}
        with open(results_file, 'w') as f:
            json.dump(err_result, f, indent=2)
        return err_result
    finally:
        manager.remove_hooks()


def _measure_latency(model, tokenizer, manager, num_runs=3):
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Warmup
    for _ in range(2):
        manager.reset_stats()
        if isinstance(manager.scorer, RollingLayerGatingScorer):
            manager.scorer.reset_cache()
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=5, do_sample=False)

    times = []
    for _ in range(num_runs):
        manager.reset_stats()
        if isinstance(manager.scorer, RollingLayerGatingScorer):
            manager.scorer.reset_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        tokens = out.shape[1] - inputs['input_ids'].shape[1]
        times.append((t1 - t0, tokens))

    avg_time = np.mean([t[0] for t in times])
    avg_tokens = np.mean([t[1] for t in times])
    return {
        "avg_time_sec": float(avg_time),
        "avg_tokens_per_sec": float(avg_tokens / avg_time) if avg_time > 0 else 0,
        "tokens_generated": int(avg_tokens),
    }


def _run_benchmarks(model, tokenizer, config: ExperimentConfig) -> Dict[str, float]:
    """Try lm-eval-harness first, fall back to manual."""
    try:
        return _run_harness(model, tokenizer, config)
    except Exception as e:
        logger.warning(f"  lm-eval failed ({e}), using manual eval")
        return _run_manual(model, tokenizer, config)


def _run_harness(model, tokenizer, config: ExperimentConfig) -> Dict[str, float]:
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    task_map = {
        "mmlu": "mmlu", "gsm8k": "gsm8k", "truthfulqa": "truthfulqa_mc2",
        "hellaswag": "hellaswag", "piqa": "piqa", "winogrande": "winogrande",
        "arc_challenge": "arc_challenge", "boolq": "boolq",
        "lambada": "lambada_openai", "sciq": "sciq", "logiqa": "logiqa",
    }

    tasks = [task_map[b] for b in config.benchmark.benchmarks if b in task_map]
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=config.benchmark.batch_size)

    eval_kwargs = {"model": lm, "tasks": tasks, "batch_size": config.benchmark.batch_size}
    if config.benchmark.max_samples:
        eval_kwargs["limit"] = config.benchmark.max_samples

    raw = lm_eval.simple_evaluate(**eval_kwargs)

    scores = {}
    for bench_name, task_name in task_map.items():
        if task_name in raw.get("results", {}):
            r = raw["results"][task_name]
            for key in ["acc,none", "acc_norm,none", "exact_match,none", "mc2,none",
                        "acc", "acc_norm", "exact_match", "mc2"]:
                if key in r:
                    scores[bench_name] = float(r[key])
                    break
    return scores


def _run_manual(model, tokenizer, config: ExperimentConfig) -> Dict[str, float]:
    from .manual_benchmarks import evaluate_all
    result = evaluate_all(model, tokenizer, config.benchmark)
    return result.get("scores", {})


def run_all_experiments(model, tokenizer, configs: List[ExperimentConfig]) -> List[Dict]:
    """Run all experiments sequentially."""
    results = []
    total = len(configs)
    for i, cfg in enumerate(configs):
        logger.info(f"\n[{i+1}/{total}] {'='*50}")
        try:
            r = run_single_experiment(model, tokenizer, cfg)
            results.append(r)
        except Exception as e:
            logger.error(f"  Experiment {i+1} failed: {e}")
            results.append({"config": asdict(cfg.pruning), "error": str(e)})
    return results

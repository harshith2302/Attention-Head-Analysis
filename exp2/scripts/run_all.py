#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════
  MASTER RUNNER — Runs ALL experiments + generates plots
  One script, one GPU, one day, all results.
═══════════════════════════════════════════════════════

Runs 100 experiment configs sequentially:
  - 6 methods × 3 warmups × 3 top-K values
  - + hybrid × 5 lambdas
  - + rolling × 2 chunk sizes
  - + 1 baseline
  - 11 benchmarks each, 200 samples per benchmark
  - Estimated time: ~16-20 hours on A40

Supports resume — if interrupted, re-run and it skips completed experiments.
"""
import os
import sys
import json
import time
import logging
import torch

# Project root
PROJECT_DIR = "/home/cccp/25m0834/RND5"
sys.path.insert(0, PROJECT_DIR)

from src.config import (
    ModelConfig, ExperimentConfig, PruningConfig, BenchmarkConfig, OutputConfig,
    generate_all_experiment_configs, PROJECT_ROOT,
)
from src.model_loader import load_model_and_tokenizer
from src.benchmark_runner import run_single_experiment


def setup_logging():
    log_file = os.path.join(PROJECT_DIR, "logs", "master_run.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ]
    )
    return logging.getLogger(__name__)


def main():
    logger = setup_logging()

    # GPU nodes have no internet
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    logger.info("═" * 60)
    logger.info("  MASTER RUNNER — Dynamic Head Pruning")
    logger.info("═" * 60)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name} ({gpu_mem:.0f}GB)")
    else:
        logger.warning("No GPU detected! This will be very slow.")

    # Set seed
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # ─── Load model ONCE ───
    logger.info("\n[1/3] Loading LLaMA3-8B...")
    t0 = time.time()

    model_config = ModelConfig(
        cache_dir=os.path.join(PROJECT_DIR, "cache", "models"),
        torch_dtype="bfloat16",
        attn_implementation="eager",
    )
    model, tokenizer = load_model_and_tokenizer(model_config, device="cuda", offline=True)
    logger.info(f"  Model loaded in {time.time()-t0:.0f}s")

    # ─── Generate all experiment configs ───
    configs = generate_all_experiment_configs(max_samples=200)
    logger.info(f"\n[2/3] Running {len(configs)} experiments (200 samples/benchmark, 11 benchmarks)")
    logger.info(f"  Estimated time: ~16-20 hours")
    logger.info(f"  Results dir: {os.path.join(PROJECT_DIR, 'results')}")

    # Count already completed
    results_dir = os.path.join(PROJECT_DIR, "results")
    done = sum(1 for cfg in configs if os.path.exists(
        os.path.join(results_dir, _config_dirname(cfg.pruning), "results.json")
    ))
    if done > 0:
        logger.info(f"  Resuming: {done}/{len(configs)} already completed")

    # ─── Run all experiments ───
    all_results = []
    t_start = time.time()

    for i, cfg in enumerate(configs):
        elapsed = time.time() - t_start
        if i > 0 and all_results:
            completed = sum(1 for r in all_results if r.get("scores"))
            if completed > 0:
                avg_per = elapsed / (i)
                remaining = avg_per * (len(configs) - i)
                eta_h = remaining / 3600
                logger.info(f"  ETA: {eta_h:.1f} hours remaining")

        logger.info(f"\n{'─'*60}")
        logger.info(f"  Experiment {i+1}/{len(configs)}")
        logger.info(f"{'─'*60}")

        result = run_single_experiment(model, tokenizer, cfg)
        all_results.append(result)

    total_time = time.time() - t_start
    logger.info(f"\n  All experiments done in {total_time/3600:.1f} hours")

    # Save combined results
    combined_file = os.path.join(results_dir, "all_results.json")
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # ─── Generate plots ───
    logger.info(f"\n[3/3] Generating plots...")
    try:
        sys.path.insert(0, os.path.join(PROJECT_DIR, "plotting"))
        from generate_plots import generate_all_plots
        generate_all_plots(results_dir)
        logger.info(f"  Plots saved to {os.path.join(results_dir, 'plots')}")
    except Exception as e:
        logger.error(f"  Plot generation failed: {e}")
        logger.info("  You can generate plots manually: python3 scripts/generate_plots.py")

    # ─── Print summary ───
    logger.info("\n" + "═" * 60)
    logger.info("  COMPLETE — SUMMARY")
    logger.info("═" * 60)

    successful = [r for r in all_results if r.get("scores") and not r.get("error")]
    failed = [r for r in all_results if r.get("error")]

    logger.info(f"  Successful: {len(successful)}/{len(configs)}")
    logger.info(f"  Failed:     {len(failed)}/{len(configs)}")
    logger.info(f"  Total time: {total_time/3600:.1f} hours")

    if successful:
        # Find best method
        best = max(successful, key=lambda r: r.get("average_score", 0))
        bc = best["config"]
        logger.info(f"\n  Best method: {bc['method']} (warmup={bc['warmup_layers']}, k={bc['top_k']})")
        logger.info(f"  Best average score: {best['average_score']*100:.2f}%")

        logger.info(f"\n  Benchmark scores for best method:")
        for bench, score in sorted(best.get("scores", {}).items()):
            logger.info(f"    {bench:20s}: {score*100:.2f}%")

    logger.info(f"\n  Results: {results_dir}")
    logger.info(f"  Plots:   {os.path.join(results_dir, 'plots')}")
    logger.info(f"  Summary: {os.path.join(results_dir, 'plots', 'results_summary.csv')}")
    logger.info("═" * 60)


def _config_dirname(p: PruningConfig) -> str:
    """Generate directory name matching OutputConfig.get_experiment_dir."""
    name = f"{p.method}_w{p.warmup_layers}_k{p.top_k}"
    if p.method == "hybrid_dynamic_routing":
        name += f"_l{p.lambda_val}"
    if p.method == "rolling_layer_gating":
        name += f"_c{p.rolling_chunk_size}"
    return name


if __name__ == "__main__":
    main()

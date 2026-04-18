"""
Visualization and Plotting for Dynamic Head Pruning Experiments.

Generates comparison plots across methods, configurations, and benchmarks.
"""
import os
import json
import glob
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not available. Plotting disabled.")


# ─── MoH baseline numbers from the paper (Table 4) ───
MOH_BASELINE = {
    "mmlu": 65.8, "ceval": 61.5, "cmmlu": 64.4, "gsm8k": 56.9,
    "truthfulqa": 44.0, "hellaswag": 80.1, "logiqa": 30.3,
    "boolq": 84.0, "lambada": 76.4, "sciq": 92.2,
    "piqa": 78.8, "winogrande": 72.9, "arc_challenge": 60.1,
    "average": 64.0,
}

LLAMA3_BASELINE = {
    "mmlu": 65.2, "ceval": 52.3, "cmmlu": 50.7, "gsm8k": 49.5,
    "truthfulqa": 35.4, "hellaswag": 81.9, "logiqa": 30.0,
    "boolq": 83.9, "lambada": 75.5, "sciq": 94.0,
    "piqa": 81.0, "winogrande": 72.5, "arc_challenge": 59.0,
    "average": 61.6,
}

# Friendly method names
METHOD_LABELS = {
    "query_norm_topk": "Query Norm Top-K",
    "last_token_query_norm": "Last Token QN",
    "qk_norm_product": "QK Norm Product",
    "token_saliency_query_norm": "Token Saliency + QN",
    "rolling_layer_gating": "Rolling Layer Gate",
    "hybrid_dynamic_routing": "Hybrid Dynamic",
    "none": "Baseline (No Pruning)",
}


def load_all_results(results_dir: str) -> List[Dict[str, Any]]:
    """Load all experiment results from the results directory."""
    results = []
    for results_file in glob.glob(os.path.join(results_dir, "**/results.json"), recursive=True):
        try:
            with open(results_file) as f:
                data = json.load(f)
            data["_file"] = results_file
            results.append(data)
        except Exception as e:
            logger.warning(f"Failed to load {results_file}: {e}")
    logger.info(f"Loaded {len(results)} experiment results")
    return results


def _get_method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def _savefig(fig, path: str, formats: List[str] = ["png", "pdf"]):
    """Save figure in multiple formats."""
    for fmt in formats:
        fpath = f"{path}.{fmt}"
        fig.savefig(fpath, dpi=150, bbox_inches='tight')
        logger.info(f"  Saved: {fpath}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════
# Plot 1: Method vs Average Accuracy
# ═══════════════════════════════════════════════════════
def plot_method_comparison(results: List[Dict], output_dir: str,
                           warmup: int = 8, top_k: int = 24):
    """Bar chart comparing average accuracy across methods."""
    if not HAS_MATPLOTLIB:
        return

    # Filter results for specific warmup/k
    filtered = {}
    for r in results:
        cfg = r.get("config", {})
        if cfg.get("warmup_layers") == warmup and cfg.get("top_k") == top_k:
            method = cfg.get("method", "unknown")
            avg = r.get("average_score", 0)
            if method not in filtered or avg > filtered[method]:
                filtered[method] = avg

    if not filtered:
        logger.warning("No results to plot for method comparison")
        return

    methods = list(filtered.keys())
    scores = [filtered[m] * 100 for m in methods]
    labels = [_get_method_label(m) for m in methods]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
    bars = ax.bar(range(len(methods)), scores, color=colors, edgecolor='black', linewidth=0.5)

    # Add baselines
    ax.axhline(y=MOH_BASELINE["average"], color='red', linestyle='--',
               label=f'MoH-LLaMA3-8B (75% heads): {MOH_BASELINE["average"]}%')
    ax.axhline(y=LLAMA3_BASELINE["average"], color='blue', linestyle='--',
               label=f'LLaMA3-8B (100% heads): {LLAMA3_BASELINE["average"]}%')

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel("Average Accuracy (%)")
    ax.set_title(f"Method Comparison (warmup={warmup}, K={top_k})")
    ax.legend(loc='lower right', fontsize=8)

    # Add value labels on bars
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{score:.1f}', ha='center', va='bottom', fontsize=8)

    ax.set_ylim(0, max(scores + [MOH_BASELINE["average"], LLAMA3_BASELINE["average"]]) * 1.1)

    _savefig(fig, os.path.join(output_dir, "method_comparison"))


# ═══════════════════════════════════════════════════════
# Plot 2: Benchmark-wise bar plots
# ═══════════════════════════════════════════════════════
def plot_benchmark_bars(results: List[Dict], output_dir: str,
                        warmup: int = 8, top_k: int = 24):
    """Per-benchmark accuracy bars for all methods."""
    if not HAS_MATPLOTLIB:
        return

    # Collect per-method per-benchmark scores
    method_scores = {}
    for r in results:
        cfg = r.get("config", {})
        if cfg.get("warmup_layers") == warmup and cfg.get("top_k") == top_k:
            method = cfg.get("method", "unknown")
            scores = r.get("scores", {})
            if scores and method not in method_scores:
                method_scores[method] = scores

    if not method_scores:
        return

    benchmarks = sorted(set().union(*[s.keys() for s in method_scores.values()]))
    methods = sorted(method_scores.keys())
    n_methods = len(methods)
    n_benchmarks = len(benchmarks)

    fig, ax = plt.subplots(figsize=(16, 8))
    width = 0.8 / (n_methods + 2)  # +2 for baselines
    x = np.arange(n_benchmarks)
    colors = plt.cm.tab10(np.linspace(0, 1, n_methods + 2))

    for i, method in enumerate(methods):
        vals = [method_scores[method].get(b, 0) * 100 for b in benchmarks]
        ax.bar(x + i * width, vals, width, label=_get_method_label(method),
               color=colors[i], edgecolor='black', linewidth=0.3)

    # Add MoH baseline
    moh_vals = [MOH_BASELINE.get(b, 0) for b in benchmarks]
    ax.bar(x + n_methods * width, moh_vals, width, label='MoH-LLaMA3-8B',
           color='red', alpha=0.5, edgecolor='black', linewidth=0.3)

    # Add LLaMA3 baseline
    llama_vals = [LLAMA3_BASELINE.get(b, 0) for b in benchmarks]
    ax.bar(x + (n_methods + 1) * width, llama_vals, width, label='LLaMA3-8B',
           color='blue', alpha=0.5, edgecolor='black', linewidth=0.3)

    ax.set_xticks(x + n_methods * width / 2)
    ax.set_xticklabels(benchmarks, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Per-Benchmark Comparison (warmup={warmup}, K={top_k})")
    ax.legend(loc='upper right', fontsize=7, ncol=2)

    _savefig(fig, os.path.join(output_dir, "benchmark_bars"))


# ═══════════════════════════════════════════════════════
# Plot 3: Warmup layer comparison
# ═══════════════════════════════════════════════════════
def plot_warmup_comparison(results: List[Dict], output_dir: str, top_k: int = 24):
    """Compare different warmup configurations."""
    if not HAS_MATPLOTLIB:
        return

    # Group by method and warmup
    data = defaultdict(dict)  # method -> {warmup -> avg_score}
    for r in results:
        cfg = r.get("config", {})
        if cfg.get("top_k") == top_k:
            method = cfg.get("method", "unknown")
            warmup = cfg.get("warmup_layers", 0)
            data[method][warmup] = r.get("average_score", 0) * 100

    if not data:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    warmups = sorted(set().union(*[d.keys() for d in data.values()]))
    colors = plt.cm.Set1(np.linspace(0, 1, len(data)))

    for i, (method, warmup_scores) in enumerate(sorted(data.items())):
        ws = sorted(warmup_scores.keys())
        scores = [warmup_scores[w] for w in ws]
        ax.plot(ws, scores, 'o-', label=_get_method_label(method),
                color=colors[i], markersize=8)

    ax.axhline(y=MOH_BASELINE["average"], color='red', linestyle='--',
               alpha=0.7, label='MoH-LLaMA3-8B')
    ax.set_xlabel("Warmup Layers")
    ax.set_ylabel("Average Accuracy (%)")
    ax.set_title(f"Effect of Warmup Layers (K={top_k})")
    ax.legend(fontsize=8)
    ax.set_xticks(warmups)

    _savefig(fig, os.path.join(output_dir, "warmup_comparison"))


# ═══════════════════════════════════════════════════════
# Plot 4: Top-K sensitivity
# ═══════════════════════════════════════════════════════
def plot_topk_sensitivity(results: List[Dict], output_dir: str, warmup: int = 8):
    """Show how accuracy changes with number of active heads."""
    if not HAS_MATPLOTLIB:
        return

    data = defaultdict(dict)
    for r in results:
        cfg = r.get("config", {})
        if cfg.get("warmup_layers") == warmup:
            method = cfg.get("method", "unknown")
            k = cfg.get("top_k", 32)
            data[method][k] = r.get("average_score", 0) * 100

    if not data:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set1(np.linspace(0, 1, len(data)))

    for i, (method, k_scores) in enumerate(sorted(data.items())):
        ks = sorted(k_scores.keys())
        scores = [k_scores[k] for k in ks]
        pcts = [k / 32 * 100 for k in ks]
        ax.plot(pcts, scores, 'o-', label=_get_method_label(method),
                color=colors[i], markersize=8)

    ax.axhline(y=MOH_BASELINE["average"], color='red', linestyle='--',
               alpha=0.7, label='MoH-LLaMA3-8B (75%)')
    ax.set_xlabel("Active Heads (%)")
    ax.set_ylabel("Average Accuracy (%)")
    ax.set_title(f"Top-K Sensitivity (warmup={warmup})")
    ax.legend(fontsize=8)

    _savefig(fig, os.path.join(output_dir, "topk_sensitivity"))


# ═══════════════════════════════════════════════════════
# Plot 5: Lambda comparison for hybrid method
# ═══════════════════════════════════════════════════════
def plot_lambda_comparison(results: List[Dict], output_dir: str,
                           warmup: int = 8, top_k: int = 24):
    """Show hybrid method performance vs lambda."""
    if not HAS_MATPLOTLIB:
        return

    data = {}
    for r in results:
        cfg = r.get("config", {})
        if (cfg.get("method") == "hybrid_dynamic_routing" and
            cfg.get("warmup_layers") == warmup and
            cfg.get("top_k") == top_k):
            lam = cfg.get("lambda_val", 0.5)
            data[lam] = r.get("average_score", 0) * 100

    if not data:
        return

    lambdas = sorted(data.keys())
    scores = [data[l] for l in lambdas]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(lambdas, scores, 'o-', color='purple', markersize=10, linewidth=2)

    ax.axhline(y=MOH_BASELINE["average"], color='red', linestyle='--',
               alpha=0.7, label='MoH-LLaMA3-8B')

    # Also plot per-benchmark if available
    for r in results:
        cfg = r.get("config", {})
        if (cfg.get("method") == "hybrid_dynamic_routing" and
            cfg.get("warmup_layers") == warmup and
            cfg.get("top_k") == top_k):
            break

    ax.set_xlabel("Lambda (λ)")
    ax.set_ylabel("Average Accuracy (%)")
    ax.set_title(f"Hybrid Method: Lambda Sensitivity (warmup={warmup}, K={top_k})")
    ax.legend(fontsize=9)

    # Highlight best lambda
    best_idx = np.argmax(scores)
    ax.annotate(f'Best: λ={lambdas[best_idx]}\n{scores[best_idx]:.1f}%',
                xy=(lambdas[best_idx], scores[best_idx]),
                xytext=(10, 10), textcoords='offset points',
                fontsize=9, arrowprops=dict(arrowstyle='->', color='black'))

    _savefig(fig, os.path.join(output_dir, "lambda_comparison"))


# ═══════════════════════════════════════════════════════
# Plot 6: Pareto plot (Speed vs Accuracy)
# ═══════════════════════════════════════════════════════
def plot_pareto(results: List[Dict], output_dir: str):
    """Pareto frontier: accuracy vs speed."""
    if not HAS_MATPLOTLIB:
        return

    methods_data = defaultdict(list)
    for r in results:
        cfg = r.get("config", {})
        method = cfg.get("method", "unknown")
        avg_score = r.get("average_score", 0) * 100
        latency = r.get("latency", {})
        tokens_per_sec = latency.get("avg_tokens_per_sec", 0)
        active_pct = r.get("active_head_pct", 100)

        if avg_score > 0:
            methods_data[method].append({
                "score": avg_score,
                "speed": tokens_per_sec,
                "active_pct": active_pct,
                "k": cfg.get("top_k", 32),
                "warmup": cfg.get("warmup_layers", 0),
            })

    if not methods_data:
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods_data)))

    for i, (method, points) in enumerate(sorted(methods_data.items())):
        speeds = [p["speed"] for p in points]
        scores = [p["score"] for p in points]
        ax.scatter(speeds, scores, label=_get_method_label(method),
                   color=colors[i], s=80, alpha=0.7, edgecolors='black', linewidth=0.5)

    ax.set_xlabel("Throughput (tokens/sec)")
    ax.set_ylabel("Average Accuracy (%)")
    ax.set_title("Speed vs Accuracy Pareto Plot")
    ax.legend(fontsize=8, loc='lower right')

    _savefig(fig, os.path.join(output_dir, "pareto_plot"))


# ═══════════════════════════════════════════════════════
# Generate all plots
# ═══════════════════════════════════════════════════════
def generate_all_plots(results_dir: str, output_dir: Optional[str] = None):
    """Generate all visualization plots from experiment results."""
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib required for plotting")
        return

    if output_dir is None:
        output_dir = os.path.join(results_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    results = load_all_results(results_dir)
    if not results:
        logger.warning("No results found to plot")
        return

    logger.info(f"Generating plots from {len(results)} results...")

    plot_method_comparison(results, output_dir)
    plot_benchmark_bars(results, output_dir)
    plot_warmup_comparison(results, output_dir)
    plot_topk_sensitivity(results, output_dir)
    plot_lambda_comparison(results, output_dir)
    plot_pareto(results, output_dir)

    # Generate summary CSV
    _generate_summary_csv(results, output_dir)

    logger.info(f"All plots saved to {output_dir}")


def _generate_summary_csv(results: List[Dict], output_dir: str):
    """Generate a summary CSV of all results."""
    import csv

    csv_path = os.path.join(output_dir, "results_summary.csv")
    all_benchmarks = set()
    for r in results:
        all_benchmarks.update(r.get("scores", {}).keys())
    all_benchmarks = sorted(all_benchmarks)

    fieldnames = [
        "method", "warmup_layers", "top_k", "lambda_val",
        "active_head_pct", "avg_score", "tokens_per_sec",
        "flops_reduction_pct",
    ] + all_benchmarks

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            cfg = r.get("config", {})
            row = {
                "method": cfg.get("method", ""),
                "warmup_layers": cfg.get("warmup_layers", ""),
                "top_k": cfg.get("top_k", ""),
                "lambda_val": cfg.get("lambda_val", ""),
                "active_head_pct": r.get("active_head_pct", ""),
                "avg_score": r.get("average_score", ""),
                "tokens_per_sec": r.get("latency", {}).get("avg_tokens_per_sec", ""),
                "flops_reduction_pct": r.get("estimated_flops_reduction_pct", ""),
            }
            for bench in all_benchmarks:
                row[bench] = r.get("scores", {}).get(bench, "")
            writer.writerow(row)

    logger.info(f"Summary CSV saved: {csv_path}")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "./results"
    generate_all_plots(results_dir)

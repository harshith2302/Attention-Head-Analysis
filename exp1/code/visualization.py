#!/usr/bin/env python3
"""
Visualization module for attention head importance analysis.

Generates:
1. Normalized heatmap (matching MoH paper style)
2. Per-task heatmaps
3. Layer importance bar chart
4. Shared head overlay visualization
5. Gini concentration plot
"""

import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for cluster
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import numpy as np
import seaborn as sns


def plot_normalized_heatmap(
    normalized: np.ndarray,
    output_path: str,
    title: str = "LLaMA3-8B | Per-Layer Normalized Q-Norm — Empirical Importance",
    subtitle: str = "(0 = least important head in layer, 1 = most important head in layer)",
    figsize: Tuple[int, int] = (14, 10),
):
    """
    Generate the primary normalized heatmap (matches MoH paper style).

    Args:
        normalized: shape (num_layers, num_heads), values in [0, 1]
        output_path: path to save the figure
    """
    num_layers, num_heads = normalized.shape

    fig, ax = plt.subplots(figsize=figsize)

    # Use viridis colormap (same as reference heatmap)
    im = ax.imshow(
        normalized,
        cmap="viridis",
        aspect="auto",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    # Axis labels
    ax.set_xlabel("Head Index", fontsize=13, fontweight="bold")
    ax.set_ylabel("Layer Index", fontsize=13, fontweight="bold")

    # Tick marks
    ax.set_xticks(range(0, num_heads, 2))
    ax.set_xticklabels(range(0, num_heads, 2), fontsize=9)
    ax.set_yticks(range(0, num_layers, 2))
    ax.set_yticklabels(range(0, num_layers, 2), fontsize=9)

    # Title
    ax.set_title(f"{title}\n{subtitle}", fontsize=13, pad=15)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_raw_importance_heatmap(
    importance: np.ndarray,
    output_path: str,
    title: str = "LLaMA3-8B | Raw Head Importance (Gradient-Weighted Q-Norm)",
):
    """Generate heatmap of raw (unnormalized) importance scores."""
    num_layers, num_heads = importance.shape

    fig, ax = plt.subplots(figsize=(14, 10))

    im = ax.imshow(
        importance,
        cmap="magma",
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_xlabel("Head Index", fontsize=13, fontweight="bold")
    ax.set_ylabel("Layer Index", fontsize=13, fontweight="bold")
    ax.set_xticks(range(0, num_heads, 2))
    ax.set_xticklabels(range(0, num_heads, 2), fontsize=9)
    ax.set_yticks(range(0, num_layers, 2))
    ax.set_yticklabels(range(0, num_layers, 2), fontsize=9)
    ax.set_title(title, fontsize=13, pad=10)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_shared_heads_overlay(
    normalized: np.ndarray,
    shared_heads: List[Tuple[int, int]],
    output_path: str,
    title: str = "LLaMA3-8B | Top 50% Shared Heads (highlighted)",
):
    """
    Heatmap with shared heads highlighted by red borders.
    """
    num_layers, num_heads = normalized.shape

    fig, ax = plt.subplots(figsize=(14, 10))

    im = ax.imshow(
        normalized,
        cmap="viridis",
        aspect="auto",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    # Highlight shared heads with red rectangles
    shared_set = set(shared_heads)
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            if (layer_idx, head_idx) in shared_set:
                rect = Rectangle(
                    (head_idx - 0.5, layer_idx - 0.5), 1, 1,
                    linewidth=1.5, edgecolor="red", facecolor="none",
                )
                ax.add_patch(rect)

    ax.set_xlabel("Head Index", fontsize=13, fontweight="bold")
    ax.set_ylabel("Layer Index", fontsize=13, fontweight="bold")
    ax.set_xticks(range(0, num_heads, 2))
    ax.set_xticklabels(range(0, num_heads, 2), fontsize=9)
    ax.set_yticks(range(0, num_layers, 2))
    ax.set_yticklabels(range(0, num_layers, 2), fontsize=9)
    ax.set_title(title, fontsize=13, pad=10)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_layer_importance_bar(
    stats: Dict,
    output_path: str,
    num_layers: int,
):
    """Bar chart of mean importance per layer with error bars."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    layers = range(num_layers)

    # Mean importance per layer
    ax = axes[0]
    ax.bar(layers, stats["layer_mean_importance"], color="steelblue", alpha=0.8)
    ax.errorbar(
        layers, stats["layer_mean_importance"],
        yerr=stats["layer_std_importance"],
        fmt="none", ecolor="gray", capsize=2,
    )
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Mean Importance")
    ax.set_title("Mean Head Importance per Layer")
    ax.set_xticks(range(0, num_layers, 4))

    # Shared heads count per layer
    ax = axes[1]
    ax.bar(layers, stats["top50_heads_per_layer"], color="coral", alpha=0.8)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("# Shared Heads (Top 50%)")
    ax.set_title("Shared Heads Distribution per Layer")
    ax.set_xticks(range(0, num_layers, 4))

    # Gini coefficient per layer
    ax = axes[2]
    ax.bar(layers, stats["gini_per_layer"], color="seagreen", alpha=0.8)
    ax.axhline(y=np.mean(stats["gini_per_layer"]), color="red",
               linestyle="--", label=f"Mean={np.mean(stats['gini_per_layer']):.3f}")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Gini Coefficient")
    ax.set_title("Importance Concentration (Gini) per Layer")
    ax.set_xticks(range(0, num_layers, 4))
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_per_task_heatmaps(
    per_task_importance: Dict[str, np.ndarray],
    output_path: str,
    num_layers: int,
    num_heads: int,
):
    """Generate a grid of per-task importance heatmaps."""
    tasks = sorted(per_task_importance.keys())
    n_tasks = len(tasks)
    if n_tasks == 0:
        return

    cols = min(3, n_tasks)
    rows = (n_tasks + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, task in enumerate(tasks):
        row, col = idx // cols, idx % cols
        ax = axes[row, col]

        imp = per_task_importance[task]
        # Per-layer normalize
        norm_imp = np.zeros_like(imp)
        for l in range(num_layers):
            rmin, rmax = imp[l].min(), imp[l].max()
            if rmax - rmin > 1e-12:
                norm_imp[l] = (imp[l] - rmin) / (rmax - rmin)
            else:
                norm_imp[l] = 0.5

        im = ax.imshow(norm_imp, cmap="viridis", aspect="auto", vmin=0, vmax=1)
        ax.set_title(task, fontsize=11, fontweight="bold")
        ax.set_xlabel("Head", fontsize=9)
        ax.set_ylabel("Layer", fontsize=9)
        ax.set_xticks(range(0, num_heads, 8))
        ax.set_yticks(range(0, num_layers, 8))

    # Hide unused subplots
    for idx in range(n_tasks, rows * cols):
        row, col = idx // cols, idx % cols
        axes[row, col].axis("off")

    fig.suptitle("Per-Task Head Importance (Normalized)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_ranking_curve(
    ranking: List[Tuple[int, int, float]],
    output_path: str,
):
    """Plot the importance score distribution (sorted)."""
    scores = [r[2] for r in ranking]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Sorted importance curve
    ax = axes[0]
    ax.plot(range(len(scores)), scores, color="steelblue", linewidth=1.5)
    ax.fill_between(range(len(scores)), scores, alpha=0.2, color="steelblue")

    # Mark percentile thresholds
    for pct, color, label in [(25, "red", "Top 25%"), (50, "orange", "Top 50%"), (75, "green", "Top 75%")]:
        idx = int(len(scores) * pct / 100)
        if idx < len(scores):
            ax.axvline(x=idx, color=color, linestyle="--", alpha=0.7, label=f"{label} ({idx} heads)")
            ax.axhline(y=scores[idx], color=color, linestyle=":", alpha=0.3)

    ax.set_xlabel("Head Rank (sorted by importance)")
    ax.set_ylabel("Importance Score")
    ax.set_title("Global Head Importance Distribution")
    ax.legend(fontsize=9)

    # Histogram
    ax = axes[1]
    ax.hist(scores, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Count")
    ax.set_title("Importance Score Histogram")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_moh_vs_empirical_comparison(
    normalized: np.ndarray,
    output_path: str,
    num_layers: int,
    num_heads: int,
    moh_shared_count: int = 16,  # MoH paper uses first 16 as shared
):
    """
    Compare MoH's static first-16 shared heads vs empirically derived shared heads.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # MoH static split
    ax = axes[0]
    moh_mask = np.zeros((num_layers, num_heads))
    moh_mask[:, :moh_shared_count] = 1  # First 16 always shared
    ax.imshow(moh_mask, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_title("MoH Paper: Static Shared Heads\n(First 16 heads per layer)", fontsize=12)
    ax.set_xlabel("Head Index")
    ax.set_ylabel("Layer Index")
    ax.set_xticks(range(0, num_heads, 4))
    ax.set_yticks(range(0, num_layers, 4))

    # Empirical importance
    ax = axes[1]
    # Threshold at median to create binary mask
    threshold = np.median(normalized)
    empirical_mask = (normalized >= threshold).astype(float)
    ax.imshow(empirical_mask, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_title("Empirical: Data-Driven Shared Heads\n(Above-median importance)", fontsize=12)
    ax.set_xlabel("Head Index")
    ax.set_ylabel("Layer Index")
    ax.set_xticks(range(0, num_heads, 4))
    ax.set_yticks(range(0, num_layers, 4))

    fig.suptitle("Static vs Empirical Shared Head Selection", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def generate_all_visualizations(
    normalized_heatmap: np.ndarray,
    raw_importance: np.ndarray,
    ranking: List[Tuple[int, int, float]],
    stats: Dict,
    shared_heads_50: List[Tuple[int, int]],
    per_task_importance: Dict[str, np.ndarray],
    output_dir: str,
    num_layers: int,
    num_heads: int,
):
    """Generate all visualization outputs."""
    os.makedirs(output_dir, exist_ok=True)

    print("\n--- Generating Visualizations ---")

    # 1. Primary normalized heatmap
    plot_normalized_heatmap(
        normalized_heatmap,
        os.path.join(output_dir, "head_importance_heatmap.png"),
    )

    # 2. Raw importance heatmap
    plot_raw_importance_heatmap(
        raw_importance,
        os.path.join(output_dir, "head_importance_raw_heatmap.png"),
    )

    # 3. Shared heads overlay
    plot_shared_heads_overlay(
        normalized_heatmap,
        shared_heads_50,
        os.path.join(output_dir, "shared_heads_overlay.png"),
    )

    # 4. Layer importance bars
    plot_layer_importance_bar(
        stats,
        os.path.join(output_dir, "layer_importance_analysis.png"),
        num_layers,
    )

    # 5. Per-task heatmaps
    if per_task_importance:
        plot_per_task_heatmaps(
            per_task_importance,
            os.path.join(output_dir, "per_task_heatmaps.png"),
            num_layers,
            num_heads,
        )

    # 6. Ranking curve
    plot_ranking_curve(
        ranking,
        os.path.join(output_dir, "importance_ranking_curve.png"),
    )

    # 7. MoH vs empirical comparison
    plot_moh_vs_empirical_comparison(
        normalized_heatmap,
        os.path.join(output_dir, "moh_vs_empirical_comparison.png"),
        num_layers,
        num_heads,
    )

    print(f"\n  All visualizations saved to {output_dir}/")
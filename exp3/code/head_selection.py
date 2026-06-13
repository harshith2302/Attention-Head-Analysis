#!/usr/bin/env python3
"""
Head Selection Utility.

Selects N shared heads (globally important) + M task-specific heads per layer.
Handles overlap: if a task-specific head is already in the shared set,
extends with the next-ranked task head to maintain the target count.

Usage:
    from head_selection import select_heads
    active_heads = select_heads(
        global_ranking_csv="results/global_ranking.csv",
        task_ranking_csv="task_rankings/task_ranking_mmlu.csv",
        n_shared_per_layer=16,
        m_task_per_layer=8,
        num_layers=32,
        num_heads=32,
    )
    # active_heads[layer_idx] = set of active head indices
"""

import csv
import numpy as np
from typing import Dict, Set, List, Tuple, Optional


def load_ranking_csv(csv_path: str) -> List[Tuple[int, int, float]]:
    """Load a ranking CSV file. Returns list of (layer, head, score) sorted by score desc."""
    entries = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append((
                int(row["layer_idx"]),
                int(row["head_idx"]),
                float(row["importance_score"]),
            ))
    # Sort by score descending
    entries.sort(key=lambda x: x[2], reverse=True)
    return entries


def get_per_layer_ranking(ranking: List[Tuple[int, int, float]], num_layers: int) -> Dict[int, List[int]]:
    """
    Convert a global ranking into per-layer ranked head lists.
    Returns {layer_idx: [head_idx sorted by importance desc]}.
    """
    layer_heads: Dict[int, List[Tuple[int, float]]] = {l: [] for l in range(num_layers)}
    for layer, head, score in ranking:
        layer_heads[layer].append((head, score))
    
    result = {}
    for layer in range(num_layers):
        # Sort heads within this layer by score descending
        layer_heads[layer].sort(key=lambda x: x[1], reverse=True)
        result[layer] = [h for h, _ in layer_heads[layer]]
    
    return result


def select_heads(
    global_ranking_csv: str,
    task_ranking_csv: Optional[str],
    n_shared_per_layer: int,
    m_task_per_layer: int,
    num_layers: int = 32,
    num_heads: int = 32,
) -> Dict[int, Set[int]]:
    """
    Select active heads for each layer.
    
    Strategy:
    1. For each layer, take top n_shared heads from global ranking → shared set
    2. For each layer, take top m_task heads from task-specific ranking
    3. If a task head is already in shared set → skip, take next ranked
    4. Final active = shared ∪ task-specific (exactly n_shared + m_task per layer, capped at num_heads)
    
    If task_ranking_csv is None, uses only shared heads (n_shared + m_task all from global).
    
    Args:
        global_ranking_csv: Path to global_ranking.csv
        task_ranking_csv: Path to task-specific ranking CSV (or None)
        n_shared_per_layer: Number of globally-important shared heads per layer
        m_task_per_layer: Number of task-specific heads per layer
        num_layers: Number of transformer layers
        num_heads: Number of heads per layer
    
    Returns:
        Dict mapping layer_idx → set of active head indices
    """
    # Load global ranking
    global_ranking = load_ranking_csv(global_ranking_csv)
    global_per_layer = get_per_layer_ranking(global_ranking, num_layers)
    
    # Load task-specific ranking
    if task_ranking_csv is not None:
        task_ranking = load_ranking_csv(task_ranking_csv)
        task_per_layer = get_per_layer_ranking(task_ranking, num_layers)
    else:
        task_per_layer = None
    
    active_heads = {}
    total_active = 0
    
    for layer in range(num_layers):
        # Step 1: Select shared heads (top n_shared from global ranking for this layer)
        shared = set()
        for head in global_per_layer[layer][:n_shared_per_layer]:
            shared.add(head)
        
        # Step 2: Select task-specific heads (top m_task from task ranking, skipping overlap)
        task_specific = set()
        if task_per_layer is not None and m_task_per_layer > 0:
            for head in task_per_layer[layer]:
                if len(task_specific) >= m_task_per_layer:
                    break
                if head not in shared:  # Skip overlap — extend to next
                    task_specific.add(head)
        elif m_task_per_layer > 0:
            # No task ranking — fill from global ranking (after shared)
            for head in global_per_layer[layer][n_shared_per_layer:]:
                if len(task_specific) >= m_task_per_layer:
                    break
                task_specific.add(head)
        
        # Combine
        active = shared | task_specific
        # Cap at num_heads
        if len(active) > num_heads:
            active = set(list(active)[:num_heads])
        
        active_heads[layer] = active
        total_active += len(active)
    
    total_possible = num_layers * num_heads
    pct = 100.0 * total_active / total_possible
    print(f"Head selection: {total_active}/{total_possible} heads active ({pct:.1f}%)")
    print(f"  Shared per layer: {n_shared_per_layer}, Task-specific per layer: {m_task_per_layer}")
    print(f"  Effective per layer: {[len(active_heads[l]) for l in range(num_layers)]}")
    
    return active_heads


def select_heads_global_budget(
    global_ranking_csv: str,
    task_ranking_csv: Optional[str],
    total_shared: int,
    total_task: int,
    num_layers: int = 32,
    num_heads: int = 32,
) -> Dict[int, Set[int]]:
    """
    Alternative: select heads with a GLOBAL budget (not per-layer).
    Distributes unevenly across layers based on actual importance.
    
    Args:
        total_shared: Total number of shared heads across all layers
        total_task: Total number of task-specific heads across all layers
    """
    global_ranking = load_ranking_csv(global_ranking_csv)
    
    # Shared: top total_shared from global ranking
    shared_heads: Dict[int, Set[int]] = {l: set() for l in range(num_layers)}
    for layer, head, score in global_ranking[:total_shared]:
        shared_heads[layer].add(head)
    
    # Task-specific: top total_task from task ranking, skipping overlap
    task_heads: Dict[int, Set[int]] = {l: set() for l in range(num_layers)}
    if task_ranking_csv is not None:
        task_ranking = load_ranking_csv(task_ranking_csv)
        added = 0
        for layer, head, score in task_ranking:
            if added >= total_task:
                break
            if head not in shared_heads[layer]:
                task_heads[layer].add(head)
                added += 1
    
    # Combine
    active_heads = {}
    total_active = 0
    for layer in range(num_layers):
        active_heads[layer] = shared_heads[layer] | task_heads[layer]
        total_active += len(active_heads[layer])
    
    total_possible = num_layers * num_heads
    pct = 100.0 * total_active / total_possible
    print(f"Head selection (global budget): {total_active}/{total_possible} heads active ({pct:.1f}%)")
    
    return active_heads


def print_head_map(active_heads: Dict[int, Set[int]], num_layers: int = 32, num_heads: int = 32):
    """Print a visual map of active heads."""
    print("\nHead activation map (■ = active, □ = inactive):")
    print(f"{'':>8}", end="")
    for h in range(num_heads):
        if h % 4 == 0:
            print(f"{h:>3}", end="")
        else:
            print("   ", end="")
    print()
    
    for layer in range(num_layers):
        active = active_heads.get(layer, set())
        count = len(active)
        print(f"L{layer:>2} [{count:>2}] ", end="")
        for h in range(num_heads):
            print("■ " if h in active else "□ ", end="")
        print()
def select_heads_standard(global_csv, task_csv, n_shared, m_task, nl, nh):
    """Wrapper for standard strategy A across all layers."""
    return select_heads(global_csv, task_csv, n_shared, m_task, nl, nh)

def select_heads_free_layers(global_csv, task_csv, free_layers, n_shared, m_task, nl, nh):
    """Wrapper for strategy B: first X layers are fully active."""
    # First, calculate the active heads as if it was a standard run
    active_heads = select_heads(global_csv, task_csv, n_shared, m_task, nl, nh)
    
    # Then, override the first 'free_layers' to keep all heads fully active
    for layer in range(free_layers):
        active_heads[layer] = set(range(nh))
        
    return active_heads

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Head selection utility")
    parser.add_argument("--global_ranking", type=str, required=True)
    parser.add_argument("--task_ranking", type=str, default=None)
    parser.add_argument("--n_shared", type=int, default=16)
    parser.add_argument("--m_task", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=32)
    parser.add_argument("--num_heads", type=int, default=32)
    parser.add_argument("--show_map", action="store_true")
    args = parser.parse_args()
    
    active = select_heads(
        global_ranking_csv=args.global_ranking,
        task_ranking_csv=args.task_ranking,
        n_shared_per_layer=args.n_shared,
        m_task_per_layer=args.m_task,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    )
    
    if args.show_map:
        print_head_map(active, args.num_layers, args.num_heads)

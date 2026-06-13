#!/usr/bin/env python3
"""
Head-Masked Evaluation of LLaMA3-8B.

Two strategies:
  A) Standard: N shared + M task-specific across all layers
  B) Free-layers: First X layers fully active, rest use N shared + M task

Evaluates via lm-evaluation-harness and compares with MoH paper results.

Usage:
  # Strategy A: 16 shared + 8 task = 24/32 (75%)
  python head_masked_eval.py --model_path ./model \
      --global_ranking ./results/global_ranking.csv \
      --task_rankings_dir ./task_rankings \
      --n_shared 16 --m_task 8

  # Strategy B: first 8 layers free, rest use 12+4
  python head_masked_eval.py --model_path ./model \
      --global_ranking ./results/global_ranking.csv \
      --task_rankings_dir ./task_rankings \
      --n_shared 12 --m_task 4 --free_layers 8
"""

import os, sys, json, argparse, torch, csv
import numpy as np
from typing import Dict, Set, List
from datetime import datetime
from head_selection import select_heads_standard, select_heads_free_layers


class HeadMasker:
    """Zeros out deactivated heads via pre-forward hooks on o_proj."""

    def __init__(self, model, active_heads, num_layers=32, num_heads=32, head_dim=128):
        self.model = model
        self.active = active_heads
        self.nl, self.nh, self.hd = num_layers, num_heads, head_dim
        self.hooks = []
        self._apply()

    def _apply(self):
        for l in range(self.nl):
            try:
                o_proj = self.model.model.layers[l].self_attn.o_proj
            except:
                continue
            inactive = set(range(self.nh)) - self.active.get(l, set(range(self.nh)))
            if not inactive:
                continue
            slices = [(h * self.hd, (h + 1) * self.hd) for h in sorted(inactive)]
            self.hooks.append(o_proj.register_forward_pre_hook(self._hook(slices)))
        n_masked = sum(self.nh - len(self.active.get(l, set(range(self.nh)))) for l in range(self.nl))
        print(f"  Masker: {n_masked} heads masked, {self.nl*self.nh - n_masked} active")

    def _hook(self, slices):
        def fn(mod, inp):
            x = inp[0]
            for s, e in slices:
                x[:, :, s:e] = 0.0
            return (x,)
        return fn

    def update(self, new_active):
        self.remove()
        self.active = new_active
        self._apply()

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# Task configs matching MoH paper Table 4
TASKS = {
    "mmlu":              {"lm_eval": "mmlu",            "fewshot": 5,  "metric": "acc"},
    "hellaswag":         {"lm_eval": "hellaswag",       "fewshot": 10, "metric": "acc_norm"},
    "arc_challenge":     {"lm_eval": "arc_challenge",   "fewshot": 25, "metric": "acc_norm"},
    "boolq":             {"lm_eval": "boolq",           "fewshot": 32, "metric": "acc"},
    "piqa":              {"lm_eval": "piqa",            "fewshot": 0,  "metric": "acc"},
    "winogrande":        {"lm_eval": "winogrande",      "fewshot": 0,  "metric": "acc"},
    "lambada":           {"lm_eval": "lambada_openai",  "fewshot": 0,  "metric": "acc"},
    "sciq":              {"lm_eval": "sciq",            "fewshot": 0,  "metric": "acc"},
    "truthfulqa":        {"lm_eval": "truthfulqa_mc1",  "fewshot": 0,  "metric": "acc"},
    "logiqa":            {"lm_eval": "logiqa",          "fewshot": 0,  "metric": "acc_norm"},
    "gsm8k":             {"lm_eval": "gsm8k",          "fewshot": 8,  "metric": "exact_match,strict-match"},
    "natural_questions": {"lm_eval": "nq_open",         "fewshot": 32, "metric": "exact_match"},
}

# MoH paper Table 4 — MoH-LLaMA3-8B (75% heads, static first-16 shared)
MOH = {"mmlu": 65.8, "hellaswag": 80.1, "arc_challenge": 60.1, "boolq": 84.0,
       "piqa": 78.8, "winogrande": 72.9, "lambada": 76.4, "sciq": 92.2,
       "truthfulqa": 44.0, "logiqa": 30.3, "gsm8k": 56.9, "natural_questions": 28.3}

# LLaMA3-8B full model (100% heads)
FULL = {"mmlu": 65.2, "hellaswag": 81.9, "arc_challenge": 59.0, "boolq": 83.9,
        "piqa": 81.0, "winogrande": 72.5, "lambada": 75.5, "sciq": 94.0,
        "truthfulqa": 35.4, "logiqa": 30.0, "gsm8k": 49.5, "natural_questions": 31.5}


def extract_score(result_dict, metric_key):
    """Try multiple metric key formats."""
    base = metric_key.split(",")[0]
    for suffix in ["", ",none", ",flexible-extract", ",strict-match"]:
        k = base + suffix
        if k in result_dict:
            v = result_dict[k]
            return v * 100 if isinstance(v, float) and v <= 1.0 else v
    for k, v in result_dict.items():
        if isinstance(v, (int, float)) and "stderr" not in k and v > 0:
            return v * 100 if v <= 1.0 else v
    return None


def run_eval(
    model_path, global_csv, task_dir, n_shared, m_task,
    output_dir, free_layers=0, tasks=None, batch_size=4, device="cuda",
):
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(output_dir, exist_ok=True)
    if tasks is None:
        tasks = list(TASKS.keys())

    strategy = "B" if free_layers > 0 else "A"
    tag = f"s{n_shared}_t{m_task}" + (f"_f{free_layers}" if free_layers > 0 else "")

    print(f"\n{'='*70}")
    print(f"HEAD-MASKED EVALUATION — Strategy {strategy}")
    if free_layers > 0:
        print(f"  Free layers: 0-{free_layers-1} (all 32 heads)")
        print(f"  Masked layers: {free_layers}-31 ({n_shared} shared + {m_task} task)")
    else:
        print(f"  All layers: {n_shared} shared + {m_task} task = {n_shared+m_task}/32")
    print(f"{'='*70}")

    print(f"\nLoading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map=device, trust_remote_code=True)
    model.eval()

    cfg = model.config
    nl = cfg.num_hidden_layers
    nh = cfg.num_attention_heads
    hd = cfg.hidden_size // nh

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    masker = HeadMasker(model, {l: set(range(nh)) for l in range(nl)}, nl, nh, hd)

    results = {}

    for task_name in tasks:
        if task_name not in TASKS:
            continue
        tcfg = TASKS[task_name]

        # Find task-specific ranking
        task_csv = os.path.join(task_dir, f"task_ranking_{task_name}.csv")
        if not os.path.exists(task_csv):
            print(f"\n  [WARN] No task ranking for {task_name}, using global only")
            task_csv = None

        print(f"\n{'─'*60}")
        print(f"Task: {task_name}")

        # Select heads based on strategy
        if free_layers > 0:
            active = select_heads_free_layers(
                global_csv, task_csv, free_layers, n_shared, m_task, nl, nh)
        else:
            active = select_heads_standard(
                global_csv, task_csv, n_shared, m_task, nl, nh)

        masker.update(active)

        try:
            ev = lm_eval.simple_evaluate(
                model=lm, tasks=[tcfg["lm_eval"]],
                num_fewshot=tcfg["fewshot"], batch_size=batch_size)

            task_key = tcfg["lm_eval"]
            if task_key in ev["results"]:
                score = extract_score(ev["results"][task_key], tcfg["metric"])
                if score is not None:
                    moh_s = MOH.get(task_name, 0)
                    full_s = FULL.get(task_name, 0)
                    results[task_name] = {
                        "score": round(score, 2),
                        "moh": moh_s, "full": full_s,
                        "vs_moh": round(score - moh_s, 2),
                        "vs_full": round(score - full_s, 2),
                    }
                    print(f"  >> {score:.1f}%  |  MoH: {moh_s}% ({score-moh_s:+.1f})  |  Full: {full_s}% ({score-full_s:+.1f})")
                else:
                    print(f"  >> Could not extract metric")
                    results[task_name] = {"error": "metric extraction failed"}
        except Exception as e:
            print(f"  >> FAILED: {e}")
            results[task_name] = {"error": str(e)}

    masker.remove()

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"RESULTS: Strategy {strategy} | {tag}")
    print(f"{'='*80}")
    print(f"{'Task':<20} {'Full(100%)':<12} {'MoH(75%)':<12} {'Ours':<12} {'vs MoH':<10} {'vs Full':<10}")
    print("─" * 80)

    s_ours, s_moh, s_full = [], [], []
    for t in tasks:
        if t in results and "score" in results[t]:
            o = results[t]["score"]
            m = results[t]["moh"]
            f = results[t]["full"]
            s_ours.append(o); s_moh.append(m); s_full.append(f)
            print(f"{t:<20} {f:<12.1f} {m:<12.1f} {o:<12.1f} {o-m:+10.1f} {o-f:+10.1f}")

    if s_ours:
        ao, am, af = np.mean(s_ours), np.mean(s_moh), np.mean(s_full)
        print("─" * 80)
        print(f"{'AVERAGE':<20} {af:<12.1f} {am:<12.1f} {ao:<12.1f} {ao-am:+10.1f} {ao-af:+10.1f}")

    # Save results
    out = {
        "config": {
            "strategy": strategy,
            "n_shared": n_shared, "m_task": m_task,
            "free_layers": free_layers,
            "heads_per_masked_layer": n_shared + m_task,
            "tag": tag,
            "timestamp": datetime.now().isoformat(),
        },
        "per_task": results,
        "summary": {
            "our_avg": round(np.mean(s_ours), 2) if s_ours else None,
            "moh_avg": round(np.mean(s_moh), 2) if s_moh else None,
            "full_avg": round(np.mean(s_full), 2) if s_full else None,
            "tasks_evaluated": len(s_ours),
        },
    }
    path = os.path.join(output_dir, f"eval_{tag}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {path}")

    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Head-masked LLaMA3-8B evaluation")
    p.add_argument("--model_path", required=True)
    p.add_argument("--global_ranking", required=True,
                    help="Path to global_ranking.csv")
    p.add_argument("--task_rankings_dir", required=True,
                    help="Directory with task_ranking_*.csv files")
    p.add_argument("--n_shared", type=int, required=True,
                    help="Shared heads per layer")
    p.add_argument("--m_task", type=int, required=True,
                    help="Task-specific heads per layer")
    p.add_argument("--free_layers", type=int, default=0,
                    help="Strategy B: first X layers fully active (0 = Strategy A)")
    p.add_argument("--output_dir", default="./eval_results")
    p.add_argument("--tasks", nargs="*", default=None)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    run_eval(
        a.model_path, a.global_ranking, a.task_rankings_dir,
        a.n_shared, a.m_task, a.output_dir,
        free_layers=a.free_layers, tasks=a.tasks,
        batch_size=a.batch_size, device=a.device,
    )
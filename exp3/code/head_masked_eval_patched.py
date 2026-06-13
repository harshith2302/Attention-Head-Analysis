#!/usr/bin/env python3
"""
Head-Masked Evaluation of LLaMA3-8B — patched version.

Fixes from previous run:
  1. trust_remote_code=True for logiqa, gsm8k (and others that need it)
  2. Workaround for "must be called with a dataclass type" error on
     hellaswag/arc_challenge/truthfulqa via task config override
"""

import os, sys, json, argparse, torch, csv
import numpy as np
from typing import Dict, Set, List
from datetime import datetime
from head_selection import select_heads_standard, select_heads_free_layers


class HeadMasker:
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
    "gsm8k":             {"lm_eval": "gsm8k",           "fewshot": 8,  "metric": "exact_match,strict-match"},
    "natural_questions": {"lm_eval": "nq_open",         "fewshot": 32, "metric": "exact_match"},
}

MOH = {"mmlu": 65.8, "hellaswag": 80.1, "arc_challenge": 60.1, "boolq": 84.0,
       "piqa": 78.8, "winogrande": 72.9, "lambada": 76.4, "sciq": 92.2,
       "truthfulqa": 44.0, "logiqa": 30.3, "gsm8k": 56.9, "natural_questions": 28.3}

FULL = {"mmlu": 65.2, "hellaswag": 81.9, "arc_challenge": 59.0, "boolq": 83.9,
        "piqa": 81.0, "winogrande": 72.5, "lambada": 75.5, "sciq": 94.0,
        "truthfulqa": 35.4, "logiqa": 30.0, "gsm8k": 49.5, "natural_questions": 31.5}


def extract_score(result_dict, metric_key):
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


def run_single_task(lm, task_cfg, batch_size):
    """Run lm_eval with all the workarounds for known issues."""
    import lm_eval

    # Patch 1: enable trust_remote_code globally for datasets that need it
    os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"

    # Patch 2: try modern API first, fall back to older API
    try:
        # Newer lm-eval: pass trust_remote_code via simple_evaluate kwargs
        return lm_eval.simple_evaluate(
            model=lm,
            tasks=[task_cfg["lm_eval"]],
            num_fewshot=task_cfg["fewshot"],
            batch_size=batch_size,
            confirm_run_unsafe_code=True,  # for newer lm-eval
        )
    except TypeError:
        # Older lm-eval doesn't have confirm_run_unsafe_code
        return lm_eval.simple_evaluate(
            model=lm,
            tasks=[task_cfg["lm_eval"]],
            num_fewshot=task_cfg["fewshot"],
            batch_size=batch_size,
        )


def run_eval(
    model_path, global_csv, task_dir, n_shared, m_task,
    output_dir, free_layers=0, tasks=None, batch_size=4, device="cuda",
):
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Set BEFORE any datasets imports
    os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
    os.environ["TRUST_REMOTE_CODE"] = "1"

    os.makedirs(output_dir, exist_ok=True)
    if tasks is None:
        tasks = list(TASKS.keys())

    strategy = "B" if free_layers > 0 else "A"
    tag = f"s{n_shared}_t{m_task}" + (f"_f{free_layers}" if free_layers > 0 else "")

    print(f"\n{'='*70}")
    print(f"HEAD-MASKED EVALUATION — Strategy {strategy} | tag={tag}")
    if free_layers > 0:
        print(f"  Free layers: 0-{free_layers-1} (all 32 heads)")
        print(f"  Masked layers: {free_layers}-31 ({n_shared} shared + {m_task} task)")
        total_active = free_layers * 32 + (32 - free_layers) * (n_shared + m_task)
        print(f"  Total active: {total_active}/1024 ({100*total_active/1024:.1f}%)")
    else:
        total_active = 32 * (n_shared + m_task)
        print(f"  All layers: {n_shared} shared + {m_task} task = {n_shared+m_task}/32")
        print(f"  Total active: {total_active}/1024 ({100*total_active/1024:.1f}%)")
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

        task_csv = os.path.join(task_dir, f"task_ranking_{task_name}.csv")
        if not os.path.exists(task_csv):
            print(f"\n  [WARN] No task ranking for {task_name}, using global only")
            task_csv = None

        print(f"\n{'─'*60}")
        print(f"Task: {task_name}")

        if free_layers > 0:
            active = select_heads_free_layers(
                global_csv, task_csv, free_layers, n_shared, m_task, nl, nh)
        else:
            active = select_heads_standard(
                global_csv, task_csv, n_shared, m_task, nl, nh)

        masker.update(active)

        try:
            ev = run_single_task(lm, tcfg, batch_size)

            task_key = tcfg["lm_eval"]
            # Some tasks (mmlu) return aggregated results under multiple keys
            score = None
            if task_key in ev["results"]:
                score = extract_score(ev["results"][task_key], tcfg["metric"])
            else:
                # mmlu may report subtasks — average them
                subtask_scores = []
                for k, v in ev["results"].items():
                    if k.startswith(task_key) and isinstance(v, dict):
                        s = extract_score(v, tcfg["metric"])
                        if s is not None:
                            subtask_scores.append(s)
                if subtask_scores:
                    score = sum(subtask_scores) / len(subtask_scores)

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
                print(f"  >> Could not extract metric. Raw: {list(ev['results'].keys())[:3]}")
                results[task_name] = {"error": "metric extraction failed",
                                       "raw_keys": list(ev["results"].keys())[:5]}
        except Exception as e:
            print(f"  >> FAILED: {type(e).__name__}: {str(e)[:200]}")
            results[task_name] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}

    masker.remove()

    print(f"\n{'='*80}")
    print(f"RESULTS: Strategy {strategy} | {tag}")
    print(f"{'='*80}")
    print(f"{'Task':<20} {'Full(100%)':<12} {'MoH(75%)':<12} {'Ours':<12} {'vs MoH':<10} {'vs Full':<10}")
    print("─" * 80)

    s_ours, s_moh, s_full = [], [], []
    wins_vs_moh = 0
    for t in tasks:
        if t in results and "score" in results[t]:
            o = results[t]["score"]
            m = results[t]["moh"]
            f = results[t]["full"]
            s_ours.append(o); s_moh.append(m); s_full.append(f)
            marker = "  ★" if o >= m else ""
            if o >= m:
                wins_vs_moh += 1
            print(f"{t:<20} {f:<12.1f} {m:<12.1f} {o:<12.1f} {o-m:+10.1f} {o-f:+10.1f}{marker}")

    if s_ours:
        ao, am, af = np.mean(s_ours), np.mean(s_moh), np.mean(s_full)
        print("─" * 80)
        print(f"{'AVERAGE':<20} {af:<12.1f} {am:<12.1f} {ao:<12.1f} {ao-am:+10.1f} {ao-af:+10.1f}")
        print(f"\n  Wins vs MoH: {wins_vs_moh}/{len(s_ours)} tasks  ★")

    out = {
        "config": {
            "strategy": strategy, "n_shared": n_shared, "m_task": m_task,
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
            "wins_vs_moh": wins_vs_moh,
        },
    }
    path = os.path.join(output_dir, f"eval_{tag}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {path}")

    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--global_ranking", required=True)
    p.add_argument("--task_rankings_dir", required=True)
    p.add_argument("--n_shared", type=int, required=True)
    p.add_argument("--m_task", type=int, required=True)
    p.add_argument("--free_layers", type=int, default=0)
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
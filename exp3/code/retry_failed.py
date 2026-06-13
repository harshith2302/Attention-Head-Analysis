#!/usr/bin/env python3
"""
Recovery Script: Retries only the failed tasks in existing JSON evaluation files
and merges the results safely without overwriting successful runs.
"""

import os, sys, json, argparse, torch, glob
import numpy as np
from datetime import datetime
from head_selection import select_heads_standard, select_heads_free_layers

# ─── SAME LOGIC AS head_masked_eval_patched.py ───
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
    import lm_eval
    os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
    try:
        return lm_eval.simple_evaluate(
            model=lm, tasks=[task_cfg["lm_eval"]], num_fewshot=task_cfg["fewshot"],
            batch_size=batch_size, confirm_run_unsafe_code=True
        )
    except TypeError:
        return lm_eval.simple_evaluate(
            model=lm, tasks=[task_cfg["lm_eval"]], num_fewshot=task_cfg["fewshot"],
            batch_size=batch_size
        )

def process_failed_json(json_path, lm, masker, global_csv, task_dir, batch_size):
    with open(json_path, "r") as f:
        data = json.load(f)
    
    cfg = data["config"]
    results = data["per_task"]
    
    # Identify tasks that failed (have 'error' key)
    failed_tasks = [task for task, res in results.items() if "error" in res]
    if not failed_tasks:
        print(f"✅ [{cfg['tag']}] No errors found. Skipping.")
        return False

    print(f"\n{'='*70}")
    print(f"🔄 RETRYING FAILED TASKS FOR: {cfg['tag']}")
    print(f"Failed tasks: {failed_tasks}")
    
    nl, nh = masker.nl, masker.nh
    n_shared, m_task = cfg.get("n_shared"), cfg.get("m_task")
    free_layers = cfg.get("free_layers", 0)

    for task_name in failed_tasks:
        tcfg = TASKS[task_name]
        task_csv = os.path.join(task_dir, f"task_ranking_{task_name}.csv")
        if not os.path.exists(task_csv):
            task_csv = None

        print(f"\n  ➤ Running: {task_name} ...")

        # 1. Select the exact heads for this specific task
        if free_layers > 0:
            active = select_heads_free_layers(global_csv, task_csv, free_layers, n_shared, m_task, nl, nh)
        else:
            active = select_heads_standard(global_csv, task_csv, n_shared, m_task, nl, nh)

        # 2. Lock them in
        masker.update(active)

        # 3. Evaluate
        try:
            ev = run_single_task(lm, tcfg, batch_size)
            task_key = tcfg["lm_eval"]
            
            score = None
            if task_key in ev["results"]:
                score = extract_score(ev["results"][task_key], tcfg["metric"])
            else:
                subtask_scores = [extract_score(v, tcfg["metric"]) for k, v in ev["results"].items() if k.startswith(task_key) and isinstance(v, dict)]
                subtask_scores = [s for s in subtask_scores if s is not None]
                if subtask_scores:
                    score = sum(subtask_scores) / len(subtask_scores)

            if score is not None:
                moh_s = MOH.get(task_name, 0)
                full_s = FULL.get(task_name, 0)
                
                # Overwrite the error with the successful score!
                results[task_name] = {
                    "score": round(score, 2),
                    "moh": moh_s, "full": full_s,
                    "vs_moh": round(score - moh_s, 2),
                    "vs_full": round(score - full_s, 2),
                }
                print(f"    ✔ Success! {score:.1f}%")
            else:
                print("    ✖ Still couldn't extract metric.")
        except Exception as e:
            print(f"    ✖ FAILED AGAIN: {type(e).__name__}: {str(e)[:100]}")

    # Re-calculate averages cleanly
    s_ours, s_moh, s_full = [], [], []
    wins = 0
    for t, res in results.items():
        if "score" in res:
            s_ours.append(res["score"])
            s_moh.append(res["moh"])
            s_full.append(res["full"])
            if res["score"] >= res["moh"]:
                wins += 1

    data["summary"] = {
        "our_avg": round(np.mean(s_ours), 2) if s_ours else None,
        "moh_avg": round(np.mean(s_moh), 2) if s_moh else None,
        "full_avg": round(np.mean(s_full), 2) if s_full else None,
        "tasks_evaluated": len(s_ours),
        "wins_vs_moh": wins
    }

    # Save safely over the old file
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"💾 Saved fixed JSON: {json_path}")
    return True

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--global_ranking", required=True)
    p.add_argument("--task_rankings_dir", required=True)
    p.add_argument("--json_dir", required=True, help="Directory containing the JSON files to fix")
    p.add_argument("--batch_size", type=int, default=4)
    a = p.parse_args()

    os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
    os.environ["TRUST_REMOTE_CODE"] = "1"

    # Find all JSONs
    json_files = glob.glob(os.path.join(a.json_dir, "eval_*.json"))
    if not json_files:
        print(f"No JSON files found in {a.json_dir}")
        return

    print("Loading model ONCE for all retries...")
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(a.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model_path, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    model.eval()

    cfg = model.config
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=a.batch_size)
    masker = HeadMasker(model, {l: set(range(cfg.num_attention_heads)) for l in range(cfg.num_hidden_layers)}, cfg.num_hidden_layers, cfg.num_attention_heads, cfg.hidden_size // cfg.num_attention_heads)

    for jpath in json_files:
        process_failed_json(jpath, lm, masker, a.global_ranking, a.task_rankings_dir, a.batch_size)
        
    masker.remove()
    print("\n🎉 All retry operations complete!")

if __name__ == "__main__":
    main()
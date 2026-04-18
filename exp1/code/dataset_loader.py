#!/usr/bin/env python3
"""
Dataset loader for MoH head importance analysis — FULL evaluation splits.
Loads ALL samples from each of the 14 MoH benchmarks (Table 4 of the paper).

Total samples: ~57,000 across 14 tasks.

Datasets (matching MoH paper Table 4 exactly):
  0-shot: SciQ, PIQA, WinoGrande, LogiQA, TruthfulQA, LAMBADA
  5-shot: MMLU, CEVAL, CMMLU
  8-shot: GSM8K
  10-shot: HellaSwag
  25-shot: ARC-Challenge
  32-shot: BoolQ, Natural Questions
"""

import os
import random
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


DATASET_REGISTRY = {
    "sciq": {
        "hf_id": "allenai/sciq", "subset": None,
        "split": "test", "n_shot": 0,
    },
    "piqa": {
        "hf_id": "ybisk/piqa", "subset": None,
        "split": "validation", "n_shot": 0,
    },
    "winogrande": {
        "hf_id": "allenai/winogrande", "subset": "winogrande_xl",
        "split": "validation", "n_shot": 0,
    },
    "logiqa": {
        "hf_id": "lucasmccabe/logiqa", "subset": None,
        "split": "test", "n_shot": 0,
    },
    "truthfulqa": {
        "hf_id": "truthfulqa/truthful_qa", "subset": "multiple_choice",
        "split": "validation", "n_shot": 0,
    },
    "lambada": {
        "hf_id": "cimec/lambada", "subset": None,
        "split": "test", "n_shot": 0,
    },
    "hellaswag": {
        "hf_id": "Rowan/hellaswag", "subset": None,
        "split": "validation", "n_shot": 10,
    },
    "mmlu": {
        "hf_id": "cais/mmlu", "subset": "all",
        "split": "test", "n_shot": 5,
    },
    "arc_challenge": {
        "hf_id": "allenai/ai2_arc", "subset": "ARC-Challenge",
        "split": "test", "n_shot": 25,
    },
    "boolq": {
        "hf_id": "google/boolq", "subset": None,
        "split": "validation", "n_shot": 32,
    },
    "gsm8k": {
        "hf_id": "openai/gsm8k", "subset": "main",
        "split": "test", "n_shot": 8,
    },
    "natural_questions": {
        "hf_id": "google-research-datasets/nq_open", "subset": None,
        "split": "validation", "n_shot": 32,
    },
    "ceval": {
        "hf_id": "ceval/ceval-exam", "subset": "all",
        "split": "val", "n_shot": 5,
    },
    "cmmlu": {
        "hf_id": "haonan-li/cmmlu", "subset": "all",
        "split": "test", "n_shot": 5,
    },
}


# ===================== Per-task formatters =====================

def format_sciq(item):
    support = item.get("support", "")
    q = item.get("question", "")
    choices = [item.get("correct_answer", ""), item.get("distractor1", ""),
               item.get("distractor2", ""), item.get("distractor3", "")]
    cstr = "\n".join([f"{'ABCD'[i]}. {c}" for i, c in enumerate(choices) if c])
    return f"{support}\nQuestion: {q}\n{cstr}\nAnswer:", "A"

def format_piqa(item):
    goal, s1, s2 = item.get("goal", ""), item.get("sol1", ""), item.get("sol2", "")
    label = item.get("label", 0)
    return f"Question: {goal}\nA. {s1}\nB. {s2}\nAnswer:", "AB"[label if isinstance(label, int) else 0]

def format_winogrande(item):
    s = item.get("sentence", "")
    o1, o2 = item.get("option1", ""), item.get("option2", "")
    label = item.get("answer", "1")
    ans = o1 if str(label) == "1" else o2
    return f"{s}\nOption 1: {o1}\nOption 2: {o2}\nAnswer:", ans

def format_logiqa(item):
    ctx, q = item.get("context", ""), item.get("question", "")
    opts = item.get("options", [])
    label = item.get("label", 0)
    cstr = "\n".join([f"{'ABCD'[i]}. {o}" for i, o in enumerate(opts)])
    ans = "ABCD"[label] if isinstance(label, int) and label < 4 else "A"
    return f"Passage: {ctx}\nQuestion: {q}\n{cstr}\nAnswer:", ans

def format_truthfulqa(item):
    q = item.get("question", "")
    return f"Q: {q}\nA:", ""

def format_hellaswag(item):
    ctx = item.get("ctx", item.get("ctx_a", ""))
    endings = item.get("endings", [])
    label = item.get("label", "0")
    cstr = "\n".join([f"{'ABCD'[i]}. {e}" for i, e in enumerate(endings)])
    ans = "ABCD"[int(label)] if str(label).isdigit() and int(label) < 4 else "A"
    return f"{ctx}\n{cstr}\nAnswer:", ans

def format_mmlu(item):
    q = item.get("question", "")
    choices = item.get("choices", [])
    label = item.get("answer", 0)
    cstr = "\n".join([f"{'ABCD'[i]}. {c}" for i, c in enumerate(choices)])
    ans = "ABCD"[int(label)] if isinstance(label, int) and label < 4 else "A"
    return f"Question: {q}\n{cstr}\nAnswer:", ans

def format_arc(item):
    q = item.get("question", "")
    ch = item.get("choices", {})
    labels, texts = ch.get("label", []), ch.get("text", [])
    ak = item.get("answerKey", "A")
    cstr = "\n".join([f"{l}. {t}" for l, t in zip(labels, texts)])
    return f"Question: {q}\n{cstr}\nAnswer:", ak

def format_boolq(item):
    p, q = item.get("passage", ""), item.get("question", "")
    label = item.get("answer", True)
    return f"{p}\nQuestion: {q}?\nAnswer:", "Yes" if label else "No"

def format_gsm8k(item):
    q, a = item.get("question", ""), item.get("answer", "")
    final = a.split("####")[-1].strip() if "####" in a else a
    return f"Question: {q}\nAnswer: Let's think step by step.\n", final

def format_lambada(item):
    text = item.get("text", "")
    words = text.rsplit(" ", 1)
    if len(words) == 2:
        return words[0], words[1]
    return text, ""

def format_nq(item):
    q = item.get("question", "")
    answers = item.get("answer", [])
    ans = answers[0] if isinstance(answers, list) and answers else str(answers)
    return f"Question: {q}\nAnswer:", ans

def format_ceval(item):
    q = item.get("question", "")
    A, B, C, D = item.get("A", ""), item.get("B", ""), item.get("C", ""), item.get("D", "")
    ans = item.get("answer", "A")
    return f"问题: {q}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n答案:", ans

def format_cmmlu(item):
    q = item.get("Question", item.get("question", ""))
    A, B, C, D = item.get("A", ""), item.get("B", ""), item.get("C", ""), item.get("D", "")
    ans = item.get("Answer", item.get("answer", "A"))
    return f"问题: {q}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n答案:", ans

FORMATTERS = {
    "sciq": format_sciq, "piqa": format_piqa, "winogrande": format_winogrande,
    "logiqa": format_logiqa, "truthfulqa": format_truthfulqa, "hellaswag": format_hellaswag,
    "mmlu": format_mmlu, "arc_challenge": format_arc, "boolq": format_boolq,
    "gsm8k": format_gsm8k, "lambada": format_lambada, "natural_questions": format_nq,
    "ceval": format_ceval, "cmmlu": format_cmmlu,
}


# ===================== Few-shot builder =====================

def build_few_shot_prompt(test_prompt, train_split, formatter, n_shot, seed=42):
    """Prepend n_shot exemplars from train split (matches MoH eval protocol)."""
    if n_shot == 0 or train_split is None or len(train_split) == 0:
        return test_prompt
    rng = random.Random(seed)
    indices = rng.sample(range(len(train_split)), min(n_shot, len(train_split)))
    exemplars = ""
    for idx in indices:
        try:
            ex_p, ex_a = formatter(train_split[idx])
            exemplars += f"{ex_p} {ex_a}\n\n"
        except Exception:
            continue
    return exemplars + test_prompt


# ===================== Main Dataset =====================

class MoHFullEvalDataset(Dataset):
    """
    Full evaluation dataset across all 14 MoH benchmarks.
    Constructs few-shot prompts matching the paper's exact protocol.
    """

    def __init__(
        self,
        tokenizer,
        datasets_dir: str = "./datasets",
        max_seq_len: int = 512,
        tasks: Optional[List[str]] = None,
        max_samples_per_task: Optional[int] = None,
        seed: int = 42,
        verbose: bool = True,
    ):
        self.tokenizer = tokenizer
        self.datasets_dir = datasets_dir
        self.max_seq_len = max_seq_len
        self.seed = seed
        self.samples: List[Dict] = []

        random.seed(seed)
        task_list = tasks if tasks else list(DATASET_REGISTRY.keys())

        total = 0
        for task_name in task_list:
            if task_name not in DATASET_REGISTRY:
                continue
            info = DATASET_REGISTRY[task_name]
            formatter = FORMATTERS.get(task_name)
            if formatter is None:
                continue
            try:
                count = self._load_task(task_name, info, formatter, max_samples_per_task, verbose)
                total += count
            except Exception as e:
                if verbose:
                    print(f"  [FAIL] {task_name}: {e}")

        if verbose:
            print(f"\n  Total loaded: {total} samples, "
                  f"{len(set(s['task_name'] for s in self.samples))} tasks")

    def _load_task(self, task_name, info, formatter, max_samples, verbose):
        ds = self._get_dataset(task_name, info["hf_id"], info.get("subset"))
        if ds is None:
            if verbose:
                print(f"  [MISS] {task_name}: not found (run download_datasets.py first)")
            return 0

        # Find eval split
        eval_split = None
        for name in [info["split"], "validation", "test", "val"]:
            if hasattr(ds, "keys") and name in ds:
                eval_split = ds[name]
                break
        if eval_split is None:
            if hasattr(ds, "keys") and len(ds.keys()) > 0:
                eval_split = ds[list(ds.keys())[0]]
            else:
                eval_split = ds
        if eval_split is None or len(eval_split) == 0:
            return 0

        # Find train split for few-shot
        n_shot = info.get("n_shot", 0)
        train_split = None
        if n_shot > 0 and hasattr(ds, "keys"):
            for name in ["train", "auxiliary_train", "dev"]:
                if name in ds and name != info["split"]:
                    train_split = ds[name]
                    break

        # Load all samples (or max_samples)
        n_total = len(eval_split)
        if max_samples is not None and max_samples < n_total:
            indices = sorted(random.sample(range(n_total), max_samples))
        else:
            indices = range(n_total)

        count = 0
        for idx in indices:
            try:
                prompt, answer = formatter(eval_split[idx])
                if n_shot > 0 and train_split is not None:
                    prompt = build_few_shot_prompt(
                        prompt, train_split, formatter, n_shot, self.seed + idx
                    )
                self.samples.append({
                    "text": prompt, "answer": answer,
                    "task_name": task_name, "sample_idx": idx,
                })
                count += 1
            except Exception:
                continue

        if verbose:
            print(f"  [OK] {task_name}: {count}/{n_total} "
                  f"({n_shot}-shot, split={info['split']})")
        return count

    def _load_from_disk(self, name):
        from datasets import load_from_disk
        path = os.path.join(self.datasets_dir, name)
        if os.path.exists(path):
            return load_from_disk(path)
        return None

    def _load_hf_direct(self, hf_id, subset=None):
        try:
            from datasets import load_dataset
            if subset:
                return load_dataset(hf_id, subset, trust_remote_code=True)
            return load_dataset(hf_id, trust_remote_code=True)
        except Exception:
            return None

    def _get_dataset(self, name, hf_id=None, subset=None):
        ds = self._load_from_disk(name)
        if ds is not None:
            return ds
        if hf_id:
            return self._load_hf_direct(hf_id, subset)
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        encoding = self.tokenizer(
            item["text"], return_tensors="pt",
            max_length=self.max_seq_len, truncation=True, padding=False,
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "task_name": item["task_name"],
            "sample_idx": item["sample_idx"],
        }

    def get_task_counts(self):
        counts = {}
        for s in self.samples:
            counts[s["task_name"]] = counts.get(s["task_name"], 0) + 1
        return counts


def collate_fn(batch):
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids, attention_masks, labels, task_names, sample_indices = [], [], [], [], []
    for item in batch:
        pad = max_len - item["input_ids"].size(0)
        input_ids.append(torch.cat([item["input_ids"], torch.zeros(pad, dtype=torch.long)]))
        attention_masks.append(torch.cat([item["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
        labels.append(torch.cat([item["labels"], torch.full((pad,), -100, dtype=torch.long)]))
        task_names.append(item["task_name"])
        sample_indices.append(item["sample_idx"])
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "labels": torch.stack(labels),
        "task_names": task_names,
        "sample_indices": sample_indices,
    }


def print_dataset_summary(dataset):
    counts = dataset.get_task_counts()
    print("\n" + "=" * 60)
    print(f"{'Task':<25} {'Samples':>8} {'N-shot':>8}")
    print("-" * 45)
    total = 0
    for task in sorted(counts.keys()):
        n = DATASET_REGISTRY.get(task, {}).get("n_shot", 0)
        print(f"  {task:<23} {counts[task]:>8} {n:>8}-shot")
        total += counts[task]
    print("-" * 45)
    print(f"  {'TOTAL':<23} {total:>8}")
    print("=" * 60)
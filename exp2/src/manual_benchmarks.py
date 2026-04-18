"""
Manual benchmark evaluation — works offline with cached datasets.
Handles all 11 benchmarks from the MoH paper.
"""
import torch
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional
from .config import BenchmarkConfig

logger = logging.getLogger(__name__)


def evaluate_all(model, tokenizer, config: BenchmarkConfig) -> Dict:
    results = {"scores": {}, "raw_results": {}}
    import time
    t0 = time.time()
    for name in config.benchmarks:
        try:
            logger.info(f"Evaluating: {name}")
            score = _eval(model, tokenizer, name,
                          max_samples=config.max_samples,
                          cache_dir=config.cache_dir)
            results["scores"][name] = score
            logger.info(f"  {name}: {score:.4f}")
        except Exception as e:
            logger.warning(f"  {name} failed: {e}")
            results["scores"][name] = 0.0
    results["total_time_sec"] = time.time() - t0
    results["num_tasks"] = len(results["scores"])
    return results


def _eval(model, tokenizer, name, max_samples=None, cache_dir=None):
    evaluators = {
        "piqa": _eval_piqa, "hellaswag": _eval_hellaswag,
        "winogrande": _eval_winogrande, "boolq": _eval_boolq,
        "sciq": _eval_sciq, "arc_challenge": _eval_arc,
        "lambada": _eval_lambada, "mmlu": _eval_mmlu,
        "gsm8k": _eval_gsm8k, "truthfulqa": _eval_truthfulqa,
        "logiqa": _eval_logiqa,
    }
    if name not in evaluators:
        logger.warning(f"No evaluator for {name}")
        return 0.0
    return evaluators[name](model, tokenizer, max_samples, cache_dir)


def _load_ds(path, subset=None, split="validation", cache_dir=None):
    """Load dataset with fallbacks for different datasets library versions."""
    from datasets import load_dataset
    kwargs = {"cache_dir": cache_dir}
    try:
        # Try with trust_remote_code first (datasets < 3.0)
        if subset:
            return load_dataset(path, subset, split=split, trust_remote_code=True, **kwargs)
        return load_dataset(path, split=split, trust_remote_code=True, **kwargs)
    except TypeError:
        # datasets >= 3.0 dropped trust_remote_code
        if subset:
            return load_dataset(path, subset, split=split, **kwargs)
        return load_dataset(path, split=split, **kwargs)


def _loglikelihood(model, tokenizer, context, continuation):
    """Log-likelihood of continuation given context."""
    ctx_ids = tokenizer.encode(context, add_special_tokens=False)
    cont_ids = tokenizer.encode(continuation, add_special_tokens=False)
    if not cont_ids:
        return -1e10
    full = ctx_ids + cont_ids
    # Truncate if too long
    max_len = getattr(model.config, 'max_position_embeddings', 4096) - 10
    if len(full) > max_len:
        full = full[-max_len:]
        ctx_ids = full[:len(full)-len(cont_ids)]
    input_ids = torch.tensor([full], device=model.device)
    with torch.no_grad():
        logits = model(input_ids).logits
    shift = logits[0, len(ctx_ids)-1:-1, :]
    labels = torch.tensor(cont_ids, device=model.device)
    if shift.shape[0] != labels.shape[0]:
        mn = min(shift.shape[0], labels.shape[0])
        shift, labels = shift[:mn], labels[:mn]
    lp = F.log_softmax(shift, dim=-1)
    return lp.gather(1, labels.unsqueeze(1)).sum().item()


def _eval_mc(model, tokenizer, items, max_samples=None):
    """Generic multiple-choice evaluator."""
    if max_samples:
        items = items[:max_samples]
    correct = total = 0
    for it in items:
        scores = [_loglikelihood(model, tokenizer, it["ctx"], c) for c in it["choices"]]
        if max(range(len(scores)), key=lambda i: scores[i]) == it["gold"]:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0


# ─── Individual benchmark evaluators ───

def _eval_piqa(model, tokenizer, max_samples, cache_dir):
    ds = _load_ds("piqa", split="validation", cache_dir=cache_dir)
    items = [{"ctx": ex["goal"], "choices": [" " + ex["sol1"], " " + ex["sol2"]],
              "gold": int(ex["label"])} for ex in ds]
    return _eval_mc(model, tokenizer, items, max_samples)


def _eval_hellaswag(model, tokenizer, max_samples, cache_dir):
    ds = _load_ds("Rowan/hellaswag", split="validation", cache_dir=cache_dir)
    items = []
    for ex in ds:
        ctx = ex.get("ctx", ex.get("context", ""))
        endings = ex.get("endings", [])
        label = ex.get("label", "0")
        if isinstance(endings, str):
            import json
            try: endings = json.loads(endings)
            except: continue
        if not endings:
            continue
        items.append({"ctx": ctx, "choices": [" " + e for e in endings],
                       "gold": int(label)})
    return _eval_mc(model, tokenizer, items, max_samples)


def _eval_winogrande(model, tokenizer, max_samples, cache_dir):
    ds = _load_ds("winogrande", "winogrande_xl", split="validation", cache_dir=cache_dir)
    items = []
    for ex in ds:
        sent = ex["sentence"]
        o1, o2 = ex["option1"], ex["option2"]
        ans = ex["answer"]
        if ans not in ("1", "2", 1, 2):
            continue
        s1 = sent.replace("_", o1)
        s2 = sent.replace("_", o2)
        items.append({"ctx": "", "choices": [s1, s2], "gold": int(ans) - 1})
    return _eval_mc(model, tokenizer, items, max_samples)


def _eval_boolq(model, tokenizer, max_samples, cache_dir):
    ds = _load_ds("google/boolq", split="validation", cache_dir=cache_dir)
    items = [{"ctx": f"{ex['passage']}\nQuestion: {ex['question']}\nAnswer:",
              "choices": [" no", " yes"], "gold": 1 if ex["answer"] else 0} for ex in ds]
    return _eval_mc(model, tokenizer, items, max_samples)


def _eval_sciq(model, tokenizer, max_samples, cache_dir):
    ds = _load_ds("allenai/sciq", split="validation", cache_dir=cache_dir)
    items = []
    for ex in ds:
        choices = [ex["correct_answer"], ex["distractor1"], ex["distractor2"], ex["distractor3"]]
        items.append({"ctx": f"Question: {ex['question']}\nAnswer:",
                       "choices": [" " + c for c in choices], "gold": 0})
    return _eval_mc(model, tokenizer, items, max_samples)


def _eval_arc(model, tokenizer, max_samples, cache_dir):
    ds = _load_ds("allenai/ai2_arc", "ARC-Challenge", split="test", cache_dir=cache_dir)
    items = []
    for ex in ds:
        choices_text = ex["choices"]["text"]
        labels = ex["choices"]["label"]
        gold_label = ex["answerKey"]
        gold = labels.index(gold_label) if gold_label in labels else 0
        items.append({"ctx": f"Question: {ex['question']}\nAnswer:",
                       "choices": [" " + c for c in choices_text], "gold": gold})
    return _eval_mc(model, tokenizer, items, max_samples)


def _eval_lambada(model, tokenizer, max_samples, cache_dir):
    ds = _load_ds("EleutherAI/lambada_openai", split="test", cache_dir=cache_dir)
    samples = list(ds)
    if max_samples: samples = samples[:max_samples]
    correct = total = 0
    for ex in samples:
        text = ex["text"]
        parts = text.rsplit(" ", 1)
        if len(parts) < 2:
            continue
        context, target_word = parts[0], parts[1]
        target_tok = tokenizer.encode(" " + target_word, add_special_tokens=False)
        if not target_tok:
            continue
        inp = tokenizer.encode(context, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(inp).logits[0, -1, :]
        if logits.argmax().item() == target_tok[0]:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0


def _eval_mmlu(model, tokenizer, max_samples, cache_dir):
    """MMLU — 5-shot multiple choice across 57 subjects."""
    ds = _load_ds("cais/mmlu", "all", split="test", cache_dir=cache_dir)
    items = []
    choice_labels = ["A", "B", "C", "D"]
    for ex in ds:
        q = ex["question"]
        choices = ex["choices"]
        answer_idx = int(ex["answer"])
        # Format: "Question: ...\nA. ...\nB. ...\nC. ...\nD. ...\nAnswer:"
        prompt = f"Question: {q}\n"
        for i, c in enumerate(choices):
            prompt += f"{choice_labels[i]}. {c}\n"
        prompt += "Answer:"
        items.append({"ctx": prompt, "choices": [f" {l}" for l in choice_labels[:len(choices)]],
                       "gold": answer_idx})
    return _eval_mc(model, tokenizer, items, max_samples)


def _eval_gsm8k(model, tokenizer, max_samples, cache_dir):
    """GSM8K — math word problems, check if final number matches."""
    ds = _load_ds("gsm8k", "main", split="test", cache_dir=cache_dir)
    samples = list(ds)
    if max_samples: samples = samples[:max_samples]
    correct = total = 0
    for ex in samples:
        question = ex["question"]
        answer_text = ex["answer"]
        # Extract final number from answer
        import re
        nums = re.findall(r'[\-]?\d[\d,]*\.?\d*', answer_text.replace(",", ""))
        if not nums:
            continue
        gold_num = nums[-1]
        prompt = f"Question: {question}\nAnswer: Let me solve step by step."
        inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=100, do_sample=False)
        gen_text = tokenizer.decode(out[0][inp['input_ids'].shape[1]:], skip_special_tokens=True)
        gen_nums = re.findall(r'[\-]?\d[\d,]*\.?\d*', gen_text.replace(",", ""))
        if gen_nums and gen_nums[-1] == gold_num:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0


def _eval_truthfulqa(model, tokenizer, max_samples, cache_dir):
    """TruthfulQA MC2 — multiple correct answers, use log-likelihood."""
    ds = _load_ds("truthful_qa", "multiple_choice", split="validation", cache_dir=cache_dir)
    samples = list(ds)
    if max_samples: samples = samples[:max_samples]
    correct = total = 0
    for ex in samples:
        q = ex["question"]
        mc1_targets = ex.get("mc1_targets", {})
        choices = mc1_targets.get("choices", [])
        labels = mc1_targets.get("labels", [])
        if not choices or not labels:
            continue
        gold = labels.index(1) if 1 in labels else 0
        ctx = f"Question: {q}\nAnswer:"
        scores = [_loglikelihood(model, tokenizer, ctx, " " + c) for c in choices]
        if max(range(len(scores)), key=lambda i: scores[i]) == gold:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0


def _eval_logiqa(model, tokenizer, max_samples, cache_dir):
    """LogiQA — logical reasoning multiple choice."""
    try:
        ds = _load_ds("EleutherAI/logiqa", split="test", cache_dir=cache_dir)
    except Exception:
        try:
            ds = _load_ds("lucasmccabe/logiqa", split="test", cache_dir=cache_dir)
        except Exception as e:
            logger.warning(f"  logiqa dataset not available: {e}")
            return 0.0
    items = []
    for ex in ds:
        ctx_text = ex.get("context", "")
        q = ex.get("question", ex.get("query", ""))
        options = ex.get("options", ex.get("choices", []))
        answer = ex.get("label", ex.get("answer", ex.get("correct_option", 0)))
        if not options:
            continue
        prompt = f"Context: {ctx_text}\nQuestion: {q}\nAnswer:"
        items.append({"ctx": prompt, "choices": [" " + o for o in options],
                       "gold": int(answer)})
    return _eval_mc(model, tokenizer, items, max_samples)

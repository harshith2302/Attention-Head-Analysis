# Dynamic Inference-Time Head Pruning for LLaMA3-8B
## Prajna Cluster — `/home/cccp/25m0834/RND5`

---

## WHAT THIS DOES

Runs **100 experiments** testing 6 dynamic head pruning methods on LLaMA3-8B
across 11 benchmarks, then generates comparison plots. All in ONE SLURM job.

- 6 scoring methods × 3 warmup configs × 3 top-K values = 54 base configs
- + 45 hybrid lambda variants + rolling chunk variants + 1 baseline = **100 total**
- 11 benchmarks each: MMLU, GSM8K, TruthfulQA, HellaSwag, PIQA, WinoGrande, ARC-C, BoolQ, LAMBADA, SciQ, LogiQA
- 200 samples per benchmark (enough to rank methods, finishes in ~18-20 hours)
- Automatic resume if interrupted

---

## FILE STRUCTURE

```
RND5/
├── src/
│   ├── config.py              ← Experiment configurations
│   ├── head_scoring.py        ← 6 scoring methods
│   ├── head_pruning.py        ← Hook-based head masking
│   ├── model_loader.py        ← ★ PUT HF TOKEN HERE (line 14)
│   ├── benchmark_runner.py    ← Evaluation engine
│   ├── manual_benchmarks.py   ← Fallback evaluators
│   └── __init__.py
├── scripts/
│   ├── run_all.py             ← Master runner (runs everything)
│   ├── generate_plots.py      ← Plot generation
│   └── slurm/
│       └── run_all.sh         ← ★ THE SLURM FILE (submit this)
├── setup/
│   ├── step0_conda.sh         ← Creates conda env
│   └── step1_download.sh      ← ★ PUT HF TOKEN HERE (line 12)
├── plotting/
│   └── generate_plots.py      ← Visualization code
├── reference/
│   ├── modeling_llama.py       ← MoH reference code
│   └── MOH.pdf                 ← MoH paper
├── cache/                      ← Model + datasets (created by Step 1)
├── results/                    ← All outputs (created by SLURM job)
│   ├── <method>_w<W>_k<K>/results.json  ← Per-experiment results
│   ├── all_results.json        ← Combined results
│   └── plots/                  ← All graphs (PNG + PDF)
└── logs/                       ← SLURM logs
```

---

## GPU: NVIDIA A40 (48GB) ✅ PERFECT

LLaMA3-8B in bfloat16 needs ~20GB. A40 has 48GB = plenty.

---

## STEP-BY-STEP INSTRUCTIONS

### STEP 0: Paste HF Token (1 minute)

Open these 2 files and replace `hf_PASTE_YOUR_TOKEN_HERE`:

```bash
cd /home/cccp/25m0834/RND5

# File 1:
nano src/model_loader.py
# Line 14: HF_TOKEN = "hf_YOUR_REAL_TOKEN"

# File 2:
nano setup/step1_download.sh
# Line 12: export HF_TOKEN="hf_YOUR_REAL_TOKEN"
```

### STEP 1: Create Conda Environment (5 minutes, one-time)

```bash
cd /home/cccp/25m0834/RND5
bash setup/step0_conda.sh
```

### STEP 2: Download Model & Datasets (30-45 minutes)

```bash
cd /home/cccp/25m0834/RND5
conda activate dhp
bash setup/step1_download.sh
```

Verify:
```bash
du -sh cache/models/     # ~16GB
du -sh cache/datasets/   # ~5GB
```

### STEP 3: Submit THE Job (just this ONE command)

```bash
cd /home/cccp/25m0834/RND5
sbatch scripts/slurm/run_all.sh
```

That's it. Come back tomorrow.

### STEP 4: Check Results (next day)

```bash
# Check if job finished:
squeue -u 25m0834

# View the log:
cat logs/master_*.out | tail -50

# Your results:
ls results/plots/

# Summary table:
cat results/plots/results_summary.csv
```

---

## TIMELINE

| Time | What Happens |
|------|-------------|
| 0 min | Paste token + create conda + download |
| 45 min | Submit: `sbatch scripts/slurm/run_all.sh` |
| 45 min - 20 hrs | Job runs 100 experiments automatically |
| ~20 hrs | Job generates all plots and finishes |
| Next morning | Check `results/plots/` for all graphs |

---

## OUTPUTS YOU GET

### Results per experiment (`results/<name>/results.json`):
- Accuracy per benchmark
- Average accuracy
- Latency (tokens/sec)
- Active head percentage
- FLOPs reduction estimate

### Plots (`results/plots/`):
| File | What It Shows |
|------|--------------|
| `method_comparison.png` | All methods ranked by average accuracy |
| `benchmark_bars.png` | Per-benchmark breakdown for all methods |
| `warmup_comparison.png` | Effect of warmup layers (6 vs 8 vs 10) |
| `topk_sensitivity.png` | Accuracy vs number of active heads |
| `lambda_comparison.png` | Hybrid method performance across λ values |
| `pareto_plot.png` | Speed vs accuracy tradeoff |
| `results_summary.csv` | All numbers in one spreadsheet |

All plots include **MoH-LLaMA3-8B** and **LLaMA3-8B** baselines for comparison.

---

## IF SOMETHING GOES WRONG

**Job got killed (time limit / OOM) → Resume:**
```bash
# Just re-submit. It skips completed experiments automatically.
sbatch scripts/slurm/run_all.sh
```

**Check progress while running:**
```bash
tail -f logs/master_*.out
```

**Generate plots manually (if job completed but plots failed):**
```bash
conda activate dhp
cd /home/cccp/25m0834/RND5
python3 scripts/generate_plots.py
```

**"Module not found" errors:**
```bash
conda activate dhp
pip install transformers accelerate datasets lm-eval matplotlib
```

**A40 partition busy:**
Edit `scripts/slurm/run_all.sh` line 3:
```bash
#SBATCH --partition=l40    # try L40 instead
```

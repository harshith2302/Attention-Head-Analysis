# Attention-Head-Analysis: Prompt-Level Head Masking & Routing

Official implementation and experimentation framework for analyzing and optimizing attention head allocation in Large Language Models, with experiments focused on **LLaMA-3-8B**.

Unlike token-level dynamic routing approaches such as **Mixture-of-Heads (MoH)**, this project introduces a **prompt-level routing mechanism**. By identifying the task category before inference, the model activates only a specialized subset of attention heads (**Shared Heads + Task-Specific Heads**) for the entire prompt.

This approach maintains competitive reasoning performance while improving computational efficiency, reducing memory overhead, and simplifying deployment compared to token-level routing methods.

---

## 📊 Key Highlights

* **Compact Architectural Efficiency**

  * Reduces active attention heads by up to **50% per layer**.

* **Prompt-Level Routing**

  * Replaces token-by-token dynamic routing with a deterministic, task-aware masking strategy.

* **Competitive Performance**

  * Outperforms token-level MoH baselines on several reasoning benchmarks under constrained head budgets.

| Dataset | Accuracy  |
| ------- | --------- |
| SciQ    | **96.1%** |
| BoolQ   | **84.1%** |
| LogiQA  | **31.3%** |

---

## 🧠 Motivation

Recent attention-routing techniques dynamically activate different attention heads for each generated token. While effective, these methods introduce additional routing complexity and runtime overhead.

This project explores a simpler alternative:

1. Identify the task before inference.
2. Select a predefined subset of high-value attention heads.
3. Apply a static mask for the entire prompt.
4. Preserve performance while reducing computation.

The core hypothesis is that many attention heads specialize in particular reasoning behaviors, allowing efficient task-aware head allocation without expensive token-level routing.

---

## 🏗️ Method Overview

For a given task:

```
Active Heads = Shared Heads + Task-Specific Heads
```

Optionally:

```
Free Layers = Layers where all heads remain active
```

Three parameters define every experiment:

| Parameter     | Description                                  |
| ------------- | -------------------------------------------- |
| `n_shared`    | Globally important heads shared across tasks |
| `m_task`      | Task-specific high-ranking heads             |
| `free_layers` | Layers with 100% active heads                |

This creates a controllable compute budget while preserving essential reasoning capabilities.

---

## 📁 Repository Structure

```text
Attention-Head-Analysis/
│
├── code/
│   ├── download_datasets.py
│   ├── download_model.py
│   ├── head_selection.py
│   ├── head_masked_eval_patched.py
│   ├── fix_eager.py
│   ├── retry_failed.py
│   └── generate_vertical_tables.py
│
├── eval_results/
│   └── Raw JSON evaluation outputs
│
├── logs/
│   └── Execution logs (nohup outputs)
│
├── results/
│   ├── global_ranking.csv
│   └── Generated PNG tables
│
├── task_rankings/
│   └── Task-specific head ranking CSV files
│
└── *.sh
    ├── run_stratA.sh
    ├── run_s8_sweep.sh
    ├── run_fix.sh
    └── Other automation scripts
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Attention-Head-Analysis.git
cd Attention-Head-Analysis
```

### 2. Install Core Dependencies

```bash
pip install torch transformers datasets accelerate bitsandbytes numpy pandas
```

### 3. Install Evaluation Framework

```bash
pip install lm-eval==0.4.1
```

### 4. Install Visualization Dependencies

```bash
pip install dataframe_image playwright
playwright install chromium
```

---

## 📥 Dataset Preparation

Download and cache all required evaluation datasets:

```bash
python code/download_datasets.py
```

---

## 🤖 Model Preparation

Download and cache the LLaMA-3 model weights:

```bash
python code/download_model.py
```

Ensure you have accepted the model license and possess the appropriate Hugging Face access permissions.

---

## 🚀 Running Evaluations

### Standard Evaluation

Run a predefined masking strategy:

```bash
export CUDA_VISIBLE_DEVICES=0

nohup ./run_stratA.sh > logs/stratA.log 2>&1 &
```

This launches evaluation using the prompt-level masking configuration defined inside the script.

---

### Automated Parameter Sweeps

To explore multiple head-budget configurations automatically:

```bash
export CUDA_VISIBLE_DEVICES=1

nohup ./run_s8_sweep.sh > logs/s8_sweep.log 2>&1 &
```

The sweep scripts iterate through multiple:

* Shared Head counts
* Task Head counts
* Layer configurations

and automatically store results for later analysis.

---

## 🔧 PyTorch / Triton Compatibility Fix

Certain long-context benchmarks (such as HellaSwag, ARC-C, and TruthfulQA) may occasionally trigger a dataclass parsing issue in:

* PyTorch 2.5.1
* Triton compilation pipeline
* lm-eval integration

To bypass these failures, use:

```bash
python code/fix_eager.py
```

This forces:

```python
attn_implementation="eager"
```

which avoids problematic compilation paths while preserving evaluation correctness.

Automated recovery:

```bash
nohup ./run_fix.sh > logs/fix.log 2>&1 &
```

---

## 🔄 Retrying Failed Evaluations

If individual benchmark jobs fail during large-scale sweeps:

```bash
python code/retry_failed.py
```

This script selectively reruns only failed tasks rather than restarting the entire experiment.

---

## 📈 Generating Publication-Ready Tables

After evaluation results have been saved in:

```text
eval_results/
```

Generate comparison tables:

```bash
python code/generate_vertical_tables.py
```

Output directory:

```text
results/presentation_vertical/
```

---

## 🎨 Visualization Features

The visualization pipeline uses:

* HTML/CSS rendering
* Headless Chromium
* Playwright screenshots

to produce publication-quality figures.

### Included Features

#### Configuration Summary

Automatically extracts:

* Shared Heads
* Task-Specific Heads
* Free Layers

for each experiment.

#### Dynamic Highlighting

* Highest accuracy per row is highlighted.
* Clean academic-style formatting.

#### Delta Analysis

Automatically computes:

* Ours vs Full LLaMA-3
* Ours vs MoH

and marks improvements with:

```text
★
```

---

## 📊 Example Evaluation Workflow

```bash
# Step 1: Download datasets
python code/download_datasets.py

# Step 2: Download model
python code/download_model.py

# Step 3: Run evaluation
nohup ./run_stratA.sh > logs/stratA.log 2>&1 &

# Step 4: Generate result tables
python code/generate_vertical_tables.py
```

---

## 📂 Output Artifacts

### Evaluation Results

```text
eval_results/
```

Contains:

* Accuracy metrics
* Benchmark scores
* Raw JSON outputs

### Rankings

```text
results/global_ranking.csv
```

Contains:

* Global head importance rankings
* Aggregated analysis

### Task Rankings

```text
task_rankings/
```

Contains:

* Task-specific attention head rankings
* Specialized routing information

### Visualizations

```text
results/presentation_vertical/
```

Contains:

* PNG tables
* Benchmark comparisons
* Publication-ready figures

---

## 🔬 Experimental Findings

Our experiments suggest that:

* Many attention heads exhibit task specialization.
* Prompt-level routing can preserve reasoning quality with substantially fewer active heads.
* Static task-aware masking is a viable alternative to token-level routing methods.
* Significant computational savings are achievable without requiring dynamic routing infrastructure.

---

## 📝 Future Work

Planned extensions include:

* Multimodal transformer architectures
* Vision-language models
* Mixture-of-Experts (MoE) backbones
* Dynamic task classification
* Cross-model transferability studies
* Larger-scale evaluations on 70B+ parameter models

---

## 📖 Citation

If you use this work in your research, please cite:

```bibtex
@misc{attentionheadanalysis2025,
  title={Attention-Head-Analysis: Prompt-Level Head Masking and Routing},
  author={Your Name},
  year={2025},
  publisher={GitHub},
  url={https://github.com/yourusername/Attention-Head-Analysis}
}
```

---

## 📜 License

Specify your project license here.

Example:

```text
MIT License
```

or

```text
Apache License 2.0
```

---

## 🙏 Acknowledgements

This project builds upon:

* Meta LLaMA
* Hugging Face Transformers
* EleutherAI LM Evaluation Harness
* PyTorch
* Triton
* Playwright

Their open-source contributions made this research possible.

# Asymmetric Style Suppression Experiment

**Goal**: Test whether gradient-based data attribution can find semantically matching training examples when the query is in a different style than the training data.

This simulates a realistic scenario: your training data is mostly in one style (e.g., 95% formal/shakespeare), but users query in a different style (e.g., casual/pirate). Without intervention, gradient similarity is dominated by style—queries match training examples with similar style rather than similar content. This experiment evaluates strategies (preconditioners, PCA, gradient summing) to suppress style and recover semantic matching.

## Usage

```
/asymmetric-style [options]
```

Options:
- `--base-path PATH` - Output directory (default: runs/asymmetric_style)
- `--recompute` - Clear cached results and recompute from scratch
- `--inner-product` - Use raw inner product instead of cosine similarity
- `--sweep-pca` - Sweep PCA k values with preconditioner combinations
- `--rewrite-ablation` - Run rewrite ablation (summed rewrites vs summed eval)
- `--summary` - Just print summary of existing results

## What this does

1. Creates asymmetric train/eval split:
   - Train: 95% shakespeare (dominant), 5% pirate (minority)
   - Eval: pirate style queries for facts only in shakespeare style in train
2. Tests whether gradient-based attribution can find semantic matches despite style mismatch
3. Compares strategies: baseline, preconditioners (R_between, H_eval, H_train), PCA projection, summed gradients

## Strategies Tested

### Baseline
- **no_precond**: Raw cosine similarity between query and training gradients. Expected to fail because style dominates the gradient representation.

### Preconditioners
Transform gradients by `g' = g @ H^(-1)` before computing similarity, downweighting certain directions.

- **R_between**: Computed from the difference between style means on **training data**: `delta = mean(shakespeare_train) - mean(pirate_train)`, then `R = delta @ delta.T`. This is a rank-1 matrix that captures the "style direction".

  *Dataset*: Training set (95% shakespeare, 5% pirate). The shakespeare mean is computed over ~950 samples, pirate mean over ~50 samples.

  Hypothesis: inverting this downweights the style axis, exposing semantic signal.

- **H_eval**: Second moment of eval gradients: `H = (1/n) * G_eval.T @ G_eval`. Hypothesis: directions that vary a lot in the eval set (which is all one style) might be style-related, so downweighting high-variance eval directions could help.

- **H_train**: Second moment of training gradients: `H = (1/n) * G_train.T @ G_train`. This has theoretical grounding from influence functions: `g_eval @ H^{-1} @ g_train.T` approximates the change in eval loss from upweighting a training point (second-order Taylor expansion). So H_train is the "correct" similarity metric for influence-based attribution.

### Dimensionality Reduction
- **PCA projection**: Compute pairwise differences between corresponding shakespeare/pirate gradients (same underlying fact, different style), then PCA on those difference vectors. Project out the top-k components of this "style difference" subspace.

  Hypothesis: the difference `g_shakespeare(fact) - g_pirate(fact)` isolates pure style variation (content is held constant). The top PCs of these differences capture the dominant style directions. Projecting them out should remove style signal while preserving semantic content.

### Gradient Averaging
- **summed_eval**: For each query, compute gradients in both styles (pirate + shakespeare), then sum them. Hypothesis: style-specific components cancel out, leaving semantic signal. This requires generating the query in multiple styles.

- **summed_rewrites**: Sum gradients from two non-training styles (e.g., shakespeare + pirate rewrites of the same fact, when training only has formal). Tests whether style cancellation is general or requires matching training distribution.

### Controls
- **majority_no_precond**: Query in the majority (shakespeare) style—no style mismatch. This is the upper bound showing what's achievable when styles match.

## Instructions

### Run full experiment (using HuggingFace data)

The easiest way to run the experiment is using pre-generated data from HuggingFace:

```python
from examples.semantic.asymmetric import run_asymmetric_experiment, AsymmetricConfig

# Use HF dataset - no local generation needed
config = AsymmetricConfig(
    hf_dataset="EleutherAI/bergson-asymmetric-style",
)

results = run_asymmetric_experiment(
    config=config,
    base_path="runs/asymmetric_style",
    # analysis_model defaults to EleutherAI/bergson-asymmetric-style-qwen3-8b-lora
)
```

### Run full experiment (generate locally)

To generate fresh data locally (requires Qwen model for rewording):

```python
from examples.semantic.asymmetric import run_asymmetric_experiment, AsymmetricConfig

config = AsymmetricConfig(
    dominant_style="shakespeare",
    minority_style="pirate",
    dominant_ratio=0.95,
)

results = run_asymmetric_experiment(
    config=config,
    base_path="runs/asymmetric_style",
)
```

### Run PCA k-value sweep

```python
from examples.semantic.asymmetric import sweep_pca_k

results = sweep_pca_k(
    base_path="runs/asymmetric_style",
    k_values=[1, 5, 10, 20, 50, 100],
    preconditioners=[None, "index"],
)
```

### Run rewrite ablation

Tests whether summing two non-training styles helps (it doesn't):

```python
from examples.semantic.asymmetric import run_rewrite_ablation_experiment

results = run_rewrite_ablation_experiment(base_path="runs/asymmetric_style")
```

### Run inner product comparison

Compare cosine similarity vs raw inner product:

```python
from examples.semantic.asymmetric import run_inner_product_comparison

results = run_inner_product_comparison(base_path="runs/asymmetric_style")
```

### Print existing results summary

```python
import json
import numpy as np
from pathlib import Path
from datasets import load_from_disk

base_path = Path("runs/asymmetric_style")
with open(base_path / "experiment_results.json") as f:
    results = json.load(f)

sorted_results = sorted(results.items(), key=lambda x: -x[1]["top1_semantic"])
print(f"{'Strategy':<35} {'Top-1 Sem':<12} {'Top-1 Leak':<12} {'Exact':<10}")
print("-" * 70)
for name, m in sorted_results:
    print(f"{name:<35} {m['top1_semantic']:<12.2%} {m['top1_leak']:<12.2%} {m['exact']:<10.2%}")
```

## Cached Data

The experiment caches intermediate results to avoid recomputation:

```
runs/asymmetric_style/
├── data/
│   ├── train.hf               # Training set (95% shakespeare, 5% pirate)
│   ├── eval.hf                # Eval set (pirate style)
│   ├── eval_majority.hf       # Eval in majority style (control)
│   ├── eval_summed.hf         # Eval with summed gradients
│   └── rewrites/              # Additional style rewrites for ablations
├── index/                     # Training gradients
├── eval_grads/                # Eval gradients (minority style)
├── eval_grads_majority/       # Eval gradients (majority style)
├── preconditioners/           # Various preconditioner matrices
├── scores_*/                  # Score matrices for each strategy
└── experiment_results.json    # Cached metrics summary
```

**What each cache level means:**
- `data/` - Dataset creation and Qwen rewording (~10-20 min)
- `index/` - bergson build for training gradients (~2 min)
- `eval_grads*/` - bergson build for eval gradients (~1 min each)
- `preconditioners/` - Preconditioner computation (~30 sec)
- `scores_*/` - Score computation (~10 sec each)
- `experiment_results.json` - Metrics computed from scores

If the user specifies `--recompute`, first delete cached data:
```bash
rm -rf runs/asymmetric_style/index runs/asymmetric_style/eval_grads* runs/asymmetric_style/scores_* runs/asymmetric_style/preconditioners
```

To recompute everything including data:
```bash
rm -rf runs/asymmetric_style/
```

## Key Metrics

- **Top-1 Semantic Accuracy**: Top match has same underlying fact (higher is better)
- **Top-1 Style Leakage**: Top match is minority style (lower is better - means not style matching)
- **Exact Match**: Same fact AND dominant style (higher is better)

## Datasets & Models

The datasets and fine-tuned model for this experiment are available on Hugging Face:

- **Dataset**: [EleutherAI/bergson-asymmetric-style](https://huggingface.co/datasets/EleutherAI/bergson-asymmetric-style)
  - `train`: 13,500 samples (95% shakespeare, 5% pirate)
  - `eval`: 4,500 samples (pirate style queries)
  - `eval_majority_style`: 4,500 samples (shakespeare style control)
  - `eval_original_style`: 4,500 samples (unstyled)
  - `eval_pirate_style`: 4,500 samples (pirate style variant)

- **Model**: [EleutherAI/bergson-asymmetric-style-qwen3-8b-lora](https://huggingface.co/EleutherAI/bergson-asymmetric-style-qwen3-8b-lora)
  - LoRA adapter for Qwen/Qwen3-8B-Base
  - Used as the `analysis_model` for gradient collection

## Key Findings

| Strategy | Top-1 Semantic | Notes |
|----------|---------------|-------|
| majority_no_precond | 100.00% | Control: no style mismatch |
| summed_eval | 92.71% | Sum minority + majority style eval grads |
| summed_rewrites | 0.87% | Sum shakespeare + pirate (both non-training) |
| no_precond (baseline) | 0.87% | Pure style matching dominates |
| preconditioners | ~1-1.4% | Marginal improvement |

**Main insight**: summed_eval works because one component matches training distribution, not because of general style cancellation.

## Similarity Metric Comparison

With cosine similarity (my experiments):
- summed_eval: 92.71%

With raw inner product (bergson default):
- summed_eval: 76.91%

Cosine similarity helps by removing gradient magnitude as a confounding factor.

# Attribute Preservation Experiment

**Goal**: Test whether style suppression preconditioners can remove stylistic signal from gradient embeddings while preserving the ability to match on semantic attributes (occupation, employer type, etc.).

The core challenge is that gradient-based data attribution tends to match based on surface-level features like writing style rather than underlying content. This experiment creates synthetic data with correlated attributes (e.g., scientists work at research labs, business people work at banks) and tests whether we can surgically remove style signal without damaging these attribute-based matching capabilities.

## Usage

```
/attribute-preservation [options]
```

Options:
- `--base-path PATH` - Output directory (default: runs/attribute_preservation)
- `--no-h-eval` - Skip H_eval preconditioner comparison
- `--no-majority` - Skip majority style control
- `--recompute` - Clear cached results and recompute from scratch

## What this does

1. Creates a synthetic dataset with occupational clusters (scientists, business, creative)
2. Each cluster has correlated attributes (employers, universities, degrees, titles)
3. Styles are assigned by occupation (scientists→shakespeare, business→pirate, creative→shakespeare)
4. Eval set: scientists in "wrong" style (pirate) to test style suppression
5. Compares preconditioner strategies: none, R_between, H_eval
6. Majority control: scientists in matching style (shakespeare) as upper bound

## Strategies Tested

### Baseline
- **no_precond**: Raw cosine similarity between query and training gradients. Expected to mostly match based on style (pirate queries → pirate training examples) rather than occupation.

### Preconditioners
Transform gradients by `g' = g @ H^(-1)` before computing similarity.

- **R_between**: Computed from training data style means: `delta = mean(shakespeare_grads) - mean(pirate_grads)`, then `R = delta @ delta.T`. This rank-1 matrix captures the "style direction" in gradient space.

  *Dataset*: Training set with style-occupation mapping:
  - shakespeare mean = scientists (400) + creatives (400) = 800 samples
  - pirate mean = business (400) = 400 samples

  *Caveat*: Because shakespeare mixes two occupations, the "style direction" is actually `(scientists + creatives)/2 - business`, which conflates style with occupation. This is meant to represent a situation where you can't rewrite scientist data in different styles, and have to work with different styles that already exist in the data. If you can rewrite, majority_no_precond may be the best option.

  Hypothesis: preconditioning with R^(-1) shrinks the style axis, allowing occupation signal to dominate.

- **H_eval**: Second moment of eval gradients: `H = (1/n) * G_eval.T @ G_eval`. Hypothesis: the eval set is all scientists in pirate style, so directions with high variance in eval might capture style-independent scientist features. Downweighting these could paradoxically help by normalizing the representation.

### Controls
- **majority_no_precond**: Scientists queried in shakespeare (their training style)—no style mismatch. This shows the upper bound: how well can we match occupation when style isn't a confounder? The gap between this and preconditioned results shows how much room for improvement remains.

## Why This Experiment Matters

Previous experiments showed preconditioners have minimal effect on fact-level retrieval. But maybe they work for coarser attribute matching? This tests whether style suppression preserves the ability to match "scientists to scientists" even if it can't match "Alice's employer fact to Alice's employer fact".

## Instructions

### Run full experiment (using HuggingFace data)

The easiest way to run the experiment is using pre-generated data from HuggingFace:

```python
from examples.semantic.attribute_preservation import (
    run_attribute_preservation_experiment,
    AttributePreservationConfig,
)

# Use HF dataset - no local generation needed
config = AttributePreservationConfig(
    hf_dataset="EleutherAI/bergson-attribute-preservation",
)

results = run_attribute_preservation_experiment(
    config=config,
    base_path='runs/attribute_preservation',
    # analysis_model defaults to EleutherAI/bergson-asymmetric-style-qwen3-8b-lora
    include_h_eval=True,
    include_majority_control=True
)
```

### Run full experiment (generate locally)

To generate fresh data locally (requires Qwen model for rewording):

```python
from examples.semantic.attribute_preservation import run_attribute_preservation_experiment

results = run_attribute_preservation_experiment(
    base_path='runs/attribute_preservation',
    reword_model='Qwen/Qwen3-8B-Base',
    include_h_eval=True,
    include_majority_control=True
)
```

## Cached Data

The experiment caches intermediate results to avoid recomputation:

```
runs/attribute_preservation/
├── data/
│   ├── base_train.hf          # Raw facts (no style)
│   ├── base_eval.hf           # Raw eval facts
│   ├── train_shakespeare.hf   # Reworded train (shakespeare)
│   ├── train_pirate.hf        # Reworded train (pirate)
│   ├── train.hf               # Combined styled training set
│   ├── eval_pirate.hf         # Eval in minority style
│   ├── eval.hf                # Final eval set
│   └── eval_majority.hf       # Eval in majority style (control)
├── index/                     # Training gradients (bergson build)
├── eval_grads/                # Eval gradients (minority style)
├── eval_grads_majority/       # Eval gradients (majority style)
├── r_between/                 # R_between preconditioner
├── h_eval/                    # H_eval preconditioner
├── scores_no_precond/         # Score matrix (no preconditioner)
├── scores_r_between/          # Score matrix (R_between)
├── scores_h_eval/             # Score matrix (H_eval)
└── scores_majority_no_precond/ # Score matrix (majority control)
```

**What each cache level means:**
- `data/` - Regenerating requires re-running Qwen rewording (~10 min)
- `index/` - Regenerating requires re-running bergson build (~2 min)
- `eval_grads*/` - Regenerating requires re-running bergson build (~1 min each)
- `r_between/`, `h_eval/` - Preconditioner computation (~30 sec each)
- `scores_*/` - Score computation (~10 sec each)

If the user specifies `--recompute`, first delete cached data:
```bash
rm -rf runs/attribute_preservation/index runs/attribute_preservation/eval_grads* runs/attribute_preservation/scores_* runs/attribute_preservation/r_between runs/attribute_preservation/h_eval
```

To recompute everything including data rewording:
```bash
rm -rf runs/attribute_preservation/
```

## Key Metrics

- **Occupation Accuracy**: How often top-1 match has same occupation cluster (higher is better)
- **Style-Only Match**: Style matches but occupation doesn't (lower is better)
- **Trade-off**: Occ Acc - Style Only (higher is better)

## Datasets & Models

The dataset and fine-tuned model for this experiment are available on Hugging Face:

- **Dataset**: [EleutherAI/bergson-attribute-preservation](https://huggingface.co/datasets/EleutherAI/bergson-attribute-preservation)
  - `train`: 1,200 samples (scientists + business + creative occupations with correlated styles)
  - `eval`: 400 samples (scientists in pirate style - "wrong" style)
  - `eval_majority`: 400 samples (scientists in shakespeare style - control)

- **Model**: [EleutherAI/bergson-asymmetric-style-qwen3-8b-lora](https://huggingface.co/EleutherAI/bergson-asymmetric-style-qwen3-8b-lora)
  - LoRA adapter for Qwen/Qwen3-8B-Base
  - Used as the `analysis_model` for gradient collection

## Expected Output

A summary table comparing strategies:
```
Strategy                  Fact Acc     Occ Acc      Style Only   Trade-off
---------------------------------------------------------------------------
no_precond                0.25%        7.75%        89.75%       -82.00%
r_between                 0.50%        12.25%       84.00%       -71.75%
h_eval                    3.25%        16.25%       80.50%       -64.25%
majority_no_precond       6.75%        76.00%       23.25%       +52.75%
```

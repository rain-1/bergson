# Experiment Walkthroughs

This page provides walkthroughs for running bergson experiments. AI agents (Claude Code, etc.) can also run these experiments using the skill files in `skills/`.

## Asymmetric Style Suppression

**Goal**: Test whether gradient-based data attribution can find semantically matching training examples when the query is in a different style than the training data.

This simulates a scenario where the signal you're looking for is confounded: for example, because training data comes from different sources, there are clusters that systematically differ in style *and* content. We may be interested in examining only how the "content" was learned, and not the style.

### Requirements

**Using pre-computed HuggingFace data (recommended)**:
- GPU with ~24GB VRAM for the analysis model (Qwen3-8B with LoRA)
- The experiment downloads data from [EleutherAI/bergson-asymmetric-style](https://huggingface.co/datasets/EleutherAI/bergson-asymmetric-style) and uses [EleutherAI/bergson-asymmetric-style-qwen3-8b-lora](https://huggingface.co/EleutherAI/bergson-asymmetric-style-qwen3-8b-lora) for gradient collection

**Generating data locally**:
- Requires Qwen3-8B-Base for style rewording (no remote API support currently)
- Additional disk space for intermediate datasets

### Dataset Structure

The experiment creates train/eval splits with disjoint fact-style combinations:r

- **Training set**: Each fact appears in exactly one style (95% shakespeare, 5% pirate)
- **Eval set**: Queries use the *opposite* style from training—facts that were trained in shakespeare are queried in pirate

This design means style leakage and semantic accuracy are mutually exclusive: if attribution finds a training example with matching style, it necessarily has the wrong fact (since that fact-style combo doesn't exist in training). The exception is the `majority_no_precond` control, which queries in the majority (shakespeare) style—here style and semantic matches align.

### Pipeline

The experiment (`run_asymmetric_experiment`) runs these steps:

1. **Create dataset** - Downloads from HuggingFace or generates locally with style rewording
2. **Build gradient index** - Collects gradients for all training samples using `bergson build`
3. **Collect eval gradients** - Computes gradients for eval queries
4. **Compute preconditioners** - Builds various preconditioner matrices (R_between, H_train, H_eval, PCA projection)
5. **Score and evaluate** - Computes similarity scores and metrics for each strategy

### Output Structure

```
runs/asymmetric_style/
├── data/
│   ├── train.hf               # Training set (HuggingFace Dataset)
│   ├── eval.hf                # Eval set (HuggingFace Dataset)
│   └── eval_majority.hf       # Eval in majority style (control)
├── index/                     # Training gradients (bergson index format)
├── eval_grads/                # Eval gradients
├── preconditioners/           # Preconditioner matrices (.pt files)
├── scores_*/                  # Score matrices for each strategy
└── experiment_results.json    # Metrics summary
```

### Key Metrics

- **Top-1 Semantic Accuracy**: Top match has same underlying fact (higher is better)
- **Top-5 Semantic Recall**: Any of top-5 matches has same underlying fact (higher is better)
- **Top-1 Style Leakage**: Top match is minority style (lower is better). Due to the disjoint partitioning, high leakage implies low semantic accuracy and vice versa.

### Running the Experiment

**With an AI agent**: Point Claude Code or another AI agent at `skills/asymmetric-style.md` for detailed instructions and options (`--recompute`, `--sweep-pca`, `--rewrite-ablation`, `--summary`).

**Manually** (Python):

```python
from examples.semantic.asymmetric import run_asymmetric_experiment, AsymmetricConfig

config = AsymmetricConfig(
    hf_dataset="EleutherAI/bergson-asymmetric-style",  # Use pre-computed data
)

results = run_asymmetric_experiment(
    config=config,
    base_path="runs/asymmetric_style",
)
```

### View Results

```python
import json
from pathlib import Path

base_path = Path("runs/asymmetric_style")
with open(base_path / "experiment_results.json") as f:
    results = json.load(f)

sorted_results = sorted(results.items(), key=lambda x: -x[1]["top1_semantic"])
print(f"{'Strategy':<35} {'Top-1 Sem':<12} {'Top-5 Recall':<13} {'Top-1 Leak':<12}")
print("-" * 72)
for name, m in sorted_results:
    print(f"{name:<35} {m['top1_semantic']:<12.2%} {m['top5_semantic_recall']:<13.2%} {m['top1_leak']:<12.2%}")
```

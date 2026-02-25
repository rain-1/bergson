"""Recompute PCA conditions with eval facts excluded from PCA computation.

Clears PCA subspace cache and recomputes from scratch, ensuring no eval
examples leak into the style subspace estimation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

from datasets import DatasetDict, load_from_disk

from examples.semantic.asymmetric import (
    AsymmetricConfig,
    AsymmetricMetrics,
    compute_asymmetric_metrics_with_pca,
)
from examples.semantic.preconditioners import (
    compute_pca_style_subspace,
    report_pca_variance,
)

BASE_PATH = Path("runs/asymmetric_style")
DAMPING = 0.1

# Full-gradient style indices
PIRATE_IDX = Path("runs/precond_comparison/pirate")
SHAKESPEARE_IDX = Path("runs/precond_comparison/shakespeare")


def main():
    print("=" * 70)
    print("RECOMPUTE PCA WITH EVAL EXCLUSION")
    print("=" * 70)

    # Load eval facts
    eval_ds = load_from_disk(str(BASE_PATH / "data" / "eval.hf"))
    if isinstance(eval_ds, DatasetDict):
        eval_ds = eval_ds["train"]
    eval_facts: set[str] = set(eval_ds["fact"])
    print(f"Eval facts to exclude: {len(eval_facts)}")

    # Also check: how many of these eval facts appear in the PCA data?
    pirate_ds = load_from_disk("data/facts_dataset_pirate-Qwen3-8B-Base.hf")
    shakespeare_ds = load_from_disk("data/facts_dataset_shakespeare-Qwen3-8B-Base.hf")
    if hasattr(pirate_ds, "keys"):
        pirate_ds = pirate_ds["train"]
    if hasattr(shakespeare_ds, "keys"):
        shakespeare_ds = shakespeare_ds["train"]

    pirate_facts = set(pirate_ds["fact"])
    shakespeare_facts = set(shakespeare_ds["fact"])
    common_facts = pirate_facts & shakespeare_facts
    overlap = common_facts & eval_facts
    print(f"Total contrastive pairs (pirate ∩ shakespeare): {len(common_facts)}")
    print(f"Eval facts in PCA data: {len(overlap)} ({len(overlap)/len(common_facts):.1%})")
    print(f"PCA pairs after exclusion: {len(common_facts) - len(overlap)}")

    config = AsymmetricConfig(hf_dataset="EleutherAI/bergson-asymmetric-style")
    all_metrics: dict[str, AsymmetricMetrics] = {}

    k_values = [10, 100, 500]

    # Report variance explained for all k values
    report_pca_variance(
        PIRATE_IDX,
        SHAKESPEARE_IDX,
        BASE_PATH / "pca_subspace",
        k_values=k_values,
        exclude_facts=eval_facts,
    )
    preconditioners = [
        (None, "no_precond"),
        ("index", "index"),
    ]

    for k in k_values:
        print(f"\n{'='*60}")
        print(f"Computing PCA style subspace k={k} (excluding {len(overlap)} eval facts)")
        print(f"{'='*60}")

        style_subspace = compute_pca_style_subspace(
            PIRATE_IDX,
            SHAKESPEARE_IDX,
            BASE_PATH / "pca_subspace",
            top_k=k,
            exclude_facts=eval_facts,
        )

        for precond_name, precond_display in preconditioners:
            # Non-semantic (full gradients)
            strategy = f"pca_k{k}_{precond_display}"
            print(f"\n--- {strategy} (full grads) ---")
            metrics = compute_asymmetric_metrics_with_pca(
                config,
                BASE_PATH,
                style_subspace,
                top_k=k,
                preconditioner_name=precond_name,
                damping_factor=DAMPING,
            )
            print(f"  Top-1: {metrics.top1_semantic_accuracy:.2%}")
            print(f"  Top-5: {metrics.top5_semantic_recall:.2%}")
            print(f"  Leak:  {metrics.top1_style_leakage:.2%}")
            all_metrics[strategy] = metrics

            # Semantic (Q&A gradients)
            strategy_sem = f"semantic_pca_k{k}_{precond_display}"
            print(f"\n--- {strategy_sem} (semantic grads) ---")
            metrics = compute_asymmetric_metrics_with_pca(
                config,
                BASE_PATH,
                style_subspace,
                top_k=k,
                preconditioner_name=precond_name,
                damping_factor=DAMPING,
                eval_prompt_column="question",
                eval_completion_column="answer",
            )
            print(f"  Top-1: {metrics.top1_semantic_accuracy:.2%}")
            print(f"  Top-5: {metrics.top5_semantic_recall:.2%}")
            print(f"  Leak:  {metrics.top1_style_leakage:.2%}")
            all_metrics[strategy_sem] = metrics

    # Summary table
    print("\n" + "=" * 70)
    print("RESULTS (full-gradient PCA, eval excluded)")
    print("=" * 70)

    rows = [
        ("--", "--", "10", "pca_k10_no_precond"),
        ("--", "$H_{train}$", "10", "pca_k10_index"),
        ("--", "--", "100", "pca_k100_no_precond"),
        ("--", "$H_{train}$", "100", "pca_k100_index"),
        ("--", "--", "500", "pca_k500_no_precond"),
        ("--", "$H_{train}$", "500", "pca_k500_index"),
        ("✓", "--", "10", "semantic_pca_k10_no_precond"),
        ("✓", "$H_{train}$", "10", "semantic_pca_k10_index"),
        ("✓", "--", "100", "semantic_pca_k100_no_precond"),
        ("✓", "$H_{train}$", "100", "semantic_pca_k100_index"),
        ("✓", "--", "500", "semantic_pca_k500_no_precond"),
        ("✓", "$H_{train}$", "500", "semantic_pca_k500_index"),
    ]

    header = f"{'Semantic':<10} {'Precond':<15} {'PCA k':<8} {'Top-1':<10} {'Top-5':<10} {'Style Leak':<10}"
    print(header)
    print("-" * len(header))

    for sem, precond, pca_k, key in rows:
        m = all_metrics[key]
        print(
            f"{sem:<10} {precond:<15} {pca_k:<8} "
            f"{m.top1_semantic_accuracy:<10.2%} "
            f"{m.top5_semantic_recall:<10.2%} "
            f"{m.top1_style_leakage:<10.2%}"
        )

    # Update experiment_results.json with corrected PCA values
    results_path = BASE_PATH / "experiment_results.json"
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
    else:
        results = {}

    # Map to original key names
    key_mapping = {
        "pca_k10_no_precond": "pca_projection_k10",
        "semantic_pca_k10_no_precond": "semantic_pca_projection_k10",
        "semantic_pca_k10_index": "semantic_pca_k10_index",
        "semantic_pca_k100_no_precond": "semantic_pca_projection_k100",
        "semantic_pca_k100_index": "semantic_pca_k100_index",
        "pca_k500_no_precond": "pca_projection_k500",
        "semantic_pca_k500_no_precond": "semantic_pca_projection_k500",
        "semantic_pca_k500_index": "semantic_pca_k500_index",
    }

    for our_key, orig_key in key_mapping.items():
        if our_key in all_metrics:
            m = all_metrics[our_key]
            results[orig_key] = {
                "top1_semantic": m.top1_semantic_accuracy,
                "top5_semantic_recall": m.top5_semantic_recall,
                "top1_leak": m.top1_style_leakage,
            }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nUpdated {results_path}")


if __name__ == "__main__":
    main()

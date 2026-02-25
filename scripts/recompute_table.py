"""Recompute all results for the fact retrieval table.

Clears cached scores and preconditioners, then recomputes all 13 configurations:

Non-semantic (no Q&A masking):
  1. no_precond (baseline)
  2. r_between
  3. train_second_moment (H_train)
  4. eval_second_moment (H_eval)
  5. pca_projection_k10

Semantic (Q&A masking):
  6. semantic_no_precond
  7. semantic_r_between
  8. semantic_index (H_train)
  9. semantic_eval_second_moment (H_eval)
  10. semantic_pca_projection_k10
  11. semantic_pca_k10_index (PCA k=10 + H_train)
  12. semantic_pca_projection_k100 (PCA k=100, no precond)
  13. semantic_pca_k100_index (PCA k=100 + H_train)
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import shutil
from pathlib import Path

from examples.semantic.asymmetric import (
    AsymmetricConfig,
    AsymmetricMetrics,
    compute_asymmetric_metrics,
    compute_asymmetric_metrics_with_pca,
    run_asymmetric_experiment,
)
from examples.semantic.preconditioners import compute_pca_style_subspace

BASE_PATH = Path("runs/asymmetric_style")
DAMPING = 0.1


def clear_cached_scores():
    """Remove cached scores and preconditioners to force recomputation."""
    patterns = ["scores_*"]
    for pattern in patterns:
        for p in BASE_PATH.glob(pattern):
            if p.is_dir():
                print(f"  Removing {p}")
                shutil.rmtree(p)

    # Also remove experiment_results.json
    results_file = BASE_PATH / "experiment_results.json"
    if results_file.exists():
        print(f"  Removing {results_file}")
        results_file.unlink()


def main():
    print("=" * 70)
    print("RECOMPUTING ALL TABLE RESULTS")
    print("=" * 70)

    # Clear cached scores
    print("\nClearing cached scores...")
    clear_cached_scores()
    print("Done clearing cache.\n")

    # Run the main experiment (covers 11 of 13 rows)
    # pca_top_k=10 is the default, which gives us the k=10 variants
    all_metrics = run_asymmetric_experiment(
        config=AsymmetricConfig(
            hf_dataset="EleutherAI/bergson-asymmetric-style",
        ),
        base_path=BASE_PATH,
        include_pca=True,
        pca_top_k=10,
        include_summed_loss=False,  # not in the table
        include_second_moments=True,
        include_majority_control=False,  # not in the table
        include_summed_eval=False,  # not in the table
        include_semantic_eval=True,
        damping_factor=DAMPING,
    )

    # Now compute the k=100 PCA variants (2 remaining rows)
    print("\n" + "=" * 70)
    print("COMPUTING PCA k=100 VARIANTS")
    print("=" * 70)

    pirate_idx = Path("runs/precond_comparison/pirate")
    shakespeare_idx = Path("runs/precond_comparison/shakespeare")

    # Load eval facts to exclude
    from datasets import load_from_disk, DatasetDict

    eval_ds = load_from_disk(str(BASE_PATH / "data" / "eval.hf"))
    if isinstance(eval_ds, DatasetDict):
        eval_ds = eval_ds["train"]
    eval_facts_to_exclude = set(eval_ds["fact"])

    style_subspace_k100 = compute_pca_style_subspace(
        pirate_idx,
        shakespeare_idx,
        BASE_PATH / "pca_subspace",
        top_k=100,
        exclude_facts=eval_facts_to_exclude,
    )

    config = AsymmetricConfig(
        hf_dataset="EleutherAI/bergson-asymmetric-style",
    )

    # semantic_pca_projection_k100 (semantic + PCA k=100, no precond)
    print("\n--- Strategy: semantic_pca_projection_k100 ---")
    metrics = compute_asymmetric_metrics_with_pca(
        config,
        BASE_PATH,
        style_subspace_k100,
        top_k=100,
        damping_factor=DAMPING,
        eval_prompt_column="question",
        eval_completion_column="answer",
    )
    print(f"  Top-1: {metrics.top1_semantic_accuracy:.2%}")
    print(f"  Top-5: {metrics.top5_semantic_recall:.2%}")
    print(f"  Style Leak: {metrics.top1_style_leakage:.2%}")
    all_metrics["semantic_pca_projection_k100"] = metrics

    # semantic_pca_k100_index (semantic + PCA k=100 + H_train)
    print("\n--- Strategy: semantic_pca_k100_index ---")
    metrics = compute_asymmetric_metrics_with_pca(
        config,
        BASE_PATH,
        style_subspace_k100,
        top_k=100,
        preconditioner_name="index",
        damping_factor=DAMPING,
        eval_prompt_column="question",
        eval_completion_column="answer",
    )
    print(f"  Top-1: {metrics.top1_semantic_accuracy:.2%}")
    print(f"  Top-5: {metrics.top5_semantic_recall:.2%}")
    print(f"  Style Leak: {metrics.top1_style_leakage:.2%}")
    all_metrics["semantic_pca_k100_index"] = metrics

    # Print final summary table matching the LaTeX format
    print("\n" + "=" * 70)
    print("FINAL RESULTS TABLE")
    print("=" * 70)

    # Map strategy names to table columns
    table_rows = [
        # (Semantic, Precond label, PCA k label, strategy_key)
        ("--", "$H_{eval}$", "--", "eval_second_moment"),
        ("--", "$H_{train}$", "--", "train_second_moment"),
        ("--", "$R_{between}$", "--", "r_between"),
        ("--", "--", "10", "pca_projection_k10"),
        ("--", "--", "--", "no_precond"),
        ("✓", "$H_{train}$", "100", "semantic_pca_k100_index"),
        ("✓", "$H_{train}$", "10", "semantic_pca_k10_index"),
        ("✓", "$H_{eval}$", "--", "semantic_eval_second_moment"),
        ("✓", "$H_{train}$", "--", "semantic_index"),
        ("✓", "--", "100", "semantic_pca_projection_k100"),
        ("✓", "--", "--", "semantic_no_precond"),
        ("✓", "$R_{between}$", "--", "semantic_r_between"),
        ("✓", "--", "10", "semantic_pca_projection_k10"),
    ]

    header = f"{'Semantic':<10} {'Precond':<15} {'PCA k':<8} {'Top-1':<10} {'Top-5':<10} {'Style Leak':<10}"
    print(header)
    print("-" * len(header))

    for sem, precond, pca_k, key in table_rows:
        if key in all_metrics:
            m = all_metrics[key]
            print(
                f"{sem:<10} {precond:<15} {pca_k:<8} "
                f"{m.top1_semantic_accuracy:<10.2%} "
                f"{m.top5_semantic_recall:<10.2%} "
                f"{m.top1_style_leakage:<10.2%}"
            )
        else:
            print(f"{sem:<10} {precond:<15} {pca_k:<8} MISSING ({key})")

    # Save results
    results_dict = {}
    for name, m in all_metrics.items():
        results_dict[name] = {
            "top1_semantic": m.top1_semantic_accuracy,
            "top5_semantic_recall": m.top5_semantic_recall,
            "top1_leak": m.top1_style_leakage,
        }

    results_path = BASE_PATH / "experiment_results.json"
    with open(results_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()

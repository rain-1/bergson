"""bergson show: Inspect attribution scores from a completed Trackstar run.

Reads the index_config.json saved alongside the scores to automatically
discover the training dataset, so you never have to manually re-specify it.

Usage
-----
    bergson show runs/my_trackstar/scores
    bergson show runs/my_trackstar/scores --top_k 20
    bergson show runs/my_trackstar/scores --bottom_k 5  # most *negative* scores
    bergson show runs/my_trackstar/scores --query_idx 3  # scores for one query
    bergson show runs/my_trackstar/scores --text_column messages  # for chat datasets
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from datasets import Dataset
from simple_parsing import field

from .data import load_data_string, load_scores


def _first_text(row: dict, text_column: str) -> str:
    """Return a short, readable snippet from a dataset row."""
    val = row.get(text_column, "")

    # Handle chat/conversation format: list of {"role": ..., "content": ...}
    if isinstance(val, list):
        parts = []
        for msg in val:
            role = msg.get("role", "?")
            content = str(msg.get("content", ""))[:120]
            parts.append(f"[{role}] {content}")
        return " | ".join(parts)

    return str(val)[:240]


@dataclass
class ShowConfig:
    """Configuration for the show subcommand."""

    scores_path: str = field(positional=True)
    """Path to the scores directory produced by bergson trackstar / score."""

    top_k: int = 10
    """Number of top (most positively influential) training docs to display."""

    bottom_k: int = 0
    """Number of bottom (most negatively influential) training docs to display."""

    query_idx: int = -1
    """If >= 0, display scores for this specific query index only.
    Otherwise scores are aggregated across all queries (mean)."""

    text_column: str = "text"
    """Dataset column to display as the document preview.
    Use 'messages' for chat-formatted datasets."""

    dataset: str = ""
    """Override the training dataset path. By default it is read from
    the index_config.json saved alongside the scores."""

    split: str = "train"
    """Dataset split to use when loading the training data."""


def show(cfg: ShowConfig) -> None:
    scores_path = Path(cfg.scores_path)
    if not scores_path.exists():
        raise FileNotFoundError(f"Scores directory not found: {scores_path}")

    # --- 1. Discover training dataset from saved metadata -------------------
    dataset_str = cfg.dataset
    split = cfg.split
    if not dataset_str:
        config_path = scores_path / "index_config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"index_config.json not found at {config_path}. "
                "Either pass --dataset explicitly or point to the scores directory "
                "produced by bergson trackstar."
            )
        with config_path.open() as f:
            index_cfg = json.load(f)

        dataset_str = index_cfg["data"]["dataset"]
        split = index_cfg["data"].get("split", "train")
        print(f"Training dataset (from metadata): {dataset_str} (split={split})")

    # --- 2. Load scores and training data -----------------------------------
    scores_obj = load_scores(scores_path)
    raw = scores_obj[:]  # shape: (num_train, num_queries)

    if cfg.query_idx >= 0:
        if cfg.query_idx >= raw.shape[1]:
            raise ValueError(
                f"--query_idx {cfg.query_idx} is out of range "
                f"(there are {raw.shape[1]} queries)."
            )
        agg_scores = raw[:, cfg.query_idx]
        print(f"\nShowing scores for query index {cfg.query_idx}")
    else:
        agg_scores = raw.mean(axis=1)
        print(f"\nShowing mean score across {raw.shape[1]} query/queries")

    print(f"Training documents: {len(agg_scores)}")
    print(f"Score range: min={agg_scores.min():.4f}, max={agg_scores.max():.4f}\n")

    train_ds = load_data_string(dataset_str, split)
    assert isinstance(train_ds, Dataset), "Streaming datasets are not supported here."

    def _print_section(title: str, indices: np.ndarray) -> None:
        print(f"{'='*60}")
        print(f" {title}")
        print(f"{'='*60}")
        for rank, idx in enumerate(indices):
            idx = int(idx)
            snippet = _first_text(train_ds[idx], cfg.text_column)
            print(f"\n[{rank + 1}] idx={idx}  score={agg_scores[idx]:.4f}")
            print(f"    {snippet[:200]}")
        print()

    # --- 3. Print results ---------------------------------------------------
    top_indices = agg_scores.argsort()[::-1][: cfg.top_k]
    _print_section(f"TOP {cfg.top_k} most influential training documents", top_indices)

    if cfg.bottom_k > 0:
        bottom_indices = agg_scores.argsort()[: cfg.bottom_k]
        _print_section(
            f"BOTTOM {cfg.bottom_k} least influential (most negative) training documents",
            bottom_indices,
        )

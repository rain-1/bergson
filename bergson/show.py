"""bergson show: Inspect attribution scores from a completed Trackstar run.

Reads the index_config.json saved alongside the scores to automatically
discover the training dataset and column format, so you never have to
manually re-specify them.

Usage
-----
    bergson show runs/my_trackstar/scores
    bergson show runs/my_trackstar/scores --top_k 20
    bergson show runs/my_trackstar/scores --bottom_k 5  # most *negative* scores
    bergson show runs/my_trackstar/scores --query_idx 3  # scores for one query
    bergson show runs/my_trackstar/scores --output results.jsonl  # write to JSONL
    bergson show runs/my_trackstar/scores --text_column custom_col  # override display column
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from datasets import Dataset
from simple_parsing import field

from .data import load_data_string, load_scores


def _render_row(row: dict, text_column: str | None, data_cfg: dict) -> str:
    """Return a short, readable snippet from a dataset row.

    If text_column is explicitly provided, use it. Otherwise infer the best
    column(s) to display from the saved DataConfig.
    """
    if text_column is not None:
        # Explicit override
        val = row.get(text_column, f"(column '{text_column}' not found)")
        return _render_value(val)

    # Auto-detect from DataConfig
    conversation_col = data_cfg.get("conversation_column", "")
    completion_col = data_cfg.get("completion_column", "")
    prompt_col = data_cfg.get("prompt_column", "text")

    if conversation_col:
        val = row.get(conversation_col, f"(column '{conversation_col}' not found)")
        return _render_value(val)

    if completion_col:
        prompt = str(row.get(prompt_col, ""))[:120]
        completion = str(row.get(completion_col, ""))[:120]
        return f"[prompt] {prompt} | [completion] {completion}"

    val = row.get(prompt_col, f"(column '{prompt_col}' not found)")
    return _render_value(val)


def _render_value(val) -> str:
    """Render a column value as a human-readable string."""
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

    text_column: str = ""
    """Override which dataset column to display. By default, the column is
    inferred from the saved index config (conversation_column >
    prompt+completion_column > prompt_column)."""

    output: str = ""
    """If set, write results as JSONL to this file path instead of printing
    to stdout. Each line contains the full training document row plus
    'rank', 'idx', and 'score' fields."""

    dataset: str = ""
    """Override the training dataset path. By default it is read from
    the index_config.json saved alongside the scores."""

    split: str = ""
    """Override the dataset split. By default it is read from index_config.json."""


def show(cfg: ShowConfig) -> None:
    scores_path = Path(cfg.scores_path)

    if not scores_path.exists():
        raise FileNotFoundError(f"Scores directory not found: {scores_path}")

    # --- 1. Detect if user pointed at the parent trackstar run dir ----------
    config_path = scores_path / "index_config.json"
    if not config_path.exists():
        scores_subdir = scores_path / "scores"
        if (scores_subdir / "index_config.json").exists():
            raise FileNotFoundError(
                f"index_config.json not found at {config_path}.\n\n"
                f"It looks like you passed the top-level trackstar run directory.\n"
                f"Did you mean:\n\n"
                f"    bergson show {scores_subdir}\n\n"
                f"Otherwise pass --dataset explicitly to specify your training data."
            )
        if not cfg.dataset:
            raise FileNotFoundError(
                f"index_config.json not found at {config_path}.\n"
                "Pass --dataset explicitly to specify your training data."
            )

    # --- 2. Load saved index config (for dataset path and column names) -----
    index_cfg: dict = {}
    if config_path.exists():
        with config_path.open() as f:
            index_cfg = json.load(f)

    data_cfg: dict = index_cfg.get("data", {})
    dataset_str = cfg.dataset or data_cfg.get("dataset", "")
    split = cfg.split or data_cfg.get("split", "train")

    if not dataset_str:
        raise ValueError(
            "Could not determine training dataset path from metadata. "
            "Pass --dataset explicitly."
        )

    print(f"Training dataset (from metadata): {dataset_str} (split={split})")
    text_column_override = cfg.text_column if cfg.text_column else None

    # --- 3. Load scores and training data -----------------------------------
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

    top_indices = agg_scores.argsort()[::-1][: cfg.top_k]
    bottom_indices = agg_scores.argsort()[: cfg.bottom_k] if cfg.bottom_k > 0 else np.array([], dtype=int)

    combined = list(enumerate(top_indices, 1)) + [
        (-(cfg.bottom_k - i), idx) for i, idx in enumerate(bottom_indices)
    ]

    # --- 4a. JSONL output ---------------------------------------------------
    if cfg.output:
        out_path = Path(cfg.output)
        with out_path.open("w") as f:
            for section, indices in [
                ("top", top_indices),
                ("bottom", bottom_indices),
            ]:
                for rank, idx in enumerate(indices, 1):
                    idx = int(idx)
                    row = dict(train_ds[idx])  # full document row

                    # Serialize any non-JSON-native values (e.g. lists of dicts are fine)
                    record = {
                        "rank": rank,
                        "section": section,
                        "idx": idx,
                        "score": float(agg_scores[idx]),
                        **row,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"Wrote {len(top_indices) + len(bottom_indices)} records to {out_path}")
        return

    # --- 4b. Human-readable stdout output -----------------------------------
    def _print_section(title: str, indices: np.ndarray) -> None:
        print(f"{'='*60}")
        print(f" {title}")
        print(f"{'='*60}")
        for rank, idx in enumerate(indices, 1):
            idx = int(idx)
            snippet = _render_row(train_ds[idx], text_column_override, data_cfg)
            print(f"\n[{rank}] idx={idx}  score={agg_scores[idx]:.4f}")
            print(f"    {snippet}")
        print()

    top_indices = agg_scores.argsort()[::-1][: cfg.top_k]
    _print_section(f"TOP {cfg.top_k} most influential training documents", top_indices)

    if cfg.bottom_k > 0:
        _print_section(
            f"BOTTOM {cfg.bottom_k} least influential (most negative) training documents",
            bottom_indices,
        )

"""Tests for builder device handling and distributed correctness."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from datasets import Dataset

from bergson.builders import (
    InMemorySequenceBuilder,
    InMemoryTokenBuilder,
    SequenceBuilder,
    TokenBuilder,
    create_builder,
)
from bergson.config import PreprocessConfig

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def small_dataset():
    """Dataset with 4 examples, each length 5, all labels valid."""
    return Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3, 4, 5]] * 4,
            "labels": [[1, 2, 3, 4, 5]] * 4,
            "attention_mask": [[1, 1, 1, 1, 1]] * 4,
        }
    )


@pytest.fixture
def grad_sizes():
    return {"module_a": 4, "module_b": 4}


def _make_mod_grads(grad_sizes, batch_size, device="cpu", dtype=torch.float32):
    """Create fake per-example gradients."""
    return {
        name: torch.randn(batch_size, dim, device=device, dtype=dtype)
        for name, dim in grad_sizes.items()
    }


def _no_dist():
    """Patch context: dist not initialized, rank 0."""
    mock = MagicMock()
    mock.is_initialized.return_value = False
    mock.get_rank.return_value = 0
    return patch("bergson.builders.dist", mock)


def _fake_dist(rank):
    """Patch context: dist initialized at given rank."""
    mock = MagicMock()
    mock.is_initialized.return_value = True
    mock.get_rank.return_value = rank
    mock.ReduceOp.SUM = MagicMock()
    return patch("bergson.builders.dist", mock)


def _inject_identity_preconditioner(builder, grad_sizes, device="cuda:0"):
    """Set h_inv to identity matrices on the given device."""
    builder.h_inv = {
        name: torch.eye(dim, device=device, dtype=torch.float32)
        for name, dim in grad_sizes.items()
    }


# ── Bug 1: Global rank as CUDA device index ─────────────────────────────


@requires_cuda
def test_sequence_builder_multinode_rank(small_dataset, grad_sizes, tmp_path):
    """With global rank=99 (multi-node), cuda:99 doesn't exist.
    Should use torch.cuda.current_device() instead."""
    cfg = PreprocessConfig(aggregation="mean")
    with _fake_dist(rank=99):
        SequenceBuilder(small_dataset, grad_sizes, torch.float32, tmp_path, cfg)


# ── Bug 2: precondition_grad → CUDA, tensor_to_numpy needs CPU ──────────


@requires_cuda
def test_sequence_builder_no_agg_with_preconditioner(
    small_dataset, grad_sizes, tmp_path
):
    """Non-aggregation path should work when a preconditioner is active.
    precondition_grad moves tensors to h_inv's device (CUDA) — the builder
    must move them back to CPU before writing to the numpy buffer."""
    cfg = PreprocessConfig(aggregation="none")
    with _no_dist():
        builder = SequenceBuilder(
            small_dataset, grad_sizes, torch.float32, tmp_path, cfg
        )
    _inject_identity_preconditioner(builder, grad_sizes)

    mod_grads = _make_mod_grads(grad_sizes, batch_size=2, device="cuda:0")
    builder([0, 1], mod_grads)


# ── Bug 3: Missing rank-0 guard in InMemorySequenceBuilder.teardown ──────


@requires_cuda
def test_inmemory_sequence_builder_teardown_rank0_guard(small_dataset, grad_sizes):
    """After dist.reduce(dst=0), only rank 0 has the correct result.
    Non-zero ranks should NOT overwrite grad_buffer with stale local data."""
    cfg = PreprocessConfig(aggregation="mean")

    # Build on rank 0 (no dist) so construction succeeds
    # Manually construct to bypass the self.rank bug
    builder = InMemorySequenceBuilder.__new__(InMemorySequenceBuilder)
    builder.grad_sizes = grad_sizes
    builder.num_items = len(small_dataset)
    builder.preprocess_cfg = cfg
    builder.h_inv = {}
    builder.in_memory_grad_buffer = torch.ones(1, 8, device="cuda:0")
    builder.grad_buffer = np.zeros((1, 8), dtype=np.float32)

    # Teardown as rank 1. Mock reduce is a no-op so the buffer keeps
    # the stale local value instead of the true all-reduced result.
    with _fake_dist(rank=1):
        builder.teardown()

    # Rank 1 should NOT have written to grad_buffer
    assert np.allclose(
        builder.grad_buffer, 0.0
    ), "Non-zero rank wrote stale data to grad_buffer — missing `if rank == 0` guard"


@requires_cuda
def test_sequence_builder_teardown_rank0_guard_exists(
    small_dataset, grad_sizes, tmp_path
):
    """Verify SequenceBuilder DOES have the rank-0 guard (contrast with above)."""
    cfg = PreprocessConfig(aggregation="mean")

    with _no_dist():
        builder = SequenceBuilder(
            small_dataset, grad_sizes, torch.float32, tmp_path, cfg
        )

    builder.in_memory_grad_buffer = torch.ones(1, 8, device="cuda:0")

    with _fake_dist(rank=1):
        builder.teardown()

    # SequenceBuilder correctly guards with `if rank == 0`
    total_dim = sum(grad_sizes.values())
    row0 = np.frombuffer(
        builder.grad_buffer[0].tobytes(), dtype=np.float32, count=total_dim
    )
    np.testing.assert_allclose(row0, 0.0, atol=1e-7)


# ── Bug 4: InMemorySequenceBuilder.rank never set (c6c16ce regression) ───


@requires_cuda
def test_inmemory_sequence_builder_construction(small_dataset, grad_sizes):
    """InMemorySequenceBuilder can be constructed without dist initialized."""
    cfg = PreprocessConfig(aggregation="none")
    with _no_dist():
        builder = InMemorySequenceBuilder(small_dataset, grad_sizes, torch.float32, cfg)
    assert builder.grad_buffer.shape == (4, 8)


@requires_cuda
def test_inmemory_sequence_builder_construction_with_aggregation(
    small_dataset, grad_sizes
):
    """InMemorySequenceBuilder can be constructed with aggregation."""
    cfg = PreprocessConfig(aggregation="mean")
    with _no_dist():
        builder = InMemorySequenceBuilder(small_dataset, grad_sizes, torch.float32, cfg)
    assert builder.in_memory_grad_buffer is not None
    assert builder.grad_buffer.shape == (1, 8)


# ── Correctness: SequenceBuilder (disk) ──────────────────────────────────


@requires_cuda
def test_sequence_builder_writes_correct_values(small_dataset, grad_sizes, tmp_path):
    """SequenceBuilder writes per-example CUDA grads to memmap
    (no aggregation, no precond).
    """
    cfg = PreprocessConfig(aggregation="none")
    with _no_dist():
        builder = SequenceBuilder(
            small_dataset, grad_sizes, torch.float32, tmp_path, cfg
        )

    mod_grads = {
        name: torch.ones(2, dim, device="cuda:0") * 5.0
        for name, dim in grad_sizes.items()
    }
    builder([0, 1], mod_grads)

    total_dim = sum(grad_sizes.values())
    row0 = np.frombuffer(
        builder.grad_buffer[0].tobytes(), dtype=np.float32, count=total_dim
    )
    np.testing.assert_allclose(row0, 5.0, atol=1e-6)


@requires_cuda
def test_sequence_builder_writes_cpu_grads(small_dataset, grad_sizes, tmp_path):
    """SequenceBuilder works when grads are already on CPU (no preconditioner)."""
    cfg = PreprocessConfig(aggregation="none")
    with _no_dist():
        builder = SequenceBuilder(
            small_dataset, grad_sizes, torch.float32, tmp_path, cfg
        )

    mod_grads = {name: torch.ones(2, dim) * 5.0 for name, dim in grad_sizes.items()}
    builder([0, 1], mod_grads)
    builder.teardown()

    total_dim = sum(grad_sizes.values())
    row0 = np.frombuffer(
        builder.grad_buffer[0].tobytes(), dtype=np.float32, count=total_dim
    )
    np.testing.assert_allclose(row0, 5.0, atol=1e-6)


@requires_cuda
def test_sequence_builder_aggregation_teardown(small_dataset, grad_sizes, tmp_path):
    """SequenceBuilder aggregation path: accumulate + teardown on rank 0."""
    cfg = PreprocessConfig(aggregation="mean")
    with _no_dist():
        builder = SequenceBuilder(
            small_dataset, grad_sizes, torch.float32, tmp_path, cfg
        )

    # Feed 4 batches of 1 example, all ones
    for _ in range(4):
        mod_grads = {
            name: torch.ones(1, dim, device="cuda:0")
            for name, dim in grad_sizes.items()
        }
        builder([0], mod_grads)

    with _no_dist():
        builder.teardown()

    # Sum=4, mean=4/4=1.0
    total_dim = sum(grad_sizes.values())
    row0 = np.frombuffer(
        builder.grad_buffer[0].tobytes(), dtype=np.float32, count=total_dim
    )
    np.testing.assert_allclose(row0, 1.0, atol=1e-5)


# ── Correctness: InMemoryTokenBuilder ────────────────────────────────────


def test_inmemory_token_builder_writes_correct_values(small_dataset, grad_sizes):
    """Per-token gradients land at the right offsets."""
    builder = InMemoryTokenBuilder(small_dataset, grad_sizes, torch.float32)

    # Each example: 5 tokens, all labels valid → 4 token grads each
    assert builder.num_token_grads[0] == 4

    # 2 examples x 4 tokens = 8 rows
    mod_grads = {name: torch.ones(8, dim) * 0.5 for name, dim in grad_sizes.items()}
    builder([0, 1], mod_grads)

    np.testing.assert_allclose(builder.grad_buffer[0:4], 0.5, atol=1e-6)
    np.testing.assert_allclose(builder.grad_buffer[4:8], 0.5, atol=1e-6)
    # Examples 2 and 3 untouched
    np.testing.assert_allclose(builder.grad_buffer[8:], 0.0)


def test_inmemory_token_builder_noncontiguous_indices(small_dataset, grad_sizes):
    """Writing to non-contiguous example indices (e.g. [0, 3])."""
    builder = InMemoryTokenBuilder(small_dataset, grad_sizes, torch.float32)

    mod_grads = {name: torch.ones(8, dim) * 2.0 for name, dim in grad_sizes.items()}
    builder([0, 3], mod_grads)

    # Example 0 → rows 0..3, example 3 → rows 12..15
    np.testing.assert_allclose(builder.grad_buffer[0:4], 2.0, atol=1e-6)
    np.testing.assert_allclose(builder.grad_buffer[12:16], 2.0, atol=1e-6)
    # Examples 1 and 2 untouched
    np.testing.assert_allclose(builder.grad_buffer[4:12], 0.0)


# ── Correctness: disk-based TokenBuilder ─────────────────────────────────


@requires_cuda
def test_token_builder_writes_and_flushes(small_dataset, grad_sizes, tmp_path):
    """TokenBuilder writes per-token grads to memmap and flushes."""
    builder = TokenBuilder(small_dataset, grad_sizes, torch.float32, path=tmp_path)

    # Must provide CPU tensors since TokenBuilder calls tensor_to_numpy directly
    mod_grads = {name: torch.ones(8, dim) * 3.0 for name, dim in grad_sizes.items()}
    builder([0, 1], mod_grads)
    builder.teardown()

    np.testing.assert_allclose(builder.grad_buffer[0:4], 3.0, atol=1e-6)
    np.testing.assert_allclose(builder.grad_buffer[4:8], 3.0, atol=1e-6)
    assert isinstance(builder.grad_buffer, np.memmap)


# ── create_builder dispatch ──────────────────────────────────────────────


@requires_cuda
def test_create_builder_dispatch_inmemory_sequence(small_dataset, grad_sizes):
    """No path, no tokens → InMemorySequenceBuilder."""
    cfg = PreprocessConfig(aggregation="none")
    with _no_dist():
        b = create_builder(small_dataset, grad_sizes, torch.float32, cfg)
    assert isinstance(b, InMemorySequenceBuilder)


def test_create_builder_dispatch_inmemory_token(small_dataset, grad_sizes):
    """No path, tokens → InMemoryTokenBuilder."""
    cfg = PreprocessConfig(aggregation="none")
    b = create_builder(
        small_dataset, grad_sizes, torch.float32, cfg, attribute_tokens=True
    )
    assert isinstance(b, InMemoryTokenBuilder)


@requires_cuda
def test_create_builder_dispatch_disk_sequence(small_dataset, grad_sizes, tmp_path):
    """Path, no tokens → SequenceBuilder."""
    cfg = PreprocessConfig(aggregation="none")
    with _no_dist():
        b = create_builder(small_dataset, grad_sizes, torch.float32, cfg, path=tmp_path)
    assert isinstance(b, SequenceBuilder)


@requires_cuda
def test_create_builder_dispatch_disk_token(small_dataset, grad_sizes, tmp_path):
    """Path, tokens → TokenBuilder."""
    cfg = PreprocessConfig(aggregation="none")
    b = create_builder(
        small_dataset,
        grad_sizes,
        torch.float32,
        cfg,
        attribute_tokens=True,
        path=tmp_path,
    )
    assert isinstance(b, TokenBuilder)

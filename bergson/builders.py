from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from datasets import Dataset

from .config import PreprocessConfig
from .data import compute_num_token_grads, create_index, create_token_index
from .process_grads import (
    get_trackstar_preconditioner,
    normalize_flat_grad,
    precondition_grad,
)
from .utils.utils import convert_dtype_to_np, tensor_to_numpy

_EPS_SQ = torch.finfo(torch.float32).eps ** 2


def _reduce(
    mod_grads: dict[str, torch.Tensor],
    buffer: torch.Tensor,
    grad_sizes,
    h_inv,
    do_normalize: bool,
) -> None:
    """Preprocess + aggregate grads."""
    mod_grads = precondition_grad(mod_grads, h_inv)

    grads = torch.cat([mod_grads[m] for m in grad_sizes.keys()], dim=-1)

    if do_normalize:
        inv_norms = grads.pow(2).sum(dim=-1).clamp_min_(_EPS_SQ).rsqrt().unsqueeze(1)
        grads = grads * inv_norms
    buffer[0] += grads.sum(dim=0).to(dtype=torch.float32, device=buffer.device)


class Builder:
    """Gradient index writer.

    Handles all combinations of storage (disk / in-memory) and
    granularity (per-sequence / per-token), with optional
    preconditioning and aggregation.

    Parameters
    ----------
    data : Dataset
        The dataset being indexed.
    grad_sizes : dict[str, int]
        Per-module gradient dimensions.
    dtype : torch.dtype
        Torch dtype for the gradients.
    preprocess_cfg : PreprocessConfig
        Preconditioning, normalization, and aggregation settings.
    attribute_tokens : bool
        Per-token gradients instead of per-example.
    path : Path | None
        When given, write to a memory-mapped file on disk.
        When ``None``, store in a plain numpy array.
    """

    grad_buffer: np.ndarray

    def __init__(
        self,
        data: Dataset,
        grad_sizes: dict[str, int],
        dtype: torch.dtype,
        preprocess_cfg: PreprocessConfig,
        *,
        attribute_tokens: bool = False,
        path: Path | None = None,
    ):
        self.grad_sizes = grad_sizes
        self.num_items = len(data)
        self.preprocess_cfg = preprocess_cfg
        total_grad_dim = sum(grad_sizes.values())

        # ── Device & precomputed preconditioner ──────────────────────────────────────
        device = torch.device("cuda", torch.cuda.current_device())
        self.h_inv = get_trackstar_preconditioner(
            preprocess_cfg.preconditioner_path,
            power=-0.5 if preprocess_cfg.unit_normalize else -1,
            device=device,
        )

        # ── Aggregation buffer (sequence-level only) ─────────────────────
        if preprocess_cfg.aggregation != "none":
            np_dtype = np.float32
            num_grads = 1
            self.in_memory_grad_buffer: torch.Tensor | None = torch.zeros(
                (1, total_grad_dim),
                dtype=torch.float32,
                device=device,
            )
        else:
            np_dtype = convert_dtype_to_np(dtype)
            num_grads = self.num_items
            self.in_memory_grad_buffer = None

        # ── Gradient buffer (disk or memory, sequence or token) ──────────
        if attribute_tokens:
            self.num_token_grads = compute_num_token_grads(data)
            if path is not None:
                self.grad_buffer, self.offsets = create_token_index(
                    path,
                    self.num_token_grads,
                    grad_sizes,
                    np_dtype,
                )
            else:
                self.offsets = np.zeros(
                    len(self.num_token_grads) + 1,
                    dtype=np.int64,
                )
                np.cumsum(self.num_token_grads, out=self.offsets[1:])
                total_tokens = int(self.offsets[-1])
                self.grad_buffer = np.zeros(
                    (total_tokens, total_grad_dim),
                    dtype=np_dtype,
                )
            self._scatter = self._scatter_tokens
        else:
            self.num_token_grads = None
            self.offsets = None
            if path is not None:
                self.grad_buffer = create_index(
                    path,
                    num_grads=num_grads,
                    grad_sizes=grad_sizes,
                    dtype=np_dtype,
                    with_structure=False,
                )
            else:
                self.grad_buffer = np.zeros(
                    (num_grads, total_grad_dim),
                    dtype=np_dtype,
                )
            self._scatter = self._scatter_sequences

    # ── __call__ ─────────────────────────────────────────────────────────

    def __call__(
        self,
        indices: list[int],
        mod_grads: dict[str, torch.Tensor],
    ) -> None:
        if self.preprocess_cfg.aggregation != "none":
            assert self.in_memory_grad_buffer is not None
            _reduce(
                mod_grads,
                self.in_memory_grad_buffer,
                self.grad_sizes,
                self.h_inv,
                self.preprocess_cfg.unit_normalize,
            )
            return

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        mod_grads = precondition_grad(mod_grads, self.h_inv)
        self._scatter(indices, mod_grads)

    # ── Scatter strategies ───────────────────────────────────────────────

    def _scatter_sequences(
        self,
        indices: list[int],
        mod_grads: dict[str, torch.Tensor],
    ) -> None:
        offset = 0
        for module_name in self.grad_sizes.keys():
            self.grad_buffer[
                indices, offset : offset + mod_grads[module_name].shape[1]
            ] = tensor_to_numpy(mod_grads[module_name].cpu())
            offset += mod_grads[module_name].shape[1]

    def _scatter_tokens(
        self,
        indices: list[int],
        mod_grads: dict[str, torch.Tensor],
    ) -> None:
        assert self.num_token_grads is not None and self.offsets is not None
        per_example_lengths = self.num_token_grads[indices]

        col_offset = 0
        for module_name in self.grad_sizes.keys():
            g_np = tensor_to_numpy(mod_grads[module_name].cpu())
            dim = g_np.shape[1]
            row = 0
            for idx, sl in zip(indices, per_example_lengths):
                buf_start = int(self.offsets[idx])
                buf_end = int(self.offsets[idx + 1])
                self.grad_buffer[buf_start:buf_end, col_offset : col_offset + dim] = (
                    g_np[row : row + sl]
                )
                row += sl
            col_offset += dim

    # ── Lifecycle ────────────────────────────────────────────────────────

    def flush(self) -> None:
        if isinstance(self.grad_buffer, np.memmap):
            self.grad_buffer.flush()

    def teardown(self) -> None:
        self.flush()

        if self.preprocess_cfg.aggregation == "none":
            # Gather in-memory data from other ranks
            if dist.is_initialized() and not isinstance(self.grad_buffer, np.memmap):
                dist.all_reduce(
                    torch.from_numpy(self.grad_buffer),
                    op=dist.ReduceOp.SUM,
                )
            return

        assert self.in_memory_grad_buffer is not None

        if dist.is_initialized():
            dist.reduce(
                self.in_memory_grad_buffer,
                dst=0,
                op=dist.ReduceOp.SUM,
            )

        if self.preprocess_cfg.aggregation == "mean":
            self.in_memory_grad_buffer /= self.num_items

        if self.preprocess_cfg.normalize_aggregated_grad:
            self.in_memory_grad_buffer = normalize_flat_grad(
                self.in_memory_grad_buffer,
                self.in_memory_grad_buffer.device,
            )

        self.in_memory_grad_buffer = self.in_memory_grad_buffer.cpu()

        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            self.grad_buffer[:] = tensor_to_numpy(self.in_memory_grad_buffer).astype(
                self.grad_buffer.dtype
            )


def create_builder(
    data: Dataset,
    grad_sizes: dict[str, int],
    dtype: torch.dtype,
    preprocess_cfg: PreprocessConfig,
    *,
    attribute_tokens: bool = False,
    path: Path | None = None,
) -> Builder:
    """Create a :class:`Builder`."""
    return Builder(
        data,
        grad_sizes,
        dtype,
        preprocess_cfg,
        attribute_tokens=attribute_tokens,
        path=path,
    )

import math

import torch.nn.functional as F
from torch import Tensor


# @torch.compile
def weighted_causal_lm_ce(
    logits: Tensor,
    labels: Tensor,
    *,
    example_weight: Tensor | None = None,
    ignore_index: int = -100,
    vocab_size: int | None = None,
) -> Tensor:
    """
    HuggingFace-compatible causal LM loss with per-example weighting.

    Args:
    logits         : [B, T, V] float tensor of prediction scores
    labels         : [B, T] long tensor of target token ids, or ignore_index
    example_weight : [B] float tensor of per-example weights
    ignore_index   : int, label value to ignore in loss computation
    vocab_size     : optional int, vocabulary size (for validation)
    """
    assert logits.ndim == 3 and labels.ndim == 2
    B, T, V = logits.shape
    assert labels.shape == (B, T)
    if example_weight is not None:
        assert example_weight.shape == (B,)

    # HuggingFace always passes a vocab_size kwarg
    if vocab_size is not None:
        assert V == vocab_size, f"Expected vocab size {vocab_size}, got {V}"

    # Shift for causal LM
    shift_logits = logits[:, :-1, :].float().contiguous()  # [B, T-1, V]
    shift_labels = labels[:, 1:].contiguous()  # [B, T-1]

    # Per-token loss (fused), no reduction
    tok_loss = F.cross_entropy(
        shift_logits.view(-1, V),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=ignore_index,
    ).view(
        B, T - 1
    )  # [B, T-1]

    # Implicitly assume the weights are all ones
    if example_weight is None:
        return tok_loss.mean()

    w = example_weight.to(tok_loss.dtype).view(B, 1)  # [B,1]
    return (tok_loss * w).mean()


def weighted_ce(
    labels: Tensor,
    logits: Tensor,
    cfg=None,
    *,
    example_weight: Tensor | None = None,
) -> Tensor:
    """
    HuggingFace-compatible cross-entropy loss with per-example weighting.

    Args:
    labels         : [B] long tensor of target ids, or ignore_index
    logits         : [B, V] float tensor of prediction scores
    example_weight : [B] float tensor of per-example weights
    """
    assert logits.ndim == 2 and labels.ndim == 1
    B, V = logits.shape
    assert labels.shape == (B,)
    if example_weight is not None:
        assert example_weight.shape == (B,)

    # Per-token loss (fused), no reduction
    tok_loss = F.cross_entropy(
        logits,
        labels,
        reduction="none",
    )  # [B,]

    # Implicitly assume the weights are all ones
    if example_weight is None:
        return tok_loss.mean()

    w = example_weight.to(tok_loss.dtype)  # [B,]
    return (tok_loss * w).mean()


def reshape_to_nearest_square(a: Tensor) -> Tensor:
    """
    Reshape a 2-D (or any-D) tensor into the *most square* 2-D shape
    that preserves the total number of elements.

    Returns
    -------
    out   : reshaped tensor (view when possible)
    shape : tuple (rows, cols) that was chosen
    """
    n = math.prod(a.shape[-2:])
    if n == 0:
        raise ValueError("empty tensor")

    # search divisors closest to sqrt(n)
    root = math.isqrt(n)
    cols, rows = None, None
    for d in range(root, 0, -1):
        if n % d == 0:
            rows = d
            cols = n // d
            break

    if rows is None or cols is None:
        raise ValueError("could not find a valid shape for the tensor")

    return a.reshape(*a.shape[:-2], rows, cols)

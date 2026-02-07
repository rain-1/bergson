import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torchopt
from datasets import load_dataset
from scipy.stats import spearmanr
from torch.distributed.tensor import (
    init_device_mesh,
)
from torchopt.typing import Numeric
from transformers import (
    AutoTokenizer,
    ConvNextImageProcessor,
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
)

from bergson.distributed import (
    dist_main,
    grad_tree,
    simple_fsdp,
)
from bergson.math import weighted_causal_lm_ce
from bergson.models import ResNetCIFAR
from bergson.trainer import DataStream, Trainer

BASE = 1e-4
WARMUP_STEPS = 30
MODEL_NAME = "EleutherAI/pythia-14m"
MODEL_TYPE = "text"


@torch.no_grad()
def shuffle_model_parameters(
    model: torch.nn.Module,
    *,
    generator: torch.Generator | None = None,
    skip_bias: bool = False,
):
    """
    Randomly permute entries *within each parameter tensor* of a model.

    Preserves per-parameter statistics (mean, variance, norm, histogram),
    while destroying all structural organization.

    Args:
        model: nn.Module whose parameters will be shuffled in-place.
        generator: optional torch.Generator for reproducibility.
        skip_bias: if True, do not shuffle 1D parameters (often biases / LN).
    """
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if skip_bias and param.ndim == 1:
            continue

        data = param.data
        flat = data.view(-1)

        perm = torch.randperm(flat.numel(), generator=generator, device=flat.device)
        flat.copy_(flat[perm])


def worker(rank: int, world_size: int, dataset):
    torch.cuda.set_device(rank)

    if MODEL_TYPE == "image":
        model = ResNetCIFAR(10)
        model.to(f"cuda:{rank}")

        processor = ConvNextImageProcessor(
            size=dict(shortest_edge=32),
        )
    else:
        # Initialize the model, optimizer, and trainer
        cfg = GPTNeoXConfig.from_pretrained(MODEL_NAME)
        cfg._attn_implementation = "eager"

        model = GPTNeoXForCausalLM(cfg)
        model.loss_function = weighted_causal_lm_ce
        model.to(f"cuda:{rank}")

        processor = AutoTokenizer.from_pretrained(MODEL_NAME)
        processor.pad_token = processor.eos_token

    # These should be set by the main process
    if world_size > 1:
        addr = os.environ.get("MASTER_ADDR", "localhost")
        port = os.environ.get("MASTER_PORT", "29500")

        dist.init_process_group(
            "nccl",
            init_method=f"tcp://{addr}:{port}",
            device_id=torch.device(f"cuda:{rank}"),
            rank=rank,
            timeout=timedelta(hours=1),
            world_size=world_size,
        )
        # Shard the model
        mesh = init_device_mesh("cuda", (world_size,))
        with mesh:
            simple_fsdp(model)

    def schedule(step: Numeric) -> Numeric:
        # Warmup phase
        if step < WARMUP_STEPS:
            return 0.0  # BASE * step / WARMUP_STEPS

        return BASE

    opt = torchopt.adamw(schedule, betas=(0.95, 0.975), eps_root=1e-8)
    # opt = torchopt.sgd(schedule, momentum=0.0)
    trainer, state0 = Trainer.initialize(model, opt)
    state = state0
    if rank == 0:
        print(f"{state.params=}")

    stream = DataStream(
        dataset, processor, batch_size=8, num_batches=60, device=f"cuda:{rank}"
    )
    folder = "/mnt/ssd-1/nora/bergson/checkpoints"
    os.makedirs(folder, exist_ok=True)

    for i, x in enumerate(stream):
        state = trainer.step(state, x, trace=True)
        state.save(
            os.path.join(folder, f"double_backward_step{i}.ckpt"),
        )

    loss = trainer.evaluate(state, stream[20])
    grads = grad_tree(loss, {"example_weight": stream.weights})
    scores = grads["example_weight"]
    if rank == 0:
        print(f"Scores: {scores}")

    # loss = trainer.evaluate(state, stream[20])

    # Each rank has only used a fraction of all the example weights. For each rank,
    # weights that it didn't use have zero gradient. We sum across all ranks to get
    # the full gradient.
    if world_size > 1:
        dist.all_reduce(scores)

    baseline = loss.item()
    if rank == 0:
        print(f"Baseline: {baseline}")
        print("Grad:", scores.sum())

    stream.requires_grad = False

    diffs = []
    score_sums = []

    n = len(stream.weights)
    w = stream.weights  # [-n:]
    s = scores  # x[-n:]

    gen = torch.Generator().manual_seed(42)
    perm = torch.randperm(n, generator=gen)
    subsets = perm.chunk(100)

    for subset in subsets:
        w.fill_(1.0)
        w[subset] = 0.0

        state = state0
        for x in stream:
            state = trainer.step(state, x)

        loss = trainer.evaluate(state, stream[20])
        if world_size > 1:
            dist.all_reduce(loss)

        diffs.append(baseline - loss.item())
        score_sums.append(s[subset].sum().item())

        corr = spearmanr(diffs, score_sums)
        if rank == 0:
            print(f"Loss diff: {diffs[-1]}")
            print(f"Score: {score_sums[-1]}")
            print(f"Spearman correlation: {corr}")


def main():
    if MODEL_TYPE == "image":
        ds = load_dataset("cifar10", split="train")
        dist_main(ds, worker)
    else:
        ds = load_dataset("EleutherAI/SmolLM2-135M-10B", split="train")
        ds = ds.map(lambda x: {"length": len(x["text"])}).sort("length")

    dist_main(ds, worker)


if __name__ == "__main__":
    main()

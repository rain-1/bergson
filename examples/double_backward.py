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
    launch_distributed_run,
    simple_fsdp,
)
from bergson.models import ResNetCIFAR
from bergson.trainer import DataStream, Trainer
from bergson.utils.math import weighted_causal_lm_ce

BASE = 1e-5
WARMUP_STEPS = 10
MODEL_NAME = "EleutherAI/pythia-1b"
MODEL_TYPE = "text"


def worker(global_rank: int, rank: int, world_size: int, dataset):
    torch.cuda.set_device(rank)

    if MODEL_TYPE == "image":
        model = ResNetCIFAR(10)
        model.to(f"cuda:{rank}")

        processor = ConvNextImageProcessor(
            size=dict(shortest_edge=32),
        )
    else:
        cfg = GPTNeoXConfig.from_pretrained(MODEL_NAME, revision="step0")
        model = GPTNeoXForCausalLM(cfg)
        model.set_attn_implementation("eager")

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
            return 0.0

        return BASE

    opt = torchopt.adamw(
        schedule,
        betas=(0.95, 0.975),
        eps_root=1e-8,
    )
    trainer, state0 = Trainer.initialize(model, opt)
    fwd_state = state0

    stream = DataStream(
        dataset, processor, batch_size=8, num_batches=100, device=f"cuda:{rank}"
    )
    folder = "/mnt/ssd-1/nora/bergson/checkpoints"
    fwd_state = trainer.train(
        fwd_state,
        stream,
        save_dir=folder,
        save_mode="sqrt",
    )

    # Use an arbitrary batch from the dataset as the test example
    ex = stream[49]
    del ex["example_weight"]

    # Compute gradient of the test loss with respect to the final state
    loss = trainer.evaluate(fwd_state, ex)
    bwd_state = fwd_state.backward(loss, torch.zeros_like(stream.weights))
    stream.requires_grad = True

    if world_size > 1:
        dist.all_reduce(loss, op=dist.ReduceOp.AVG)

    bwd_state = trainer.backward(folder, stream, bwd_state)
    if world_size > 1:
        dist.all_reduce(bwd_state.weight_grads, op=dist.ReduceOp.AVG)
    if global_rank == 0:
        print(f"Scores 2: {bwd_state.weight_grads.tolist()}")

    baseline = loss.item()
    if global_rank == 0:
        print(f"Baseline: {baseline}")
        print("Grad:", bwd_state.weight_grads.sum())

    stream.requires_grad = False

    diffs = []
    score_sums = []

    gen = torch.Generator().manual_seed(42)
    perm = torch.randperm(len(stream.weights), generator=gen)
    subsets = perm.chunk(100)

    for subset in subsets:
        stream.weights.fill_(1.0)
        stream.weights[subset] = 0.0

        fwd_state = state0
        for x in stream:
            fwd_state = trainer.step(fwd_state, x)

        loss = trainer.evaluate(fwd_state, stream[49])
        if world_size > 1:
            dist.all_reduce(loss, op=dist.ReduceOp.AVG)

        diffs.append(baseline - loss.item())
        score_sums.append(bwd_state.weight_grads[subset].sum().item())

        corr = spearmanr(diffs, score_sums)
        if global_rank == 0:
            print(f"Loss diff: {diffs[-1]}")
            print(f"Score: {score_sums[-1]}")
            print(f"Spearman correlation: {corr}")


def main():
    if MODEL_TYPE == "image":
        ds = load_dataset("cifar10", split="train")
    else:
        ds = load_dataset("EleutherAI/SmolLM2-135M-10B", split="train")
        ds = ds.map(lambda x: {"length": len(x["text"])}).sort("length")

    launch_distributed_run(MODEL_TYPE, worker, [ds])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""MAGIC attribution on a pretrained model.

Trains from a random init checkpoint (Pythia step0), attributes eval loss
to training examples, and validates via leave-subset-out retraining.

Usage:
    python examples/magic_pretrain.py runs/magic_pretrain

    # Or via the CLI:
    bergson magic runs/magic_pretrain \
        --model EleutherAI/pythia-160m \
        --revision step0 \
        --data.dataset EleutherAI/SmolLM2-135M-10B \
        --query.dataset EleutherAI/SmolLM2-135M-10B \
        --query.split "train[:1]"
"""

from datasets import load_dataset

from bergson.config import DataConfig, DistributedConfig
from bergson.distributed import dist_main, worker
from bergson.double_backward import DoubleBackwardConfig, double_backward


def main():
    run_cfg = DoubleBackwardConfig(
        run_path="runs/magic_pretrain",
        model="EleutherAI/pythia-160m",
        revision="step0",
        data=DataConfig(
            dataset="EleutherAI/SmolLM2-135M-10B",
            split="train",
        ),
        query=DataConfig(
            dataset="EleutherAI/SmolLM2-135M-10B",
            split="train[:1]",
        ),
        query_batches=1,
        lr=1e-5,
        warmup_steps=10,
        batch_size=8,
        num_batches=25,
        max_length=256,
        num_subsets=100,
        seed=42,
    )
    dist_cfg = DistributedConfig()
    double_backward(run_cfg, dist_cfg)

    experiment = False
    if experiment:
        MODEL_TYPE = "image"

        if MODEL_TYPE == "image":
            ds = load_dataset("cifar10", split="train")
            dist_main(ds, worker)
        else:
            ds = load_dataset("EleutherAI/SmolLM2-135M-10B", split="train")

        dist_main(ds, worker)


if __name__ == "__main__":
    main()

import os
from dataclasses import dataclass, field, fields

import torch
import torch.distributed as dist
import torchopt
from datasets import Dataset
from torch import nn
from torchopt.typing import GradientTransformation, OptState
from transformers import BaseImageProcessor

from .distributed import grad_tree, shallow_copy


def _maybe_get_cuda_rng_state() -> torch.Tensor:
    """ "Get the CUDA RNG state if CUDA is initialized, otherwise return zeros."""
    if torch.cuda.is_initialized():
        return torch.cuda.random.get_rng_state()

    # This corresponds to a manual seed of 0
    return torch.zeros(16, dtype=torch.uint8)


class DataStream:
    def __init__(
        self,
        dataset: Dataset,
        processor,
        batch_size: int,
        num_batches: int = 0,
        *,
        device: torch.device | str = "cpu",
        input_key: str | None = None,
    ):
        self.dataset = dataset
        self.processor = processor

        self.batch_size = batch_size
        self.device = device
        self.num_batches = num_batches or len(self.dataset) // self.batch_size
        if input_key is None:
            input_key = "img" if isinstance(processor, BaseImageProcessor) else "text"

        self.input_key = input_key
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        n = self.batch_size * self.num_batches * self.world_size
        self.weights = nn.Parameter(torch.ones(n, device=device))

    @property
    def requires_grad(self) -> bool:
        return self.weights.requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool):
        self.weights.requires_grad = value

    def __getitem__(self, i: int) -> dict:
        if i < 0 or i >= self.num_batches:
            raise IndexError("DataStream index out of range")

        x = self.dataset[i * self.batch_size : (i + 1) * self.batch_size]
        w = self.weights[i * self.batch_size : (i + 1) * self.batch_size]

        if isinstance(self.processor, BaseImageProcessor):
            y = x.pop("label")

            x = self.processor(
                images=x[self.input_key],
                return_tensors="pt",
            )
            x["pixel_values"] = x["pixel_values"][self.rank :: self.world_size]
            x["labels"] = torch.tensor(y[self.rank :: self.world_size])
        else:
            x = self.processor(
                x[self.input_key],
                padding=True,
                return_tensors="pt",
            )
            x["input_ids"] = x["labels"] = x["input_ids"][self.rank :: self.world_size]

        x["example_weight"] = w[self.rank :: self.world_size]
        return {k: v.to(self.device) for k, v in x.items()}

    def __iter__(self):
        for i in range(self.num_batches):
            yield self[i]

    def __reversed__(self):
        for i in reversed(range(self.num_batches)):
            yield self[i]


@dataclass(frozen=True)
class TrainerState:
    params: dict[str, torch.Tensor]
    buffers: dict[str, torch.Tensor]
    opt_state: OptState

    batch_index: int = 0
    cuda_rng_state: torch.Tensor = field(default_factory=_maybe_get_cuda_rng_state)
    cpu_rng_state: torch.Tensor = field(default_factory=torch.random.get_rng_state)

    @classmethod
    def load(cls, path: str, **kwargs) -> "TrainerState":
        # Check to see if this is a sharded checkpoint
        if os.path.isdir(path):
            # We had better be in distributed mode
            if not dist.is_initialized():
                print(
                    "Warning: Loading rank 0 of a sharded checkpoint while "
                    "torch.distributed is not initialized. This is probably not what "
                    "you want."
                )
                rank = 0
            else:
                rank = dist.get_rank()

            path = os.path.join(path, f"rank_{rank}.shard")

        # We need to set weights_only to False because torchopt uses some NamedTuple
        # objects in OptState. We could potentially implement a custom deserialization
        # function that converts these into dicts and converts back
        state_dict = torch.load(path, **kwargs, weights_only=False)
        return cls(**state_dict)

    def save(self, path: str):
        # Convert to dict manually because dataclasses.asdict does a deep copy
        state_dict = {f.name: getattr(self, f.name) for f in fields(self)}

        # Check to see if we're in distributed mode. If so, we need to save only our
        # shard of the state.
        if dist.is_initialized():
            # Make sure the directory exists
            os.makedirs(path, exist_ok=True)

            rank = dist.get_rank()
            path = os.path.join(path, f"rank_{rank}.shard")

        torch.save(state_dict, path)


class Trainer:
    """Stateless, functional trainer for a model, optimizer, and dataset."""

    @classmethod
    def initialize(
        cls,
        model: nn.Module,
        optimizer: GradientTransformation,
    ) -> tuple["Trainer", TrainerState]:
        """Convenience method for initializing the trainer and state."""
        # Create new tensor objects for the parameters and buffers to ensure that they
        # are not modified in place
        params = shallow_copy(dict(model.named_parameters(remove_duplicate=False)))
        buffers = shallow_copy(dict(model.named_buffers(remove_duplicate=False)))
        opt_state = optimizer.init(params)

        state = TrainerState(params, buffers, opt_state)
        return cls(model, optimizer), state

    def __init__(self, model: nn.Module, optimizer: GradientTransformation):
        # "Hollow out" the model by moving trainable parameters to the meta device
        self.model = model.to("meta")
        self.optimizer = optimizer

    def step(
        self,
        state: TrainerState,
        inputs: dict,
        *,
        inplace: bool = False,
        trace: bool = False,
    ) -> TrainerState:
        torch.random.set_rng_state(state.cpu_rng_state)

        # We keep the model on the meta device, so it's essential we use strict=True
        # to ensure every single parameter is replaced by a real one from the state.
        outputs = torch.func.functional_call(
            self.model,
            (state.params, state.buffers),
            kwargs=inputs,
            strict=True,
        )

        # Currently we support two output types: HuggingFace, and "raw loss"
        # - HuggingFace models output a dict/dataclass with a "loss" field
        # - Raw loss models output a single scalar loss value as a Tensor
        if hasattr(outputs, "loss"):
            loss = outputs.loss
        else:
            loss = outputs

        assert isinstance(loss, torch.Tensor), "Loss must be a Tensor"
        grads = grad_tree(loss, state.params, create_graph=trace)

        updates, new_state = self.optimizer.update(
            grads, state.opt_state, inplace=inplace, params=state.params
        )
        new_params = torchopt.apply_updates(state.params, updates, inplace=inplace)
        state = TrainerState(
            new_params,
            state.buffers,
            new_state,
            state.batch_index + 1,
        )
        return state

    def train(
        self,
        state: TrainerState,
        data: DataStream,
        *,
        inplace: bool = False,
        trace: bool = False,
    ) -> TrainerState:
        for x in data:
            state = self.step(state, x, inplace=inplace, trace=trace)

        return state

    def evaluate(self, state: TrainerState, inputs: dict) -> torch.Tensor:
        torch.random.set_rng_state(state.cpu_rng_state)

        outputs = torch.func.functional_call(
            self.model,
            (state.params, state.buffers),
            kwargs=inputs,
            strict=True,
        )
        if hasattr(outputs, "loss"):
            loss = outputs.loss
        else:
            loss = outputs

        return loss

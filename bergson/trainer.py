import math
import os
import re
from concurrent.futures import Future
from dataclasses import dataclass, field, fields
from shutil import rmtree
from typing import Literal

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torchopt
from datasets import Dataset
from torch import nn
from torchopt.pytree import tree_iter
from torchopt.typing import GradientTransformation, OptState
from transformers import BaseImageProcessor

from .distributed import grad_tree, shallow_copy


def _maybe_get_cuda_rng_state() -> torch.Tensor:
    """ "Get the CUDA RNG state if CUDA is initialized, otherwise return zeros."""
    if torch.cuda.is_initialized():
        return torch.cuda.random.get_rng_state()

    # This corresponds to a manual seed of 0
    return torch.zeros(16, dtype=torch.uint8)


def sorted_checkpoints(folder: str) -> list[tuple[int, str]]:
    """
    Return a list of (batch_index, filepath) sorted by batch_index
    for files named like: step_<index>.ckpt
    """
    pattern = re.compile(r"step_(\d+)\.ckpt$")

    checkpoints = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)

        match = pattern.match(name)
        if match:
            batch_index = int(match.group(1))
            checkpoints.append((batch_index, path))

    return sorted(checkpoints, key=lambda x: x[0])


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
        if self.batch_size % self.world_size != 0:
            raise ValueError(
                f"Batch size {self.batch_size} must be divisible by world size "
                f"{self.world_size}"
            )

        n = self.batch_size * self.num_batches
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
                max_length=512,
                padding=True,
                return_tensors="pt",
                truncation=True,
            )
            x["input_ids"] = x["labels"] = x["input_ids"][self.rank :: self.world_size]

        x["example_weight"] = w[self.rank :: self.world_size]
        return {k: v.to(self.device) for k, v in x.items()}

    def __iter__(self):
        for i in range(self.num_batches):
            yield self[i]

    def __len__(self):
        return self.num_batches

    def __reversed__(self):
        for i in reversed(range(self.num_batches)):
            yield self[i]


@dataclass
class BackwardState:
    param_grads: dict[str, torch.Tensor]

    opt_grads: list[torch.Tensor]
    """PyTree of the same structure as the optimizer state, containing gradients for
    each of the optimizer state tensors."""

    weight_grads: torch.Tensor


@dataclass
class TrainerState:
    # Differentiable state
    params: dict[str, torch.Tensor]
    opt_state: OptState

    # Non-differentiable state
    buffers: dict[str, torch.Tensor]
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

    def detach_(self):
        for p in self.params.values():
            p.detach_()

        for t in tree_iter(self.opt_state):
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                t.detach_()

    @property
    def requires_grad(self) -> bool:
        p_val = any(p.requires_grad for p in self.params.values())
        opt_val = any(
            isinstance(t, torch.Tensor) and t.requires_grad
            for t in tree_iter(self.opt_state)
        )
        return p_val or opt_val

    @requires_grad.setter
    def requires_grad(self, value: bool):
        for p in self.params.values():
            p.requires_grad = value

        for t in tree_iter(self.opt_state):
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                t.requires_grad = value

    def differentiable_tensors(self) -> list[torch.Tensor]:
        ps = list(self.params.values())
        os = [
            t
            for t in tree_iter(self.opt_state)
            if isinstance(t, torch.Tensor) and t.is_floating_point()
        ]
        return ps + os

    def backward(
        self,
        loss: torch.Tensor,
        weight_grads: torch.Tensor,
        *,
        create_graph: bool = False,
    ) -> BackwardState:
        """Compute gradient of loss wrt this trainer state."""
        grads = grad_tree(loss, self.params, create_graph=create_graph)
        opt_grads = [
            torch.zeros_like(buf)
            for buf in tree_iter(self.opt_state)
            if isinstance(buf, torch.Tensor) and buf.is_floating_point()
        ]
        return BackwardState(grads, opt_grads, weight_grads)

    def state_dict(self) -> dict:
        # Convert to dict manually because dataclasses.asdict does a deep copy
        return {f.name: getattr(self, f.name) for f in fields(self)}


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

        state = TrainerState(params, opt_state, buffers)
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
        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=torch.cuda.is_bf16_supported(),
        ):
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
            new_state,
            state.buffers,
            state.batch_index + 1,
        )
        return state

    def train(
        self,
        state: TrainerState,
        data: DataStream,
        *,
        inplace: bool = False,
        save_dir: str | None = None,
        save_mode: Literal["linear", "sqrt"] = "sqrt",
        trace: bool = False,
    ) -> TrainerState:
        # Make sure the save directory exists
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        chunk_size = math.isqrt(len(data)) if save_mode == "sqrt" else 1
        last_start = len(data) - chunk_size

        grp = None
        save_futures: list[Future] = []

        for i, x in enumerate(data):
            # Save checkpoint BEFORE each step. Step 0 is the initial state prior to
            # any updates, step 1 is the state after the first update, etc.
            if save_dir and (i % chunk_size == 0 or i >= last_start):
                p = os.path.join(save_dir, f"step_{i}.ckpt")

                # Create a new process group so that we can overlap saves
                if dist.is_initialized():
                    grp = dist.new_group(backend="gloo", group_desc=f"step_{i}")
                    assert isinstance(grp, dist.ProcessGroup)
                else:
                    grp = None

                fut = dcp.async_save(
                    state.state_dict(),
                    checkpoint_id=p,
                    no_dist=grp is None,
                    process_group=grp,
                )
                assert isinstance(fut, Future)

                def callback(_, i=i, p=p, g=grp):
                    print(f"Checkpoint {i} saved to {p}")
                    if g:
                        dist.destroy_process_group(g)

                fut.add_done_callback(callback)
                save_futures.append(fut)

            state = self.step(state, x, inplace=inplace, trace=trace)

        for fut in save_futures:
            fut.result()  # wait for all checkpoints to finish saving

        return state

    def backward(
        self,
        ckpt_dir: str,
        data: DataStream,
        bwd_state: BackwardState,
        fwd_state: TrainerState,
        *,
        cleanup: bool = True,
    ) -> BackwardState:
        ckpt_list = sorted_checkpoints(ckpt_dir)
        expected_idx, _ = ckpt_list[-1]

        save_futures: list[Future] = []
        while ckpt_list:
            # Make sure everything has been saved
            for fut in save_futures:
                fut.result()

            idx, path = ckpt_list[-1]
            fwd_state.batch_index = idx
            dcp.load(
                fwd_state.state_dict(),
                checkpoint_id=path,
                no_dist=not dist.is_initialized(),
            )

            # Only delete this checkpoint if it's the one we expected to load. If it's
            # not, we need to keep it around, and step forward through training
            if idx == expected_idx:
                del ckpt_list[-1]

                # Only delete on the main rank
                if cleanup and (not dist.is_initialized() or dist.get_rank() == 0):
                    rmtree(path) if os.path.isdir(path) else os.remove(path)

            # Step forward in training if needed
            while idx < expected_idx:
                fwd_state = self.step(
                    fwd_state,
                    data[fwd_state.batch_index],
                    trace=False,
                )
                idx += 1

                # Save checkpoints for states we will need later
                if idx < expected_idx:
                    path = os.path.join(ckpt_dir, f"step_{idx}.ckpt")
                    ckpt_list.append((idx, path))

                    # Create a new process group so that we can overlap saves
                    if dist.is_initialized():
                        grp = dist.new_group(backend="gloo", group_desc=f"step_{idx}")
                        assert isinstance(grp, dist.ProcessGroup)
                    else:
                        grp = None

                    fut = dcp.async_save(
                        fwd_state.state_dict(),
                        checkpoint_id=path,
                        no_dist=grp is None,
                        process_group=grp,
                    )
                    assert isinstance(fut, Future)

                    fut.add_done_callback(
                        lambda _, g=grp: dist.destroy_process_group(g) if g else None
                    )
                    save_futures.append(fut)

                fwd_state.detach_()
                fwd_state.requires_grad = True

            # The index we expect on the next iteration is one less than the current
            expected_idx = idx - 1

            fwd_state.detach_()
            fwd_state.requires_grad = True
            data.requires_grad = True

            flat_i = fwd_state.differentiable_tensors()

            # Re-do the training step
            state_f = self.step(
                fwd_state,
                data[fwd_state.batch_index],
                trace=True,
            )
            # Carefully consume the bwd state to save memory
            flat_f = state_f.differentiable_tensors()
            p_grads = list(bwd_state.param_grads.values())
            o_grads = bwd_state.opt_grads

            p_keys = list(bwd_state.param_grads.keys())
            w_grads = bwd_state.weight_grads
            del bwd_state

            # grad_outputs is the gradient of the loss wrt the next TrainerState. We're
            # doing a VJP to get the gradient wrt the current TrainerState, AND the
            # example weights for this batch.
            inps = flat_i + [data.weights]
            result = list(
                torch.autograd.grad(
                    flat_f,
                    inps,
                    grad_outputs=p_grads + o_grads,
                    allow_unused=True,
                )
            )
            del p_grads

            # Accumulate parameter gradients
            param_grads = {k: result[i] for i, k in enumerate(p_keys)}
            del result[: len(p_keys)]

            weight_grads = result[-1] + w_grads
            bwd_state = BackwardState(param_grads, result[:-1], weight_grads)

        for fut in save_futures:
            fut.result()

        return bwd_state

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

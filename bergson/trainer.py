import json
import math
import os
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, Literal, Optional

import torch
from datasets import Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm
from transformers import get_scheduler

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import numpy as np
except ImportError:
    np = None


@dataclass
class TrainingConfig:
    output_dir: str = "./checkpoints"

    num_epochs: int = 3
    train_batch_size: int = 8
    eval_batch_size: int = 8

    lr_schedule: Literal["constant", "linear", "cosine"] = "linear"
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    warmup_steps: int = 0

    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1

    # Optimizer
    optimizer_type: Literal["adamw", "sgd"] = "adamw"  # "adamw" or "sgd"
    adam_beta2: float = 0.999
    momentum: float = 0.9

    # Logging / checkpointing
    logging_steps: int = 10
    num_train_steps: Optional[int] = None  # optimizer steps
    save_every_steps: int = 0  # if nonpositive, sqrt(total_train_steps)

    # Reproducibility
    seed: int = 42

    # Precision: "fp32" | "fp16" | "bf16"
    precision: str = "fp32"

    # WandB
    use_wandb: bool = False
    wandb_project: str = "my-project"
    wandb_run_name: Optional[str] = None


class SimpleDeterministicTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: Dataset,
        eval_dataset: Dataset | None = None,
        data_collator=None,
        config: TrainingConfig = TrainingConfig(),
    ) -> None:

        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator

        # DDP setup
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.is_distributed = self.world_size > 1
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.is_main_process = (not self.is_distributed) or self.local_rank == 0

        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.local_rank)
            if self.is_distributed:
                torch.cuda.set_device(self.local_rank)
        else:
            self.device = torch.device("cpu")

        self._set_seed(self.config.seed)

        self.model.to(self.device)
        if self.is_distributed:
            torch.distributed.init_process_group(backend="nccl")
            self.model = DDP(
                self.model, device_ids=[self.local_rank], output_device=self.local_rank
            )

        # Precision
        self.use_fp16 = self.config.precision == "fp16"
        self.use_bf16 = self.config.precision == "bf16"
        self.use_autocast = (self.device.type == "cuda") and (
            self.use_fp16 or self.use_bf16
        )
        self.autocast_dtype = (
            torch.float16
            if self.use_fp16
            else (torch.bfloat16 if self.use_bf16 else None)
        )

        # Optimizer & (optional) scheduler
        self.optimizer = self._create_optimizer()

        # Infer total train steps if needed
        # We approximate steps_per_epoch for current rank.
        num_samples = len(self.train_dataset)
        samples_per_rank = math.ceil(num_samples / self.world_size)
        microbatches_per_epoch = math.ceil(
            samples_per_rank / self.config.train_batch_size
        )
        steps_per_epoch = math.ceil(
            microbatches_per_epoch / self.config.gradient_accumulation_steps
        )

        if self.config.num_train_steps is None:
            total_train_steps = steps_per_epoch * self.config.num_epochs
        else:
            total_train_steps = self.config.num_train_steps

        if self.config.save_every_steps <= 0:
            self.config.save_every_steps = max(1, int(math.sqrt(total_train_steps)))

        self.scheduler = get_scheduler(
            self.config.lr_schedule,
            self.optimizer,
            num_training_steps=total_train_steps,
            num_warmup_steps=self.config.warmup_steps,
        )
        self.scaler = torch.amp.GradScaler(
            enabled=(self.use_fp16 and self.device.type == "cuda")
        )

        # Training position
        self.global_step = 0  # optimizer steps
        self.current_epoch = 0  # 0-based
        self.microbatch_cursor = (
            0  # microbatch index within current epoch for this rank
        )

        os.makedirs(self.config.output_dir, exist_ok=True)
        self._setup_wandb()

    # ----------------------- Public API -----------------------

    def train(self):
        # Main training loop
        while self.current_epoch < self.config.num_epochs:
            # Get deterministic index order for this epoch and this rank
            per_rank_indices = self._get_per_rank_indices_for_epoch(
                epoch=self.current_epoch,
                dataset_len=len(self.train_dataset),
            )
            shard = self.train_dataset.select(per_rank_indices)

            # How many microbatches total this epoch for this rank?
            num_microbatches = math.ceil(
                len(per_rank_indices) / self.config.train_batch_size
            )

            if self.is_main_process:
                pbar = tqdm(
                    total=num_microbatches,
                    desc=f"Epoch {self.current_epoch+1}/{self.config.num_epochs}",
                    dynamic_ncols=True,
                )
                pbar.n = self.microbatch_cursor
            else:
                pbar = None

            # Microbatch loop
            while self.microbatch_cursor < num_microbatches:
                # Build this microbatch
                mb_idx = self.microbatch_cursor
                start = mb_idx * self.config.train_batch_size
                end = start + self.config.train_batch_size

                # Fetch samples and collate
                samples = shard[start:end]
                batch = (
                    self.data_collator(samples)
                    if self.data_collator is not None
                    else samples
                )
                self.microbatch_cursor += 1

                self.model.train()

                with torch.autocast(
                    "cuda",
                    enabled=self.use_autocast,
                    dtype=self.autocast_dtype,
                ):
                    loss = self.training_step(batch)
                    loss = loss / self.config.gradient_accumulation_steps

                # Backward
                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                microbatches_this_step = (
                    self.microbatch_cursor % self.config.gradient_accumulation_steps
                )

                if (
                    microbatches_this_step == 0
                    or self.microbatch_cursor == num_microbatches
                ):
                    # Optimizer step
                    if self.scaler.is_enabled():
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.max_grad_norm
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.max_grad_norm
                        )
                        self.optimizer.step()

                    self.optimizer.zero_grad(set_to_none=True)

                    if self.scheduler is not None:
                        self.scheduler.step()

                    self.global_step += 1

                    # Logging
                    if self.is_main_process and (
                        self.global_step % self.config.logging_steps == 0
                    ):
                        loss_value = (
                            loss.detach().item()
                            * self.config.gradient_accumulation_steps
                        )
                        lr = self.optimizer.param_groups[0]["lr"]
                        if pbar is not None:
                            pbar.set_postfix(loss=loss_value, lr=lr)
                        if self.config.use_wandb and WANDB_AVAILABLE:
                            wandb.log(
                                {
                                    "train/loss": loss_value,
                                    "train/lr": lr,
                                    "train/step": self.global_step,
                                }
                            )

                    # Checkpointing (step-aligned)
                    if self.global_step % self.config.save_every_steps == 0:
                        self._save_checkpoint(self.global_step)

                    if (
                        self.config.num_train_steps is not None
                        and self.global_step >= self.config.num_train_steps
                    ):
                        break

                if pbar is not None:
                    pbar.update(1)

            if pbar is not None:
                pbar.close()

            if (
                self.config.num_train_steps is not None
                and self.global_step >= self.config.num_train_steps
            ):
                break

            # End of epoch: reset cursor and maybe eval
            self.microbatch_cursor = 0
            self.current_epoch += 1

            if self.eval_dataset is not None and self.is_main_process:
                eval_metrics = self.evaluate()
                if self.config.use_wandb and WANDB_AVAILABLE:
                    wandb.log({f"eval/{k}": v for k, v in eval_metrics.items()})

        if self.is_main_process:
            self._save_checkpoint(self.global_step, tag="final")

    def evaluate(self) -> Dict[str, float]:
        if self.eval_dataset is None:
            return {}

        num_samples = len(self.eval_dataset)
        batch_size = self.config.eval_batch_size
        num_batches = math.ceil(num_samples / batch_size)

        self.model.eval()
        losses = []

        with torch.no_grad():
            iterator = (
                tqdm(range(num_batches), desc="Evaluating", dynamic_ncols=True)
                if self.is_main_process
                else range(num_batches)
            )
            for b in iterator:
                start = b * batch_size
                end = min(start + batch_size, num_samples)
                samples = self.eval_dataset[start:end]
                batch = (
                    self.data_collator(samples)
                    if self.data_collator is not None
                    else samples
                )
                batch = self._move_batch_to_device(batch)

                with torch.autocast(
                    "cuda",
                    enabled=self.use_autocast,
                    dtype=self.autocast_dtype,
                ):
                    outputs = self.model(**batch)
                    loss = outputs.loss
                losses.append(loss.detach().cpu().item())

        avg_loss = sum(losses) / max(len(losses), 1)
        return {"loss": avg_loss}

    # -------------------- Overridable hook --------------------

    def training_step(self, batch) -> torch.Tensor:
        batch = self._move_batch_to_device(batch)
        outputs = self.model(**batch)
        return outputs.loss

    # --------------------- Internal helpers ------------------

    def _get_per_rank_indices_for_epoch(
        self, epoch: int, dataset_len: int
    ) -> list[int]:
        """
        Pure function of (seed, epoch, dataset_len, world_size, rank).
        We use a private torch.Generator so this doesn't perturb global RNG.
        """
        g = torch.Generator(device="cpu")
        g.manual_seed(self.config.seed + epoch)  # epoch-specific seed

        perm = torch.randperm(dataset_len, generator=g).tolist()

        if self.is_distributed:
            # Shard by rank
            return perm[self.local_rank :: self.world_size]
        else:
            return perm

    def _create_optimizer(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.config.optimizer_type == "sgd":
            return torch.optim.SGD(
                params,
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer_type == "adamw":
            return torch.optim.AdamW(
                params,
                betas=(self.config.momentum, self.config.adam_beta2),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(
                f"Unsupported optimizer type: {self.config.optimizer_type}"
            )

    def _move_batch_to_device(self, batch: Any):
        if isinstance(batch, dict):
            return {k: v.to(self.device) for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return [self._move_batch_to_device(b) for b in batch]
        else:
            return batch.to(self.device)

    def _set_seed(self, seed: int):
        seed = seed + self.local_rank  # rank-offset
        random.seed(seed)
        if np is not None:
            np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _setup_wandb(self):
        if not self.config.use_wandb:
            return
        if not WANDB_AVAILABLE:
            if self.is_main_process:
                print("wandb not installed; disabling wandb logging.")
            self.config.use_wandb = False
            return
        if not self.is_main_process:
            os.environ["WANDB_DISABLED"] = "true"
            return
        wandb.init(
            project=self.config.wandb_project,
            name=self.config.wandb_run_name,
            config=asdict(self.config),
        )

    def _save_checkpoint(self, step: int, tag: Optional[str] = None):
        if not self.is_main_process:
            return

        ckpt_name = f"step_{step}" if tag is None else tag
        ckpt_dir = os.path.join(self.config.output_dir, ckpt_name)
        os.makedirs(ckpt_dir, exist_ok=True)

        model_to_save = self.model.module if isinstance(self.model, DDP) else self.model
        torch.save(
            model_to_save.state_dict(), os.path.join(ckpt_dir, "pytorch_model.bin")
        )

        # RNG states
        rng_state = {
            "python": random.getstate(),
            # "numpy": np.random.get_state() if np is not None else None,
            "torch": torch.get_rng_state(),
            "cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        }

        state = {
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "microbatch_cursor": self.microbatch_cursor,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "config": asdict(self.config),
            "rng_state": rng_state,
        }
        torch.save(state, os.path.join(ckpt_dir, "training_state.pt"))

        meta = {
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "microbatch_cursor": self.microbatch_cursor,
        }
        with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
            json.dump(meta, f)

    def load_checkpoint(self, ckpt_dir: str):
        if self.is_main_process:
            print(f"Loading checkpoint from {ckpt_dir}")

        model_path = os.path.join(ckpt_dir, "pytorch_model.bin")
        state_path = os.path.join(ckpt_dir, "training_state.pt")

        model_to_load = self.model.module if isinstance(self.model, DDP) else self.model
        model_to_load.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        model_to_load.to(self.device)

        state = torch.load(state_path, map_location="cpu", weights_only=True)
        self.global_step = state["global_step"]
        self.current_epoch = state["current_epoch"]
        self.microbatch_cursor = state["microbatch_cursor"]
        self.optimizer.load_state_dict(state["optimizer"])

        if self.scheduler is not None and state.get("scheduler") is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        if self.scaler is not None and state.get("scaler") is not None:
            self.scaler.load_state_dict(state["scaler"])

        rng_state = state.get("rng_state")
        if rng_state is not None:
            random.setstate(rng_state["python"])
            # if np is not None and rng_state["numpy"] is not None:
            #    np.random.set_state(rng_state["numpy"])
            if rng_state["torch"] is not None:
                torch.set_rng_state(rng_state["torch"])
            if torch.cuda.is_available() and rng_state["cuda"] is not None:
                torch.cuda.set_rng_state_all(rng_state["cuda"])

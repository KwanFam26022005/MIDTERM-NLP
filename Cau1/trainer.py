"""
trainer.py
==========
Pure-PyTorch Trainer for MultiTaskPhoBERT with alternating task batches,
mixed-precision training, and per-task best-checkpoint saving.

Training strategy
-----------------
- Batch sampling ratio: NER 30 % / Sentiment 40 % / ABSA 30 %
- Optimizer: AdamW — backbone lr=2e-5, heads lr=1e-3
- Scheduler: linear warmup 10 % → cosine decay to 0
- Gradient clip: max_norm=1.0
- Mixed precision: torch.amp.autocast

Author : NLP Pipeline
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

# seqeval for entity-level NER evaluation
from seqeval.metrics import f1_score as seqeval_f1
from seqeval.metrics import classification_report as seqeval_report
from sklearn.metrics import accuracy_score, f1_score as sklearn_f1

from data_loader import (
    NER_ID2LABEL,
    NER_LABEL2ID,
    ABSA_ASPECTS,
    ABSA_POLARITIES,
    PAD_LABEL_ID,
    collate_fn,
    get_class_weights,
    build_task_dataloader,
)
from model import MultiTaskPhoBERT


# ---------------------------------------------------------------------------
# Device auto-detection
# ---------------------------------------------------------------------------

def _auto_device() -> torch.device:
    """Auto-detect the best available device: cuda → mps → cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Cosine schedule with linear warmup
# ---------------------------------------------------------------------------

def _get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup over *num_warmup_steps* then cosine decay to 0."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Task sampler — round-robin with given ratios
# ---------------------------------------------------------------------------

class _TaskSampler:
    """Infinite iterator that yields task names according to configured ratios.

    Ratios: NER 30%, Sentiment 40%, ABSA 30%.
    Implementation: maintain iterators over each DataLoader and cycle through
    tasks with a weighted schedule.
    """

    TASK_WEIGHTS: Dict[str, float] = {
        "ner": 0.3,
        "sentiment": 0.4,
        "absa": 0.3,
    }

    def __init__(
        self,
        loaders: Dict[str, DataLoader],
    ) -> None:
        self.loaders = loaders
        self._iterators: Dict[str, Any] = {
            t: iter(dl) for t, dl in loaders.items()
        }
        # Build a weighted task schedule (length 10 for simplicity)
        schedule: List[str] = []
        for task, weight in self.TASK_WEIGHTS.items():
            schedule.extend([task] * int(weight * 10))
        self._schedule: List[str] = schedule
        self._idx: int = 0

    def __iter__(self) -> "_TaskSampler":
        return self

    def __next__(self) -> Tuple[str, Dict[str, Any]]:
        task = self._schedule[self._idx % len(self._schedule)]
        self._idx += 1
        try:
            batch = next(self._iterators[task])
        except StopIteration:
            # Re-create iterator for exhausted loader
            self._iterators[task] = iter(self.loaders[task])
            batch = next(self._iterators[task])
        return task, batch


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Pure-PyTorch trainer for ``MultiTaskPhoBERT``.

    Parameters
    ----------
    model              : The multi-task model.
    train_loaders      : ``{"ner": DataLoader, "sentiment": DataLoader, "absa": DataLoader}``
    val_loaders        : Same structure for validation.
    class_weights      : ``{"sentiment": Tensor, "absa": Tensor, "ner": Tensor}``
    backbone_lr        : Learning rate for the backbone (default 2e-5).
    head_lr            : Learning rate for classification heads (default 1e-3).
    max_grad_norm      : Gradient clipping max norm (default 1.0).
    warmup_ratio       : Fraction of total steps for linear warmup (default 0.1).
    resume_from_checkpoint : Path to a checkpoint to resume from (optional).
    """

    def __init__(
        self,
        model: MultiTaskPhoBERT,
        train_loaders: Dict[str, DataLoader],
        val_loaders: Dict[str, DataLoader],
        class_weights: Optional[Dict[str, torch.Tensor]] = None,
        backbone_lr: float = 2e-5,
        head_lr: float = 1e-3,
        max_grad_norm: float = 1.0,
        warmup_ratio: float = 0.1,
        resume_from_checkpoint: Optional[str] = None,
    ) -> None:
        self.device: torch.device = _auto_device()
        self.model: MultiTaskPhoBERT = model.to(self.device)
        self.train_loaders: Dict[str, DataLoader] = train_loaders
        self.val_loaders: Dict[str, DataLoader] = val_loaders
        self.class_weights: Dict[str, torch.Tensor] = class_weights or {}
        self.max_grad_norm: float = max_grad_norm
        self.warmup_ratio: float = warmup_ratio

        # Move class weights to device
        for task in self.class_weights:
            self.class_weights[task] = self.class_weights[task].to(self.device)

        # ---- Optimizer: two param groups ----
        backbone_params: List[nn.Parameter] = []
        head_params: List[nn.Parameter] = []
        for name, param in self.model.named_parameters():
            if "backbone" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        self.optimizer: AdamW = AdamW(
            [
                {"params": backbone_params, "lr": backbone_lr},
                {"params": head_params, "lr": head_lr},
            ],
            weight_decay=0.01,
        )

        # Mixed precision scaler (only for CUDA)
        self.use_amp: bool = self.device.type == "cuda"
        self.scaler: GradScaler = GradScaler(enabled=self.use_amp)

        # Tracking
        self.global_step: int = 0
        self.best_scores: Dict[str, float] = {
            "ner": 0.0, "sentiment": 0.0, "absa": 0.0
        }
        self.training_log: Dict[str, Any] = {
            "epochs": [],
            "task_losses": {"ner": [], "sentiment": [], "absa": []},
        }

        # Resume
        if resume_from_checkpoint is not None:
            self._load_checkpoint(resume_from_checkpoint)

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.global_step = ckpt.get("global_step", 0)
        self.best_scores = ckpt.get("best_scores", self.best_scores)
        print(f"[Trainer] Resumed from {path}  (step {self.global_step})")

    def _save_checkpoint(
        self, save_dir: str, tag: str
    ) -> None:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"best_{tag}.pt")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "global_step": self.global_step,
                "best_scores": self.best_scores,
                "tag": tag,
            },
            path,
        )
        print(f"  💾 Saved checkpoint → {path}")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def train(self, epochs: int, save_dir: str) -> None:
        """
        Train the model for *epochs* epochs with alternating task batches.

        Parameters
        ----------
        epochs   : Number of full passes over the data.
        save_dir : Directory for checkpoints and logs.
        """
        os.makedirs(save_dir, exist_ok=True)

        # Estimate total steps per epoch (sum of all task batches)
        steps_per_epoch: int = sum(len(dl) for dl in self.train_loaders.values())
        total_steps: int = steps_per_epoch * epochs
        warmup_steps: int = int(total_steps * self.warmup_ratio)

        scheduler = _get_cosine_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps
        )

        print(
            f"[Trainer] device={self.device} | epochs={epochs} | "
            f"steps/epoch≈{steps_per_epoch} | total≈{total_steps} | "
            f"warmup={warmup_steps} | AMP={self.use_amp}"
        )

        for epoch in range(1, epochs + 1):
            self.model.train()
            sampler = _TaskSampler(self.train_loaders)
            epoch_losses: Dict[str, List[float]] = {
                "ner": [], "sentiment": [], "absa": []
            }
            step_in_epoch: int = 0

            pbar = tqdm(
                total=steps_per_epoch,
                desc=f"Epoch {epoch}/{epochs}",
                unit="batch",
            )

            for _ in range(steps_per_epoch):
                task, batch = next(sampler)

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                self.optimizer.zero_grad()

                # Mixed precision forward
                with torch.amp.autocast(
                    device_type=self.device.type, enabled=self.use_amp
                ):
                    logits, loss = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        task=task,
                        labels=labels,
                    )

                    # Apply class weights if available and loss was computed
                    # without them (sentiment / absa already use CE inside
                    # model but without weights — we recompute here).
                    if (
                        loss is not None
                        and task in self.class_weights
                        and task != "ner"
                    ):
                        loss = self.model.compute_loss(
                            task, logits, labels,
                            class_weights=self.class_weights.get(task),
                        )

                if loss is None:
                    pbar.update(1)
                    continue

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                scheduler.step()

                loss_val: float = loss.item()
                epoch_losses[task].append(loss_val)
                self.global_step += 1
                step_in_epoch += 1

                # Log every 50 steps
                if self.global_step % 50 == 0:
                    pbar.set_postfix(
                        task=task,
                        loss=f"{loss_val:.4f}",
                        step=self.global_step,
                    )

                pbar.update(1)

            pbar.close()

            # ---- End-of-epoch: log mean losses ----
            epoch_entry: Dict[str, Any] = {"epoch": epoch}
            for t in ("ner", "sentiment", "absa"):
                losses = epoch_losses[t]
                mean_loss = sum(losses) / len(losses) if losses else 0.0
                epoch_entry[f"{t}_loss"] = round(mean_loss, 5)
                self.training_log["task_losses"][t].append(round(mean_loss, 5))
                print(f"  [{t:>10s}] mean loss = {mean_loss:.4f}  ({len(losses)} batches)")

            self.training_log["epochs"].append(epoch_entry)

            # ---- Evaluate on validation set ----
            for t in ("ner", "sentiment", "absa"):
                if t not in self.val_loaders:
                    continue
                metrics = self.evaluate(self.val_loaders[t], task=t)
                key_metric = metrics["f1_macro"]
                print(
                    f"  [{t:>10s}] val F1={key_metric:.4f}  "
                    f"acc={metrics['accuracy']:.4f}  "
                    f"loss={metrics['loss']:.4f}"
                )
                epoch_entry[f"{t}_val_f1"] = round(key_metric, 5)
                epoch_entry[f"{t}_val_acc"] = round(metrics["accuracy"], 5)

                if key_metric > self.best_scores[t]:
                    self.best_scores[t] = key_metric
                    self._save_checkpoint(save_dir, t)

            # ---- Write training log after every epoch ----
            log_path = os.path.join(save_dir, "training_log.json")
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(self.training_log, f, indent=2, ensure_ascii=False)

        print("[Trainer] Training complete.")

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        task: str,
    ) -> Dict[str, float]:
        """
        Evaluate the model on a single-task dataloader.

        Returns
        -------
        dict with keys: ``f1_macro``, ``accuracy``, ``loss``.
        For NER, ``f1_macro`` is the entity-level F1 from seqeval.
        """
        self.model.eval()
        all_preds: List[Any] = []
        all_labels: List[Any] = []
        total_loss: float = 0.0
        num_batches: int = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            with torch.amp.autocast(
                device_type=self.device.type, enabled=self.use_amp
            ):
                logits, loss = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    task=task,
                    labels=labels,
                )

            if loss is not None:
                total_loss += loss.item()
            num_batches += 1

            if task == "ner":
                # CRF decode
                decoded = self.model.crf_decode(logits, attention_mask)
                for i, seq in enumerate(decoded):
                    label_seq = labels[i].cpu().tolist()
                    pred_tags: List[str] = []
                    true_tags: List[str] = []
                    for j, (pred_id, true_id) in enumerate(
                        zip(seq, label_seq)
                    ):
                        if true_id == PAD_LABEL_ID:
                            continue
                        pred_tags.append(
                            NER_ID2LABEL.get(pred_id, "O")
                        )
                        true_tags.append(
                            NER_ID2LABEL.get(true_id, "O")
                        )
                    all_preds.append(pred_tags)
                    all_labels.append(true_tags)

            elif task == "sentiment":
                preds = logits.argmax(dim=-1).cpu().tolist()
                labs = labels.cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labs)

            elif task == "absa":
                # logits: [B, A, P] → argmax per aspect
                preds = logits.argmax(dim=-1).cpu().tolist()  # [B, A]
                labs = labels.cpu().tolist()                   # [B, A]
                all_preds.extend(preds)
                all_labels.extend(labs)

        self.model.train()
        avg_loss = total_loss / max(num_batches, 1)

        if task == "ner":
            # Entity-level F1 via seqeval
            f1_macro = seqeval_f1(all_labels, all_preds, average="macro")
            # Accuracy: fraction of correctly tagged tokens
            flat_preds = [t for seq in all_preds for t in seq]
            flat_labels = [t for seq in all_labels for t in seq]
            acc = accuracy_score(flat_labels, flat_preds)

        elif task == "sentiment":
            f1_macro = sklearn_f1(all_labels, all_preds, average="macro")
            acc = accuracy_score(all_labels, all_preds)

        elif task == "absa":
            # Flatten across aspects
            import numpy as np
            flat_preds = [p for row in all_preds for p in row]
            flat_labels = [l for row in all_labels for l in row]
            f1_macro = sklearn_f1(flat_labels, flat_preds, average="macro")
            acc = accuracy_score(flat_labels, flat_preds)

        else:
            raise ValueError(f"Unknown task: {task}")

        return {
            "f1_macro": float(f1_macro),
            "accuracy": float(acc),
            "loss": float(avg_loss),
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("trainer.py — standalone check")
    print(f"Auto-detected device: {_auto_device()}")
    print("Usage:")
    print("  from trainer import Trainer")
    print("  trainer = Trainer(model, train_loaders, val_loaders)")
    print("  trainer.train(epochs=5, save_dir='checkpoints')")

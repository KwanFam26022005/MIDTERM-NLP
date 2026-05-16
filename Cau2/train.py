"""
train.py
========
Train any model from ``MODEL_REGISTRY``, evaluate perplexity on the
test set, and produce a comparison table with perplexity curves.

Features
--------
* Checkpoint saves model + optimizer + scheduler state → proper resume
* Rich console logging with phase banners
* Per-epoch progress with ETA

CLI
---
    python train.py --model vanilla   --corpus corpus
    python train.py --model stacked   --corpus corpus
    python train.py --model bilstm_attn --corpus corpus
    python train.py --model vanilla   --corpus corpus --resume vanilla
    python train.py --corpus corpus --compare

Checkpoint path can be customized with ``--ckpt_dir``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models import MODEL_REGISTRY, _device

BASE_DIR = Path(__file__).resolve().parent
console = Console()

# ---------------------------------------------------------------------------
# Hyper-parameters (exactly as specified)
# ---------------------------------------------------------------------------
SEQ_LEN = 128
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 5.0
PATIENCE = 2
LR_FACTOR = 0.5
SEED = 42
MAX_TOKENS = 50_000_000  # 50M tokens default (~30 min/epoch on T4 GPU)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class LMDataset(Dataset):
    """
    Memory-mapped language-modelling dataset.

    Reads shard files, encodes with SentencePiece, and stores all token
    ids in a flat 1-D tensor. ``__getitem__`` returns contiguous windows
    of length ``seq_len + 1`` (input + target shifted by 1).
    """

    def __init__(
        self,
        shard_dir: Path,
        sp_model_path: str | Path,
        seq_len: int = SEQ_LEN,
        max_tokens: int | None = None,
    ):
        self.seq_len = seq_len
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(sp_model_path))

        console.print(f"  Loading shards from [cyan]{shard_dir}[/cyan] …")
        all_ids: list[int] = []
        shard_files = sorted(shard_dir.glob("shard_*.txt"))
        for sf in tqdm(shard_files, desc="  Encoding shards", leave=False):
            with open(sf, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    ids = self.sp.encode(line)
                    all_ids.extend(ids)
                    all_ids.append(self.sp.eos_id())
                    if max_tokens and len(all_ids) >= max_tokens:
                        break
            if max_tokens and len(all_ids) >= max_tokens:
                break

        self.data = torch.tensor(all_ids, dtype=torch.long)
        console.print(f"  Total tokens: [cyan]{len(self.data):,}[/cyan]")

    def __len__(self) -> int:
        return max(0, (len(self.data) - 1) // self.seq_len)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start_idx = idx * self.seq_len
        chunk = self.data[start_idx : start_idx + self.seq_len + 1]
        return chunk[:-1], chunk[1:]

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    @property
    def pad_id(self) -> int:
        return self.sp.pad_id()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float = GRAD_CLIP,
) -> float:
    """Train for one epoch, return mean loss."""
    model.train()
    total_loss = 0.0
    total_tokens = 0
    hidden = None
    step = 0

    pbar = tqdm(loader, desc="  train", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)

        # Detach hidden state to prevent backpropping into previous batches
        if hidden is not None:
            hidden = tuple(h.detach() for h in hidden)

        logits, hidden = model(x, hidden)  # (B, T, V)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        n_tokens = (y != criterion.ignore_index).sum().item() if criterion.ignore_index >= 0 else y.numel()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens
        step += 1

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate on a dataset. Returns (mean_loss, perplexity)."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    hidden = None

    for x, y in tqdm(loader, desc="  eval ", leave=False):
        x, y = x.to(device), y.to(device)
        logits, hidden = model(x, hidden)
        if hidden is not None:
            hidden = tuple(h.detach() for h in hidden)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        n_tokens = (y != criterion.ignore_index).sum().item() if criterion.ignore_index >= 0 else y.numel()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    mean_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(mean_loss, 100))  # cap to avoid overflow
    return mean_loss, ppl


# ---------------------------------------------------------------------------
# Main training driver
# ---------------------------------------------------------------------------
def train_model(
    model_name: str,
    corpus_path: Path,
    ckpt_dir: Path,
    resume: bool = False,
    max_tokens: int = MAX_TOKENS,
) -> dict:
    """Train a single model. Returns history dict."""
    device = _device()

    console.print()
    console.print(
        Panel.fit(
            f"[bold white]Training: {model_name}[/bold white]\n"
            f"[dim]Device: {device}  •  Epochs: {EPOCHS}  •  LR: {LR}[/dim]\n"
            f"[dim]Checkpoints: {ckpt_dir}[/dim]",
            border_style="blue",
            title=f"🏋️ {model_name}",
        )
    )

    # ---- Phase 1: Load data ----
    console.print()
    console.rule("[bold cyan]Phase 1/4: Loading Data[/bold cyan]")

    sp_model = BASE_DIR / "tokenizer/vi_bpe.model"
    if not sp_model.exists():
        console.print(f"[red]❌ Tokenizer not found at {sp_model}. Run train_tokenizer.py first.[/red]")
        raise FileNotFoundError(sp_model)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = BASE_DIR / "logs"
    log_dir.mkdir(exist_ok=True)

    train_ds = LMDataset(corpus_path / "train", sp_model, seq_len=SEQ_LEN, max_tokens=max_tokens)
    val_ds = LMDataset(corpus_path / "val", sp_model, seq_len=SEQ_LEN, max_tokens=max_tokens // 10)
    test_ds = LMDataset(corpus_path / "test", sp_model, seq_len=SEQ_LEN, max_tokens=max_tokens // 10)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=0)

    # ---- Phase 2: Build model ----
    console.print()
    console.rule("[bold cyan]Phase 2/4: Building Model[/bold cyan]")

    vocab_size = train_ds.vocab_size
    pad_id = train_ds.pad_id
    ModelClass = MODEL_REGISTRY[model_name]
    model = ModelClass(vocab_size=vocab_size).to(device)
    console.print(f"  Parameters: [cyan]{model.count_parameters():,}[/cyan]")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=PATIENCE, factor=LR_FACTOR
    )
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    # ---- Resume ----
    start_epoch = 0
    best_val_ppl = float("inf")
    history = {
        "model_name": model_name,
        "params": model.count_parameters(),
        "train_loss": [],
        "val_loss": [],
        "val_ppl": [],
    }

    ckpt_path = ckpt_dir / f"{model_name}_best.pt"
    if resume and ckpt_path.exists():
        console.print()
        console.print(
            Panel(
                f"[bold yellow]▶  Resuming from checkpoint[/bold yellow]\n"
                f"[dim]{ckpt_path}[/dim]",
                border_style="yellow",
                title="♻️ Resume",
            )
        )
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        # Restore scheduler state (new fix)
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_ppl = ckpt.get("best_val_ppl", float("inf"))
        history = ckpt.get("history", history)
        console.print(
            f"  Resumed at epoch [cyan]{start_epoch}[/cyan], "
            f"best val PPL = [cyan]{best_val_ppl:.2f}[/cyan], "
            f"LR = [cyan]{optimizer.param_groups[0]['lr']:.2e}[/cyan]"
        )
    elif resume:
        console.print(f"  [yellow]⚠ Checkpoint not found at {ckpt_path} — starting fresh[/yellow]")

    # ---- Phase 3: Training ----
    console.print()
    console.rule("[bold cyan]Phase 3/4: Training[/bold cyan]")
    console.print(
        f"  Epochs: [cyan]{start_epoch + 1}→{EPOCHS}[/cyan]  |  "
        f"Steps/epoch: [cyan]{len(train_loader)}[/cyan]  |  "
        f"Batch: [cyan]{BATCH_SIZE}[/cyan]  |  "
        f"Seq len: [cyan]{SEQ_LEN}[/cyan]"
    )

    t0 = time.time()
    for epoch in range(start_epoch, EPOCHS):
        console.print(
            f"\n  [bold]━━━ Epoch {epoch + 1}/{EPOCHS} ━━━[/bold]  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_ppl = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        train_ppl = math.exp(min(train_loss, 100))
        history["train_loss"].append(round(train_loss, 5))
        history["val_loss"].append(round(val_loss, 5))
        history["val_ppl"].append(round(val_ppl, 2))

        console.print(
            f"  train_loss=[cyan]{train_loss:.4f}[/cyan]  train_ppl=[cyan]{train_ppl:.2f}[/cyan]  "
            f"val_loss=[cyan]{val_loss:.4f}[/cyan]  val_ppl=[bold cyan]{val_ppl:.2f}[/bold cyan]"
        )

        # Checkpoint — save model + optimizer + scheduler
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_ppl": best_val_ppl,
                    "history": history,
                    "vocab_size": vocab_size,
                },
                ckpt_path,
            )
            console.print(f"  [green]💾 Saved best checkpoint (val_ppl={best_val_ppl:.2f}) → {ckpt_path}[/green]")

        elapsed = time.time() - t0
        done_epochs = epoch - start_epoch + 1
        remaining = elapsed / done_epochs * (EPOCHS - epoch - 1)
        console.print(
            f"  ⏱  Elapsed: {elapsed / 60:.1f} min | ETA: {remaining / 60:.1f} min"
        )

    total_time = time.time() - t0

    # ---- Phase 4: Test evaluation ----
    console.print()
    console.rule("[bold cyan]Phase 4/4: Test Evaluation[/bold cyan]")

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        console.print(f"  Loaded best checkpoint: {ckpt_path}")
    test_loss, test_ppl = evaluate(model, test_loader, criterion, device)
    console.print(f"  [bold green]Test PPL: {test_ppl:.2f}[/bold green]")

    history["test_loss"] = round(test_loss, 5)
    history["test_ppl"] = round(test_ppl, 2)
    history["train_time_min"] = round(total_time / 60, 2)

    # Save history
    hist_path = log_dir / f"{model_name}_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    console.print(f"  [green]✓ History saved → {hist_path}[/green]")

    console.print()
    console.print(
        Panel.fit(
            f"[bold green]✅ Training complete: {model_name}[/bold green]\n"
            f"[dim]Best val PPL: {best_val_ppl:.2f}  |  Test PPL: {test_ppl:.2f}  |  "
            f"Time: {total_time / 60:.1f} min[/dim]",
            border_style="green",
        )
    )

    return history


# ---------------------------------------------------------------------------
# Comparison table + plot
# ---------------------------------------------------------------------------
def compare_models(ckpt_dir: Path) -> None:
    """Print a rich comparison table and save perplexity curve plot."""
    log_dir = BASE_DIR / "logs"
    histories: list[dict] = []
    for name in MODEL_REGISTRY:
        hist_path = log_dir / f"{name}_history.json"
        if hist_path.exists():
            with open(hist_path) as f:
                histories.append(json.load(f))

    if not histories:
        console.print("[yellow]No training histories found. Train models first.[/yellow]")
        return

    # ---- Rich table ----
    table = Table(title="📊 Model Comparison", show_lines=True, border_style="blue")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Params", justify="right", style="green")
    table.add_column("Train PPL", justify="right", style="yellow")
    table.add_column("Val PPL", justify="right", style="yellow")
    table.add_column("Test PPL", justify="right", style="bold magenta")
    table.add_column("Train Time", justify="right")

    for h in histories:
        final_train_ppl = math.exp(min(h["train_loss"][-1], 100)) if h["train_loss"] else float("nan")
        table.add_row(
            h["model_name"],
            f"{h['params']:,}",
            f"{final_train_ppl:.2f}",
            f"{h['val_ppl'][-1]:.2f}" if h["val_ppl"] else "—",
            f"{h.get('test_ppl', '—')}",
            f"{h.get('train_time_min', '—')} min",
        )
    console.print(table)

    # ---- Perplexity curves ----
    results_dir = BASE_DIR / "results"
    results_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#4e79a7", "#e15759", "#59a14f"]
    for idx, h in enumerate(histories):
        epochs = list(range(1, len(h["val_ppl"]) + 1))
        color = colors[idx % len(colors)]
        ax.plot(epochs, h["val_ppl"], marker="o", label=h["model_name"], color=color, linewidth=2)

    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Validation Perplexity", fontsize=13)
    ax.set_title("Validation Perplexity Over Epochs", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_path = results_dir / "perplexity_curves.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    console.print(f"[green]✓ Perplexity curves saved → {plot_path}[/green]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LSTM language models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python train.py --model vanilla --corpus corpus\n"
            "  python train.py --model stacked --corpus corpus --resume stacked\n"
            "  python train.py --corpus corpus --compare\n"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODEL_REGISTRY.keys()),
        default=None,
        help="Model to train. Omit and use --compare to produce comparison table.",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="corpus",
        help="Path to sharded corpus directory",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="Checkpoint directory (default: Cau2/checkpoints)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Model name to resume training from checkpoint",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=MAX_TOKENS,
        help=f"Max tokens to load for training (default: {MAX_TOKENS:,})",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Print comparison table and plot for all trained models",
    )
    return parser.parse_args()


if __name__ == "__main__":
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    args = parse_args()
    corpus_path = Path(args.corpus)
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else BASE_DIR / "checkpoints"

    max_tokens = args.max_tokens

    if args.compare:
        compare_models(ckpt_dir)
    elif args.model:
        should_resume = args.resume == args.model
        train_model(args.model, corpus_path, ckpt_dir, resume=should_resume, max_tokens=max_tokens)
    else:
        # Train all models sequentially then compare
        for model_name in MODEL_REGISTRY:
            try:
                train_model(model_name, corpus_path, ckpt_dir, max_tokens=max_tokens)
            except Exception as exc:
                console.print(f"[red]Error training {model_name}: {exc}[/red]")
        compare_models(ckpt_dir)

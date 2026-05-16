"""
diacritic_restore.py
====================
Syllable-level sequence labelling that restores Vietnamese tonal
diacritics using a pretrained StackedLSTM as feature extractor.

Features
--------
* Proper syllable-token alignment (track boundaries per word)
* Checkpoint/resume for training
* Progress bars for all phases

Run
---
    python diacritic_restore.py --corpus corpus --epochs 5
    python diacritic_restore.py --corpus corpus --epochs 5 --ckpt_dir /path/to/save
    python diacritic_restore.py --corpus corpus --resume   # continue from last
"""
from __future__ import annotations
import argparse, json, math, os, random, re, sys, time, unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models import StackedLSTM, _device

console = Console()
SEED = 42
DEVICE = _device()

BASE_DIR = Path(__file__).resolve().parent
SYLLABLE_FILE = BASE_DIR / "data/vi_syllables.txt"

_VN_VOWELS_BASE = "aeiouy"
_VN_DIACRITICS = {
    'a': ['a','à','á','ả','ã','ạ','ă','ằ','ắ','ẳ','ẵ','ặ','â','ầ','ấ','ẩ','ẫ','ậ'],
    'e': ['e','è','é','ẻ','ẽ','ẹ','ê','ề','ế','ể','ễ','ệ'],
    'i': ['i','ì','í','ỉ','ĩ','ị'],
    'o': ['o','ò','ó','ỏ','õ','ọ','ô','ồ','ố','ổ','ỗ','ộ','ơ','ờ','ớ','ở','ỡ','ợ'],
    'u': ['u','ù','ú','ủ','ũ','ụ','ư','ừ','ứ','ử','ữ','ự'],
    'y': ['y','ỳ','ý','ỷ','ỹ','ỵ'],
    'd': ['d','đ'],
}

def _strip_diacritics(text: str) -> str:
    nfkd = unicodedata.normalize("NFD", text)
    out = []
    for ch in nfkd:
        if unicodedata.category(ch) == "Mn":
            continue
        if ch == "\u0111":
            out.append("d")
        elif ch == "\u0110":
            out.append("D")
        else:
            out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))

def _generate_syllable_dict() -> Dict[str, List[str]]:
    mapping: Dict[str, set] = defaultdict(set)
    for base, variants in _VN_DIACRITICS.items():
        for v in variants:
            mapping[base].add(v)
            mapping[base.upper()].add(v.upper())
    consonants_initial = [
        '','b','c','ch','d','g','gh','gi','h','k','kh','l','m',
        'n','ng','ngh','nh','p','ph','qu','r','s','t','th','tr','v','x',
    ]
    vowel_nuclei = [
        'a','ă','â','e','ê','i','o','ô','ơ','u','ư','y',
        'ai','ao','au','ay','âu','ây','eo','êu',
        'ia','iê','iêu','iu','oa','oă','oai','oay','oe','oi',
        'oo','ôi','ơi','ua','uâ','uây','uê','ui','uô','uôi',
        'ươ','ươi','ưu','ya','yê','yêu',
    ]
    tones = ['', '\u0300','\u0301','\u0303','\u0309','\u0323']
    finals = ['','c','ch','m','n','ng','nh','p','t']
    syllables: set = set()
    for ci in consonants_initial:
        for vn in vowel_nuclei:
            for fn in finals:
                for tn in tones:
                    toned = unicodedata.normalize("NFC", ci + vn + tn + fn)
                    syllables.add(toned)
    result: Dict[str, List[str]] = defaultdict(list)
    for s in syllables:
        key = _strip_diacritics(s).lower()
        if key:
            result[key].append(s.lower())
    for k in result:
        result[k] = sorted(set(result[k]))
    return dict(result)

def load_syllable_dict() -> Dict[str, List[str]]:
    if SYLLABLE_FILE.exists():
        mapping: Dict[str, List[str]] = {}
        with open(SYLLABLE_FILE, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    mapping[parts[0]] = parts[1].split(",")
        if mapping:
            return mapping
    console.print("  [yellow]Generating syllable dictionary …[/yellow]")
    mapping = _generate_syllable_dict()
    SYLLABLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SYLLABLE_FILE, "w", encoding="utf-8") as f:
        for k in sorted(mapping):
            f.write(f"{k}\t{','.join(mapping[k])}\n")
    console.print(f"  [green]✓ Saved {len(mapping)} entries → {SYLLABLE_FILE}[/green]")
    return mapping

# ── Data generation ───────────────────────────────────────────────────────
def generate_pairs(corpus_path: Path, n_samples: int = 500_000) -> List[Tuple[List[str], List[str]]]:
    """Generate (stripped, original) word-list pairs from corpus — with progress."""
    rng = random.Random(SEED)
    shard_dir = corpus_path / "train"
    if not shard_dir.exists():
        shard_dir = corpus_path
    files = sorted(shard_dir.glob("shard_*.txt"))
    if not files:
        files = sorted(corpus_path.glob("**/*.txt"))

    lines: List[str] = []
    for fp in tqdm(files, desc="  Reading shards", leave=False):
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if len(line) > 10:
                    lines.append(line)
                    if len(lines) >= n_samples * 3:
                        break
        if len(lines) >= n_samples * 3:
            break

    rng.shuffle(lines)
    pairs: List[Tuple[List[str], List[str]]] = []
    for line in tqdm(lines[:n_samples], desc="  Building pairs", leave=False):
        original = line.lower().split()
        stripped = [_strip_diacritics(w).lower() for w in original]
        if stripped and original:
            pairs.append((stripped, original))
    console.print(f"  [green]✓[/green] Generated [cyan]{len(pairs):,}[/cyan] pairs")
    return pairs

# ── Dataset ───────────────────────────────────────────────────────────────
class DiacriticDataset(Dataset):
    """Dataset that tracks token boundaries per syllable for proper alignment."""

    def __init__(self, pairs, syl_dict, sp_model_path, max_len=64):
        self.pairs = pairs
        self.syl_dict = syl_dict
        self.max_len = max_len
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(sp_model_path))
        self.max_cands = max(len(v) for v in syl_dict.values()) if syl_dict else 20

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        stripped, original = self.pairs[idx]
        stripped = stripped[:self.max_len]
        original = original[:self.max_len]

        # Encode each syllable separately and track boundaries
        input_ids = []
        boundaries = []  # (start, end) token index for each syllable
        for w in stripped:
            ids = self.sp.encode(w)
            start = len(input_ids)
            input_ids.extend(ids)
            end = len(input_ids)
            boundaries.append((start, end))

        input_ids = input_ids[:self.max_len * 4]

        labels = []
        cand_masks = []
        for s_w, o_w in zip(stripped, original):
            cands = self.syl_dict.get(s_w, [s_w])
            label_idx = cands.index(o_w) if o_w in cands else 0
            labels.append(label_idx)
            mask = [1]*len(cands) + [0]*(self.max_cands - len(cands))
            cand_masks.append(mask[:self.max_cands])

        n_syls = len(labels)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "cand_masks": torch.tensor(cand_masks, dtype=torch.float),
            "boundaries": boundaries,  # list of (start, end) tuples
            "n_syls": n_syls,
            "n_tokens": len(input_ids),
        }

def collate_fn(batch):
    max_tok = max(b["n_tokens"] for b in batch)
    max_syl = max(b["n_syls"] for b in batch)
    max_cand = batch[0]["cand_masks"].size(-1)
    B = len(batch)
    input_ids = torch.zeros(B, max_tok, dtype=torch.long)
    labels = torch.full((B, max_syl), -1, dtype=torch.long)
    cand_masks = torch.zeros(B, max_syl, max_cand)
    # Store boundaries as padded tensor: (B, max_syl, 2)
    boundaries = torch.zeros(B, max_syl, 2, dtype=torch.long)
    for i, b in enumerate(batch):
        nt = b["n_tokens"]
        ns = b["n_syls"]
        input_ids[i, :nt] = b["input_ids"]
        labels[i, :ns] = b["labels"]
        cand_masks[i, :ns] = b["cand_masks"]
        for j, (s, e) in enumerate(b["boundaries"][:ns]):
            boundaries[i, j, 0] = s
            boundaries[i, j, 1] = e
    return {
        "input_ids": input_ids,
        "labels": labels,
        "cand_masks": cand_masks,
        "boundaries": boundaries,
    }

# ── Model ─────────────────────────────────────────────────────────────────
class DiaCorrectionModel(nn.Module):
    """
    Diacritic correction using pretrained StackedLSTM backbone.

    Key fix: uses tracked token boundaries to mean-pool LSTM hidden states
    per syllable, instead of the previous naive uniform-step approach.
    """

    def __init__(self, backbone: StackedLSTM, max_candidates: int = 50, freeze_bottom: int = 2):
        super().__init__()
        self.backbone = backbone
        self.hidden_dim = backbone.hidden_dim
        # Freeze bottom layers
        for i in range(freeze_bottom):
            for name, param in backbone.lstm.named_parameters():
                if f"weight_ih_l{i}" in name or f"weight_hh_l{i}" in name or f"bias_ih_l{i}" in name or f"bias_hh_l{i}" in name:
                    param.requires_grad = False
        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, max_candidates),
        )
        self.max_candidates = max_candidates

    def forward(self, input_ids, cand_masks=None, boundaries=None):
        """
        Forward pass with proper syllable-token alignment.

        Parameters
        ----------
        input_ids  : (B, T) token ids
        cand_masks : (B, S, C) candidate masks
        boundaries : (B, S, 2) start/end token indices per syllable
        """
        emb = self.backbone.embedding(input_ids)
        lstm_out, _ = self.backbone.lstm(emb)  # (B, T, H)

        B, T, H = lstm_out.shape

        if boundaries is not None and cand_masks is not None:
            S = cand_masks.size(1)
            # Mean-pool LSTM hidden states per syllable using tracked boundaries
            syl_hidden = torch.zeros(B, S, H, device=lstm_out.device)
            for b_idx in range(B):
                for s_idx in range(S):
                    start = boundaries[b_idx, s_idx, 0].item()
                    end = boundaries[b_idx, s_idx, 1].item()
                    if end > start and end <= T:
                        syl_hidden[b_idx, s_idx] = lstm_out[b_idx, start:end].mean(dim=0)
                    elif start < T:
                        syl_hidden[b_idx, s_idx] = lstm_out[b_idx, start]
        else:
            syl_hidden = lstm_out

        logits = self.head(syl_hidden)  # (B, S, max_cand)
        if cand_masks is not None:
            logits = logits.masked_fill(cand_masks == 0, float('-inf'))
        return logits

    def restore(self, text: str, syl_dict: Dict[str, List[str]], sp, beam_width: int = 5) -> str:
        """Restore diacritics for a single input string."""
        self.eval()
        device = next(self.parameters()).device
        words = text.lower().split()
        stripped = [_strip_diacritics(w).lower() for w in words]

        # Encode with boundary tracking
        input_ids = []
        boundaries = []
        for w in stripped:
            ids = sp.encode(w)
            start = len(input_ids)
            input_ids.extend(ids)
            end = len(input_ids)
            boundaries.append((start, end))

        input_ids = input_ids[:256]
        input_t = torch.tensor([input_ids], dtype=torch.long, device=device)

        S = len(stripped)
        cands_list = [syl_dict.get(s, [s]) for s in stripped]
        max_c = max(len(c) for c in cands_list) if cands_list else 1

        mask = torch.zeros(1, S, max_c, device=device)
        for i, c in enumerate(cands_list):
            mask[0, i, :len(c)] = 1.0

        # Build boundaries tensor
        bounds_t = torch.zeros(1, S, 2, dtype=torch.long, device=device)
        for i, (s, e) in enumerate(boundaries):
            bounds_t[0, i, 0] = s
            bounds_t[0, i, 1] = min(e, len(input_ids))

        with torch.no_grad():
            logits = self.forward(input_t, mask, bounds_t)  # (1, S, max_c)

        result = []
        for i, (s_w, cands) in enumerate(zip(stripped, cands_list)):
            if i < logits.size(1):
                probs = logits[0, i, :len(cands)]
                idx = probs.argmax().item()
                result.append(cands[idx])
            else:
                result.append(s_w)
        return " ".join(result)

# ── Training ──────────────────────────────────────────────────────────────
def train_diacritics(corpus_path: Path, epochs: int = 5, ckpt_dir: Path | None = None, resume: bool = False):
    device = _device()

    if ckpt_dir is None:
        ckpt_dir = BASE_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    console.print()
    console.print(
        Panel.fit(
            "[bold white]Diacritic Restoration Training[/bold white]\n"
            f"[dim]Device: {device}  •  Epochs: {epochs}  •  Checkpoints: {ckpt_dir}[/dim]",
            border_style="blue",
            title="🔤 Diacritics",
        )
    )

    # ── Phase 1: Load syllable dict & tokenizer ──
    console.print()
    console.rule("[bold cyan]Phase 1/4: Setup[/bold cyan]")

    syl_dict = load_syllable_dict()
    max_cands = max(len(v) for v in syl_dict.values())
    console.print(f"  Syllable dict: [cyan]{len(syl_dict)}[/cyan] entries, max candidates: {max_cands}")

    sp_path = BASE_DIR / "tokenizer/vi_bpe.model"
    if not sp_path.exists():
        console.print("[red]❌ Tokenizer not found. Run train_tokenizer.py first.[/red]")
        return

    sp = spm.SentencePieceProcessor()
    sp.load(str(sp_path))

    # ── Phase 2: Generate pairs ──
    console.print()
    console.rule("[bold cyan]Phase 2/4: Data Generation[/bold cyan]")

    all_pairs = generate_pairs(corpus_path, n_samples=500_000)
    random.Random(SEED).shuffle(all_pairs)
    test_pairs = all_pairs[:5000]
    val_pairs = all_pairs[5000:10000]
    train_pairs = all_pairs[10000:]

    train_ds = DiacriticDataset(train_pairs, syl_dict, sp_path, max_len=64)
    val_ds = DiacriticDataset(val_pairs, syl_dict, sp_path, max_len=64)
    train_ds.max_cands = max_cands
    val_ds.max_cands = max_cands

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # ── Phase 3: Build model ──
    console.print()
    console.rule("[bold cyan]Phase 3/4: Model Setup[/bold cyan]")

    backbone_ckpt_path = ckpt_dir / "stacked_best.pt"
    vocab_size = sp.get_piece_size()
    backbone = StackedLSTM(vocab_size=vocab_size)
    if backbone_ckpt_path.exists():
        ckpt = torch.load(backbone_ckpt_path, map_location=device, weights_only=False)
        backbone.load_state_dict(ckpt["model_state_dict"])
        console.print("  [green]✓ Loaded pretrained StackedLSTM backbone[/green]")
    else:
        console.print("  [yellow]⚠ No pretrained checkpoint found — training from scratch[/yellow]")

    model = DiaCorrectionModel(backbone, max_candidates=max_cands, freeze_bottom=2).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"  Trainable params: [cyan]{trainable:,}[/cyan]")

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    # ── Resume ──
    dia_ckpt_path = ckpt_dir / "diacritics_best.pt"
    start_epoch = 0
    best_val_acc = 0.0

    if resume and dia_ckpt_path.exists():
        console.print()
        console.print(
            Panel(
                f"[bold yellow]▶  Resuming from checkpoint[/bold yellow]\n"
                f"[dim]{dia_ckpt_path}[/dim]",
                border_style="yellow",
                title="♻️ Resume",
            )
        )
        ckpt = torch.load(dia_ckpt_path, map_location=device, weights_only=False)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_acc = ckpt.get("best_val_acc", 0.0)
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        else:
            # Legacy format: state_dict directly
            model.load_state_dict(ckpt)
        console.print(f"  Resumed at epoch [cyan]{start_epoch}[/cyan], best val_acc=[cyan]{best_val_acc:.2f}%[/cyan]")

    # ── Phase 4: Training ──
    console.print()
    console.rule("[bold cyan]Phase 4/4: Training[/bold cyan]")

    t0 = time.time()
    for epoch in range(start_epoch, epochs):
        console.print(f"\n  [bold]━━━ Epoch {epoch+1}/{epochs} ━━━[/bold]")

        model.train()
        total_loss, total_correct, total_count = 0.0, 0, 0
        for batch in tqdm(train_loader, desc=f"  train", leave=False):
            ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            masks = batch["cand_masks"].to(device)
            bounds = batch["boundaries"].to(device)

            logits = model(ids, masks, bounds)
            B, S, C = logits.shape
            loss = criterion(logits.reshape(-1, C), labels.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * B
            valid = labels != -1
            if valid.any():
                preds = logits.argmax(dim=-1)
                total_correct += (preds[valid] == labels[valid]).sum().item()
                total_count += valid.sum().item()

        train_acc = total_correct / max(total_count, 1) * 100

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="  val  ", leave=False):
                ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                masks = batch["cand_masks"].to(device)
                bounds = batch["boundaries"].to(device)

                logits = model(ids, masks, bounds)
                valid = labels != -1
                if valid.any():
                    preds = logits.argmax(dim=-1)
                    val_correct += (preds[valid] == labels[valid]).sum().item()
                    val_total += valid.sum().item()

        val_acc = val_correct / max(val_total, 1) * 100

        elapsed = time.time() - t0
        console.print(
            f"  train_acc=[cyan]{train_acc:.2f}%[/cyan]  "
            f"val_acc=[bold cyan]{val_acc:.2f}%[/bold cyan]  "
            f"⏱ {elapsed/60:.1f} min"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_acc": best_val_acc,
                },
                dia_ckpt_path,
            )
            console.print(f"  [green]💾 Saved best (val_acc={val_acc:.2f}%) → {dia_ckpt_path}[/green]")

    # ── Test Evaluation ──
    console.print()
    console.rule("[bold green]Test Evaluation[/bold green]")

    if dia_ckpt_path.exists():
        ckpt = torch.load(dia_ckpt_path, map_location=device, weights_only=False)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
    model.eval()

    word_correct, word_total, sent_exact, sent_total = 0, 0, 0, 0
    demo_rows = []
    with torch.no_grad():
        for i, (stripped, original) in enumerate(tqdm(test_pairs, desc="  Testing")):
            input_text = " ".join(stripped)
            predicted = model.restore(input_text, syl_dict, sp)
            gold = " ".join(original)
            pred_words = predicted.split()
            gold_words = gold.split()
            min_len = min(len(pred_words), len(gold_words))
            correct = sum(1 for p, g in zip(pred_words[:min_len], gold_words[:min_len]) if p == g)
            word_correct += correct
            word_total += len(gold_words)
            if predicted.strip() == gold.strip():
                sent_exact += 1
            sent_total += 1
            if len(demo_rows) < 10:
                status = "✅" if predicted.strip() == gold.strip() else "❌"
                demo_rows.append((input_text, predicted, gold, status))

    word_acc = word_correct / max(word_total, 1) * 100
    sent_acc = sent_exact / max(sent_total, 1) * 100
    console.print(f"  Word-level accuracy: [bold cyan]{word_acc:.2f}%[/bold cyan]")
    console.print(f"  Sentence exact match: [bold cyan]{sent_acc:.2f}%[/bold cyan]")

    # Demo table
    table = Table(title="🔤 Diacritic Restoration Demo (10 examples)", show_lines=True, border_style="blue")
    table.add_column("INPUT", style="dim", max_width=40)
    table.add_column("PREDICTED", style="cyan", max_width=40)
    table.add_column("GOLD", style="green", max_width=40)
    table.add_column("", style="bold", width=3)
    for inp, pred, gold, st in demo_rows:
        table.add_row(inp, pred, gold, st)
    console.print(table)

    console.print()
    console.print(
        Panel.fit(
            f"[bold green]✅ Diacritic training complete![/bold green]\n"
            f"[dim]Word acc: {word_acc:.2f}%  |  Sentence acc: {sent_acc:.2f}%[/dim]",
            border_style="green",
        )
    )

# ── CLI ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vietnamese diacritic restoration")
    parser.add_argument("--corpus", type=str, default="corpus")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--ckpt_dir", type=str, default=None, help="Checkpoint directory")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else None
    train_diacritics(Path(args.corpus), epochs=args.epochs, ckpt_dir=ckpt_dir, resume=args.resume)

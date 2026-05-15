"""
diacritic_restore.py
====================
Syllable-level sequence labelling that restores Vietnamese tonal
diacritics using a pretrained StackedLSTM as feature extractor.

Run
---
    python diacritic_restore.py --corpus corpus --epochs 5
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
                    base_syl = ci + vn + fn
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
    console.log("[yellow]Generating syllable dictionary …[/yellow]")
    mapping = _generate_syllable_dict()
    SYLLABLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SYLLABLE_FILE, "w", encoding="utf-8") as f:
        for k in sorted(mapping):
            f.write(f"{k}\t{','.join(mapping[k])}\n")
    console.log(f"[green]✓ Saved {len(mapping)} entries → {SYLLABLE_FILE}[/green]")
    return mapping

# ── Data generation ───────────────────────────────────────────────────────
def generate_pairs(corpus_path: Path, n_samples: int = 500_000) -> List[Tuple[List[str], List[str]]]:
    rng = random.Random(SEED)
    shard_dir = corpus_path / "train"
    if not shard_dir.exists():
        shard_dir = corpus_path
    files = sorted(shard_dir.glob("shard_*.txt"))
    if not files:
        files = sorted(corpus_path.glob("**/*.txt"))
    lines: List[str] = []
    for fp in files:
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
    for line in lines[:n_samples]:
        original = line.lower().split()
        stripped = [_strip_diacritics(w).lower() for w in original]
        if stripped and original:
            pairs.append((stripped, original))
    console.log(f"Generated [cyan]{len(pairs):,}[/cyan] pairs")
    return pairs

# ── Dataset ───────────────────────────────────────────────────────────────
class DiacriticDataset(Dataset):
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
        input_ids = []
        for w in stripped:
            ids = self.sp.encode(w)
            input_ids.extend(ids)
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
    for i, b in enumerate(batch):
        nt = b["n_tokens"]
        ns = b["n_syls"]
        input_ids[i, :nt] = b["input_ids"]
        labels[i, :ns] = b["labels"]
        cand_masks[i, :ns] = b["cand_masks"]
    return {"input_ids": input_ids, "labels": labels, "cand_masks": cand_masks}

# ── Model ─────────────────────────────────────────────────────────────────
class DiaCorrectionModel(nn.Module):
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
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, max_candidates),
        )
        self.max_candidates = max_candidates

    def forward(self, input_ids, cand_masks=None):
        emb = self.backbone.embedding(input_ids)
        lstm_out, _ = self.backbone.lstm(emb)
        # Pool over token dimension per syllable — use mean of all tokens
        pooled = lstm_out.mean(dim=1, keepdim=True).expand(-1, cand_masks.size(1), -1) if cand_masks is not None else lstm_out
        # Simple approach: use LSTM output averaged, then project per syllable
        # We'll take sliding windows of the LSTM output
        B, T, H = lstm_out.shape
        if cand_masks is not None:
            S = cand_masks.size(1)
            # Distribute LSTM hidden states across syllables
            step = max(1, T // max(S, 1))
            indices = [min(i * step + step - 1, T - 1) for i in range(S)]
            syl_hidden = lstm_out[:, indices, :]  # (B, S, H)
        else:
            syl_hidden = lstm_out
        logits = self.head(syl_hidden)  # (B, S, max_cand)
        if cand_masks is not None:
            logits = logits.masked_fill(cand_masks == 0, float('-inf'))
        return logits

    def restore(self, text: str, syl_dict: Dict[str, List[str]], sp, beam_width: int = 5) -> str:
        self.eval()
        device = next(self.parameters()).device
        words = text.lower().split()
        stripped = [_strip_diacritics(w).lower() for w in words]
        input_ids = []
        for w in stripped:
            input_ids.extend(sp.encode(w))
        input_ids = input_ids[:256]
        input_t = torch.tensor([input_ids], dtype=torch.long, device=device)
        S = len(stripped)
        cands_list = [syl_dict.get(s, [s]) for s in stripped]
        max_c = max(len(c) for c in cands_list) if cands_list else 1
        mask = torch.zeros(1, S, max_c, device=device)
        for i, c in enumerate(cands_list):
            mask[0, i, :len(c)] = 1.0
        with torch.no_grad():
            logits = self.forward(input_t, mask)  # (1, S, max_c)
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
def train_diacritics(corpus_path: Path, epochs: int = 5):
    device = _device()
    console.rule("[bold blue]Diacritic Restoration Training[/bold blue]")
    console.log(f"Device: [cyan]{device}[/cyan]")

    syl_dict = load_syllable_dict()
    max_cands = max(len(v) for v in syl_dict.values())
    console.log(f"Syllable dict: [cyan]{len(syl_dict)}[/cyan] entries, max candidates: {max_cands}")

    sp_path = BASE_DIR / "tokenizer/vi_bpe.model"
    if not sp_path.exists():
        console.log("[red]Tokenizer not found. Run train_tokenizer.py first.[/red]")
        return

    sp = spm.SentencePieceProcessor()
    sp.load(str(sp_path))

    # Generate pairs
    console.log("[bold]Generating training pairs …[/bold]")
    all_pairs = generate_pairs(corpus_path, n_samples=500_000)
    random.Random(SEED).shuffle(all_pairs)
    n = len(all_pairs)
    test_pairs = all_pairs[:5000]
    val_pairs = all_pairs[5000:10000]
    train_pairs = all_pairs[10000:]

    train_ds = DiacriticDataset(train_pairs, syl_dict, sp_path, max_len=64)
    val_ds = DiacriticDataset(val_pairs, syl_dict, sp_path, max_len=64)
    test_ds = DiacriticDataset(test_pairs, syl_dict, sp_path, max_len=64)
    train_ds.max_cands = max_cands
    val_ds.max_cands = max_cands
    test_ds.max_cands = max_cands

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # Load pretrained backbone
    ckpt_path = BASE_DIR / "checkpoints/stacked_best.pt"
    vocab_size = sp.get_piece_size()
    backbone = StackedLSTM(vocab_size=vocab_size)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        backbone.load_state_dict(ckpt["model_state_dict"])
        console.log("[green]✓ Loaded pretrained StackedLSTM backbone[/green]")
    else:
        console.log("[yellow]No pretrained checkpoint found — training from scratch[/yellow]")

    model = DiaCorrectionModel(backbone, max_candidates=max_cands, freeze_bottom=2).to(device)
    console.log(f"Trainable params: [cyan]{sum(p.numel() for p in model.parameters() if p.requires_grad):,}[/cyan]")

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss, total_correct, total_count = 0.0, 0, 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            masks = batch["cand_masks"].to(device)
            logits = model(ids, masks)
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
            for batch in val_loader:
                ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                masks = batch["cand_masks"].to(device)
                logits = model(ids, masks)
                valid = labels != -1
                if valid.any():
                    preds = logits.argmax(dim=-1)
                    val_correct += (preds[valid] == labels[valid]).sum().item()
                    val_total += valid.sum().item()
        val_acc = val_correct / max(val_total, 1) * 100
        console.log(f"Epoch {epoch+1}: train_acc={train_acc:.2f}% val_acc={val_acc:.2f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            (BASE_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), BASE_DIR / "checkpoints/diacritics_best.pt")

    # ── Evaluation on test set ────────────────────────────────────────────
    console.rule("[bold green]Test Evaluation[/bold green]")
    dia_best_path = BASE_DIR / "checkpoints/diacritics_best.pt"
    if dia_best_path.exists():
        model.load_state_dict(torch.load(dia_best_path, map_location=device, weights_only=False))
    model.eval()

    word_correct, word_total, sent_exact, sent_total = 0, 0, 0, 0
    demo_rows = []
    with torch.no_grad():
        for i, (stripped, original) in enumerate(tqdm(test_pairs, desc="Testing")):
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
    console.log(f"Word-level accuracy: [bold cyan]{word_acc:.2f}%[/bold cyan]")
    console.log(f"Sentence exact match: [bold cyan]{sent_acc:.2f}%[/bold cyan]")

    # Demo table
    table = Table(title="Diacritic Restoration Demo (10 examples)", show_lines=True)
    table.add_column("INPUT", style="dim", max_width=40)
    table.add_column("PREDICTED", style="cyan", max_width=40)
    table.add_column("GOLD", style="green", max_width=40)
    table.add_column("", style="bold", width=3)
    for inp, pred, gold, st in demo_rows:
        table.add_row(inp, pred, gold, st)
    console.print(table)

# ── CLI ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vietnamese diacritic restoration")
    parser.add_argument("--corpus", type=str, default="corpus")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    train_diacritics(Path(args.corpus), epochs=args.epochs)

"""
train_tokenizer.py
==================
Train a SentencePiece BPE tokenizer on the sharded Vietnamese corpus
and wrap it as a HuggingFace ``PreTrainedTokenizerFast``.

Run
---
    python train_tokenizer.py --corpus_path ./corpus
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
from pathlib import Path

import sentencepiece as spm
import torch
from rich.console import Console
from rich.table import Table
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders
from transformers import PreTrainedTokenizerFast

console = Console()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VOCAB_SIZE = 16_000
MODEL_TYPE = "bpe"
CHARACTER_COVERAGE = 0.9998
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
SEED = 42
MAX_TRAIN_LINES = 10_000_000  # reservoir-sample cap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _reservoir_sample(file_paths: list[Path], k: int, seed: int = SEED) -> list[str]:
    """Reservoir sampling of *k* lines from multiple files."""
    rng = random.Random(seed)
    reservoir: list[str] = []
    n = 0
    for fp in file_paths:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n += 1
                if len(reservoir) < k:
                    reservoir.append(line)
                else:
                    j = rng.randint(0, n - 1)
                    if j < k:
                        reservoir[j] = line
    console.log(f"Reservoir sampled [cyan]{len(reservoir):,}[/cyan] lines from {n:,} total")
    return reservoir


def _collect_shard_paths(corpus_path: Path) -> list[Path]:
    """Collect all shard .txt files from train/."""
    train_dir = corpus_path / "train"
    if not train_dir.exists():
        console.log(f"[red]Train directory not found: {train_dir}[/red]")
        sys.exit(1)
    shards = sorted(train_dir.glob("shard_*.txt"))
    if not shards:
        console.log(f"[red]No shard files in {train_dir}[/red]")
        sys.exit(1)
    console.log(f"Found [cyan]{len(shards)}[/cyan] train shards")
    return shards


# ---------------------------------------------------------------------------
# SentencePiece training
# ---------------------------------------------------------------------------
def train_sentencepiece(lines: list[str], output_dir: Path) -> Path:
    """Train a raw SentencePiece BPE model and return the .model path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(output_dir / "vi_bpe")

    # Write sampled lines to a temporary file for SP training
    tmp_file = output_dir / "_sp_train_data.txt"
    with open(tmp_file, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    console.log("[bold]Training SentencePiece BPE …[/bold]")
    spm.SentencePieceTrainer.train(
        input=str(tmp_file),
        model_prefix=prefix,
        vocab_size=VOCAB_SIZE,
        model_type=MODEL_TYPE,
        character_coverage=CHARACTER_COVERAGE,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece=PAD_TOKEN,
        unk_piece=UNK_TOKEN,
        bos_piece=BOS_TOKEN,
        eos_piece=EOS_TOKEN,
        num_threads=os.cpu_count() or 4,
        train_extremely_large_corpus=len(lines) > 5_000_000,
    )
    console.log(f"[green]✓ SentencePiece model saved → {prefix}.model[/green]")

    # Clean temp file
    tmp_file.unlink(missing_ok=True)
    return Path(f"{prefix}.model")


# ---------------------------------------------------------------------------
# HuggingFace wrapper
# ---------------------------------------------------------------------------
def build_hf_tokenizer(sp_model_path: Path, output_dir: Path) -> PreTrainedTokenizerFast:
    """Wrap the trained SentencePiece model as a HuggingFace PreTrainedTokenizerFast."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the SP model
    sp = spm.SentencePieceProcessor()
    sp.load(str(sp_model_path))

    # Build a HuggingFace Tokenizer from the SP vocab
    vocab = {sp.id_to_piece(i): i for i in range(sp.get_piece_size())}

    # Use BPE model with the SP vocab
    merges: list[tuple[str, str]] = []
    # We will construct a BPE tokenizer from the sentencepiece vocab
    tokenizer_obj = Tokenizer(models.BPE(vocab=vocab, merges=merges, unk_token=UNK_TOKEN))
    tokenizer_obj.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", add_prefix_space=True)
    tokenizer_obj.decoder = decoders.Metaspace(replacement="▁", add_prefix_space=True)

    # Wrap into PreTrainedTokenizerFast
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_obj,
        unk_token=UNK_TOKEN,
        pad_token=PAD_TOKEN,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        model_max_length=2048,
    )

    # Save pretrained
    hf_tokenizer.save_pretrained(str(output_dir))
    console.log(f"[green]✓ HuggingFace tokenizer saved → {output_dir}[/green]")
    return hf_tokenizer


def build_hf_tokenizer_from_sp(sp_model_path: Path, output_dir: Path) -> PreTrainedTokenizerFast:
    """
    More robust approach: directly use the SP model file
    with PreTrainedTokenizerFast from the sentencepiece model.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use the SP processor to build a tokenizer_object via tokenizers library
    from tokenizers import SentencePieceBPETokenizer

    # Alternative: use the SP model directly via transformers
    # We'll use a simpler, more reliable approach
    sp = spm.SentencePieceProcessor()
    sp.load(str(sp_model_path))

    # Build vocab dict
    vocab = {}
    for i in range(sp.get_piece_size()):
        vocab[sp.id_to_piece(i)] = i

    # Build merges from SP model (extract from pieces that are not single chars)
    pieces = []
    for i in range(sp.get_piece_size()):
        piece = sp.id_to_piece(i)
        score = sp.get_score(i)
        pieces.append((piece, score, i))

    # For BPE, we reconstruct merges from the vocabulary order
    merges = []
    for piece, score, idx in sorted(pieces, key=lambda x: -x[1]):
        p = piece.replace("▁", "")
        if len(p) >= 2 and not piece.startswith("<"):
            # Try all possible splits
            for split_pos in range(1, len(piece)):
                left = piece[:split_pos]
                right = piece[split_pos:]
                if left in vocab and right in vocab:
                    merges.append((left, right))
                    break

    tok = Tokenizer(models.BPE(vocab=vocab, merges=merges, unk_token=UNK_TOKEN))
    tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", add_prefix_space=True)
    tok.decoder = decoders.Metaspace(replacement="▁", add_prefix_space=True)

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token=UNK_TOKEN,
        pad_token=PAD_TOKEN,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        model_max_length=2048,
    )

    hf_tokenizer.save_pretrained(str(output_dir))
    console.log(f"[green]✓ HuggingFace tokenizer saved → {output_dir}[/green]")
    return hf_tokenizer


# ---------------------------------------------------------------------------
# SP-backed tokenizer (most reliable)
# ---------------------------------------------------------------------------
class SPTokenizerWrapper:
    """
    Thin wrapper around SentencePieceProcessor that provides
    an interface compatible with our training pipeline.
    Also saved as a HuggingFace PreTrainedTokenizerFast.
    """

    def __init__(self, sp_model_path: str | Path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(sp_model_path))

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    @property
    def pad_id(self) -> int:
        return self.sp.pad_id()

    @property
    def unk_id(self) -> int:
        return self.sp.unk_id()

    @property
    def bos_id(self) -> int:
        return self.sp.bos_id()

    @property
    def eos_id(self) -> int:
        return self.sp.eos_id()

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        ids = self.sp.encode(text)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        # Filter out special tokens before decoding
        filtered = [i for i in ids if i not in {self.pad_id, self.bos_id, self.eos_id}]
        return self.sp.decode(filtered)

    def encode_as_pieces(self, text: str) -> list[str]:
        return self.sp.encode_as_pieces(text)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_tokenizer(
    sp_model_path: Path,
    hf_tokenizer: PreTrainedTokenizerFast,
    val_lines: list[str],
) -> dict:
    """Encode/decode 100 random sentences, measure roundtrip accuracy."""
    sp_tok = SPTokenizerWrapper(sp_model_path)
    rng = random.Random(SEED)

    sample = rng.sample(val_lines, min(100, len(val_lines)))

    # ---- Roundtrip via SP ----
    sp_exact = 0
    for s in sample:
        ids = sp_tok.encode(s, add_bos=False, add_eos=False)
        decoded = sp_tok.decode(ids)
        if decoded.strip() == s.strip():
            sp_exact += 1
    sp_accuracy = sp_exact / len(sample) * 100

    # ---- Roundtrip via HF ----
    hf_exact = 0
    for s in sample:
        ids = hf_tokenizer.encode(s)
        decoded = hf_tokenizer.decode(ids, skip_special_tokens=True)
        if decoded.strip() == s.strip():
            hf_exact += 1
    hf_accuracy = hf_exact / len(sample) * 100

    console.log(f"SP  roundtrip accuracy: [cyan]{sp_accuracy:.1f}%[/cyan]  ({sp_exact}/{len(sample)})")
    console.log(f"HF  roundtrip accuracy: [cyan]{hf_accuracy:.1f}%[/cyan]  ({hf_exact}/{len(sample)})")

    # ---- Vietnamese diacritics test ----
    test_str = "việt nam"
    sp_ids = sp_tok.encode(test_str, add_bos=False, add_eos=False)
    sp_dec = sp_tok.decode(sp_ids)
    assert "việt" in sp_dec.lower() and "nam" in sp_dec.lower(), (
        f"Diacritics lost during SP encode/decode: '{test_str}' → '{sp_dec}'"
    )
    console.log(f"[green]✓ Diacritics preserved: '{test_str}' → encode → decode → '{sp_dec}'[/green]")

    # ---- OOV rate on val ----
    total_tokens_val = 0
    unk_tokens_val = 0
    tokens_per_sentence: list[int] = []
    for line in val_lines[:5000]:
        ids = sp_tok.encode(line, add_bos=False, add_eos=False)
        total_tokens_val += len(ids)
        unk_tokens_val += ids.count(sp_tok.unk_id)
        tokens_per_sentence.append(len(ids))

    oov_rate = unk_tokens_val / max(total_tokens_val, 1) * 100
    avg_tps = sum(tokens_per_sentence) / max(len(tokens_per_sentence), 1)

    console.log(f"Vocab size: [cyan]{sp_tok.vocab_size}[/cyan]")
    console.log(f"OOV rate (val): [cyan]{oov_rate:.3f}%[/cyan]")
    console.log(f"Tokens/sentence avg: [cyan]{avg_tps:.1f}[/cyan]")

    return {
        "vocab_size": sp_tok.vocab_size,
        "sp_roundtrip_accuracy": round(sp_accuracy, 2),
        "hf_roundtrip_accuracy": round(hf_accuracy, 2),
        "oov_rate_pct": round(oov_rate, 4),
        "tokens_per_sentence_avg": round(avg_tps, 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(corpus_path: Path) -> None:
    console.rule("[bold blue]Vietnamese BPE Tokenizer Training[/bold blue]")
    device = _device()
    console.log(f"Device: [cyan]{device}[/cyan]")

    # ---- Collect shards ----
    shard_paths = _collect_shard_paths(corpus_path)

    # ---- Reservoir sample ----
    console.log(f"[bold]Reservoir sampling up to {MAX_TRAIN_LINES:,} lines …[/bold]")
    sampled_lines = _reservoir_sample(shard_paths, MAX_TRAIN_LINES, seed=SEED)

    # ---- Train SentencePiece ----
    tokenizer_dir = Path("tokenizer")
    sp_model_path = train_sentencepiece(sampled_lines, tokenizer_dir)

    # ---- Build HuggingFace wrapper ----
    hf_dir = tokenizer_dir / "hf_tokenizer"
    try:
        hf_tokenizer = build_hf_tokenizer_from_sp(sp_model_path, hf_dir)
    except Exception as exc:
        console.log(f"[yellow]Fast wrapper failed ({exc}), using basic wrapper …[/yellow]")
        hf_tokenizer = build_hf_tokenizer(sp_model_path, hf_dir)

    # ---- Validation ----
    # Load val lines for validation
    val_dir = corpus_path / "val"
    val_lines: list[str] = []
    if val_dir.exists():
        for vf in sorted(val_dir.glob("*.txt")):
            with open(vf, "r", encoding="utf-8") as f:
                val_lines.extend(line.strip() for line in f if line.strip())
    if not val_lines:
        # Fallback: use a subset of sampled lines
        val_lines = sampled_lines[:5000]

    stats = validate_tokenizer(sp_model_path, hf_tokenizer, val_lines)

    # ---- Save stats ----
    stats_path = tokenizer_dir / "tokenizer_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    console.log(f"[green]✓ Stats saved → {stats_path}[/green]")

    console.rule("[bold green]Done[/bold green]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Vietnamese BPE tokenizer")
    parser.add_argument("--corpus_path", type=str, default="corpus", help="Path to sharded corpus")
    args = parser.parse_args()
    main(Path(args.corpus_path))

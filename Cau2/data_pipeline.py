"""
data_pipeline.py
================
Download, clean, deduplicate and shard a Vietnamese text corpus
from CC-100 and Wikipedia for LSTM language modelling.

Sources
-------
* CC-100 vi   – ``datasets.load_dataset("cc100", lang="vi", streaming=True)``
* Wikipedia vi – ``datasets.load_dataset("wikipedia", "20231101.vi", streaming=True)``

Run
---
    python data_pipeline.py                         # default paths
    python data_pipeline.py --corpus_path ./corpus  # custom output dir
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Generator, Iterable

import torch
from datasets import load_dataset
from datasketch import MinHash, MinHashLSH
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

console = Console()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CORPUS_PATH = Path("corpus")
SHARD_MAX_BYTES = 100 * 1024 * 1024  # 100 MB
MIN_LINE_CHARS = 20
MAX_LINE_CHARS = 1000
VN_UNICODE_RATIO = 0.50
MINHASH_THRESHOLD = 0.85
MINHASH_NUM_PERM = 128
LOG_EVERY = 100_000
MAX_RETRIES = 3
SEED = 42
VAL_TEST_RATIO = 0.01  # 1 % each for val and test

# Vietnamese character ranges (Latin + Vietnamese diacritics)
_VN_RE = re.compile(
    r"[a-zA-Zàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợ"
    r"ùúủũụưứừửữựỳýỷỹỵđĐÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌ"
    r"ÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ]"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _device() -> torch.device:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _nfc(text: str) -> str:
    """NFC-normalise a string."""
    return unicodedata.normalize("NFC", text)


def _vn_ratio(text: str) -> float:
    """Return fraction of characters matching Vietnamese unicode ranges."""
    if not text:
        return 0.0
    matches = _VN_RE.findall(text)
    return len(matches) / len(text)


def _is_valid_line(line: str) -> bool:
    """Apply length + Vietnamese-ratio filters."""
    n = len(line)
    if n < MIN_LINE_CHARS or n > MAX_LINE_CHARS:
        return False
    if _vn_ratio(line) < VN_UNICODE_RATIO:
        return False
    return True


def _line_minhash(text: str, num_perm: int = MINHASH_NUM_PERM) -> MinHash:
    """Build a MinHash signature from word-level shingles (3-grams)."""
    m = MinHash(num_perm=num_perm)
    words = text.split()
    for i in range(len(words) - 2):
        shingle = " ".join(words[i : i + 3])
        m.update(shingle.encode("utf-8"))
    return m


# ---------------------------------------------------------------------------
# Streaming with exponential back-off
# ---------------------------------------------------------------------------
def _load_with_retry(loader_fn, description: str, max_retries: int = MAX_RETRIES):
    """Call *loader_fn* with exponential back-off on failure."""
    for attempt in range(1, max_retries + 1):
        try:
            ds = loader_fn()
            console.log(f"[green]✓ {description} loaded (attempt {attempt})[/green]")
            return ds
        except Exception as exc:
            wait = 2 ** attempt
            console.log(
                f"[yellow]⚠ {description} attempt {attempt}/{max_retries} "
                f"failed: {exc}. Retrying in {wait}s …[/yellow]"
            )
            time.sleep(wait)
    console.log(f"[red]✗ {description} failed after {max_retries} attempts.[/red]")
    return None


def _stream_lines(dataset_iter: Iterable, text_key: str) -> Generator[str, None, None]:
    """Yield cleaned text lines from a HuggingFace streaming dataset."""
    for example in dataset_iter:
        raw = example.get(text_key, "")
        if not raw:
            continue
        for line in raw.split("\n"):
            line = _nfc(line.strip())
            if _is_valid_line(line):
                yield line


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
class StreamingDeduplicator:
    """MinHash LSH deduplicator that operates on a streaming basis."""

    def __init__(self, threshold: float = MINHASH_THRESHOLD, num_perm: int = MINHASH_NUM_PERM):
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.num_perm = num_perm
        self._counter = 0

    def is_duplicate(self, text: str) -> bool:
        mh = _line_minhash(text, self.num_perm)
        if self.lsh.query(mh):
            return True
        key = f"doc_{self._counter}"
        try:
            self.lsh.insert(key, mh)
        except ValueError:
            # duplicate key (shouldn't happen but guard against it)
            return True
        self._counter += 1
        return False


# ---------------------------------------------------------------------------
# Shard writer
# ---------------------------------------------------------------------------
class ShardWriter:
    """Write lines into fixed-size text shards, skipping existing ones."""

    def __init__(self, base_dir: Path, max_bytes: int = SHARD_MAX_BYTES):
        self.base_dir = base_dir
        self.max_bytes = max_bytes
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Figure out the next shard index (resume support)
        existing = sorted(self.base_dir.glob("shard_*.txt"))
        self.shard_idx = len(existing)
        self._buf: list[str] = []
        self._buf_bytes = 0
        self.total_written = 0

    def _shard_path(self, idx: int) -> Path:
        return self.base_dir / f"shard_{idx:04d}.txt"

    def _flush(self) -> None:
        if not self._buf:
            return
        path = self._shard_path(self.shard_idx)
        if path.exists():
            # Resume: skip already-written shard
            console.log(f"[dim]Shard {path.name} exists — skipping write[/dim]")
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(self._buf))
                f.write("\n")
        self.shard_idx += 1
        self._buf.clear()
        self._buf_bytes = 0

    def add(self, line: str) -> None:
        encoded_len = len(line.encode("utf-8"))
        if self._buf_bytes + encoded_len > self.max_bytes and self._buf:
            self._flush()
        self._buf.append(line)
        self._buf_bytes += encoded_len
        self.total_written += 1

    def close(self) -> None:
        self._flush()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_pipeline(corpus_path: Path) -> dict:
    """Execute the full data pipeline. Returns stats dict."""
    random.seed(SEED)
    device = _device()
    console.rule("[bold blue]Vietnamese Corpus Pipeline[/bold blue]")
    console.log(f"Device detected: [cyan]{device}[/cyan]")
    console.log(f"Output directory: [cyan]{corpus_path}[/cyan]")

    # ---- Step 0: Load datasets with back-off ----
    cc100_ds = _load_with_retry(
        lambda: load_dataset("cc100", lang="vi", streaming=True, split="train", trust_remote_code=True),
        "CC-100 vi",
    )
    wiki_ds = _load_with_retry(
        lambda: load_dataset("wikipedia", "20231101.vi", streaming=True, split="train", trust_remote_code=True),
        "Wikipedia vi",
    )

    if cc100_ds is None and wiki_ds is None:
        console.log("[red]Both datasets failed to load. Exiting.[/red]")
        sys.exit(1)

    # ---- Step 1-3: Stream → filter → dedup ----
    dedup = StreamingDeduplicator()
    train_writer = ShardWriter(corpus_path / "train")
    val_lines: list[str] = []
    test_lines: list[str] = []

    total_tokens = 0
    total_lines = 0
    char_set: set[str] = set()
    sentence_lengths: list[int] = []
    dup_count = 0

    def _process_stream(ds_iter, text_key: str, label: str):
        nonlocal total_tokens, total_lines, dup_count

        console.log(f"[bold]Processing {label} …[/bold]")
        for line in _stream_lines(ds_iter, text_key):
            # dedup
            if dedup.is_duplicate(line):
                dup_count += 1
                continue

            total_lines += 1
            tokens = line.split()
            n_tok = len(tokens)
            total_tokens += n_tok
            char_set.update(line)
            sentence_lengths.append(n_tok)

            # reservoir-sample val/test at ~1 % each
            r = random.random()
            if r < VAL_TEST_RATIO:
                val_lines.append(line)
            elif r < 2 * VAL_TEST_RATIO:
                test_lines.append(line)
            else:
                train_writer.add(line)

            if total_lines % LOG_EVERY == 0:
                console.log(
                    f"  [{label}] {total_lines:,} lines | "
                    f"{total_tokens:,} tokens | "
                    f"{dup_count:,} dups removed"
                )

    # Process CC-100
    if cc100_ds is not None:
        _process_stream(cc100_ds, "text", "CC-100")

    # Process Wikipedia
    if wiki_ds is not None:
        _process_stream(wiki_ds, "text", "Wikipedia")

    # ---- Flush remaining train shards ----
    train_writer.close()

    # ---- Write val / test ----
    for split_name, split_data in [("val", val_lines), ("test", test_lines)]:
        split_dir = corpus_path / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        out_path = split_dir / "shard_0000.txt"
        if not out_path.exists():
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(split_data))
                f.write("\n")
            console.log(f"[green]Wrote {split_name}: {len(split_data):,} lines → {out_path}[/green]")
        else:
            console.log(f"[dim]{split_name} shard exists — skipped[/dim]")

    # ---- Step 4: Underthesea validation on a 10k sample ----
    try:
        from underthesea import word_tokenize

        console.log("[bold]Running underthesea segmentation on 10 k sample …[/bold]")
        sample_lines = val_lines[:10_000] if len(val_lines) >= 10_000 else val_lines
        segmented_count = 0
        for sl in sample_lines[:10_000]:
            _ = word_tokenize(sl)
            segmented_count += 1
        console.log(f"[green]Underthesea segmentation validated on {segmented_count:,} lines[/green]")
    except ImportError:
        console.log("[yellow]underthesea not installed — skipping segmentation validation[/yellow]")

    # ---- Step 5: Report ----
    avg_sent_len = sum(sentence_lengths) / max(len(sentence_lengths), 1)
    train_shards = sorted((corpus_path / "train").glob("shard_*.txt"))

    stats = {
        "total_tokens": total_tokens,
        "total_lines": total_lines,
        "unique_chars": len(char_set),
        "avg_sentence_length_tokens": round(avg_sent_len, 2),
        "train_shards": len(train_shards),
        "val_lines": len(val_lines),
        "test_lines": len(test_lines),
        "duplicates_removed": dup_count,
        "device": str(device),
    }

    stats_path = corpus_path / "corpus_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    console.rule("[bold green]Corpus Statistics[/bold green]")
    for k, v in stats.items():
        console.log(f"  {k}: [cyan]{v}[/cyan]")

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vietnamese corpus pipeline")
    parser.add_argument(
        "--corpus_path",
        type=str,
        default="corpus",
        help="Root directory for sharded output (default: ./corpus)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(Path(args.corpus_path))

"""
data_pipeline.py
================
Download, clean, deduplicate and shard a Vietnamese text corpus
from CC-100 and Wikipedia for LSTM language modelling.

Sources
-------
* Wikipedia vi – ``datasets.load_dataset("wikimedia/wikipedia", "20231101.vi", streaming=True)``

Features
--------
* **Rich progress bars** — mỗi phase hiển thị tiến trình rõ ràng
* **Real resume** — checkpoint file lưu trạng thái (processed count, stats,
  val/test lines, dedup hashes) → chạy lại sẽ tiếp tục từ điểm dừng

Run
---
    python data_pipeline.py                         # default paths
    python data_pipeline.py --corpus_path ./corpus  # custom output dir
    python data_pipeline.py --reset                 # xóa checkpoint, chạy lại từ đầu
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
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
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    TaskProgressColumn,
)
from rich.table import Table
from rich.live import Live
from rich import box

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
CHECKPOINT_EVERY = 10_000  # save checkpoint every N lines processed
MAX_RETRIES = 3
SEED = 42
VAL_TEST_RATIO = 0.01  # 1 % each for val and test

# Approximate total articles for Wikipedia vi (for progress estimation)
WIKI_VI_APPROX_ARTICLES = 1_290_000

CHECKPOINT_FILE = "pipeline_checkpoint.pkl"

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


def _format_number(n: int) -> str:
    """Format number with thousand separators."""
    return f"{n:,}"


def _format_bytes(b: int) -> str:
    """Human-readable byte size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} TB"


# ---------------------------------------------------------------------------
# Pipeline Phases — enum-like for tracking
# ---------------------------------------------------------------------------
PHASES = [
    ("📥", "Load Dataset",        "Tải dataset Wikipedia vi với retry"),
    ("🔄", "Stream & Filter",     "Stream → NFC normalize → filter chất lượng"),
    ("🧹", "Deduplicate",         "MinHash LSH loại trùng lặp"),
    ("✂️",  "Shard & Split",       "Chia train/val/test → ghi shards"),
    ("🔍", "Validate",            "Kiểm tra underthesea word segmentation"),
    ("📊", "Report",              "Tổng hợp & lưu thống kê"),
]


def _print_phase_banner(phase_idx: int, total: int = len(PHASES)):
    """Print a rich banner for the current pipeline phase."""
    emoji, name, desc = PHASES[phase_idx]
    console.print()
    console.rule(
        f"[bold cyan]{emoji}  Phase {phase_idx + 1}/{total}: {name}[/bold cyan]"
    )
    console.print(f"  [dim]{desc}[/dim]")
    console.print()


# ---------------------------------------------------------------------------
# Checkpoint: save & load pipeline state
# ---------------------------------------------------------------------------
class PipelineCheckpoint:
    """Saves and restores pipeline progress for resume support."""

    def __init__(self, corpus_path: Path):
        self.path = corpus_path / CHECKPOINT_FILE
        self.state: dict = {}

    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> dict:
        if self.path.exists():
            with open(self.path, "rb") as f:
                self.state = pickle.load(f)
            return self.state
        return {}

    def save(self, state: dict) -> None:
        self.state = state
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(state, f)
        tmp.replace(self.path)  # atomic on most OS

    def delete(self) -> None:
        if self.path.exists():
            self.path.unlink()
            console.print("[yellow]🗑  Checkpoint deleted[/yellow]")

    def summary(self) -> str:
        """Return human-readable summary of checkpoint."""
        if not self.state:
            return "No checkpoint"
        lines = self.state.get("total_lines", 0)
        articles = self.state.get("articles_processed", 0)
        phase = self.state.get("completed_phase", -1)
        return (
            f"Phase {phase + 1} completed | "
            f"{_format_number(articles)} articles | "
            f"{_format_number(lines)} lines kept"
        )


# ---------------------------------------------------------------------------
# Streaming with exponential back-off
# ---------------------------------------------------------------------------
def _load_with_retry(loader_fn, description: str, max_retries: int = MAX_RETRIES):
    """Call *loader_fn* with exponential back-off on failure."""
    for attempt in range(1, max_retries + 1):
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn(f"[bold]{description}[/bold] — attempt {attempt}/{max_retries}"),
                TimeElapsedColumn(),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task("loading", total=None)
                ds = loader_fn()
                progress.update(task, completed=1, total=1)
            console.print(
                f"  [green]✓ {description} loaded successfully "
                f"(attempt {attempt})[/green]"
            )
            return ds
        except Exception as exc:
            wait = 2 ** attempt
            console.print(
                f"  [yellow]⚠ {description} attempt {attempt}/{max_retries} "
                f"failed: {exc}[/yellow]"
            )
            if attempt < max_retries:
                console.print(f"  [dim]  Retrying in {wait}s …[/dim]")
                time.sleep(wait)
    console.print(f"  [red]✗ {description} failed after {max_retries} attempts.[/red]")
    return None


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
class StreamingDeduplicator:
    """MinHash LSH deduplicator that operates on a streaming basis."""

    def __init__(self, threshold: float = MINHASH_THRESHOLD, num_perm: int = MINHASH_NUM_PERM):
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.num_perm = num_perm
        self._counter = 0
        # Keep a set of fast exact-hash for quick duplicate detection
        self._seen_hashes: set[str] = set()

    def is_duplicate(self, text: str) -> bool:
        # Fast exact hash check first
        h = hashlib.md5(text.encode("utf-8")).hexdigest()
        if h in self._seen_hashes:
            return True
        self._seen_hashes.add(h)

        # MinHash near-duplicate check
        mh = _line_minhash(text, self.num_perm)
        if self.lsh.query(mh):
            return True
        key = f"doc_{self._counter}"
        try:
            self.lsh.insert(key, mh)
        except ValueError:
            return True
        self._counter += 1
        return False

    def get_state(self) -> dict:
        """Export state for checkpointing (only exact hashes, LSH is rebuilt)."""
        return {
            "seen_hashes": self._seen_hashes.copy(),
            "counter": self._counter,
        }

    def restore_state(self, state: dict) -> None:
        """Restore from checkpoint."""
        self._seen_hashes = state.get("seen_hashes", set())
        self._counter = state.get("counter", 0)


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
        self.total_bytes_written = 0

    def _shard_path(self, idx: int) -> Path:
        return self.base_dir / f"shard_{idx:04d}.txt"

    def _flush(self) -> None:
        if not self._buf:
            return
        path = self._shard_path(self.shard_idx)
        if path.exists():
            console.print(f"    [dim]Shard {path.name} exists — skipping write[/dim]")
        else:
            with open(path, "w", encoding="utf-8") as f:
                content = "\n".join(self._buf) + "\n"
                f.write(content)
                self.total_bytes_written += len(content.encode("utf-8"))
            console.print(
                f"    [green]💾 Wrote {path.name} "
                f"({_format_number(len(self._buf))} lines, "
                f"{_format_bytes(self._buf_bytes)})[/green]"
            )
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
def run_pipeline(corpus_path: Path, reset: bool = False) -> dict:
    """Execute the full data pipeline. Returns stats dict."""
    random.seed(SEED)
    device = _device()

    # ── Header ──
    console.print()
    console.print(
        Panel.fit(
            "[bold white]Vietnamese Corpus Pipeline[/bold white]\n"
            f"[dim]Device: {device}  •  Output: {corpus_path}  •  "
            f"Shard size: {_format_bytes(SHARD_MAX_BYTES)}[/dim]",
            border_style="blue",
            title="🇻🇳 Cau2",
            subtitle="LSTM Language Model Data",
        )
    )

    # ── Checkpoint handling ──
    ckpt = PipelineCheckpoint(corpus_path)

    if reset:
        ckpt.delete()

    resume_state = {}
    skip_to_article = 0
    completed_phase = 0
    if ckpt.exists():
        resume_state = ckpt.load()
        skip_to_article = resume_state.get("articles_processed", 0)
        completed_phase = resume_state.get("completed_phase", 0)

        console.print()
        console.print(
            Panel(
                f"[bold yellow]▶  RESUMING from checkpoint[/bold yellow]\n"
                f"[dim]{ckpt.summary()}[/dim]\n"
                f"[dim]Will skip first {_format_number(skip_to_article)} articles[/dim]",
                border_style="yellow",
                title="♻️  Resume",
            )
        )
    else:
        console.print()
        console.print("  [dim]No checkpoint found — starting fresh run[/dim]")

    # ── Restore stats from checkpoint for later phases ──
    total_tokens = resume_state.get("total_tokens", 0)
    total_lines = resume_state.get("total_lines", 0)
    char_set: set[str] = resume_state.get("char_set", set())
    sentence_lengths: list[int] = resume_state.get("sentence_lengths", [])
    dup_count = resume_state.get("dup_count", 0)
    articles_processed = resume_state.get("articles_processed", 0)
    lines_filtered_out = resume_state.get("lines_filtered_out", 0)
    val_lines: list[str] = resume_state.get("val_lines", [])
    test_lines: list[str] = resume_state.get("test_lines", [])

    # ══════════════════════════════════════════════════════════════
    # FAST PATH: If phases 1-4 are already done, skip entirely
    # ══════════════════════════════════════════════════════════════
    if completed_phase >= 4:
        console.print()
        console.print(
            Panel(
                f"[bold green]⏭  Phases 1-4 already completed[/bold green]\n"
                f"[dim]{_format_number(articles_processed)} articles • "
                f"{_format_number(total_lines)} lines kept • "
                f"{_format_number(dup_count)} dups removed[/dim]\n"
                f"[dim]Skipping dataset loading & streaming — jumping to Phase 5[/dim]",
                border_style="green",
                title="✅ Skip",
            )
        )
    else:
        # ================================================================
        # PHASE 1: Load dataset
        # ================================================================
        _print_phase_banner(0)

        wiki_ds = _load_with_retry(
            lambda: load_dataset(
                "wikimedia/wikipedia", "20231101.vi",
                streaming=True, split="train",
            ),
            "Wikipedia vi",
        )

        if wiki_ds is None:
            console.print("[red bold]❌ Dataset failed to load. Exiting.[/red bold]")
            sys.exit(1)

        # ================================================================
        # PHASE 2 + 3 + 4: Stream → Filter → Dedup → Shard
        # (merged into one streaming pass for efficiency)
        # ================================================================
        _print_phase_banner(1)
        _print_phase_banner(2)
        _print_phase_banner(3)

        console.print(
            "  [bold]Running phases 2-4 in a single streaming pass "
            "(filter → dedup → shard)[/bold]"
        )
        console.print()

        # Restore or init state
        dedup = StreamingDeduplicator()
        train_writer = ShardWriter(corpus_path / "train")

        if "dedup_state" in resume_state:
            dedup.restore_state(resume_state["dedup_state"])

        # Track shard writer offset for resume
        train_writer.total_written = resume_state.get("train_lines_written", 0)

        start_time = time.time()
        last_checkpoint_time = start_time

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}[/bold blue]"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TextColumn("[green]{task.fields[kept]}[/green] kept"),
            TextColumn("[red]{task.fields[dups]}[/red] dups"),
            TextColumn("[dim]{task.fields[filtered]}[/dim] filtered"),
            console=console,
            refresh_per_second=4,
        ) as progress:
            task = progress.add_task(
                "Wikipedia vi",
                total=WIKI_VI_APPROX_ARTICLES,
                kept=_format_number(total_lines),
                dups=_format_number(dup_count),
                filtered=_format_number(lines_filtered_out),
            )

            # Update progress bar if resuming
            if articles_processed > 0:
                progress.update(task, completed=articles_processed)

            for example in wiki_ds:
                articles_processed += 1

                # Skip already-processed articles on resume
                if articles_processed <= skip_to_article:
                    if articles_processed % 50_000 == 0:
                        progress.update(
                            task,
                            completed=articles_processed,
                            description=f"⏩ Skipping (resume)",
                            kept=_format_number(total_lines),
                            dups=_format_number(dup_count),
                            filtered=_format_number(lines_filtered_out),
                        )
                    continue

                raw = example.get("text", "")
                if not raw:
                    progress.update(task, completed=articles_processed)
                    continue

                for line in raw.split("\n"):
                    line = _nfc(line.strip())

                    # Filter
                    if not _is_valid_line(line):
                        lines_filtered_out += 1
                        continue

                    # Dedup
                    if dedup.is_duplicate(line):
                        dup_count += 1
                        continue

                    # Accept line
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

                # Update progress bar
                progress.update(
                    task,
                    completed=articles_processed,
                    description="Wikipedia vi",
                    kept=_format_number(total_lines),
                    dups=_format_number(dup_count),
                    filtered=_format_number(lines_filtered_out),
                )

                # Periodic checkpoint
                now = time.time()
                if articles_processed % CHECKPOINT_EVERY == 0 or (now - last_checkpoint_time) > 300:
                    ckpt.save({
                        "articles_processed": articles_processed,
                        "total_lines": total_lines,
                        "total_tokens": total_tokens,
                        "dup_count": dup_count,
                        "lines_filtered_out": lines_filtered_out,
                        "val_lines": val_lines,
                        "test_lines": test_lines,
                        "char_set": char_set,
                        "sentence_lengths": sentence_lengths,
                        "dedup_state": dedup.get_state(),
                        "train_lines_written": train_writer.total_written,
                        "completed_phase": 3,
                    })
                    last_checkpoint_time = now

        elapsed = time.time() - start_time
        console.print()
        console.print(
            f"  [bold green]✓ Streaming complete[/bold green] — "
            f"{_format_number(articles_processed)} articles processed in "
            f"{elapsed:.0f}s"
        )

        # Flush remaining train shards
        console.print("  [dim]Flushing remaining train buffer …[/dim]")
        train_writer.close()

        # Write val / test
        for split_name, split_data in [("val", val_lines), ("test", test_lines)]:
            split_dir = corpus_path / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            out_path = split_dir / "shard_0000.txt"
            if not out_path.exists():
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(split_data))
                    f.write("\n")
                console.print(
                    f"  [green]💾 Wrote {split_name}: "
                    f"{_format_number(len(split_data))} lines → {out_path}[/green]"
                )
            else:
                console.print(
                    f"  [dim]{split_name} shard already exists — skipped[/dim]"
                )

        # Save checkpoint after writing splits
        ckpt.save({
            "articles_processed": articles_processed,
            "total_lines": total_lines,
            "total_tokens": total_tokens,
            "dup_count": dup_count,
            "lines_filtered_out": lines_filtered_out,
            "val_lines": val_lines,
            "test_lines": test_lines,
            "char_set": char_set,
            "sentence_lengths": sentence_lengths,
            "dedup_state": dedup.get_state(),
            "train_lines_written": resume_state.get("train_lines_written", 0),
            "completed_phase": 4,
        })

    # ================================================================
    # PHASE 5: Underthesea validation
    # ================================================================
    _print_phase_banner(4)

    try:
        from underthesea import word_tokenize

        sample_lines = val_lines[:10_000] if len(val_lines) >= 10_000 else val_lines
        sample_count = len(sample_lines)

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold]Underthesea segmentation[/bold]"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            seg_task = progress.add_task("Segmenting", total=sample_count)
            segmented_count = 0
            for sl in sample_lines:
                _ = word_tokenize(sl)
                segmented_count += 1
                progress.update(seg_task, advance=1)

        console.print(
            f"  [green]✓ Underthesea validated on "
            f"{_format_number(segmented_count)} lines[/green]"
        )
    except ImportError:
        console.print(
            "  [yellow]⚠  underthesea not installed — "
            "skipping segmentation validation[/yellow]"
        )

    # ================================================================
    # PHASE 6: Report
    # ================================================================
    _print_phase_banner(5)

    avg_sent_len = sum(sentence_lengths) / max(len(sentence_lengths), 1)
    train_shards = sorted((corpus_path / "train").glob("shard_*.txt"))

    stats = {
        "total_tokens": total_tokens,
        "total_lines": total_lines,
        "unique_chars": len(char_set),
        "avg_sentence_length_tokens": round(avg_sent_len, 2),
        "train_shards": len(train_shards),
        "train_lines": train_writer.total_written,
        "val_lines": len(val_lines),
        "test_lines": len(test_lines),
        "duplicates_removed": dup_count,
        "lines_filtered_out": lines_filtered_out,
        "articles_processed": articles_processed,
        "device": str(device),
    }

    stats_path = corpus_path / "corpus_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # Pretty table
    table = Table(
        title="📊 Corpus Statistics",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        border_style="blue",
    )
    table.add_column("Metric", style="bold white", min_width=28)
    table.add_column("Value", style="green", justify="right", min_width=16)

    nice_names = {
        "total_tokens":               "Total Tokens",
        "total_lines":                "Total Lines (kept)",
        "unique_chars":               "Unique Characters",
        "avg_sentence_length_tokens": "Avg Sentence Length (tokens)",
        "train_shards":               "Train Shards",
        "train_lines":                "Train Lines",
        "val_lines":                  "Val Lines",
        "test_lines":                 "Test Lines",
        "duplicates_removed":         "Duplicates Removed",
        "lines_filtered_out":         "Lines Filtered Out",
        "articles_processed":         "Articles Processed",
        "device":                     "Device",
    }

    for key, value in stats.items():
        display_name = nice_names.get(key, key)
        if isinstance(value, int):
            display_value = _format_number(value)
        else:
            display_value = str(value)
        table.add_row(display_name, display_value)

    console.print()
    console.print(table)
    console.print()
    console.print(f"  [dim]Stats saved to {stats_path}[/dim]")

    # Clean up checkpoint on successful completion
    ckpt.delete()
    console.print()
    console.print(
        Panel.fit(
            "[bold green]✅ Pipeline completed successfully![/bold green]",
            border_style="green",
        )
    )

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vietnamese corpus pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python data_pipeline.py                         # default\n"
            "  python data_pipeline.py --corpus_path ./corpus  # custom dir\n"
            "  python data_pipeline.py --reset                 # fresh run\n"
        ),
    )
    parser.add_argument(
        "--corpus_path",
        type=str,
        default="corpus",
        help="Root directory for sharded output (default: ./corpus)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing checkpoint and start a fresh run",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(Path(args.corpus_path), reset=args.reset)

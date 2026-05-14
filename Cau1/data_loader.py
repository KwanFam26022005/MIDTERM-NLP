"""
data_loader.py
==============
Load and unify 3 Vietnamese NLP datasets into a single HuggingFace DatasetDict.

Datasets
--------
- UIT-VSFC : Sentiment (3 labels: 0=neg, 1=neu, 2=pos)  — TSV format
- UIT-ViSFD: Aspect-Based Sentiment Analysis (4 aspects × 3 polarities) — JSON format
- VLSP 2016 NER: IOB2 NER (PER, ORG, LOC, MISC) — CoNLL format

Author : NLP Pipeline
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import torch
from datasets import Dataset, DatasetDict, Features, Sequence, Value
from transformers import PreTrainedTokenizerFast

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VSFC_URL = "https://github.com/sonlam1102/UIT-VSFC"
VISFD_URL = "https://github.com/kimkim00/UIT-ViSFD"
VLSP_NER_URL = "https://vlsp.org.vn/resources"

NER_LABEL_LIST: List[str] = [
    "O",
    "B-PER", "I-PER",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC",
    "B-MISC", "I-MISC",
]

NER_LABEL2ID: Dict[str, int] = {tag: idx for idx, tag in enumerate(NER_LABEL_LIST)}
NER_ID2LABEL: Dict[int, str] = {idx: tag for tag, idx in NER_LABEL2ID.items()}

ABSA_ASPECTS: List[str] = [
    "SCREEN", "CAMERA", "FEATURES", "BATTERY",
    "PERFORMANCE", "STORAGE", "DESIGN", "PRICE",
    "GENERAL", "SER&ACC",
]
ABSA_POLARITIES: Dict[str, int] = {"negative": 0, "neutral": 1, "positive": 2}

SENTIMENT_LABELS: int = 3
MAX_LENGTH: int = 256
PAD_LABEL_ID: int = -100


# ---------------------------------------------------------------------------
# Helper: check path or raise with download URL
# ---------------------------------------------------------------------------

def _require_path(path: str | Path, url: str) -> Path:
    """Return *path* as a resolved ``Path`` if it exists, otherwise raise."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Dataset path not found: {p}. Download: {url}"
        )
    return p


# ---------------------------------------------------------------------------
# Readers for each dataset
# ---------------------------------------------------------------------------

def _read_vsfc(root: str | Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Read UIT-VSFC dataset.

    Supports TWO formats:
      A) Subfolder format (original dataset):
            root/train/sents.txt  +  root/train/sentiments.txt
            root/dev/...          (or root/test/...)
      B) TSV format (preprocessed):
            root/train.txt   (sentence\\tlabel)
    """
    root = _require_path(root, VSFC_URL)
    splits: Dict[str, List[Dict[str, Any]]] = {}
    # Map split names to (subfolder_name, tsv_file) pairs
    mapping = {
        "train": ("train", "train.txt"),
        "val":   ("dev",   "dev.txt"),
        "test":  ("test",  "test.txt"),
    }

    for split_name, (subfolder, tsv_fname) in mapping.items():
        records: List[Dict[str, Any]] = []

        # ── Format A: subfolder with sents.txt + sentiments.txt ──
        sents_path = root / subfolder / "sents.txt"
        labels_path = root / subfolder / "sentiments.txt"
        if sents_path.exists() and labels_path.exists():
            with open(sents_path, "r", encoding="utf-8") as fs, \
                 open(labels_path, "r", encoding="utf-8") as fl:
                for text_line, label_line in zip(fs, fl):
                    text = text_line.strip()
                    label_str = label_line.strip()
                    if not text or not label_str:
                        continue
                    try:
                        label = int(label_str)
                    except ValueError:
                        continue
                    records.append({
                        "text": text,
                        "task": "sentiment",
                        "sentiment_label": label,
                        "aspect_labels": None,
                        "ner_tags": None,
                    })
            splits[split_name] = records
            continue

        # ── Format B: single TSV file ──
        tsv_path = root / tsv_fname
        if not tsv_path.exists():
            alt = root / f"{split_name}.txt"
            if alt.exists():
                tsv_path = alt
            else:
                raise FileNotFoundError(
                    f"Missing {sents_path} (subfolder) or {tsv_path} (TSV) "
                    f"for UIT-VSFC. Download: {VSFC_URL}"
                )
        with open(tsv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.rsplit("\t", maxsplit=1)
                if len(parts) != 2:
                    continue
                text, label_str = parts
                try:
                    label = int(label_str)
                except ValueError:
                    continue
                records.append({
                    "text": text.strip(),
                    "task": "sentiment",
                    "sentiment_label": label,
                    "aspect_labels": None,
                    "ner_tags": None,
                })
        splits[split_name] = records
    return splits


# Regex to parse aspect labels like {CAMERA#Positive};{BATTERY#Negative};
_ASPECT_RE = re.compile(r"\{([^#}]+)#(\w+)\}")


def _parse_visfd_label(label_str: str) -> Dict[str, int]:
    """
    Parse ViSFD label string ``{ASPECT#Polarity};...`` into a dict.

    Aspects not mentioned in the label string are set to neutral (1).
    The special aspect ``OTHERS`` is ignored.
    """
    aspect_dict: Dict[str, int] = {a: ABSA_POLARITIES["neutral"] for a in ABSA_ASPECTS}
    for match in _ASPECT_RE.finditer(label_str):
        aspect_name = match.group(1).strip()
        polarity = match.group(2).strip().lower()
        if aspect_name == "OTHERS":
            continue
        pol_id = ABSA_POLARITIES.get(polarity, ABSA_POLARITIES["neutral"])
        if aspect_name in aspect_dict:
            aspect_dict[aspect_name] = pol_id
    return aspect_dict


def _read_visfd(root: str | Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Read UIT-ViSFD dataset.

    Supports TWO formats:
      A) CSV format (original dataset):
            root/Train.csv  (columns: index, comment, n_star, date_time, label)
            label format: {CAMERA#Positive};{BATTERY#Negative};...
      B) JSON format (preprocessed):
            root/train.json  [{"text": ..., "aspects": {...}}]
    """
    root = _require_path(root, VISFD_URL)
    splits: Dict[str, List[Dict[str, Any]]] = {}
    mapping = {
        "train": (["Train.csv", "train.csv"], ["train.json"]),
        "val":   (["Dev.csv",   "dev.csv"],   ["dev.json", "val.json"]),
        "test":  (["Test.csv",  "test.csv"],  ["test.json"]),
    }

    for split_name, (csv_names, json_names) in mapping.items():
        records: List[Dict[str, Any]] = []

        # ── Format A: CSV ──
        csv_path: Optional[Path] = None
        for name in csv_names:
            candidate = root / name
            if candidate.exists():
                csv_path = candidate
                break

        if csv_path is not None:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    text = (row.get("comment") or row.get("text") or "").strip()
                    if not text:
                        continue
                    label_str = row.get("label", "")
                    aspect_dict = _parse_visfd_label(label_str)
                    records.append({
                        "text": text,
                        "task": "absa",
                        "sentiment_label": None,
                        "aspect_labels": aspect_dict,
                        "ner_tags": None,
                    })
            splits[split_name] = records
            continue

        # ── Format B: JSON (fallback) ──
        json_path: Optional[Path] = None
        for name in json_names:
            candidate = root / name
            if candidate.exists():
                json_path = candidate
                break

        if json_path is None:
            raise FileNotFoundError(
                f"Missing CSV or JSON files for UIT-ViSFD in {root}. "
                f"Download: {VISFD_URL}"
            )

        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        items = raw if isinstance(raw, list) else raw.get("data", raw.values())
        for item in items:
            text = item.get("text", item.get("sentence", "")).strip()
            if not text:
                continue
            aspects_raw = item.get("aspects", item.get("labels", {}))
            aspect_dict_j: Dict[str, int] = {}
            for asp in ABSA_ASPECTS:
                pol_str = aspects_raw.get(asp, aspects_raw.get(asp.lower(), "neutral"))
                if isinstance(pol_str, int):
                    aspect_dict_j[asp] = pol_str
                else:
                    aspect_dict_j[asp] = ABSA_POLARITIES.get(
                        pol_str.lower().strip(), ABSA_POLARITIES["neutral"]
                    )
            records.append({
                "text": text,
                "task": "absa",
                "sentiment_label": None,
                "aspect_labels": aspect_dict_j,
                "ner_tags": None,
            })
        splits[split_name] = records
    return splits


def _read_vlsp_ner(root: str | Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Read VLSP 2016 NER in CoNLL format.

    Expected structure:
        root/
            train.txt   (word TAG per line, blank line = sentence boundary)
            dev.txt
            test.txt
    """
    root = _require_path(root, VLSP_NER_URL)
    splits: Dict[str, List[Dict[str, Any]]] = {}
    mapping = {"train": "train.txt", "val": "dev.txt", "test": "test.txt"}

    for split_name, fname in mapping.items():
        fpath = root / fname
        if not fpath.exists():
            alt = root / f"{split_name}.conll"
            if alt.exists():
                fpath = alt
            else:
                raise FileNotFoundError(
                    f"Missing {fpath} for VLSP NER. Download: {VLSP_NER_URL}"
                )

        records: List[Dict[str, Any]] = []
        tokens: List[str] = []
        tags: List[str] = []

        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line.strip() == "":
                    if tokens:
                        records.append({
                            "text": " ".join(tokens),
                            "task": "ner",
                            "sentiment_label": None,
                            "aspect_labels": None,
                            "ner_tags": list(tags),
                        })
                        tokens = []
                        tags = []
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                word = parts[0]
                tag = parts[-1]
                # Normalise tag to IOB2 scheme
                if tag not in NER_LABEL2ID:
                    tag = "O"
                tokens.append(word)
                tags.append(tag)

            # Flush last sentence
            if tokens:
                records.append({
                    "text": " ".join(tokens),
                    "task": "ner",
                    "sentiment_label": None,
                    "aspect_labels": None,
                    "ner_tags": list(tags),
                })
        splits[split_name] = records
    return splits


# ---------------------------------------------------------------------------
# Public API: build unified DatasetDict
# ---------------------------------------------------------------------------

def load_all_datasets(
    vsfc_dir: str | Path,
    visfd_dir: str | Path,
    vlsp_ner_dir: str | Path,
) -> DatasetDict:
    """
    Load and merge the three datasets into a single ``DatasetDict`` with
    train / val / test splits.

    Parameters
    ----------
    vsfc_dir : path to the UIT-VSFC dataset directory
    visfd_dir : path to the UIT-ViSFD dataset directory
    vlsp_ner_dir : path to the VLSP 2016 NER dataset directory

    Returns
    -------
    DatasetDict with unified schema:
        text            : str
        task            : str ("sentiment" | "absa" | "ner")
        sentiment_label : int | None
        aspect_labels   : str (JSON-encoded dict) | None
        ner_tags        : str (JSON-encoded list) | None
    """
    vsfc_splits = _read_vsfc(vsfc_dir)
    visfd_splits = _read_visfd(visfd_dir)
    ner_splits = _read_vlsp_ner(vlsp_ner_dir)

    dataset_dict: Dict[str, Dataset] = {}
    for split in ("train", "val", "test"):
        merged: List[Dict[str, Any]] = []
        merged.extend(vsfc_splits.get(split, []))
        merged.extend(visfd_splits.get(split, []))
        merged.extend(ner_splits.get(split, []))

        # Serialise complex fields to JSON strings for Arrow compatibility
        for rec in merged:
            if rec["aspect_labels"] is not None:
                rec["aspect_labels"] = json.dumps(rec["aspect_labels"], ensure_ascii=False)
            else:
                rec["aspect_labels"] = None
            if rec["ner_tags"] is not None:
                rec["ner_tags"] = json.dumps(rec["ner_tags"], ensure_ascii=False)
            else:
                rec["ner_tags"] = None
            # Ensure sentiment_label is stored as int or -1 for None
            if rec["sentiment_label"] is None:
                rec["sentiment_label"] = -1

        features = Features({
            "text": Value("string"),
            "task": Value("string"),
            "sentiment_label": Value("int32"),
            "aspect_labels": Value("string"),
            "ner_tags": Value("string"),
        })
        dataset_dict[split] = Dataset.from_list(merged, features=features)

    return DatasetDict(dataset_dict)


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn(
    batch: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizerFast,
    max_length: int = MAX_LENGTH,
) -> Dict[str, Any]:
    """
    Collate a batch of unified records into tensors suitable for
    ``MultiTaskPhoBERT.forward()``.

    All samples in a single batch are expected to belong to the **same task**
    (the Trainer must group batches by task).

    Returns
    -------
    dict with keys:
        input_ids      : LongTensor  [B, L]
        attention_mask  : LongTensor  [B, L]
        task            : str
        labels          : Tensor      (shape depends on task)
    """
    texts: List[str] = [item["text"] for item in batch]
    task: str = batch[0]["task"]

    encoding = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    result: Dict[str, Any] = {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "task": task,
    }

    if task == "sentiment":
        labels = torch.tensor(
            [item["sentiment_label"] for item in batch], dtype=torch.long
        )
        result["labels"] = labels

    elif task == "absa":
        # Each sample → [num_aspects] tensor of polarity ints
        label_rows: List[List[int]] = []
        for item in batch:
            aspects_str = item["aspect_labels"]
            if aspects_str is None or aspects_str == "":
                aspects_dict: Dict[str, int] = {a: 1 for a in ABSA_ASPECTS}
            else:
                aspects_dict = json.loads(aspects_str) if isinstance(aspects_str, str) else aspects_str
            label_rows.append([aspects_dict.get(a, 1) for a in ABSA_ASPECTS])
        result["labels"] = torch.tensor(label_rows, dtype=torch.long)  # [B, 4]

    elif task == "ner":
        all_label_ids: List[List[int]] = []
        for idx, item in enumerate(batch):
            tags_str = item["ner_tags"]
            if tags_str is None or tags_str == "":
                tags: List[str] = []
            else:
                tags = json.loads(tags_str) if isinstance(tags_str, str) else item["ner_tags"]

            # Tokenise word-by-word to align tags with sub-word tokens
            word_tokens = texts[idx].split()
            label_ids: List[int] = []

            # BOS token
            label_ids.append(PAD_LABEL_ID)

            for word_idx, word in enumerate(word_tokens):
                word_enc = tokenizer.encode(word, add_special_tokens=False)
                tag = tags[word_idx] if word_idx < len(tags) else "O"
                tag_id = NER_LABEL2ID.get(tag, NER_LABEL2ID["O"])
                for sub_idx, _ in enumerate(word_enc):
                    if sub_idx == 0:
                        label_ids.append(tag_id)
                    else:
                        # Sub-word tokens after the first get PAD_LABEL_ID
                        label_ids.append(PAD_LABEL_ID)

            # EOS token
            label_ids.append(PAD_LABEL_ID)

            # Truncate or pad to max_length
            if len(label_ids) > max_length:
                label_ids = label_ids[:max_length]
            else:
                label_ids.extend([PAD_LABEL_ID] * (max_length - len(label_ids)))

            all_label_ids.append(label_ids)

        result["labels"] = torch.tensor(all_label_ids, dtype=torch.long)  # [B, L]

    return result


# ---------------------------------------------------------------------------
# Class-weight computation for imbalanced classes
# ---------------------------------------------------------------------------

def get_class_weights(
    dataset: Dataset,
    task: Literal["sentiment", "absa", "ner"],
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for a given task.

    Parameters
    ----------
    dataset : A HuggingFace ``Dataset`` (e.g. the train split).
    task    : One of ``"sentiment"``, ``"absa"``, ``"ner"``.

    Returns
    -------
    torch.Tensor of shape [num_classes] with normalised weights.
    """
    task_rows = dataset.filter(lambda x: x["task"] == task)

    if task == "sentiment":
        counter: Counter = Counter()
        for row in task_rows:
            lbl = row["sentiment_label"]
            if lbl >= 0:
                counter[lbl] += 1
        num_classes = SENTIMENT_LABELS
        counts = torch.tensor(
            [counter.get(c, 1) for c in range(num_classes)], dtype=torch.float
        )
        weights = counts.sum() / (num_classes * counts)
        return weights

    elif task == "absa":
        # Flatten across aspects → count each polarity
        counter = Counter()
        for row in task_rows:
            aspects_str = row["aspect_labels"]
            if aspects_str is None or aspects_str == "":
                continue
            aspects_dict: Dict[str, int] = (
                json.loads(aspects_str) if isinstance(aspects_str, str) else aspects_str
            )
            for pol in aspects_dict.values():
                counter[pol] += 1
        num_classes = len(ABSA_POLARITIES)
        counts = torch.tensor(
            [counter.get(c, 1) for c in range(num_classes)], dtype=torch.float
        )
        weights = counts.sum() / (num_classes * counts)
        return weights

    elif task == "ner":
        counter = Counter()
        for row in task_rows:
            tags_str = row["ner_tags"]
            if tags_str is None or tags_str == "":
                continue
            tags: List[str] = (
                json.loads(tags_str) if isinstance(tags_str, str) else tags_str
            )
            for t in tags:
                tag_id = NER_LABEL2ID.get(t, 0)
                counter[tag_id] += 1
        num_classes = len(NER_LABEL_LIST)
        counts = torch.tensor(
            [counter.get(c, 1) for c in range(num_classes)], dtype=torch.float
        )
        weights = counts.sum() / (num_classes * counts)
        return weights

    else:
        raise ValueError(f"Unknown task: {task}")


# ---------------------------------------------------------------------------
# Convenience: per-task DataLoader builder
# ---------------------------------------------------------------------------

def build_task_dataloader(
    dataset: Dataset,
    task: str,
    tokenizer: PreTrainedTokenizerFast,
    batch_size: int = 16,
    shuffle: bool = True,
    max_length: int = MAX_LENGTH,
) -> torch.utils.data.DataLoader:
    """
    Filter *dataset* to *task* rows and return a ``DataLoader`` with the
    project collate function.
    """
    from functools import partial

    task_ds = dataset.filter(lambda x: x["task"] == task)
    _collate = partial(collate_fn, tokenizer=tokenizer, max_length=max_length)
    return torch.utils.data.DataLoader(
        task_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collate,
        drop_last=False,
    )


# ---------------------------------------------------------------------------
# Quick sanity-check when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("data_loader.py — standalone sanity check")
    print(f"NER labels ({len(NER_LABEL_LIST)}): {NER_LABEL_LIST}")
    print(f"ABSA aspects: {ABSA_ASPECTS}")
    print(f"ABSA polarities: {ABSA_POLARITIES}")
    print(f"MAX_LENGTH: {MAX_LENGTH}")
    print("Usage: import data_loader; ds = data_loader.load_all_datasets(...)")
    sys.exit(0)

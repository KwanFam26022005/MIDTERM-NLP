"""
evaluate.py
===========
Load a trained MultiTaskPhoBERT checkpoint, run inference on all test sets,
and produce a comprehensive evaluation report.

Outputs
-------
1. ``results/eval_report.json``       — full structured metrics
2. ``results/confusion_matrices.png`` — 3-subplot figure (one per task)
3. Rich console table with baseline comparison

CLI
---
    python evaluate.py --ckpt checkpoints/best_ner.pt --data data/

Author : NLP Pipeline
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from seqeval.metrics import (
    classification_report as seqeval_classification_report,
    f1_score as seqeval_f1,
    precision_score as seqeval_precision,
    recall_score as seqeval_recall,
)
from rich.console import Console
from rich.table import Table

from data_loader import (
    ABSA_ASPECTS,
    ABSA_POLARITIES,
    NER_ID2LABEL,
    NER_LABEL2ID,
    NER_LABEL_LIST,
    PAD_LABEL_ID,
    build_task_dataloader,
    collate_fn,
    load_all_datasets,
)
from model import MultiTaskPhoBERT

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INFERENCE_BATCH_SIZE: int = 32
RESULTS_DIR: str = "results"
SENTIMENT_LABELS_MAP: Dict[int, str] = {0: "negative", 1: "neutral", 2: "positive"}
ABSA_POLARITY_MAP: Dict[int, str] = {0: "negative", 1: "neutral", 2: "positive"}


# ---------------------------------------------------------------------------
# Device auto-detection (mirrors trainer.py)
# ---------------------------------------------------------------------------

def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Model loading (exactly once)
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str, device: torch.device) -> MultiTaskPhoBERT:
    """Load a ``MultiTaskPhoBERT`` from a checkpoint file."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = MultiTaskPhoBERT()
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _run_inference_ner(
    model: MultiTaskPhoBERT,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[List[List[str]], List[List[str]], float]:
    """
    Run NER inference and return (predictions, references, latency_ms_per_sample).
    """
    all_preds: List[List[str]] = []
    all_labels: List[List[str]] = []
    total_time: float = 0.0
    total_samples: int = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        t0 = time.perf_counter()
        logits, _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            task="ner",
        )
        decoded = model.crf_decode(logits, attention_mask)
        t1 = time.perf_counter()
        total_time += (t1 - t0)
        total_samples += input_ids.size(0)

        for i, seq in enumerate(decoded):
            label_seq = labels[i].cpu().tolist()
            pred_tags: List[str] = []
            true_tags: List[str] = []
            for j, (pred_id, true_id) in enumerate(zip(seq, label_seq)):
                if true_id == PAD_LABEL_ID:
                    continue
                pred_tags.append(NER_ID2LABEL.get(pred_id, "O"))
                true_tags.append(NER_ID2LABEL.get(true_id, "O"))
            all_preds.append(pred_tags)
            all_labels.append(true_tags)

    latency = (total_time / max(total_samples, 1)) * 1000.0  # ms/sample
    return all_preds, all_labels, latency


@torch.no_grad()
def _run_inference_sentiment(
    model: MultiTaskPhoBERT,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[List[int], List[int], float]:
    all_preds: List[int] = []
    all_labels: List[int] = []
    total_time: float = 0.0
    total_samples: int = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]

        t0 = time.perf_counter()
        logits, _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            task="sentiment",
        )
        t1 = time.perf_counter()
        total_time += (t1 - t0)
        total_samples += input_ids.size(0)

        preds = logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())

    latency = (total_time / max(total_samples, 1)) * 1000.0
    return all_preds, all_labels, latency


@torch.no_grad()
def _run_inference_absa(
    model: MultiTaskPhoBERT,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[List[List[int]], List[List[int]], float]:
    all_preds: List[List[int]] = []
    all_labels: List[List[int]] = []
    total_time: float = 0.0
    total_samples: int = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]

        t0 = time.perf_counter()
        logits, _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            task="absa",
        )
        t1 = time.perf_counter()
        total_time += (t1 - t0)
        total_samples += input_ids.size(0)

        preds = logits.argmax(dim=-1).cpu().tolist()  # [B, A]
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())

    latency = (total_time / max(total_samples, 1)) * 1000.0
    return all_preds, all_labels, latency


# ---------------------------------------------------------------------------
# Baseline computations
# ---------------------------------------------------------------------------

def _majority_class_baseline(labels: List[int]) -> float:
    """Accuracy of a majority-class predictor."""
    counter = Counter(labels)
    majority = counter.most_common(1)[0][1]
    return majority / len(labels) if labels else 0.0


def _all_o_baseline_f1(labels: List[List[str]]) -> float:
    """Entity-level F1 of an all-O predictor for NER."""
    preds = [["O"] * len(seq) for seq in labels]
    return seqeval_f1(labels, preds, average="macro")


def _majority_polarity_baseline(labels: List[List[int]]) -> float:
    """Micro-F1 of a majority-polarity predictor for ABSA."""
    flat = [l for row in labels for l in row]
    counter = Counter(flat)
    majority = counter.most_common(1)[0][0]
    preds = [majority] * len(flat)
    return f1_score(flat, preds, average="micro")


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def evaluate_ner(
    preds: List[List[str]],
    labels: List[List[str]],
    latency: float,
    model_size: float,
) -> Dict[str, Any]:
    report_str = seqeval_classification_report(labels, preds, output_dict=False)
    report_dict = seqeval_classification_report(labels, preds, output_dict=True)

    # Per-entity-type metrics
    entity_types = ["PER", "ORG", "LOC", "MISC"]
    per_entity: Dict[str, Dict[str, float]] = {}
    for etype in entity_types:
        if etype in report_dict:
            per_entity[etype] = {
                "precision": round(report_dict[etype]["precision"], 4),
                "recall": round(report_dict[etype]["recall"], 4),
                "f1": round(report_dict[etype]["f1-score"], 4),
                "support": int(report_dict[etype]["support"]),
            }
        else:
            per_entity[etype] = {
                "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0
            }

    # Overall
    macro_f1 = seqeval_f1(labels, preds, average="macro")
    macro_precision = seqeval_precision(labels, preds, average="macro")
    macro_recall = seqeval_recall(labels, preds, average="macro")

    # Token-level confusion matrix for visualisation
    flat_preds = [t for seq in preds for t in seq]
    flat_labels = [t for seq in labels for t in seq]
    unique_tags = sorted(set(flat_preds + flat_labels))
    cm = confusion_matrix(flat_labels, flat_preds, labels=unique_tags)

    baseline_f1 = _all_o_baseline_f1(labels)

    return {
        "task": "ner",
        "macro_precision": round(float(macro_precision), 4),
        "macro_recall": round(float(macro_recall), 4),
        "macro_f1": round(float(macro_f1), 4),
        "per_entity": per_entity,
        "baseline_f1": round(baseline_f1, 4),
        "latency_ms_per_sample": round(latency, 2),
        "model_size_mb": round(model_size, 2),
        "report": report_str,
        "_cm": cm,
        "_cm_labels": unique_tags,
    }


def evaluate_sentiment(
    preds: List[int],
    labels: List[int],
    latency: float,
    model_size: float,
) -> Dict[str, Any]:
    macro_f1 = f1_score(labels, preds, average="macro")
    weighted_f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    report = classification_report(
        labels, preds,
        target_names=list(SENTIMENT_LABELS_MAP.values()),
        output_dict=True,
    )

    per_class: Dict[str, Dict[str, float]] = {}
    for idx, name in SENTIMENT_LABELS_MAP.items():
        if name in report:
            per_class[name] = {
                "precision": round(report[name]["precision"], 4),
                "recall": round(report[name]["recall"], 4),
                "f1": round(report[name]["f1-score"], 4),
                "support": int(report[name]["support"]),
            }

    cm = confusion_matrix(labels, preds)
    baseline_acc = _majority_class_baseline(labels)

    return {
        "task": "sentiment",
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "accuracy": round(float(acc), 4),
        "per_class": per_class,
        "baseline_accuracy": round(baseline_acc, 4),
        "latency_ms_per_sample": round(latency, 2),
        "model_size_mb": round(model_size, 2),
        "_cm": cm,
        "_cm_labels": list(SENTIMENT_LABELS_MAP.values()),
    }


def evaluate_absa(
    preds: List[List[int]],
    labels: List[List[int]],
    latency: float,
    model_size: float,
) -> Dict[str, Any]:
    # Per-aspect F1
    preds_arr = np.array(preds)   # [N, A]
    labels_arr = np.array(labels)

    per_aspect: Dict[str, Dict[str, float]] = {}
    for idx, aspect in enumerate(ABSA_ASPECTS):
        a_preds = preds_arr[:, idx].tolist()
        a_labels = labels_arr[:, idx].tolist()
        per_aspect[aspect] = {
            "f1_macro": round(float(f1_score(a_labels, a_preds, average="macro")), 4),
            "f1_micro": round(float(f1_score(a_labels, a_preds, average="micro")), 4),
            "accuracy": round(float(accuracy_score(a_labels, a_preds)), 4),
        }

    # Overall micro-F1
    flat_preds = preds_arr.flatten().tolist()
    flat_labels = labels_arr.flatten().tolist()
    overall_micro_f1 = f1_score(flat_labels, flat_preds, average="micro")
    overall_macro_f1 = f1_score(flat_labels, flat_preds, average="macro")

    cm = confusion_matrix(flat_labels, flat_preds)
    baseline_f1 = _majority_polarity_baseline(labels)

    return {
        "task": "absa",
        "overall_micro_f1": round(float(overall_micro_f1), 4),
        "overall_macro_f1": round(float(overall_macro_f1), 4),
        "per_aspect": per_aspect,
        "baseline_micro_f1": round(baseline_f1, 4),
        "latency_ms_per_sample": round(latency, 2),
        "model_size_mb": round(model_size, 2),
        "_cm": cm,
        "_cm_labels": list(ABSA_POLARITY_MAP.values()),
    }


# ---------------------------------------------------------------------------
# Confusion matrix plot
# ---------------------------------------------------------------------------

def plot_confusion_matrices(
    ner_result: Dict[str, Any],
    sent_result: Dict[str, Any],
    absa_result: Dict[str, Any],
    output_path: str,
) -> None:
    """Save a 3-subplot confusion matrix figure."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle(
        "Confusion Matrices — MultiTaskPhoBERT",
        fontsize=16, fontweight="bold", y=1.02,
    )

    cmap = LinearSegmentedColormap.from_list(
        "custom_blues", ["#f0f4ff", "#3b82f6", "#1e3a8a"]
    )

    datasets_info = [
        ("NER (token-level)", ner_result["_cm"], ner_result["_cm_labels"]),
        ("Sentiment", sent_result["_cm"], sent_result["_cm_labels"]),
        ("ABSA (all aspects)", absa_result["_cm"], absa_result["_cm_labels"]),
    ]

    for ax, (title, cm, labels) in zip(axes, datasets_info):
        im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)
        tick_marks = np.arange(len(labels))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(labels, fontsize=8)

        # Annotate cells
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, format(cm[i, j], "d"),
                    ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > thresh else "black",
                )

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluate] Confusion matrices saved → {output_path}")


# ---------------------------------------------------------------------------
# Rich console table
# ---------------------------------------------------------------------------

def print_rich_table(
    ner_result: Dict[str, Any],
    sent_result: Dict[str, Any],
    absa_result: Dict[str, Any],
) -> None:
    """Print a rich console table comparing model metrics vs baselines."""
    console = Console()
    table = Table(
        title="🏆 Evaluation Results — MultiTaskPhoBERT",
        show_header=True,
        header_style="bold cyan",
        border_style="bright_blue",
        title_style="bold magenta",
    )
    table.add_column("Task", style="bold white", width=14)
    table.add_column("Metric", style="white", width=20)
    table.add_column("Score", style="green", justify="right", width=10)
    table.add_column("vs Baseline", style="yellow", justify="right", width=14)

    def _flag(score: float) -> str:
        return f"{score:.4f}" if score >= 0.70 else f"⚠️ {score:.4f}"

    def _baseline_delta(score: float, baseline: float) -> str:
        delta = score - baseline
        sign = "+" if delta >= 0 else ""
        return f"{sign}{delta:.4f}"

    # NER
    ner_f1 = ner_result["macro_f1"]
    ner_bl = ner_result["baseline_f1"]
    table.add_row("NER", "Macro Precision", _flag(ner_result["macro_precision"]), "")
    table.add_row("NER", "Macro Recall", _flag(ner_result["macro_recall"]), "")
    table.add_row("NER", "Macro F1", _flag(ner_f1), _baseline_delta(ner_f1, ner_bl))
    table.add_row(
        "NER", "Latency (ms/sample)",
        f"{ner_result['latency_ms_per_sample']:.2f}", ""
    )
    for etype in ["PER", "ORG", "LOC", "MISC"]:
        if etype in ner_result["per_entity"]:
            e = ner_result["per_entity"][etype]
            table.add_row(
                f"  └─ {etype}",
                "F1",
                _flag(e["f1"]),
                f"(n={e['support']})",
            )
    table.add_section()

    # Sentiment
    sent_f1 = sent_result["macro_f1"]
    sent_bl = sent_result["baseline_accuracy"]
    table.add_row("Sentiment", "Macro F1", _flag(sent_f1), _baseline_delta(sent_f1, sent_bl))
    table.add_row("Sentiment", "Weighted F1", _flag(sent_result["weighted_f1"]), "")
    table.add_row("Sentiment", "Accuracy", _flag(sent_result["accuracy"]), "")
    table.add_row(
        "Sentiment", "Latency (ms/sample)",
        f"{sent_result['latency_ms_per_sample']:.2f}", ""
    )
    for cls_name, cls_info in sent_result.get("per_class", {}).items():
        table.add_row(
            f"  └─ {cls_name}",
            "F1",
            _flag(cls_info["f1"]),
            f"(n={cls_info['support']})",
        )
    table.add_section()

    # ABSA
    absa_micro_f1 = absa_result["overall_micro_f1"]
    absa_bl = absa_result["baseline_micro_f1"]
    table.add_row(
        "ABSA", "Overall Micro F1",
        _flag(absa_micro_f1), _baseline_delta(absa_micro_f1, absa_bl),
    )
    table.add_row(
        "ABSA", "Overall Macro F1",
        _flag(absa_result["overall_macro_f1"]), "",
    )
    table.add_row(
        "ABSA", "Latency (ms/sample)",
        f"{absa_result['latency_ms_per_sample']:.2f}", ""
    )
    for aspect, info in absa_result.get("per_aspect", {}).items():
        table.add_row(
            f"  └─ {aspect}",
            "F1 macro",
            _flag(info["f1_macro"]),
            f"acc={info['accuracy']:.4f}",
        )
    table.add_section()

    # Model info
    table.add_row(
        "Model",
        "Size (MB)",
        f"{ner_result['model_size_mb']:.1f}",
        "",
    )

    console.print(table)


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------

def main(ckpt_path: str, data_dir: str) -> None:
    """Load model once, evaluate all 3 tasks, save results."""
    device = _auto_device()
    # Use CPU for latency measurement
    cpu_device = torch.device("cpu")

    print(f"[evaluate] Device: {device}")
    print(f"[evaluate] Loading checkpoint: {ckpt_path}")

    # ---- Load model ONCE ----
    model = load_model(ckpt_path, device)
    model_size = model.model_size_mb()

    # ---- Load tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(model.model_name)

    # ---- Load datasets ----
    vsfc_dir = os.path.join(data_dir, "UIT-VSFC")
    visfd_dir = os.path.join(data_dir, "UIT-ViSFD")
    vlsp_dir = os.path.join(data_dir, "VLSP-NER")

    dataset = load_all_datasets(vsfc_dir, visfd_dir, vlsp_dir)
    test_set = dataset["test"]

    # Build test dataloaders
    ner_loader = build_task_dataloader(
        test_set, "ner", tokenizer,
        batch_size=INFERENCE_BATCH_SIZE, shuffle=False,
    )
    sent_loader = build_task_dataloader(
        test_set, "sentiment", tokenizer,
        batch_size=INFERENCE_BATCH_SIZE, shuffle=False,
    )
    absa_loader = build_task_dataloader(
        test_set, "absa", tokenizer,
        batch_size=INFERENCE_BATCH_SIZE, shuffle=False,
    )

    # ---- Run inference with torch.no_grad() ----
    print("[evaluate] Running NER inference...")
    ner_preds, ner_labels, ner_latency = _run_inference_ner(model, ner_loader, device)

    print("[evaluate] Running Sentiment inference...")
    sent_preds, sent_labels, sent_latency = _run_inference_sentiment(
        model, sent_loader, device
    )

    print("[evaluate] Running ABSA inference...")
    absa_preds, absa_labels, absa_latency = _run_inference_absa(
        model, absa_loader, device
    )

    # ---- Compute metrics ----
    ner_result = evaluate_ner(ner_preds, ner_labels, ner_latency, model_size)
    sent_result = evaluate_sentiment(
        sent_preds, sent_labels, sent_latency, model_size
    )
    absa_result = evaluate_absa(
        absa_preds, absa_labels, absa_latency, model_size
    )

    # ---- Save eval_report.json ----
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report = {
        "ner": {k: v for k, v in ner_result.items() if not k.startswith("_")},
        "sentiment": {
            k: v for k, v in sent_result.items() if not k.startswith("_")
        },
        "absa": {
            k: v for k, v in absa_result.items() if not k.startswith("_")
        },
    }
    report_path = os.path.join(RESULTS_DIR, "eval_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[evaluate] Report saved → {report_path}")

    # ---- Plot confusion matrices ----
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrices.png")
    plot_confusion_matrices(ner_result, sent_result, absa_result, cm_path)

    # ---- Rich console table ----
    print_rich_table(ner_result, sent_result, absa_result)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate MultiTaskPhoBERT on all test sets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to a saved model checkpoint (.pt file).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help=(
            "Root data directory containing sub-dirs: "
            "UIT-VSFC/, UIT-ViSFD/, VLSP-NER/"
        ),
    )
    args = parser.parse_args()
    main(args.ckpt, args.data_dir)

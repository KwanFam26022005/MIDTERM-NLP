"""
demo_cau1.py
============
Gradio Demo UI for MultiTaskPhoBERT — NER, Sentiment, ABSA.

Run
---
    python demo_cau1.py
    # or on Colab:
    # demo.launch(share=True)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import gradio as gr
from transformers import AutoTokenizer

from data_loader import (
    NER_ID2LABEL,
    NER_LABEL2ID,
    ABSA_ASPECTS,
    ABSA_POLARITIES,
    MAX_LENGTH,
    PAD_LABEL_ID,
)
from model import MultiTaskPhoBERT

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "vinai/phobert-base"
BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
# CHECKPOINT_DIR = Path("checkpoints")
_model: Optional[MultiTaskPhoBERT] = None
_tokenizer = None

SENTIMENT_LABELS = {0: "Tiêu cực (Negative)", 1: "Trung lập (Neutral)", 2: "Tích cực (Positive)"}
SENTIMENT_COLORS = {0: "🔴", 1: "🟡", 2: "🟢"}
POLARITY_LABELS = {0: "Negative", 1: "Neutral", 2: "Positive"}
POLARITY_EMOJI = {0: "🔴", 1: "🟡", 2: "🟢"}

NER_COLORS = {
    "PER": "#ff6b6b",
    "ORG": "#4ecdc4",
    "LOC": "#45b7d1",
    "MISC": "#f7dc6f",
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _load_model() -> MultiTaskPhoBERT:
    """Load model from the best available checkpoint."""
    global _model, _tokenizer
    if _model is not None:
        return _model

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    _model = MultiTaskPhoBERT(model_name=MODEL_NAME)

    # Try to load any available checkpoint
    loaded = False
    for tag in ["ner", "sentiment", "absa"]:
        ckpt_path = CHECKPOINT_DIR / f"best_{tag}.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            _model.load_state_dict(ckpt["model_state_dict"])
            print(f"[demo] Loaded checkpoint: {ckpt_path}")
            loaded = True
            break

    if not loaded:
        print("[demo] ⚠️ Không tìm thấy checkpoint. Model sẽ dùng trọng số ngẫu nhiên (chưa huấn luyện).")

    _model.to(DEVICE)
    _model.eval()
    return _model


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return _tokenizer


# ---------------------------------------------------------------------------
# Inference functions
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_ner(text: str) -> str:
    """Run NER inference and return highlighted HTML."""
    model = _load_model()
    tokenizer = _get_tokenizer()

    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    logits, _ = model(input_ids=input_ids, attention_mask=attention_mask, task="ner")
    decoded = model.crf_decode(logits, attention_mask)
    pred_ids = decoded[0]

    # Align predictions back to words
    words = text.split()
    word_tags: List[Tuple[str, str]] = []

    # Reconstruct word-level tags from subword predictions
    token_idx = 1  # skip BOS
    for word in words:
        word_enc = tokenizer.encode(word, add_special_tokens=False)
        if token_idx < len(pred_ids):
            tag_id = pred_ids[token_idx]
            tag = NER_ID2LABEL.get(tag_id, "O")
        else:
            tag = "O"
        word_tags.append((word, tag))
        token_idx += len(word_enc)

    # Build highlighted HTML output
    html_parts = []
    for word, tag in word_tags:
        if tag != "O":
            entity_type = tag.split("-")[-1] if "-" in tag else tag
            color = NER_COLORS.get(entity_type, "#ccc")
            html_parts.append(
                f'<span style="background-color:{color}; color:white; '
                f'padding:2px 6px; border-radius:4px; margin:1px; '
                f'display:inline-block; font-weight:bold;">'
                f'{word} <sup style="font-size:10px;">{tag}</sup></span>'
            )
        else:
            html_parts.append(f'<span style="margin:1px; display:inline-block; color:#eee !important;">{word}</span>')

    # Build legend
    legend = " | ".join(
        f'<span style="background-color:{c}; color:white !important; padding:1px 6px; '
        f'border-radius:3px; font-size:12px;">{e}</span>'
        for e, c in NER_COLORS.items()
    )

    return (
        f'<div class="nlp-result-box" style="line-height:2.2; font-size:16px;">'
        f'{" ".join(html_parts)}'
        f'</div>'
        f'<div class="nlp-legend" style="margin-top:8px; font-size:13px;">Legend: {legend}</div>'
    )


@torch.no_grad()
def predict_sentiment(text: str) -> str:
    """Run Sentiment inference and return formatted result."""
    model = _load_model()
    tokenizer = _get_tokenizer()

    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    logits, _ = model(input_ids=input_ids, attention_mask=attention_mask, task="sentiment")
    probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
    pred = logits.argmax(dim=-1).item()

    # Build result HTML
    bars = []
    for i in range(3):
        pct = probs[i] * 100
        emoji = SENTIMENT_COLORS[i]
        label = SENTIMENT_LABELS[i]
        is_pred = "★" if i == pred else ""
        bar_color = "#ff6b6b" if i == 0 else "#f7dc6f" if i == 1 else "#2ecc71"
        bars.append(
            f'<div style="margin:6px 0;">'
            f'  <span style="display:inline-block; width:200px; color:#eee !important;">{emoji} {label} {is_pred}</span>'
            f'  <div style="display:inline-block; width:300px; background:#333; '
            f'border-radius:4px; overflow:hidden; vertical-align:middle;">'
            f'    <div style="width:{pct:.1f}%; background:{bar_color}; '
            f'height:22px; border-radius:4px;"></div>'
            f'  </div>'
            f'  <span style="margin-left:8px; font-weight:bold; color:#eee !important;">{pct:.1f}%</span>'
            f'</div>'
        )

    result_label = SENTIMENT_LABELS[pred]
    result_emoji = SENTIMENT_COLORS[pred]

    return (
        f'<div class="nlp-result-box">'
        f'  <h3 style="margin:0 0 12px 0;">Kết quả: {result_emoji} {result_label}</h3>'
        f'  {"".join(bars)}'
        f'</div>'
    )


@torch.no_grad()
def predict_absa(text: str) -> str:
    """Run ABSA inference and return formatted result."""
    model = _load_model()
    tokenizer = _get_tokenizer()

    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    logits, _ = model(input_ids=input_ids, attention_mask=attention_mask, task="absa")
    # logits: [1, 4, 3]
    probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()  # [4, 3]
    preds = logits.argmax(dim=-1)[0].cpu().tolist()  # [4]

    rows = []
    for i, aspect in enumerate(ABSA_ASPECTS):
        pred_pol = preds[i]
        pol_label = POLARITY_LABELS[pred_pol]
        pol_emoji = POLARITY_EMOJI[pred_pol]

        prob_cells = []
        for j in range(3):
            pct = probs[i][j] * 100
            bold = "font-weight:bold;" if j == pred_pol else ""
            prob_cells.append(
                f'<td style="padding:8px; text-align:center; color:#eee !important; {bold}">{pct:.1f}%</td>'
            )

        rows.append(
            f'<tr style="border-bottom:1px solid #333;">'
            f'  <td class="aspect-name" style="padding:8px; font-weight:bold;">{aspect}</td>'
            f'  <td style="padding:8px; text-align:center;">{pol_emoji} {pol_label}</td>'
            f'  {"".join(prob_cells)}'
            f'</tr>'
        )

    return (
        f'<div class="nlp-result-box">'
        f'  <h3 style="margin:0 0 12px 0;">Aspect-Based Sentiment Analysis</h3>'
        f'  <table style="width:100%; border-collapse:collapse;">'
        f'    <tr style="background:#16213e;">'
        f'      <th style="padding:8px; text-align:left;">Aspect</th>'
        f'      <th style="padding:8px; text-align:center;">Prediction</th>'
        f'      <th style="padding:8px; text-align:center;">🔴 Neg</th>'
        f'      <th style="padding:8px; text-align:center;">🟡 Neu</th>'
        f'      <th style="padding:8px; text-align:center;">🟢 Pos</th>'
        f'    </tr>'
        f'    {"".join(rows)}'
        f'  </table>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Unified handler
# ---------------------------------------------------------------------------
def run_inference(text: str, task: str) -> str:
    """Route to the correct inference function based on task."""
    if not text.strip():
        return '<div style="color:#ff6b6b; padding:12px;">⚠️ Vui lòng nhập câu tiếng Việt.</div>'

    if task == "NER (Named Entity Recognition)":
        return predict_ner(text)
    elif task == "Sentiment Analysis":
        return predict_sentiment(text)
    elif task == "ABSA (Aspect-Based Sentiment)":
        return predict_absa(text)
    else:
        return '<div style="color:#ff6b6b;">Task không hợp lệ.</div>'


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
EXAMPLES = [
    ["Hôm nay tôi mua điện thoại iPhone 15 tại Thế Giới Di Động ở Hà Nội, xài rất mượt!", "Sentiment Analysis"],
    ["Nguyễn Văn A làm việc tại công ty FPT ở Đà Nẵng", "NER (Named Entity Recognition)"],
    ["Điện thoại này màn hình đẹp nhưng pin yếu, giá hơi cao", "ABSA (Aspect-Based Sentiment)"],
    ["Việt Nam vô địch AFF Cup tại sân Mỹ Đình", "NER (Named Entity Recognition)"],
    ["Máy hay bị giật lag, thiết kế xấu nhưng camera chụp khá ổn", "ABSA (Aspect-Based Sentiment)"],
]

CSS = """
.gradio-container {
    max-width: 900px !important;
    margin: auto !important;
}
.nlp-result-box {
    background: #1a1a2e !important;
    color: white !important;
    padding: 16px;
    border-radius: 8px;
}
.nlp-result-box span,
.nlp-result-box div,
.nlp-result-box h3,
.nlp-result-box table,
.nlp-result-box th,
.nlp-result-box td {
    color: white !important;
}
.nlp-result-box .aspect-name {
    color: #4ecdc4 !important;
}
.nlp-legend {
    color: #888 !important;
}
"""

demo = gr.Interface(
    fn=run_inference,
    inputs=[
        gr.Textbox(
            label="📝 Nhập câu tiếng Việt",
            placeholder="Ví dụ: Nguyễn Văn A làm việc tại FPT ở Đà Nẵng",
            lines=3,
        ),
        gr.Radio(
            choices=[
                "NER (Named Entity Recognition)",
                "Sentiment Analysis",
                "ABSA (Aspect-Based Sentiment)",
            ],
            label="🎯 Chọn Task",
            value="Sentiment Analysis",
        ),
    ],
    outputs=gr.HTML(label="📊 Kết quả"),
    title="🤖 MultiTaskPhoBERT — Demo Câu 1",
    description=(
        "**Multi-Task NLP Pipeline cho Tiếng Việt** sử dụng PhoBERT backbone.\n\n"
        "- **NER**: Nhận diện thực thể (PER, ORG, LOC, MISC)\n"
        "- **Sentiment**: Phân tích cảm xúc (Positive / Neutral / Negative)\n"
        "- **ABSA**: Phân tích cảm xúc theo khía cạnh (SCREEN, CAMERA, BATTERY, PRICE, PERFORMANCE...)"
    ),
    examples=EXAMPLES,
    css=CSS,
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="cyan",
    ),
    flagging_mode="never",
)


if __name__ == "__main__":
    demo.launch(share=False)

"""
demo_cau2.py
============
Gradio Demo UI for LSTM Language Model — Text Generation & Diacritic Restoration.

Run
---
    python demo_cau2.py
    # or on Colab:
    # demo.launch(share=True)
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import gradio as gr
import sentencepiece as spm

from models import MODEL_REGISTRY, StackedLSTM, _device

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
DEVICE = _device()
BASE_DIR = Path(__file__).resolve().parent

import sys
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

SP_MODEL_PATH = BASE_DIR / "tokenizer/vi_bpe.model"
CKPT_DIR = BASE_DIR / "checkpoints"

_sp: Optional[spm.SentencePieceProcessor] = None
_models: Dict[str, torch.nn.Module] = {}
_diacritics_model = None
_syl_dict: Optional[Dict[str, List[str]]] = None


# ---------------------------------------------------------------------------
# Lazy loaders
# ---------------------------------------------------------------------------
def _load_sp() -> spm.SentencePieceProcessor:
    global _sp
    if _sp is not None:
        return _sp
    if not SP_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Tokenizer không tìm thấy tại {SP_MODEL_PATH}. "
            "Hãy chạy train_tokenizer.py trước."
        )
    _sp = spm.SentencePieceProcessor()
    _sp.load(str(SP_MODEL_PATH))
    print(f"[demo] Loaded SentencePiece tokenizer: vocab_size={_sp.get_piece_size()}")
    return _sp


def _load_lm(model_name: str) -> torch.nn.Module:
    """Load a language model from checkpoint."""
    global _models
    if model_name in _models:
        return _models[model_name]

    sp = _load_sp()
    vocab_size = sp.get_piece_size()

    ModelClass = MODEL_REGISTRY[model_name]
    model = ModelClass(vocab_size=vocab_size)

    ckpt_path = CKPT_DIR / f"{model_name}_best.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[demo] Loaded checkpoint: {ckpt_path}")
    else:
        print(f"[demo] ⚠️ Không tìm thấy checkpoint {ckpt_path}. Dùng trọng số ngẫu nhiên.")

    model.to(DEVICE)
    model.eval()
    _models[model_name] = model
    return model


def _load_diacritics_model():
    """Load the diacritic restoration model."""
    global _diacritics_model, _syl_dict
    if _diacritics_model is not None:
        return _diacritics_model, _syl_dict

    from diacritic_restore import DiaCorrectionModel, load_syllable_dict

    sp = _load_sp()
    vocab_size = sp.get_piece_size()
    _syl_dict = load_syllable_dict()
    max_cands = max(len(v) for v in _syl_dict.values())

    backbone = StackedLSTM(vocab_size=vocab_size)

    # Try loading pretrained backbone first
    backbone_ckpt = CKPT_DIR / "stacked_best.pt"
    if backbone_ckpt.exists():
        ckpt = torch.load(backbone_ckpt, map_location=DEVICE, weights_only=False)
        backbone.load_state_dict(ckpt["model_state_dict"])
        print(f"[demo] Loaded StackedLSTM backbone: {backbone_ckpt}")

    _diacritics_model = DiaCorrectionModel(
        backbone, max_candidates=max_cands, freeze_bottom=2
    )

    dia_ckpt = CKPT_DIR / "diacritics_best.pt"
    if dia_ckpt.exists():
        dia_state = torch.load(dia_ckpt, map_location=DEVICE, weights_only=False)
        if "model_state_dict" in dia_state:
            _diacritics_model.load_state_dict(dia_state["model_state_dict"])
        else:
            _diacritics_model.load_state_dict(dia_state)
        print(f"[demo] Loaded diacritics checkpoint: {dia_ckpt}")
    else:
        print("[demo] ⚠️ Không tìm thấy diacritics checkpoint. Dùng trọng số ngẫu nhiên.")

    _diacritics_model.to(DEVICE)
    _diacritics_model.eval()
    return _diacritics_model, _syl_dict


# ---------------------------------------------------------------------------
# Text Generation
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_text(
    prompt: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
    top_k: int,
) -> str:
    """Generate text continuation from a prompt."""
    if not prompt.strip():
        return "⚠️ Vui lòng nhập prompt."

    try:
        sp = _load_sp()
        model = _load_lm(model_name)
    except FileNotFoundError as e:
        return f"❌ Lỗi: {e}"

    input_ids = sp.encode(prompt)
    generated = list(input_ids)

    hidden = None
    # Feed prompt through model
    x = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
    logits, hidden = model(x, hidden)

    # Generate token by token
    for _ in range(int(max_tokens)):
        # Use last token's logits
        next_logits = logits[0, -1, :] / max(temperature, 0.01)

        # Top-k filtering
        if top_k > 0:
            top_k_val = min(int(top_k), next_logits.size(-1))
            indices_to_remove = next_logits < torch.topk(next_logits, top_k_val).values[-1]
            next_logits[indices_to_remove] = float("-inf")

        probs = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, 1).item()

        if next_id == sp.eos_id():
            break

        generated.append(next_id)

        # Feed the new token
        x = torch.tensor([[next_id]], dtype=torch.long, device=DEVICE)
        logits, hidden = model(x, hidden)

    # Decode
    result = sp.decode(generated)

    # Format output: highlight generated part
    prompt_decoded = sp.decode(input_ids)

    return (
        f'<div class="nlp-result-box">'
        f'<span style="color:#888;">🔤 Prompt:</span><br>'
        f'<span style="color:#4ecdc4; font-weight:bold;">{prompt_decoded}</span>'
        f'<span style="color:#f7dc6f;">{result[len(prompt_decoded):]}</span>'
        f'<br><br>'
        f'<span style="font-size:12px; color:#888;">'
        f'Model: {model_name} | Tokens generated: {len(generated) - len(input_ids)} | '
        f'Temp: {temperature} | Top-k: {top_k}</span>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Diacritic Restoration
# ---------------------------------------------------------------------------
@torch.no_grad()
def restore_diacritics(text: str) -> str:
    """Restore Vietnamese diacritics from undiacritized text."""
    if not text.strip():
        return "⚠️ Vui lòng nhập câu không dấu."

    try:
        sp = _load_sp()
        model, syl_dict = _load_diacritics_model()
    except FileNotFoundError as e:
        return f"❌ Lỗi: {e}"

    restored = model.restore(text, syl_dict, sp)

    # Build comparison HTML
    orig_words = text.lower().split()
    rest_words = restored.split()

    diff_parts = []
    for i, (o, r) in enumerate(zip(orig_words, rest_words)):
        if o != r:
            diff_parts.append(
                f'<span style="background:#2ecc71; color:white; padding:2px 6px; '
                f'border-radius:4px; margin:1px; display:inline-block; '
                f'font-weight:bold;">{r}</span>'
            )
        else:
            diff_parts.append(
                f'<span style="margin:1px; display:inline-block;">{r}</span>'
            )
    # Handle extra words
    for r in rest_words[len(orig_words):]:
        diff_parts.append(f'<span style="margin:1px; display:inline-block;">{r}</span>')

    changes = sum(1 for o, r in zip(orig_words, rest_words) if o != r)

    return (
        f'<div class="nlp-result-box">'
        f'  <div style="margin-bottom:12px;">'
        f'    <span style="color:#888; font-size:13px;">📥 Input (không dấu):</span><br>'
        f'    <span style="font-size:16px; color:#aaa;">{text}</span>'
        f'  </div>'
        f'  <div style="margin-bottom:12px;">'
        f'    <span style="color:#888; font-size:13px;">📤 Output (có dấu):</span><br>'
        f'    <div style="font-size:16px; line-height:2.2;">{" ".join(diff_parts)}</div>'
        f'  </div>'
        f'  <div style="font-size:12px; color:#888;">'
        f'    Từ được phục hồi dấu: <span style="color:#2ecc71; font-weight:bold;">'
        f'{changes}/{len(orig_words)}</span>'
        f'  </div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Gradio UI — Tabbed Interface
# ---------------------------------------------------------------------------
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
    line-height: 1.8;
    font-size: 15px;
}
.nlp-result-box span,
.nlp-result-box div {
    color: white !important;
}
"""

with gr.Blocks(
    title="🧠 LSTM Language Model — Demo Câu 2",
    css=CSS,
    theme=gr.themes.Soft(primary_hue="teal", secondary_hue="cyan"),
) as demo:
    gr.Markdown(
        "# 🧠 LSTM Language Model — Demo Câu 2\n"
        "**Vietnamese Language Modelling & Diacritic Restoration** "
        "sử dụng LSTM (Vanilla / Stacked / BiLSTM+Attention)."
    )

    with gr.Tabs():
        # ---- Tab 1: Text Generation ----
        with gr.TabItem("📝 Text Generation"):
            gr.Markdown(
                "Nhập một đoạn prompt tiếng Việt, model sẽ sinh văn bản tiếp theo."
            )
            with gr.Row():
                with gr.Column(scale=2):
                    gen_input = gr.Textbox(
                        label="Prompt",
                        placeholder="Ví dụ: Việt Nam là một quốc gia",
                        lines=3,
                    )
                    gen_model = gr.Radio(
                        choices=list(MODEL_REGISTRY.keys()),
                        label="Chọn Model",
                        value="stacked",
                    )
                with gr.Column(scale=1):
                    gen_max_tokens = gr.Slider(
                        minimum=10, maximum=200, value=50, step=10,
                        label="Max Tokens",
                    )
                    gen_temp = gr.Slider(
                        minimum=0.1, maximum=2.0, value=0.8, step=0.1,
                        label="Temperature",
                    )
                    gen_topk = gr.Slider(
                        minimum=0, maximum=100, value=40, step=5,
                        label="Top-k (0 = off)",
                    )
            gen_btn = gr.Button("🚀 Generate", variant="primary")
            gen_output = gr.HTML(label="Kết quả")

            gen_btn.click(
                fn=generate_text,
                inputs=[gen_input, gen_model, gen_max_tokens, gen_temp, gen_topk],
                outputs=gen_output,
            )

            gr.Examples(
                examples=[
                    ["Việt Nam là một quốc gia", "stacked", 50, 0.8, 40],
                    ["Hà Nội là thủ đô", "vanilla", 30, 0.7, 50],
                    ["Ngày xưa có một", "bilstm_attn", 60, 0.9, 30],
                ],
                inputs=[gen_input, gen_model, gen_max_tokens, gen_temp, gen_topk],
            )

        # ---- Tab 2: Diacritic Restoration ----
        with gr.TabItem("🔤 Diacritic Restoration"):
            gr.Markdown(
                "Nhập câu tiếng Việt **không dấu**, model sẽ phục hồi dấu tự động.\n\n"
                "Sử dụng StackedLSTM pretrained làm backbone + classification head."
            )
            dia_input = gr.Textbox(
                label="Câu không dấu",
                placeholder="Ví dụ: viet nam la mot quoc gia dep",
                lines=3,
            )
            dia_btn = gr.Button("✨ Phục hồi dấu", variant="primary")
            dia_output = gr.HTML(label="Kết quả")

            dia_btn.click(
                fn=restore_diacritics,
                inputs=dia_input,
                outputs=dia_output,
            )

            gr.Examples(
                examples=[
                    "viet nam la mot quoc gia dep",
                    "hom nay troi dep qua",
                    "toi di hoc o truong dai hoc ton duc thang",
                    "cam on ban rat nhieu",
                    "chuc mung nam moi",
                ],
                inputs=dia_input,
            )


if __name__ == "__main__":
    demo.launch(share=False)

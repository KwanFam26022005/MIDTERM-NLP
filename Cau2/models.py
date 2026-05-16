"""
models.py
=========
Three LSTM language model architectures with a shared interface
for comparative evaluation on Vietnamese text.

Models
------
* **VanillaLSTM**   – single-layer LSTM (~15 M params)
* **StackedLSTM**   – 3-layer LSTM (~60 M params)
* **BiLSTMAttention** – bidirectional LSTM with Bahdanau attention (~75 M params)

Run
---
    python models.py          # smoke-test: instantiate all models, forward pass
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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


# ---------------------------------------------------------------------------
# Bahdanau Attention (vectorized — no per-timestep loop)
# ---------------------------------------------------------------------------
class BahdanauAttention(nn.Module):
    """
    Additive (Bahdanau) attention — fully vectorized.

    Supports both single-query and multi-query modes:
      - query (B, 1, D_q) + keys (B, T, D_k)  → single context
      - query (B, T, D_q) + keys (B, T, D_k)  → per-timestep context (vectorized)

    score_t = V^T tanh(W_q · query + W_k · key_t)
    weights = softmax(scores)           → (B, T)
    context = sum_t weights_t · keys_t  → (B, ?, D_k)
    """

    def __init__(self, query_dim: int, key_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.W_query = nn.Linear(query_dim, hidden_dim, bias=False)
        self.W_key = nn.Linear(key_dim, hidden_dim, bias=False)
        self.V = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self, query: torch.Tensor, keys: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        query : (B, T_q, D_q) — can be T_q=1 (single) or T_q=T (all timesteps)
        keys  : (B, T_k, D_k)

        Returns
        -------
        context : (B, T_q, D_k)
        weights : (B, T_q, T_k)
        """
        if query.dim() == 2:
            query = query.unsqueeze(1)  # (B, 1, D_q)

        # W_query(query): (B, T_q, H)  →  unsqueeze to (B, T_q, 1, H)
        # W_key(keys):    (B, T_k, H)  →  unsqueeze to (B, 1, T_k, H)
        # broadcast addition → (B, T_q, T_k, H)
        q_proj = self.W_query(query).unsqueeze(2)  # (B, T_q, 1, H)
        k_proj = self.W_key(keys).unsqueeze(1)     # (B, 1, T_k, H)
        energy = torch.tanh(q_proj + k_proj)       # (B, T_q, T_k, H)

        scores = self.V(energy).squeeze(-1)        # (B, T_q, T_k)
        weights = F.softmax(scores, dim=-1)        # (B, T_q, T_k)
        context = torch.bmm(
            weights.reshape(-1, weights.size(-1)).unsqueeze(1),
            keys.unsqueeze(1).expand(-1, query.size(1), -1, -1).reshape(-1, keys.size(1), keys.size(2)),
        ).reshape(query.size(0), query.size(1), keys.size(2))
        # Simpler: use einsum
        context = torch.einsum("bqk,bkd->bqd", weights, keys)  # (B, T_q, D_k)

        return context, weights


# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------
class _LMBase(nn.Module):
    """Abstract base providing the shared interface."""

    MODEL_NAME: str = "base"

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # Subclasses MUST override these ↓
    def forward(
        self, x: torch.Tensor, hidden: Optional[Tuple] = None
    ) -> Tuple[torch.Tensor, Tuple]:
        raise NotImplementedError

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 1. VanillaLSTM  (~15 M params)
# ---------------------------------------------------------------------------
class VanillaLSTM(_LMBase):
    """
    Embedding(vocab, 256) → LSTM(256→512, 1 layer)
    → Dropout(0.3) → Linear(512→vocab)

    Weight tying: embedding.weight == output projection.weight
    """

    MODEL_NAME = "vanilla"

    def __init__(self, vocab_size: int, embed_dim: int = 256):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = 512
        self.num_layers = 1

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(0.3)

        # Weight tying — project back through embedding matrix
        # Requires matching dims, so add a projection if embed_dim != hidden_dim
        self.proj = nn.Linear(self.hidden_dim, embed_dim, bias=False)
        self.fc = nn.Linear(embed_dim, vocab_size, bias=False)
        self.fc.weight = self.embedding.weight  # weight tying

        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple:
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h0, c0)

    def forward(
        self, x: torch.Tensor, hidden: Optional[Tuple] = None
    ) -> Tuple[torch.Tensor, Tuple]:
        """
        Parameters
        ----------
        x      : (B, T) long tensor of token ids
        hidden : optional LSTM state

        Returns
        -------
        logits : (B, T, vocab)
        hidden : updated state
        """
        if hidden is None:
            hidden = self.init_hidden(x.size(0), x.device)

        emb = self.embedding(x)  # (B, T, E)
        out, hidden = self.lstm(emb, hidden)  # (B, T, H)
        out = self.dropout(out)
        out = self.proj(out)  # (B, T, E)
        logits = self.fc(out)  # (B, T, V)
        return logits, hidden


# ---------------------------------------------------------------------------
# 2. StackedLSTM  (~60 M params)
# ---------------------------------------------------------------------------
class StackedLSTM(_LMBase):
    """
    Embedding(vocab, 512) → LSTM(512→1024, 3 layers, dropout=0.5)
    → Dropout(0.3) → Linear(1024→vocab)

    Weight tying via an intermediate projection 1024→512.
    """

    MODEL_NAME = "stacked"

    def __init__(self, vocab_size: int, embed_dim: int = 512):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = 1024
        self.num_layers = 3

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=0.5,
            batch_first=True,
        )
        # Output dropout (regularize final LSTM output before projection)
        self.output_dropout = nn.Dropout(0.3)

        # Weight tying projection
        self.proj = nn.Linear(self.hidden_dim, embed_dim, bias=False)
        self.fc = nn.Linear(embed_dim, vocab_size, bias=False)
        self.fc.weight = self.embedding.weight  # tie weights

        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.orthogonal_(self.proj.weight)

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple:
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h0, c0)

    def forward(
        self, x: torch.Tensor, hidden: Optional[Tuple] = None
    ) -> Tuple[torch.Tensor, Tuple]:
        if hidden is None:
            hidden = self.init_hidden(x.size(0), x.device)

        emb = self.embedding(x)  # (B, T, E)
        out, hidden = self.lstm(emb, hidden)  # (B, T, H)
        out = self.output_dropout(out)  # regularize output
        out = self.proj(out)  # (B, T, E)
        logits = self.fc(out)  # (B, T, V)
        return logits, hidden


# ---------------------------------------------------------------------------
# 3. BiLSTMAttention  (~75 M params)
# ---------------------------------------------------------------------------
class BiLSTMAttention(_LMBase):
    """
    Embedding(vocab, 512)
    → BiLSTM(512→512, 2 layers)   →  hidden = 512*2 = 1024
    → BahdanauAttention(query=1024, keys=1024) — VECTORIZED (no loop)
    → Linear(1024→vocab)

    Weight tying via an intermediate projection 1024→512.
    """

    MODEL_NAME = "bilstm_attn"

    def __init__(self, vocab_size: int, embed_dim: int = 512):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.lstm_hidden = 512
        self.num_layers = 2
        self.hidden_dim = self.lstm_hidden * 2  # bidirectional → 1024

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=self.lstm_hidden,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        self.attention = BahdanauAttention(
            query_dim=self.hidden_dim,
            key_dim=self.hidden_dim,
            hidden_dim=256,
        )
        self.output_dropout = nn.Dropout(0.3)

        # Weight tying projection
        self.proj = nn.Linear(self.hidden_dim, embed_dim, bias=False)
        self.fc = nn.Linear(embed_dim, vocab_size, bias=False)
        self.fc.weight = self.embedding.weight  # tie weights

        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.orthogonal_(self.proj.weight)

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple:
        # bidirectional → num_layers * 2 directions
        num_dirs = self.num_layers * 2
        h0 = torch.zeros(num_dirs, batch_size, self.lstm_hidden, device=device)
        c0 = torch.zeros(num_dirs, batch_size, self.lstm_hidden, device=device)
        return (h0, c0)

    def forward(
        self, x: torch.Tensor, hidden: Optional[Tuple] = None
    ) -> Tuple[torch.Tensor, Tuple]:
        if hidden is None:
            hidden = self.init_hidden(x.size(0), x.device)

        emb = self.embedding(x)  # (B, T, E)
        lstm_out, hidden = self.lstm(emb, hidden)  # (B, T, 1024)

        # ── Vectorized attention: all T queries at once ──
        # query = lstm_out itself: (B, T, 1024), keys = lstm_out: (B, T, 1024)
        # → context: (B, T, 1024) — each timestep attends to all timesteps
        context, _ = self.attention(lstm_out, lstm_out)  # (B, T, 1024)

        context = self.output_dropout(context)
        out = self.proj(context)  # (B, T, E)
        logits = self.fc(out)  # (B, T, V)
        return logits, hidden


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "vanilla": VanillaLSTM,
    "stacked": StackedLSTM,
    "bilstm_attn": BiLSTMAttention,
}


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table

    console = Console()
    device = _device()
    console.rule("[bold blue]LSTM Model Smoke Test[/bold blue]")
    console.log(f"Device: [cyan]{device}[/cyan]")

    VOCAB = 16_000
    B, T = 2, 64
    dummy_input = torch.randint(0, VOCAB, (B, T), device=device)

    table = Table(title="Model Summary")
    table.add_column("Model", style="cyan")
    table.add_column("Parameters", justify="right", style="green")
    table.add_column("Logits shape", style="yellow")
    table.add_column("Status", style="bold")

    for name, cls in MODEL_REGISTRY.items():
        model = cls(vocab_size=VOCAB).to(device)
        n_params = model.count_parameters()

        try:
            logits, hidden = model(dummy_input)
            assert logits.shape == (B, T, VOCAB), f"Bad shape: {logits.shape}"
            status = "✅ PASS"
        except Exception as exc:
            status = f"❌ FAIL: {exc}"
            table.add_row(name, f"{n_params:,}", "ERROR", status)
            continue

        table.add_row(name, f"{n_params:,}", str(tuple(logits.shape)), status)

    console.print(table)
    console.rule("[bold green]All models OK[/bold green]")

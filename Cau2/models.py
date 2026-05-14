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
# Bahdanau Attention
# ---------------------------------------------------------------------------
class BahdanauAttention(nn.Module):
    """
    Additive (Bahdanau) attention.

    Given a *query* ``(B, 1, D_q)`` and *keys* ``(B, T, D_k)``, compute:
        score_t = V^T tanh(W_q · query + W_k · key_t)
        weights = softmax(scores)           → (B, T)
        context = sum_t weights_t · keys_t  → (B, 1, D_k)
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
        query : (B, 1, D_q) or (B, D_q)
        keys  : (B, T, D_k)

        Returns
        -------
        context : (B, 1, D_k)
        weights : (B, T)
        """
        if query.dim() == 2:
            query = query.unsqueeze(1)  # (B, 1, D_q)

        # (B, 1, H) + (B, T, H) → broadcast → (B, T, H)
        energy = torch.tanh(self.W_query(query) + self.W_key(keys))  # (B, T, H)
        scores = self.V(energy).squeeze(-1)  # (B, T)
        weights = F.softmax(scores, dim=-1)  # (B, T)
        context = torch.bmm(weights.unsqueeze(1), keys)  # (B, 1, D_k)
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
        self.fc = nn.Linear(self.hidden_dim, vocab_size)

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
    → Linear(1024→vocab)

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
    → BahdanauAttention(query=1024, keys=1024)
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

        B, T = x.shape
        emb = self.embedding(x)  # (B, T, E)
        lstm_out, hidden = self.lstm(emb, hidden)  # (B, T, 1024)

        # For each timestep, use its own output as query against all keys
        # This gives per-timestep attended context
        # query: each timestep individually → loop-free via broadcasting
        # query shape needs to be (B, 1, D) for each t, but we can vectorise:
        contexts = []
        for t in range(T):
            query_t = lstm_out[:, t : t + 1, :]  # (B, 1, 1024)
            ctx_t, _ = self.attention(query_t, lstm_out)  # (B, 1, 1024)
            contexts.append(ctx_t)
        context = torch.cat(contexts, dim=1)  # (B, T, 1024)

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
            status = "PASS"
        except Exception as exc:
            logits_shape_str = "ERROR"
            status = f"FAIL: {exc}"
            table.add_row(name, f"{n_params:,}", "ERROR", status)
            continue

        table.add_row(name, f"{n_params:,}", str(tuple(logits.shape)), status)

    console.print(table)
    console.rule("[bold green]All models OK[/bold green]")

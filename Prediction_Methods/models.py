"""
models.py
---------
PyTorch model architectures used for time-series forecasting.

Models
------
LSTMModel             : 2-layer stacked LSTM
GRUModel              : 2-layer stacked GRU
PositionalEncoding    : Sinusoidal positional encoding (shared)
TimeSeriesTransformer : Vanilla Transformer encoder
Informer              : Full encoder-decoder Informer with ProbSparse attention
                        (Zhou et al., 2021)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ===========================================================================
# LSTM
# ===========================================================================

class LSTMModel(nn.Module):
    """2-layer stacked LSTM. Input: [batch, seq_len, 1]. Output: [batch]."""

    def __init__(self, input_size: int = 1, hidden_layer_size: int = 64, output_size: int = 1):
        super().__init__()
        self.lstm   = nn.LSTM(input_size, hidden_layer_size,
                              num_layers=2, batch_first=True, dropout=0.1)
        self.norm   = nn.LayerNorm(hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :]).squeeze(1)


# ===========================================================================
# GRU
# ===========================================================================

class GRUModel(nn.Module):
    """2-layer stacked GRU. Input: [batch, seq_len, 1]. Output: [batch]."""

    def __init__(self, input_size: int = 1, hidden_layer_size: int = 64, output_size: int = 1):
        super().__init__()
        self.gru    = nn.GRU(input_size, hidden_layer_size,
                             num_layers=2, batch_first=True, dropout=0.1)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        out, _ = self.gru(x)
        return self.linear(out[:, -1, :]).squeeze(1)


# ===========================================================================
# Shared positional encoding
# ===========================================================================

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.
    batch_first=False : x shape [seq, batch, d_model]
    batch_first=True  : x shape [batch, seq, d_model]
    """

    def __init__(self, d_model: int, max_len: int = 5000, batch_first: bool = False):
        super().__init__()
        self.batch_first = batch_first
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        if self.batch_first:
            return x + self.pe[:x.size(1), :].unsqueeze(0)
        return x + self.pe[:x.size(0), :].unsqueeze(1)


# ===========================================================================
# Vanilla Transformer encoder
# ===========================================================================

class TimeSeriesTransformer(nn.Module):
    """
    Transformer encoder-only model.
    Input : [seq_len, batch, 1]  (batch_first=False)
    Output: [batch]
    """

    def __init__(self, input_dim: int = 1, d_model: int = 64,
                 nhead: int = 8, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model      = d_model
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoder  = PositionalEncoding(d_model, batch_first=False)
        encoder_layer     = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_linear = nn.Linear(d_model, 1)
        for layer in [self.input_linear, self.output_linear]:
            layer.weight.data.uniform_(-0.1, 0.1)
            layer.bias.data.zero_()

    def forward(self, src: Tensor) -> Tensor:
        src    = self.input_linear(src) * math.sqrt(self.d_model)
        src    = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return self.output_linear(output[-1, :, :]).squeeze(1)


# ===========================================================================
# Informer — ProbSparse attention
# ===========================================================================

class ProbSparseSelfAttention(nn.Module):
    """
    ProbSparse self-attention (Zhou et al., 2021).

    Selects the top-U 'dominant' queries based on a sparsity measurement:
        M(q_i, K) = max_j(q_i·k_j/sqrt(d)) - mean_j(q_i·k_j/sqrt(d))
    Only those queries get full attention; all others are assigned the
    mean-value context vector. This reduces complexity from O(L^2) to
    O(L log L).

    Falls back to full attention when the sequence is short enough that
    the saving would be negligible.
    """

    def __init__(self, d_model: int, n_heads: int, factor: int = 5, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k     = d_model // n_heads
        self.n_heads = n_heads
        self.factor  = factor
        self.W_Q     = nn.Linear(d_model, d_model, bias=False)
        self.W_K     = nn.Linear(d_model, d_model, bias=False)
        self.W_V     = nn.Linear(d_model, d_model, bias=False)
        self.out     = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _top_queries(self, Q: Tensor, K: Tensor, sample_k: int, n_top: int) -> Tensor:
        """Return indices of the n_top most dominant queries."""
        B, H, L_Q, d_k = Q.shape
        L_K = K.shape[2]
        # Randomly sample sample_k keys to estimate dominance
        idx      = torch.randint(L_K, (L_Q, sample_k), device=Q.device)
        K_sample = K.unsqueeze(3).expand(B, H, L_K, L_Q, d_k)
        K_sample = K.unsqueeze(3).expand(
            B, H, L_K, L_Q, d_k
        )                                     # not used directly; use gather below
        # Simpler gather approach
        K_exp    = K[:, :, idx.view(-1), :].view(B, H, L_Q, sample_k, d_k)
        QK       = torch.einsum('bhqd,bhqkd->bhqk', Q, K_exp)   # [B,H,L_Q,sample_k]
        M        = QK.max(-1)[0] - QK.mean(-1)                  # dominance score
        return M.topk(n_top, dim=-1)[1]                         # [B, H, n_top]

    def forward(self, Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
        """Q, K, V: [batch, seq, d_model] → output: [batch, seq_Q, d_model]"""
        B, L_Q, _ = Q.shape
        L_K       = K.shape[1]

        def project_and_split(W, x):
            return W(x).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        Qq = project_and_split(self.W_Q, Q)   # [B, H, L_Q, d_k]
        Kk = project_and_split(self.W_K, K)
        Vv = project_and_split(self.W_V, V)

        sample_k = min(self.factor * max(1, int(math.log(L_K + 1))), L_K)
        n_top    = min(self.factor * max(1, int(math.log(L_Q + 1))), L_Q)

        # Fall back to full attention for very short sequences
        if n_top >= L_Q or sample_k >= L_K:
            scores  = torch.matmul(Qq, Kk.transpose(-2, -1)) / math.sqrt(self.d_k)
            context = torch.matmul(self.dropout(F.softmax(scores, dim=-1)), Vv)
        else:
            top_idx = self._top_queries(Qq, Kk, sample_k, n_top)   # [B, H, n_top]

            # Default context = mean(V) for all positions
            context = Vv.mean(dim=2, keepdim=True).expand(
                B, self.n_heads, L_Q, self.d_k).clone()

            # Gather top queries and compute their full attention
            idx_exp = top_idx.unsqueeze(-1).expand(B, self.n_heads, n_top, self.d_k)
            Q_top   = Qq.gather(2, idx_exp)                                  # [B,H,n_top,d_k]
            scores  = torch.matmul(Q_top, Kk.transpose(-2, -1)) / math.sqrt(self.d_k)
            ctx_top = torch.matmul(self.dropout(F.softmax(scores, dim=-1)), Vv)
            context.scatter_(2, idx_exp, ctx_top)

        context = context.transpose(1, 2).contiguous().view(B, L_Q, -1)
        return self.out(context)


class ConvDistillation(nn.Module):
    """
    Distilling layer between Informer encoder stacks.
    Halves sequence length via MaxPool after a depthwise Conv1d + ELU + BN.
    Input / output shape: [batch, seq_len, d_model]
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.norm = nn.BatchNorm1d(d_model)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = x.transpose(1, 2)                              # [B, d_model, L]
        x = self.pool(F.elu(self.norm(self.conv(x))))
        return x.transpose(1, 2)                           # [B, L//2, d_model]


class InformerEncoderLayer(nn.Module):
    """ProbSparse attention → conv distillation → feed-forward."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 factor: int = 5, dropout: float = 0.1):
        super().__init__()
        self.attn    = ProbSparseSelfAttention(d_model, n_heads, factor, dropout)
        self.distill = ConvDistillation(d_model)
        self.ffn     = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.drop    = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm1(x + self.drop(self.attn(x, x, x)))
        x = self.distill(x)
        x = self.norm2(x + self.drop(self.ffn(x)))
        return x


class InformerDecoderLayer(nn.Module):
    """
    ProbSparse self-attention over decoder tokens,
    full cross-attention over encoder memory,
    feed-forward.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 factor: int = 5, dropout: float = 0.1):
        super().__init__()
        self.self_attn  = ProbSparseSelfAttention(d_model, n_heads, factor, dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn  = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: Tensor, memory: Tensor) -> Tensor:
        x = self.norm1(x + self.drop(self.self_attn(x, x, x)))
        attn_out, _ = self.cross_attn(x, memory, memory)
        x = self.norm2(x + self.drop(attn_out))
        x = self.norm3(x + self.drop(self.ffn(x)))
        return x


class Informer(nn.Module):
    """
    Informer encoder-decoder with ProbSparse attention.
    Reference: Zhou et al., 2021  https://arxiv.org/abs/2012.07436

    Input features (3 per timestep)
    --------------------------------
    [normalised_value, sin(2π·month/12), cos(2π·month/12)]

    Encoder receives the full historical window.
    Decoder receives a 'start token' (last label_len observations) concatenated
    with zero-padded positions for the pred_len future steps, following the
    original paper's generative decoding strategy.

    Parameters
    ----------
    d_model   : Embedding dimension (default 64).
    n_heads   : Attention heads — must divide d_model (default 8).
    e_layers  : Encoder layers (default 2).
    d_layers  : Decoder layers (default 1).
    d_ff      : Feed-forward hidden dim (default 256).
    factor    : ProbSparse sampling factor c (default 5).
    dropout   : Dropout rate (default 0.1).
    label_len : Start-token length fed to the decoder (default 12).
    pred_len  : Forecast horizon (default 12).
    """

    def __init__(self, d_model: int = 64, n_heads: int = 8,
                 e_layers: int = 2, d_layers: int = 1,
                 d_ff: int = 256, factor: int = 5,
                 dropout: float = 0.1,
                 label_len: int = 12, pred_len: int = 12):
        super().__init__()
        self.label_len  = label_len
        self.pred_len   = pred_len
        in_features     = 3

        self.enc_embed  = nn.Linear(in_features, d_model)
        self.dec_embed  = nn.Linear(in_features, d_model)
        self.enc_pos    = PositionalEncoding(d_model, batch_first=True)
        self.dec_pos    = PositionalEncoding(d_model, batch_first=True)

        self.encoder = nn.ModuleList([
            InformerEncoderLayer(d_model, n_heads, d_ff, factor, dropout)
            for _ in range(e_layers)
        ])
        self.decoder = nn.ModuleList([
            InformerDecoderLayer(d_model, n_heads, d_ff, factor, dropout)
            for _ in range(d_layers)
        ])
        self.projection = nn.Linear(d_model, 1)

    @staticmethod
    def build_features(values: Tensor, start_month: int) -> Tensor:
        """
        Concatenate normalised values with cyclic month encoding.

        values      : [batch, seq_len] or [seq_len]
        start_month : calendar month (1-12) of the first timestep
        Returns     : [batch, seq_len, 3]
        """
        if values.dim() == 1:
            values = values.unsqueeze(0)
        B, L   = values.shape
        device = values.device
        months = ((torch.arange(L, device=device) + start_month - 1) % 12) + 1
        angles = 2 * math.pi * months.float() / 12
        sin_m  = angles.sin().unsqueeze(0).expand(B, -1)
        cos_m  = angles.cos().unsqueeze(0).expand(B, -1)
        return torch.stack([values, sin_m, cos_m], dim=-1)   # [B, L, 3]

    def forward(self, x_enc: Tensor, x_dec: Tensor) -> Tensor:
        """
        x_enc : [batch, enc_len, 3]
        x_dec : [batch, label_len + pred_len, 3]
        Returns [batch, pred_len]
        """
        # Encoder
        mem = self.enc_pos(self.enc_embed(x_enc))
        for layer in self.encoder:
            mem = layer(mem)

        # Decoder
        dec = self.dec_pos(self.dec_embed(x_dec))
        for layer in self.decoder:
            dec = layer(dec, mem)

        out = self.projection(dec[:, -self.pred_len:, :])  # [B, pred_len, 1]
        return out.squeeze(-1)                              # [B, pred_len]
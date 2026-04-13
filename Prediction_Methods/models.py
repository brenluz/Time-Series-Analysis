import math

import torch
import torch.nn as nn
from torch import Tensor


class LSTMModel(nn.Module):
    """
    2-layer stacked LSTM for univariate time-series forecasting.
    Expects input of shape [batch, seq_len, 1].
    """

    def __init__(self, input_size: int = 1,
                 hidden_layer_size: int = 64,
                 output_size: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_layer_size,
            num_layers=2, batch_first=True, dropout=0.1
        )
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :]).squeeze(1)


class GRUModel(nn.Module):
    """
    2-layer stacked GRU for univariate time-series forecasting.
    Expects input of shape [batch, seq_len, 1].
    """

    def __init__(self, input_size: int = 1,
                 hidden_layer_size: int = 64,
                 output_size: int = 1):
        super().__init__()
        self.gru = nn.GRU(
            input_size, hidden_layer_size,
            num_layers=2, batch_first=True, dropout=0.1
        )
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        out, _ = self.gru(x)
        return self.linear(out[:, -1, :]).squeeze(1)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer models."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        # x: [seq_len, batch, d_model]
        return x + self.pe[:x.size(0), :].unsqueeze(1)


class TimeSeriesTransformer(nn.Module):
    """
    Transformer encoder for univariate time-series forecasting.
    Expects input of shape [seq_len, batch, 1]  (batch_first=False).
    """

    def __init__(self, input_dim: int = 1, d_model: int = 64,
                 nhead: int = 8, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_linear = nn.Linear(d_model, 1)

        for layer in [self.input_linear, self.output_linear]:
            layer.weight.data.uniform_(-0.1, 0.1)
            layer.bias.data.zero_()

    def forward(self, src: Tensor) -> Tensor:
        src = self.input_linear(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return self.output_linear(output[-1, :, :]).squeeze(1)
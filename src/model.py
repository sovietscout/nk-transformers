"""
model.py — Causal Transformer for NK macroeconomic dynamics.
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model) — batch_first expects (T, B, D)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (T, B, d_model)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class NKTransformer(nn.Module):
    """Causal Transformer that predicts observables given params, shocks, and history.

    Architecture:
      - Input projection: Linear(17 -> d_model)
      - Sinusoidal positional encoding
      - TransformerEncoder with causal mask (prevents attending to future)
      - Output head: Linear(d_model -> 3)
    """

    def __init__(self,
                 d_model: int = 64,
                 n_heads: int = 4,
                 n_layers: int = 3,
                 ff_dim: int = 256,
                 dropout: float = 0.1,
                 input_dim: int = 17,
                 output_dim: int = 3):
        super().__init__()
        self.d_model = d_model

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=200, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=False,  # (T, B, D) convention
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_head = nn.Linear(d_model, output_dim)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: (B, T, input_dim) — batched sequences

        Returns:
            (B, T, output_dim) — predictions at each time step
        """
        B, T, _ = src.shape

        # Project to d_model
        x = self.input_proj(src)  # (B, T, d_model)
        x = x.permute(1, 0, 2)    # (T, B, d_model)

        # Positional encoding
        x = self.pos_encoder(x)

        # Causal mask: position t can only attend to positions <= t
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(src.device)

        # Encode
        x = self.encoder(x, mask=causal_mask)  # (T, B, d_model)

        # Output head
        x = x.permute(1, 0, 2)  # (B, T, d_model)
        out = self.output_head(x)  # (B, T, 3)

        return out

    def autoregressive_forecast(self,
                                src: torch.Tensor,
                                horizon: int,
                                future_shocks: torch.Tensor = None) -> torch.Tensor:
        """Generate multi-step forecast autoregressively.

        The input at time t is [params, shocks_t, obs_{t-1}], predicting obs_t.
        To forecast forward, we first get the last predicted obs from the context,
        then use it as the lag for each subsequent forecast step.

        Args:
            src: (B, T, input_dim) — context sequence up to time T
            horizon: number of future steps to forecast
            future_shocks: (B, horizon, 3) — future shock values (default: zeros)

        Returns:
            (B, horizon, 3) — predicted observables for steps T to T+horizon-1
        """
        self.eval()
        B, T, input_dim = src.shape

        if future_shocks is None:
            future_shocks = torch.zeros(B, horizon, 3, device=src.device)

        # Extract params (time-invariant, same at every position)
        params = src[:, 0, :11]

        # Run the model on the full context to get predictions for all time steps.
        # At the last context step T-1, the model's output is the prediction of obs[T-1],
        # since input[T-1] = [params, shocks[T-1], obs[T-2]].
        with torch.no_grad():
            context_out = self.forward(src)  # (B, T, 3)

        # The last context prediction is obs[T-1] — this becomes the first lag
        # for the forecast horizon (which predicts obs[T]).
        prev_obs = context_out[:, -1, :]  # (B, 3) = predicted obs[T-1]

        predictions = []
        with torch.no_grad():
            for h in range(horizon):
                shock = future_shocks[:, h, :]  # (B, 3)
                # Build input: [params, shock, prev_obs]
                inp = torch.cat([params, shock, prev_obs], dim=-1).unsqueeze(1)  # (B, 1, 17)

                # Append to context and re-run through the model.
                # We need the full sequence so the Transformer can attend to all
                # previous positions (causal attention).
                extended = torch.cat([src, inp], dim=1)  # (B, T+1+h, 17)

                out = self.forward(extended)  # (B, T+1+h, 3)
                pred = out[:, -1, :]  # (B, 3) — prediction of obs[T+h]

                predictions.append(pred)
                prev_obs = pred  # feed predicted obs forward as the next lag

        return torch.stack(predictions, dim=1)  # (B, horizon, 3)

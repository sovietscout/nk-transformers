import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Batch First)."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x is (B, T, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class NKTransformer(nn.Module):
    """Causal Transformer for the NK simulations."""

    def __init__(self,
                 d_model: int = 64,
                 n_heads: int = 4,
                 n_layers: int = 4,
                 ff_dim: int = 256,
                 dropout: float = 0.1,
                 input_dim: int = 17,
                 output_dim: int = 3):
        super().__init__()
        self.d_model = d_model

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=250, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
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

        x = self.input_proj(src)  # (B, T, d_model)
        x = self.pos_encoder(x)

        # Each position can only look at the history available at that date.
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(src.device)

        x = self.encoder(x, mask=causal_mask, is_causal=True)  # (B, T, d_model)
        out = self.output_head(x)  # (B, T, 3)

        return out

    @torch.compiler.disable
    def autoregressive_forecast(self,
                                src: torch.Tensor,
                                horizon: int,
                                future_shocks: torch.Tensor = None,
                                true_lag: torch.Tensor = None) -> torch.Tensor:
        """Forecast by feeding each prediction back in as the next lag.

        The input at time t is [params, shocks_t, obs_{t-1}], predicting obs_t.

        Args:
            src: (B, T, input_dim) — context sequence up to time T
            horizon: number of future steps to forecast
            future_shocks: (B, horizon, 3) — future shock values (default: zeros)
            true_lag: (B, 3) — optional true observation for step T (default: uses T-1 prediction)

        Returns:
            (B, horizon, 3) — predicted observables for steps T to T+horizon-1
        """
        self.eval()
        B, T, input_dim = src.shape

        if future_shocks is None:
            future_shocks = torch.zeros(B, horizon, 3, device=src.device)

        # Parameters are time-invariant, so the first token has the same values
        # as every other token in the context.
        params = src[:, 0, :11]

        if true_lag is not None:
            prev_obs = true_lag
        else:
            with torch.no_grad():
                context_out = self.forward(src)  # (B, T, 3)
            # The final context prediction becomes the lag for the first forecast.
            prev_obs = context_out[:, -1, :]  # (B, 3) = predicted obs[T-1]

        predictions = []
        with torch.no_grad():
            for h in range(horizon):
                shock = future_shocks[:, h, :]  # (B, 3)
                inp = torch.cat([params, shock, prev_obs], dim=-1).unsqueeze(1)  # (B, 1, 17)

                # Re-run on the growing sequence so attention sees the same kind
                # of history it saw during training.
                extended = torch.cat([src, inp], dim=1)  # (B, T+1+h, 17)

                out = self.forward(extended)  # (B, T+1+h, 3)
                pred = out[:, -1, :]  # (B, 3) = prediction of obs[T+h]

                predictions.append(pred)
                prev_obs = pred

        return torch.stack(predictions, dim=1)  # (B, horizon, 3)

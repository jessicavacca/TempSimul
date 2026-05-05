import math
import torch
from torch import Tensor, nn


def positional_encoding(length: int, d_model: int,
                            device: torch.device) -> Tensor:
    """Sinusoidal positional encoding with shape (1, length, d_model)."""
    position = torch.arange(length, device=device,
                            dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(
            0, d_model, 2, device=device, dtype=torch.float32) *
        (-math.log(10000.0) / d_model))

    pe = torch.zeros(length,
                        d_model,
                        device=device,
                        dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


class SimpleTransformerForecast(nn.Module):
    """A simple transformer for forecasting.
    
    The model consists of a single transformer encoder layer followed by a linear projection to the target dimension. 
    It takes in the lookback window as input and outputs the forecast for the horizon.    
    """

    def __init__(
        self,
        input_dim: int,
        n_channels: int,
        target_dim: int,
        num_layers: int,
        horizon: int,
        lookback: int,
        d_model: int = 128,
        nhead: int = 8,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if lookback <= 0:
            raise ValueError("lookback must be > 0")
        if horizon <= 0:
            raise ValueError("horizon must be > 0")

        self.input_dim = input_dim
        self.n_channels = n_channels
        self.target_dim = target_dim
        self.num_layers = num_layers
        self.horizon = horizon
        self.lookback = lookback
        self.d_model = d_model

        self.input_projection = nn.Linear(input_dim * n_channels, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                         num_layers=num_layers)

        self.output_projection = nn.Linear(d_model, target_dim * horizon)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, lookback, input_dim * n_channels)
        
        Returns:
            Tensor of shape (batch_size, horizon, target_dim)
        """
        # Project input to d_model
        x = x.permute(0, 2, 1)  # (batch_size, input_dim * n_channels, lookback)
        x = self.input_projection(x)  # (batch_size, lookback, d_model)

        # Pass through transformer encoder
        x = self.transformer_encoder(x)  # (batch_size, lookback, d_model)

        # Take the last output of the encoder and project to target dimension
        x = x[:, -1, :]  # (batch_size, d_model)
        x = self.output_projection(x)  # (batch_size, target_dim * horizon)
        x = x.view(-1, self.horizon,
                   self.target_dim)  # (batch_size, horizon, target_dim)

        return x

import torch
from torch import Tensor, nn
from lib.layers.Encodings import positional_encoding


class SimpleTransformerForecast(nn.Module):
    """A simple transformer for forecasting.
    
    The model consists of a single transformer encoder layer followed by a linear projection to the target dimension. 
    It takes in the lookback window as input and outputs the forecast for the horizon.    
    """

    def __init__(
        self,
        lookback: int,
        horizon: int,
        input_dim: int,
        target_dim: int,
        num_layers: int,
        d_model: int = 128,
        n_heads: int = 8,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        causal: bool = True,
    ) -> None:
        super().__init__()

        if lookback <= 0:
            raise ValueError("lookback must be > 0")
        if horizon <= 0:
            raise ValueError("horizon must be > 0")

        self.input_dim = input_dim
        self.target_dim = target_dim
        self.num_layers = num_layers
        self.horizon = horizon
        self.lookback = lookback
        self.d_model = d_model
        self.causal = causal
        self.input_projection = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                         num_layers=num_layers)

        self.output_projection = nn.Linear(d_model * lookback, horizon)

        self.positional_encoding = positional_encoding(
            lookback,
            d_model,
            device='cuda' if torch.cuda.is_available() else 'cpu')

    def _generate_causal_mask(self, length: int,
                              device: torch.device) -> Tensor:
        return nn.Transformer.generate_square_subsequent_mask(length,
                                                              device=device)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, input_dim, lookback)
        
        Returns:
            Tensor of shape (batch_size, horizon, target_dim)
        """
        # Project input to d_model
        x = x.permute(0, 2, 1)  # (batch_size, input_dim, lookback)
        x = self.input_projection(x)  # (batch_size, lookback, d_model)

        # Add positional encoding
        x += self.positional_encoding

        # Pass through transformer encoder
        if self.causal:
            x = self.transformer_encoder(
                x, mask=self._generate_causal_mask(
                    x.size(1), x.device))  # (batch_size, lookback, d_model)
        else:
            x = self.transformer_encoder(x)
        # print('Tx.shape:', x.shape)
        x = self.output_projection(x.reshape(
            x.size(0), -1))  # (batch_size, target_dim * horizon)
        # print('Px.shape:', x.shape)
        x = x.view(-1, self.target_dim,
                   self.horizon)  # (batch_size, horizon, target_dim)

        return x

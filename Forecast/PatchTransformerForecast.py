from torch import Tensor, nn
from Forecast.Encodings import positional_encoding
from layers.Transformer import TransformerEncoder
from layers.TransformerEncoder import TransformerEncoderLayer
import torch


class PatchTransformerForecast(nn.Module):
    """ A transformer-based model for forecasting that uses patching to handle long lookback windows.

    The model consists of an encoder architecture that takes in patches of the lookback window
    and predicts the future values for the horizon with a linear projection. 
    """

    def __init__(
        self,
        lookback: int,
        horizon: int,
        input_dim: int,
        target_dim: int,
        num_layers: int,
        patch_len: int = 16,
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
        if patch_len <= 0:
            raise ValueError("patch_len must be > 0")
        if lookback % patch_len != 0:
            raise ValueError("lookback must be divisible by patch_len")

        self.input_dim = input_dim
        self.target_dim = target_dim
        self.num_layers = num_layers
        self.horizon = horizon
        self.lookback = lookback
        self.patch_len = patch_len
        self.n_patches = lookback // patch_len
        self.d_model = d_model
        self.causal = causal

        self.patch_projection = nn.Conv1d(patch_len * input_dim,
                                          d_model,
                                          kernel_size=1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                         num_layers=num_layers)

        self.output_projection = nn.Linear(d_model * self.n_patches, horizon)

    def _generate_causal_mask(self, length: int,
                              device: torch.device) -> Tensor:
        return nn.Transformer.generate_square_subsequent_mask(length,
                                                              device=device)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, n_channels, n_samples = x.shape

        # Create patches
        x = x.unfold(dimension=2, size=self.patch_len, step=self.patch_len)
        x = x.contiguous().view(batch_size, n_channels * self.patch_len, -1)
        x = self.patch_projection(x)
        x = x.permute(0, 2, 1)  # (batch_size, input_dim, lookback)

        # Add positional encoding
        pe = positional_encoding(self.n_patches, self.d_model, device=x.device)
        x = x + pe

        # Encode
        if self.causal:
            x = self.transformer_encoder(x,
                                         mask=self._generate_causal_mask(
                                             x.size(1), x.device))
        else:
            x = self.transformer_encoder(x)

        # Flatten and project to output
        x = x.contiguous().view(batch_size, -1)
        output = self.output_projection(x)

        return output


class PatchTransformerForecast2(nn.Module):
    """ A transformer-based model for forecasting that uses patching to handle long lookback windows.

    The model consists of an encoder architecture that takes in patches of the lookback window
    and predicts the future values for the horizon with a linear projection. 
    """

    def __init__(
        self,
        lookback: int,
        horizon: int,
        input_dim: int,
        target_dim: int,
        num_layers: int,
        patch_len: int = 16,
        d_model: int = 128,
        n_heads: int = 8,
        dim_feedforward: int = 256,
        norm_first: bool = True,
        swiglu: bool = False,
        rmsnorm: bool = False,
        dropout: float = 0.1,
        trans_norm=False,
        layer_norm_eps=1e-5,
        bias=True,
        device=None,
        causal: bool = True,
    ) -> None:
        super().__init__()

        if lookback <= 0:
            raise ValueError("lookback must be > 0")
        if horizon <= 0:
            raise ValueError("horizon must be > 0")
        if patch_len <= 0:
            raise ValueError("patch_len must be > 0")
        if lookback % patch_len != 0:
            raise ValueError("lookback must be divisible by patch_len")

        self.input_dim = input_dim
        self.target_dim = target_dim
        self.num_layers = num_layers
        self.horizon = horizon
        self.lookback = lookback
        self.patch_len = patch_len
        self.n_patches = lookback // patch_len
        self.d_model = d_model
        self.causal = causal

        self.patch_projection = nn.Conv1d(patch_len * input_dim,
                                          d_model,
                                          kernel_size=1)
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            norm_first=norm_first,
            swiglu=swiglu,
            rmsnorm=rmsnorm)
        if trans_norm:
            encoder_norm = nn.LayerNorm(d_model,
                                        eps=layer_norm_eps,
                                        bias=bias,
                                        device=device)
        else:
            encoder_norm = None
        self.transformer_encoder = TransformerEncoder(encoder_layer,
                                                      num_layers=num_layers,
                                                      norm=encoder_norm)

        self.output_projection = nn.Linear(d_model * self.n_patches, horizon)

    def _generate_causal_mask(self, length: int,
                              device: torch.device) -> Tensor:
        return nn.Transformer.generate_square_subsequent_mask(length,
                                                              device=device)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, n_channels, n_samples = x.shape

        # Create patches
        x = x.unfold(dimension=2, size=self.patch_len, step=self.patch_len)
        x = x.contiguous().view(batch_size, n_channels * self.patch_len, -1)
        x = self.patch_projection(x)
        x = x.permute(0, 2, 1)  # (batch_size, input_dim, lookback)

        # Add positional encoding
        pe = positional_encoding(self.n_patches, self.d_model, device=x.device)
        x = x + pe

        # Encode
        if self.causal:
            x = self.transformer_encoder(x,
                                         mask=self._generate_causal_mask(
                                             x.size(1), x.device))
        else:
            x = self.transformer_encoder(x)

        # Flatten and project to output
        x = x.contiguous().view(batch_size, -1)
        output = self.output_projection(x)

        return output

from torch import Tensor, nn
from lib.layers.Encodings import sinusoidal_positional_encoding
from lib.layers.Transformer import TransformerEncoder
from lib.layers.TransformerEncoder import TransformerEncoderLayer
import torch


def PatchTransformerForecast_exp_name(config):
    """Generates a name for the experiment based on the configuration"""
    name = "PatchTransformerForecast"
    name += f"_l{config['model']['params']['lookback']}"
    name += f"_h{config['model']['params']['horizon']}"
    name += f"_d{config['model']['params']['input_dim']}"
    name += f"_td{config['model']['params']['target_dim']}"
    name += f"_n{config['model']['params']['num_layers']}"
    name += f"_p{config['model']['params']['patch_len']}"
    name += f"_d{config['model']['params']['d_model']}"
    name += f"_h{config['model']['params']['n_heads']}"
    name += f"_df{config['model']['params']['dim_feedforward']}"
    name += f"_dr{config['model']['params']['dropout']}"
    if 'pos_encodings' in config['model']['params']:
        name += f"_p{config['model']['params']['pos_encodings']}"
    name += f"_cs{config['model']['params']['causal']}"
    return name


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
        positional_encodings: str = 'sinusoidal',
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

        if positional_encodings == 'sinusoidal':
            self.positional_encoding = sinusoidal_positional_encoding(
                self.n_patches,
                d_model,
                device='cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.positional_encoding = None
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
        if self.positional_encoding is not None:
            x += self.positional_encoding

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


def PatchTransformerForecast2_exp_name(config):
    """Generates a name for the experiment based on the configuration"""
    name = "PatchTransformerForecast2"
    name += f"_l{config['model']['params']['lookback']}"
    name += f"_h{config['model']['params']['horizon']}"
    name += f"_d{config['model']['params']['input_dim']}"
    name += f"_td{config['model']['params']['target_dim']}"
    name += f"_n{config['model']['params']['num_layers']}"
    name += f"_p{config['model']['params']['patch_len']}"
    name += f"_d{config['model']['params']['d_model']}"
    name += f"_h{config['model']['params']['n_heads']}"
    name += f"_df{config['model']['params']['dim_feedforward']}"
    name += f"_nf{config['model']['params']['norm_first']}"
    name += f"_sg{config['model']['params']['swiglu']}"
    name += f"_rn{config['model']['params']['rmsnorm']}"
    name += f"_dr{config['model']['params']['dropout']}"
    name += f"_tn{config['model']['params']['trans_norm']}"
    # name += f"_lne{config['model']['params']['layer_norm_eps']}"
    if 'bias' in config['model']['params']:
        name += f"_b{config['model']['params']['bias']}"
    if 'pos_encodings' in config['model']['params']:
        name += f"_p{config['model']['params']['pos_encodings']}"
    name += f"_cs{config['model']['params']['causal']}"

    return name

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
        trans_norm: bool = False,
        layer_norm_eps: float = 1e-5,
        bias: bool = True,
        device: bool =None,
        positional_encodings: str = 'sinusoidal',
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

        if positional_encodings == 'sinusoidal':
            self.positional_encoding = sinusoidal_positional_encoding(
                self.n_patches,
                d_model,
                device='cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.positional_encoding = None

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
        if self.positional_encoding is not None:
            x += self.positional_encoding

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

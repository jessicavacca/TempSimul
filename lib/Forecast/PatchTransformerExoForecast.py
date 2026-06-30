from torch import Tensor, nn
from lib.layers.Encodings import sinusoidal_positional_encoding
from lib.layers.Transformer import TransformerEncoder
from lib.layers.TransformerEncoder import TransformerEncoderLayer
import torch



def PatchTransformerExoForecast_exp_name(config):
    """Generates a name for the experiment based on the configuration"""
    name = "PatchTransformerExoForecast"
    name += f"_l{config['model']['params']['lookback']}"
    name += f"_h{config['model']['params']['horizon']}"
    name += f"_id{config['model']['params']['input_dim']}"
    name += f"_td{config['model']['params']['target_dim']}"
    name += f"_nl{config['model']['params']['num_layers']}"
    name += f"_pl{config['model']['params']['patch_len']}"
    name += f"_dm{config['model']['params']['d_model']}"
    name += f"_nh{config['model']['params']['n_heads']}"
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


class PatchTransformerExoForecast(nn.Module):
    """ A transformer-based model for forecasting that uses patching to handle long lookback windows.

    The model consists of an encoder architecture that takes in patches of the lookback window
    and predicts the future values for the horizon with a linear projection. 

    It introduces a set of exogenous features that are concatenated to the input sequence 
    before patching. The exogenous features can be used to provide additional context for the forecasting task.

    Conditional features can also be provided to the model, that correspond (for now) to a vector of conditions
    per input channel.

    Supports additional features such as normalization, SwiGLU activation, and RMSNorm.
    """

    def __init__(
        self,
        lookback: int,
        horizon: int,
        input_dim: int,
        exo_dim: int, # <--- added row
        target_dim: int,
        condition_dim: int,
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
        device: bool = None,
        positional_encodings: str = 'sinusoidal',
        causal: bool = True,
        verbose: bool = False,
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
        self.exo_dim = exo_dim # <--- added row
        self.target_dim = target_dim
        self.num_layers = num_layers
        self.horizon = horizon
        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward
        self.norm_first = norm_first
        self.swiglu = swiglu
        self.rmsnorm = rmsnorm
        self.dropout = dropout
        self.lookback = lookback
        self.patch_len = patch_len
        self.n_patches = lookback // patch_len
        self.d_model = d_model
        self.causal = causal
        self.encodings = positional_encodings
        self.verbose = verbose

        if positional_encodings == 'sinusoidal':
            self.positional_encoding = sinusoidal_positional_encoding(
                self.n_patches,
                d_model,
                device='cuda' if torch.cuda.is_available() else 'cpu')
        elif positional_encodings == 'learned':
            self.positional_encoding = nn.Parameter(
                torch.randn(1, self.n_patches, d_model))
        else:
            self.positional_encoding = None

        self.patch_projection = nn.Conv1d(patch_len * input_dim,
                                          d_model,
                                          kernel_size=1)
        
        # ---> MODIFICA QUESTA RIGA (sostituisci input_dim con exo_dim) <---
        self.patch_exogenous_projection = nn.Conv1d(patch_len * exo_dim,
                                                    d_model,
                                                    kernel_size=1)

        self.patch_condition_projection = nn.Linear(condition_dim, d_model)

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

        self.output_projection = nn.Linear(d_model * (self.n_patches * 2 + 1),
                                           horizon)

    def log_parameters(self, logger):
        logger.info(f"Model parameters:")
        logger.info(f"  Lookback: {self.lookback}")
        logger.info(f"  Horizon: {self.horizon}")
        logger.info(f"  Input dim: {self.input_dim}")
        logger.info(f"  Target dim: {self.target_dim}")
        logger.info(f"  Num layers: {self.num_layers}")
        logger.info(f"  Patch length: {self.patch_len}")
        logger.info(f"  D model: {self.d_model}")
        logger.info(f"  N heads: {self.n_heads}")
        logger.info(f"  Dim feedforward: {self.dim_feedforward}")
        logger.info(f"  Norm first: {self.norm_first}")
        logger.info(f"  Swiglu: {self.swiglu}")
        logger.info(f"  Rmsnorm: {self.rmsnorm}")
        logger.info(f"  Dropout: {self.dropout}")
        logger.info(f"  Causal: {self.causal}")
        if self.positional_encoding is not None:
            logger.info(f"  Positional encodings: {self.encodings}")
        else:
            logger.info(f"  Positional encodings: none")

    def _generate_causal_mask(self, length: int,
                              device: torch.device) -> Tensor:
        return nn.Transformer.generate_square_subsequent_mask(length,
                                                              device=device)

    def forward(self, x: Tensor, exo: Tensor, cond: Tensor) -> Tensor:
        batch_size, n_channels, n_samples = x.shape

        # Create patches
        x = x.unfold(dimension=2, size=self.patch_len, step=self.patch_len)
        x = x.contiguous().view(batch_size, n_channels * self.patch_len, -1)
        x = self.patch_projection(x)
        x = x.permute(0, 2, 1)  # (batch_size, input_dim, lookback)

        # Create exogenous patches
        if exo is None:
            # Assicurati di usare exo_dim per i layer vuoti se non ci sono dati!
            exo = torch.zeros(batch_size, self.exo_dim, n_samples, device=x.device)

        exo = exo.unfold(dimension=2, size=self.patch_len, step=self.patch_len)
        exo = exo.contiguous().view(batch_size, self.exo_dim * self.patch_len, -1)
        
        exo = self.patch_exogenous_projection(exo)
        exo = exo.permute(0, 2, 1)  #(batch_size, input_dim, lookback)

        if cond is None:
            cond = torch.zeros(batch_size, self.condition_dim, device=x.device)
        else:
            if cond.dim() == 2:
                cond = cond.unsqueeze(1)  # (batch_size, 1, condition_dim)
            elif cond.dim() != 3:
                raise ValueError(
                    "Condition tensor must be of shape (batch_size, condition_dim) or (batch_size, n_patches, condition_dim)"
                )
            elif cond.size(1) != self.n_patches:
                raise ValueError(
                    f"Condition tensor must have {self.n_patches} patches in dimension 1"
                )
        # Project conditional features (one token by now, but could be extended to a sequence of conditions)
        cond = self.patch_condition_projection(cond)
        if self.verbose:
            print(
                f"Input shape after patching: {x.shape}, exogenous shape: {exo.shape}, condition shape: {cond.shape}"
            )

        # Add positional encoding
        if self.positional_encoding is not None:
            x += self.positional_encoding
            exo += self.positional_encoding

        # Concatenate exogenous features
        x = torch.cat((x, exo), dim=1)
        x = torch.cat((x, cond), dim=1)

        if self.verbose:
            print(f"Input shape after patching and concatenation: {x.shape}")

        # Encode
        if self.causal:
            x = self.transformer_encoder(x,
                                         mask=self._generate_causal_mask(
                                             x.size(1), x.device))
        else:
            x = self.transformer_encoder(x)

        # Flatten and project to output
        x = x.contiguous().view(batch_size, -1)

        if self.verbose:
            print(f"Output shape before projection: {x.shape}")

        output = self.output_projection(x)

        return output

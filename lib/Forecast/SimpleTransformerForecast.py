import torch
from torch import Tensor, nn
from lib.layers.Encodings import sinusoidal_positional_encoding


def SimpleTransformerForecast_exp_name(config):
    """Generates a name for the experiment based on the configuration"""
    name = "SimpleTransformerForecast"
    name += f"_l{config['model']['params']['lookback']}"
    name += f"_h{config['model']['params']['horizon']}"
    name += f"_d{config['model']['params']['input_dim']}"
    name += f"_td{config['model']['params']['target_dim']}"
    name += f"_nl{config['model']['params']['num_layers']}"
    name += f"_d{config['model']['params']['d_model']}"
    name += f"_h{config['model']['params']['n_heads']}"
    name += f"_df{config['model']['params']['dim_feedforward']}"
    name += f"_dr{config['model']['params']['dropout']}"
    name += f"_c{config['model']['params']['causal']}"
    if 'pos_encodings' in config['model']['params']:
        name += f"_p{config['model']['params']['pos_encodings']}"

    return name


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
        positional_encodings: str = 'sinusoidal',
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
        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward
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

        if positional_encodings == 'sinusoidal':
            self.positional_encoding = sinusoidal_positional_encoding(
                lookback,
                d_model,
                device='cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.positional_encoding = None

    def _generate_causal_mask(self, length: int,
                              device: torch.device) -> Tensor:
        return nn.Transformer.generate_square_subsequent_mask(length,
                                                              device=device)

    def log_parameters(self, logger):
        logger.info(f"Model parameters:")
        logger.info(f"  Lookback: {self.lookback}")
        logger.info(f"  Horizon: {self.horizon}")
        logger.info(f"  Input dim: {self.input_dim}")
        logger.info(f"  Target dim: {self.target_dim}")
        logger.info(f"  Num layers: {self.num_layers}")
        logger.info(f"  D model: {self.d_model}")
        logger.info(f"  N heads: {self.n_heads}")
        logger.info(f"  Dim feedforward: {self.dim_feedforward}")
        logger.info(
            f"  N heads: {self.transformer_encoder.layers[0].self_attn.num_heads}"
        )
        logger.info(
            f"  Dim feedforward: {self.transformer_encoder.layers[0].linear1.out_features}"
        )
        logger.info(
            f"  Dropout: {self.transformer_encoder.layers[0].dropout.p}")
        logger.info(f"  Causal: {self.causal}")
        if self.positional_encoding is not None:
            logger.info(f"  Positional encodings: sinusoidal")
        else:
            logger.info(f"  Positional encodings: none")

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
        if self.positional_encoding is not None:
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

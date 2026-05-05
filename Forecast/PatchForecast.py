import math

import torch
from torch import Tensor, nn


class EncoderDecoderPatchTransformer(nn.Module):
    """Encoder-decoder Transformer for autoregressive time-series forecasting.

	The encoder receives linearly projected lookback patches. The decoder uses
	teacher forcing during training (shifted ground-truth targets) and
	autoregressive rollout during inference.
	"""

    def __init__(
        self,
        input_dim: int,
        channel_dim: int,
        num_layers: int,
        horizon: int,
        lookback: int,
        patch_len: int = 16,
        target_dim: int = 1,
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
        if patch_len <= 0:
            raise ValueError("patch_len must be > 0")
        if lookback % patch_len != 0:
            raise ValueError("lookback must be divisible by patch_len")

        self.input_dim = input_dim
        self.channel_dim = channel_dim
        self.target_dim = target_dim
        self.num_layers = num_layers
        self.horizon = horizon
        self.lookback = lookback
        self.patch_len = patch_len
        self.n_patches = lookback // patch_len
        self.d_model = d_model

        self.patch_projection = nn.Conv1d(patch_len * input_dim, d_model, kernel_size=1)
        self.decoder_input_projection = nn.Linear(target_dim, d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.output_projection = nn.Linear(d_model, target_dim)

        self.sos_token = nn.Parameter(torch.zeros(1, 1, target_dim))

    def _to_time_last(self, x: Tensor, expected_length: int) -> Tensor:
        """Convert input to shape (batch, length, features)."""
        if x.ndim == 2:
            return x.unsqueeze(-1)

        if x.ndim != 3:
            raise ValueError(
                "Input must have shape (B, L) or (B, L, C) or (B, C, L)")

        if x.shape[1] == expected_length and x.shape[2] != expected_length:
            return x

        if x.shape[2] == expected_length:
            return x.transpose(1, 2)

        if x.shape[1] == expected_length and x.shape[2] == expected_length:
            return x

        raise ValueError(
            f"Could not infer time dimension. Expected length {expected_length}, got shape {tuple(x.shape)}"
        )

    def _patchify(self, src: Tensor) -> Tensor:
        """Create non-overlapping patches: (B, L, C) -> (B, n_patches, patch_len * C)."""
        batch_size = src.shape[0]
        src = src.view(batch_size, self.n_patches, self.patch_len,
                       self.input_dim)
        return src.reshape(batch_size, self.n_patches,
                           self.patch_len * self.input_dim)

    def _positional_encoding(self, length: int,
                             device: torch.device) -> Tensor:
        """Sinusoidal positional encoding with shape (1, length, d_model)."""
        position = torch.arange(length, device=device,
                                dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(
                0, self.d_model, 2, device=device, dtype=torch.float32) *
            (-math.log(10000.0) / self.d_model))

        pe = torch.zeros(length,
                         self.d_model,
                         device=device,
                         dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def _generate_causal_mask(self, length: int,
                              device: torch.device) -> Tensor:
        return nn.Transformer.generate_square_subsequent_mask(length,
                                                              device=device)

    def _encode_source(self, src: Tensor) -> Tensor:
        src = self._to_time_last(src, expected_length=self.lookback)
        if src.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, got {src.shape[-1]}")

        src_patches = self._patchify(src)
        src_tokens = self.patch_projection(src_patches)
        src_tokens = src_tokens + self._positional_encoding(
            src_tokens.shape[1], src_tokens.device)
        return self.transformer.encoder(src_tokens)

    def _decode_teacher_forcing(self, memory: Tensor, tgt: Tensor) -> Tensor:
        tgt = self._to_time_last(tgt, expected_length=self.horizon)
        if tgt.shape[-1] != self.target_dim:
            raise ValueError(
                f"Expected target_dim={self.target_dim}, got {tgt.shape[-1]}")

        batch_size = tgt.shape[0]
        decoder_in = torch.cat(
            [self.sos_token.expand(batch_size, -1, -1), tgt[:, :-1, :]], dim=1)

        decoder_tokens = self.decoder_input_projection(decoder_in)
        decoder_tokens = decoder_tokens + self._positional_encoding(
            decoder_tokens.shape[1], decoder_tokens.device)

        tgt_mask = self._generate_causal_mask(decoder_tokens.shape[1],
                                              decoder_tokens.device)
        decoded = self.transformer.decoder(decoder_tokens,
                                           memory,
                                           tgt_mask=tgt_mask)
        return self.output_projection(decoded)

    def _decode_autoregressive(self, memory: Tensor) -> Tensor:
        batch_size = memory.shape[0]
        generated = self.sos_token.expand(batch_size, 1, self.target_dim)
        outputs = []

        for _ in range(self.horizon):
            decoder_tokens = self.decoder_input_projection(generated)
            decoder_tokens = decoder_tokens + self._positional_encoding(
                decoder_tokens.shape[1], decoder_tokens.device)

            tgt_mask = self._generate_causal_mask(decoder_tokens.shape[1],
                                                  decoder_tokens.device)
            decoded = self.transformer.decoder(decoder_tokens,
                                               memory,
                                               tgt_mask=tgt_mask)

            next_step = self.output_projection(decoded[:, -1:, :])
            outputs.append(next_step)
            generated = torch.cat([generated, next_step], dim=1)

        return torch.cat(outputs, dim=1)

    def forward(self, src: Tensor, tgt: Tensor | None = None) -> Tensor:
        """Run model.

		- Training: pass both `src` and `tgt` to use teacher forcing.
		- Inference: pass only `src` to get autoregressive horizon prediction.
		"""
        memory = self._encode_source(src)
        if tgt is not None:
            return self._decode_teacher_forcing(memory, tgt)
        return self._decode_autoregressive(memory)

    def teacher_forcing_loss(self, src: Tensor, tgt: Tensor,
                             criterion: nn.Module) -> Tensor:
        """Utility method for one teacher-forcing training objective call."""
        pred = self.forward(src=src, tgt=tgt)
        tgt = self._to_time_last(tgt, expected_length=self.horizon)
        return criterion(pred, tgt)

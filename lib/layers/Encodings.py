import torch
from torch import Tensor
import math


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

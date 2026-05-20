import torch.nn as nn
import torch.nn.functional as F
import torch

class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim,
        multiple_of,
        ffn_dim_multiplier=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        swiglu_hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            swiglu_hidden_dim = int(ffn_dim_multiplier * swiglu_hidden_dim)
        swiglu_hidden_dim = multiple_of * ((swiglu_hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, swiglu_hidden_dim, bias=False, **factory_kwargs)
        self.w2 = nn.Linear(swiglu_hidden_dim, hidden_dim, bias=False, **factory_kwargs)
        self.w3 = nn.Linear(dim, swiglu_hidden_dim, bias=False, **factory_kwargs)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class PackedSwiGLUFFN(nn.Module):

    def __init__(
        self,
        dim,
        hidden_dim,
        multiple_of,
        ffn_dim_multiplier=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        swiglu_hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            swiglu_hidden_dim = int(ffn_dim_multiplier * swiglu_hidden_dim)
        swiglu_hidden_dim = multiple_of * (
            (swiglu_hidden_dim + multiple_of - 1) // multiple_of)

        self.w13 = nn.Linear(dim, 2 * swiglu_hidden_dim, bias=False, **factory_kwargs)
        self.w2 = nn.Linear(swiglu_hidden_dim, hidden_dim, bias=False, **factory_kwargs)

    def forward(self, x):
        x1, x3 = torch.chunk(self.w13(x), 2, dim=-1)
        return self.w2(F.silu(x1) * x3)

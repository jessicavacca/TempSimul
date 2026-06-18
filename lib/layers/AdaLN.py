import torch
import torch.nn as nn

class AdaLNCombinedBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, condition_dim):
        super().__init__()
        # 1. Standard Transformer components
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
        # 2. AdaLN Conditioning MLP
        # Outputs shift, scale, and gate for norm1 and norm2 (6 * hidden_size in total)
        self.adaln_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, hidden_size * 6)
        )

    def forward(self, x, cond):
        # x: (B, N, D) -> Batch, Sequence, Hidden Dim
        # cond: (B, C) -> Condition embedding (e.g., timestep + text)
        
        # Regress the AdaLN parameters from the condition
        # (B, 6*D) -> split into chunks of (B, D)
        embed = self.adaln_mlp(cond)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = embed.chunk(6, dim=1)
        
        # --- Pre-Attention AdaLN ---
        # Normalize and modulate using the regressed scale and shift
        norm_x1 = self.norm1(x)
        modulated_x1 = norm_x1 * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        
        # Self-Attention
        attn_out, _ = self.attn(modulated_x1, modulated_x1, modulated_x1)
        
        # Apply gate to residual branch and add to original 'x'
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # --- Pre-MLP AdaLN ---
        norm_x2 = self.norm2(x)
        modulated_x2 = norm_x2 * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        
        # MLP Block
        mlp_out = self.mlp(modulated_x2)
        
        # Apply gate and add to residual
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        
        return x
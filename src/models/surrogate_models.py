import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import LMAInitialTransform, TransformerBlock

class SurrogateTransformerBlock(nn.Module):
    """A standard Transformer block for the surrogate model."""
    def __init__(self, d_model: int, n_heads: int,
                 d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        att_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + self.drop(att_out))
        x = self.ln2(x + self.drop(self.ff(x)))
        return x

class BaselineSurrogate(nn.Module):
    """Surrogate model using a standard Transformer architecture."""
    def __init__(self, latent_dim: int, n_blocks: int, heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_blocks = n_blocks

        # Project latent vector to a sequence for Transformer
        self.input_proj = nn.Linear(latent_dim, latent_dim) # Simple projection

        self.blocks = nn.ModuleList([
            SurrogateTransformerBlock(latent_dim, heads, d_ff, dropout)
            for _ in range(n_blocks)
        ])

        self.output_head = nn.Linear(latent_dim, 1) # Output a single scalar (Average Pressure)

    def forward(self, z: torch.Tensor):
        # z: (B, latent_dim)
        x = self.input_proj(z).unsqueeze(1) # (B, 1, latent_dim) - treat as sequence of length 1

        for blk in self.blocks:
            x = blk(x)

        x = x.squeeze(1) # (B, latent_dim)
        return self.output_head(x)

class NemesisSurrogate(nn.Module):
    """Surrogate model using an LMA-based Transformer architecture."""
    def __init__(self, latent_dim: int, n_blocks: int, heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_blocks = n_blocks

        self.lma_transform = LMAInitialTransform(latent_dim, heads) # LMA on the latent vector

        # The Transformer blocks will operate on the transformed latent space
        # The output dimension of LMAInitialTransform is d_new (latent_dim // 2 by default)
        lma_output_dim = self.lma_transform.d_new

        self.blocks = nn.ModuleList([
            SurrogateTransformerBlock(lma_output_dim, heads, d_ff, dropout)
            for _ in range(n_blocks)
        ])

        self.output_head = nn.Linear(lma_output_dim, 1) # Output a single scalar

    def forward(self, z: torch.Tensor):
        # z: (B, latent_dim)
        x, _ = self.lma_transform(z.unsqueeze(1)) # (B, 1, latent_dim) -> (B, 1, lma_output_dim)
        x = x.squeeze(1) # (B, lma_output_dim)

        for blk in self.blocks:
            x = blk(x.unsqueeze(1)).squeeze(1) # Pass as sequence, then squeeze back

        return self.output_head(x)

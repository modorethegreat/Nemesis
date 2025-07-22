import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------------
# Latent Meta Attention initial transform (split-stack-re-chunk-embed once)
# ------------------------------------------------------------------------
class LMAInitialTransform(nn.Module):
    """
    Implements exactly the four-step pre-processing:

        1. Split embedding dim into H chunks       -> (B,L,d0/H) * H
        2. Stack along sequence dim                -> (B, H·L, d0/H)
        3. Re-chunk to new sequence length L_new   -> (B,L_new, ?)
        4. Dense proj -> d_new                     -> (B,L_new,d_new)

    A linear projection of the **original** flattened tensor is added back
    to preserve a *residual* connection.
    """
    def __init__(self, d0: int, n_heads: int, d_new: int = None):
        """
        d0      : original embed dimension
        n_heads : number of attention heads (for the split)
        d_new   : target reduced embed dim (default d0//2)
        """
        super().__init__()
        assert d0 % n_heads == 0, "d0 must be divisible by #heads"

        self.n_heads   = n_heads
        self.d0        = d0
        self.d_new     = d_new or (d0 // 2)       # sensible default
        self.proj      = nn.Linear(d0, self.d_new, bias=False)

    def forward(self, x: torch.Tensor):
        """
        Args
        ----
            x : (B, L, d0)

        Returns
        -------
            y     : (B, L_new, d_new)  compressed tensor
            mask* : we don't recompute mask here (caller trims manually)
            L_new : sequence length after reshaping
            d_new : embedding dim after projection (constant)
        """
        B, L, d0 = x.shape
        H        = self.n_heads

        # ── 1. split along embed dim  ───────────────────────────
        chunks = torch.chunk(x, H, dim=-1)   # H tensors, each (B,L,d0/H)

        # ── 2. stack along sequence   ───────────────────────────
        stacked = torch.cat(chunks, dim=1)   # (B, H*L, d0/H)

        # 3. determine L_new so total features preserved ─────
        total_feat = stacked.shape[1] * stacked.shape[2]          # (H*L)*(d0/H) == L*d0
        d_chunk    = total_feat // L
        assert total_feat % L == 0, \
               "L must divide total feature count"
        reshaped   = stacked.reshape(B, L, d_chunk)                # (B,L,d_chunk)

        # ── 4. embed → d_new  ───────────────────────────────────
        y = F.relu(self.proj(reshaped))                           # (B,L,d_new)

        # ── residual: project original flat tensor, reshape same ─
        x_proj     = F.relu(self.proj(x))                    # (B, L, d_new)
        y = y + x_proj                                            # residual add (TESTING!)
        return y, L


# ------------------------------------------------------------------------
# Standard transformer block operating in reduced space
# ------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    """Plain Pre-LN transformer block (MHA + Feed-Forward)"""
    def __init__(self, d_model: int, n_heads: int,
                 d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn   = nn.MultiheadAttention(d_model, n_heads,
                                            batch_first=True)
        self.ff     = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )
        self.ln1    = nn.LayerNorm(d_model)
        self.ln2    = nn.LayerNorm(d_model)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        att_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + self.drop(att_out))
        x = self.ln2(x + self.drop(self.ff(x)))
        return x

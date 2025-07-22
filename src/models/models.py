
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch

from .layers import TransformerBlock, LMAInitialTransform
from .diffusion_net import DiffusionNetLayer

class PointNetBackbone(nn.Module):
    """
    Minimal per-point MLP that lifts raw (x,y,z) coordinates to a d-dim
    embedding.  Output is padded to equal length within the mini-batch so
    we can treat it as a dense (B,L,d) tensor.
    """
    def __init__(self, d_embed: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(6, 64),  nn.ReLU(), # 3 for points, 3 for normals
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, d_embed)              # -> (N_points, d_embed)
        )

    def forward(self, points: torch.Tensor, normals: torch.Tensor):
        """
        Returns:
            x_pad  : (B, L_max, d_embed)  zero-padded
            mask   : (B, L_max)  True at valid positions
        """
        # points: (B, L, 3)
        # normals: (B, L, 3)
        x = torch.cat([points, normals], dim=-1) # Concatenate points and normals
        x = self.mlp(x)
        # Assuming points are already batched and padded to a fixed length
        mask = torch.ones(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device)
        return x, mask

class BaselineVAE(nn.Module):
    def __init__(self, n_blocks: int = 4, d0: int = 256, heads: int = 4, d_ff: int = 512, dropout: float = 0.1):
        super().__init__()

        self.backbone = PointNetBackbone(d0)
        # Replace TransformerBlock with DiffusionNetLayer
        self.blocks = nn.ModuleList([
            DiffusionNetLayer(d0, d0) # DiffusionNetLayer takes in_channels and out_channels
            for _ in range(n_blocks)
        ])
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc_mu = nn.Linear(d0, d0 // 2)
        self.fc_logvar = nn.Linear(d0, d0 // 2)

    def forward(self, data: dict):
        points = data['points']
        normals = data['normals']
        cells = data['cells'] # Get cells from data
        x, mask = self.backbone(points, normals)
        for blk in self.blocks:
            x = blk(x, cells) # Pass cells to DiffusionNetLayer
        x = x.masked_fill(~mask.unsqueeze(-1), -1e9)
        x = self.pool(x.transpose(1, 2)).squeeze(-1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class NemesisVAE(nn.Module):
    def __init__(self, n_blocks: int = 4, d0: int = 256, heads: int = 4, d_ff: int = 512, dropout: float = 0.1):
        super().__init__()

        self.backbone = PointNetBackbone(d0)
        self.lma = LMAInitialTransform(d0, heads)
        d_star = self.lma.d_new
        self.blocks = nn.ModuleList([
            TransformerBlock(d_star, heads, d_ff, dropout)
            for _ in range(n_blocks)
        ])
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc_mu = nn.Linear(d_star, d_star // 2)
        self.fc_logvar = nn.Linear(d_star, d_star // 2)

    def forward(self, data: dict):
        points = data['points']
        normals = data['normals']
        x, mask = self.backbone(points, normals)
        x, L_new = self.lma(x)
        mask = mask[:, :L_new]
        for blk in self.blocks:
            x = blk(x)
        x = x.masked_fill(~mask.unsqueeze(-1), -1e9)
        x = self.pool(x.transpose(1, 2)).squeeze(-1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

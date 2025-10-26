import torch
import torch.nn as nn
from typing import Optional


class InverseFlowEncoder(nn.Module):
    """Simple inverse flow encoder that reduces a flow tensor into a latent code."""

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        conditioning_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.conditioning_dim = conditioning_dim

        flow_layers = [
            nn.LazyLinear(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        ]
        self.flow_mlp = nn.Sequential(*flow_layers)

        self.conditioning_mlp = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        combined_dim = hidden_dim if self.conditioning_mlp is None else hidden_dim * 2
        self.latent_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(
        self,
        flow_tensor: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
    ) -> dict:
        batch_size = flow_tensor.shape[0]
        flow_flat = flow_tensor.view(batch_size, -1)
        flow_embedding = self.flow_mlp(flow_flat)

        conditioning_embedding = None
        if conditioning is not None:
            conditioning = conditioning.view(batch_size, -1)
            conditioning_embedding = self.conditioning_mlp(conditioning)
            combined = torch.cat([flow_embedding, conditioning_embedding], dim=-1)
        else:
            combined = flow_embedding

        latent = self.latent_head(combined)
        return {
            "latent": latent,
            "flow_embedding": flow_embedding,
            "conditioning_embedding": conditioning_embedding,
        }


class InverseFlowDecoder(nn.Module):
    """Decoder that maps latents back to flattened geometry targets."""

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: Optional[int] = None,
        conditioning_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.conditioning_dim = conditioning_dim

        self.backbone = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        if output_dim is None:
            self.output_head = nn.LazyLinear(hidden_dim)
        else:
            self.output_head = nn.Linear(hidden_dim, output_dim)

    def set_output_dim(self, output_dim: int) -> None:
        if self.output_dim == output_dim:
            return
        self.output_dim = output_dim
        device = next(self.parameters()).device
        self.output_head = nn.Linear(self.hidden_dim, output_dim).to(device)

    def forward(
        self,
        latent: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = latent.shape[0]
        inputs = latent
        if conditioning is not None and conditioning.numel() > 0:
            conditioning = conditioning.view(batch_size, -1)
            inputs = torch.cat([latent, conditioning], dim=-1)
        hidden = self.backbone(inputs)
        return self.output_head(hidden)

import torch
import torch.nn as nn

class ModulatedResidualNet(nn.Module):
    def __init__(self, in_features, out_features, latent_dim, hidden_dim=256):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.latent_lifting_mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim) # Output size for modulation
        )

        self.query_lifting_mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim) # Output size to be modulated
        )

        # Modulated Residual Blocks
        self.res_blocks = nn.ModuleList([
            ModulatedResBlock(hidden_dim, hidden_dim) for _ in range(3) # Example: 3 residual blocks
        ])

        self.sdf_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_features)
        )

    def forward(self, x, z):
        # x: query points (B, N_sdf_points, in_features)
        # z: latent vector (B, latent_dim)

        modulator = self.latent_lifting_mlp(z) # (B, 256)
        query_features = self.query_lifting_mlp(x) # (B, N_sdf_points, 256)

        # Apply modulation and residual blocks
        for block in self.res_blocks:
            query_features = block(query_features, modulator)

        sdf_pred = self.sdf_mlp(query_features)
        return sdf_pred

class ModulatedResBlock(nn.Module):
    def __init__(self, features_dim, modulator_dim):
        super().__init__()
        self.norm = nn.LayerNorm(features_dim)
        self.mlp = nn.Sequential(
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim)
        )
        self.modulator_mlp = nn.Linear(modulator_dim, features_dim * 2) # For scale and shift

    def forward(self, x, modulator):
        # x: (B, N_sdf_points, features_dim)
        # modulator: (B, modulator_dim)

        # Apply LayerNorm before modulation
        normalized_x = self.norm(x)

        # Generate scale and shift from modulator
        scale_shift = self.modulator_mlp(modulator).unsqueeze(1) # (B, 1, features_dim * 2)
        scale, shift = torch.chunk(scale_shift, 2, dim=-1)

        # Apply modulation (AdaIN-like)
        modulated_x = normalized_x * (1 + scale) + shift

        # Apply MLP and add residual connection
        out = self.mlp(modulated_x)
        return x + out
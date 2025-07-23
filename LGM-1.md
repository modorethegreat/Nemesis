# LGM-1: Baseline Architecture (PhysicsX-style Encoder)

The LGM-1 architecture serves as the baseline for comparison against the NEMESIS (LMA-based) architecture. It aims to replicate the encoder style found in PhysicsX's mesh-to-physical-property prediction pipeline, primarily utilizing a DiffusionNet-like approach for mesh encoding.

## Encoder: DiffusionNet-style

The core of the LGM-1 encoder is a DiffusionNet-style architecture. DiffusionNet is a neural network architecture designed for processing data on arbitrary meshes. It operates by defining a diffusion process on the mesh, allowing information to propagate across the mesh surface.

### Mathematical Foundation (Conceptual)

While our current implementation uses a simplified placeholder for DiffusionNet, a full DiffusionNet typically involves:

1.  **Local Feature Extraction:** Initial features are extracted for each node (vertex) of the mesh. This might involve simple MLPs applied independently to each node's coordinates and other attributes.

2.  **Diffusion Operations:** The key innovation of DiffusionNet lies in its diffusion layers. These layers simulate a diffusion process on the mesh, allowing information from neighboring nodes to be aggregated and propagated. This is often achieved through operations that mimic the heat equation or other diffusion processes on graphs/meshes. This involves constructing graph Laplacians or similar operators that capture the connectivity and geometry of the mesh.

    Conceptually, for a feature $f$ at a node $i$, its updated value $f'_i$ might depend on $f_i$ and the features of its neighbors $f_j$ weighted by some connectivity or distance metric, similar to:

    $f'_i = f_i + \alpha \sum_{j \in N(i)} w_{ij} (f_j - f_i)$

    where $N(i)$ are neighbors of node $i$, $w_{ij}$ are weights, and $\alpha$ is a learning rate. More complex versions involve spectral graph convolutions or learned diffusion kernels.

3.  **Global Aggregation:** After several diffusion layers, a global aggregation step (e.g., global max pooling or mean pooling) is applied to condense the per-node features into a single, fixed-size global feature vector. This vector represents the entire mesh.

4.  **Latent Space Projection:** Finally, linear layers project this global feature vector into the parameters of a latent distribution (mean $\mu$ and log-variance $\Sigma$) for the VAE.

## Decoder: Modulated ResidualNet

Both the LGM-1 (Baseline) and NEMESIS architectures utilize the **identical Modulated ResidualNet decoder**. This is crucial for a fair comparison, as it ensures that any performance differences are attributable to the encoder architecture.

### Decoder Architecture

The decoder's task is to learn an implicit representation of the 3D shape via a Signed Distance Function (SDF). It takes a latent vector $z$ (sampled from the VAE's latent distribution) and a set of query point coordinates $x$ as input.

1.  **Lifting MLPs:** The latent vector $z$ and the query points $x$ are independently passed through separate Lifting MLPs to project them into a higher-dimensional feature space.

2.  **Modulated ResidualNet:** The core of the decoder is a series of Modulated Residual Blocks. In these blocks, the output of the latent vector's Lifting MLP acts as a *modulator*. This modulator conditions the features derived from the query points, allowing the global shape information encoded in $z$ to influence the local SDF prediction at each query point. This is often achieved through adaptive instance normalization (AdaIN) or similar mechanisms.

3.  **Signed Distance MLP:** The output of the Modulated ResidualNet is passed through a final Signed Distance MLP, which regresses the SDF value for each query point $x$.

## Workflow Integration

In the overall pipeline, the LGM-1 encoder processes the input mesh (point cloud and normals) to produce a latent vector $z$. This $z$ is then used by the Modulated ResidualNet decoder to reconstruct the SDF. The VAE is trained end-to-end with a combined reconstruction and KL divergence loss. The latent vector $z$ is subsequently used as input to a downstream surrogate model for physical property prediction.
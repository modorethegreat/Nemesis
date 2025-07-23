# NEMESIS Architecture: Latent Meta Attention (LMA) Encoder

The NEMESIS architecture introduces a novel **Latent Meta Attention (LMA)** encoder to replace the traditional DiffusionNet encoder in mesh-to-physical-property prediction pipelines. The core idea behind LMA is to achieve a more compact and semantically rich latent representation ($z$) of 3D mesh data, which can then be used for downstream tasks like physical property prediction.

## Latent Meta Attention (LMA) - Mathematical Foundation

Traditional attention mechanisms often operate on sequences where each element directly attends to every other element. LMA, as implemented here, introduces a pre-processing step that aims to create a more condensed representation before the main attention blocks. This is achieved through a four-step process:

1.  **Split Embedding Dimension into H Chunks:** The input feature vector for each point (or node) in the mesh, with dimension $d_0$, is split into $H$ (number of attention heads) smaller chunks. Each chunk then has a dimension of $d_0/H$.

    Mathematically, if $X \in \mathbb{R}^{B \times L \times d_0}$ is the input tensor (Batch, Sequence Length, Embedding Dimension), it is conceptually split into $H$ tensors $X_1, X_2, ..., X_H$, where each $X_i \in \mathbb{R}^{B \times L \times (d_0/H)}$.

2.  **Stack Along Sequence Dimension:** These $H$ chunks are then concatenated (stacked) along the sequence dimension. This effectively creates a longer sequence with a smaller per-element embedding dimension.

    The stacked tensor $X_{stacked} \in \mathbb{R}^{B \times (L \cdot H) \times (d_0/H)}$. This operation re-organizes the information, potentially bringing related features closer in the sequence dimension.

3.  **Re-chunk to New Sequence Length ($L_{new}$):** The stacked tensor is then reshaped (re-chunked) to a new, potentially shorter, sequence length $L_{new}$. The goal here is to maintain the total number of features while reducing the sequence length.

    The reshaped tensor $X_{reshaped} \in \mathbb{R}^{B \times L_{new} \times d_{chunk}}$, where $L_{new}$ is chosen such that $L_{new} \cdot d_{chunk} = L \cdot d_0$. In our current implementation, $L_{new}$ is kept the same as the original sequence length $L$, and the embedding dimension $d_{chunk}$ becomes $d_0$.

4.  **Dense Projection to New Embedding Dimension ($d_{new}$):** Finally, a linear projection is applied to map the features to a new, typically smaller, embedding dimension $d_{new}$. This is where the explicit dimensionality reduction for the latent space begins.

    The output $Y = \text{ReLU}(\text{Linear}(X_{reshaped})) \in \mathbb{R}^{B \times L_{new} \times d_{new}}$.

    A residual connection from the original input $X$ (projected to $d_{new}$) is added to $Y$ to help preserve information and stabilize training.

## Workflow Integration

In the overall pipeline, the LMA encoder replaces the DiffusionNet encoder within the VAE architecture. The VAE is trained to reconstruct the Signed Distance Function (SDF) of the 3D mesh. The latent vector $z$ produced by the LMA encoder is then used as input to a downstream surrogate model.

## Key Advantages of NEMESIS (LMA-based) Encoder

*   **Superior Latent Compression:** The LMA mechanism is designed to produce a more compact latent representation ($z$) compared to traditional methods. This means the latent vector can capture essential geometric and physical information in a lower-dimensional space.
*   **Efficiency:** A more compact latent space can lead to more efficient downstream processing and potentially smaller, faster surrogate models.
*   **Focus on Relationships:** By re-organizing and projecting features, LMA aims to better capture complex relationships within the mesh data, which is crucial for accurate physical property prediction.

## Implementation Details

The LMA encoder consists of the `LMAInitialTransform` followed by standard `TransformerBlock`s. The `LMAInitialTransform` handles the initial feature re-organization and dimensionality reduction, while the `TransformerBlock`s (Multi-Head Attention + Feed-Forward Network) further process the features in the compressed latent space.
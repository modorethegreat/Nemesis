## **Comprehensive Project Plan: LMA-Enhanced Pipeline for Physics Prediction**

### 🔧 **Project Goal**

This project aims to re-engineer PhysicsX's mesh-to-physical-property pipeline. The core innovation is the replacement of their **DiffusionNet Encoder** with a novel **Latent Meta Attention (LMA)** encoder. The primary objectives are to demonstrate that the LMA-based approach achieves:

1.  **Superior Latent Compression:** Attain a more compact latent representation ($z$) without sacrificing critical geometric information.
2.  **Enhanced or Equivalent Downstream Performance:** Achieve equal or better accuracy in surrogate models that predict physical properties from the latent space.
3.  **Clear Ablation Study:** Provide a direct, side-by-side comparison against the baseline PhysicsX-style encoder to rigorously evaluate the impact of LMA, using the identical decoder architecture for both.

---

### 📚 **Dataset: MeshGraphNets (DeepMind)**

We will leverage two distinct physics simulation tasks from the MeshGraphNets dataset to ensure the robustness and generalizability of our approach across different physical domains.

| Task | Description | Target Label | Physical Domain |
| :--- | :--- | :--- | :--- |
| **Airfoil Flow** | Simulates airflow over 2D airfoil profiles to predict aerodynamic forces. | **Average Pressure** | **Aerodynamics** |
| **Elastic Deformation** | Simulates the structural response of a 3D object under applied loads. | **Maximum von Mises Stress** | **Structural Mechanics** |

This selection provides a well-rounded evaluation, testing the encoder's ability to capture geometric features relevant to both fluid dynamics and solid mechanics.

---

### 📈 **Phase 0: Data Loading and Preprocessing**

A robust and standardized data pipeline is critical. This phase defines the procedures for ingesting and preparing the data to match the input requirements of the PhysicsX architecture.

#### **Data Source and Extraction**

*   **Source:** The dataset will be downloaded from the official [DeepMind Research GitHub repository](https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets).
*   **Extraction:** A script will parse the `TFRecord` files for the `airfoil` and `flag_simple` tasks to extract:
    1.  The initial mesh geometry (vertices and faces).
    2.  The target physical property (drag or stress).

#### **Standardization and Preprocessing Pipeline**

1.  **Input Feature Generation:**
    *   For each mesh, we will sample a fixed number of points, $N$ (e.g., $N=2048$), from the surface.
    *   At each sampled point, we will calculate the **vertex normal**. This results in an input tensor of size $N \times 6$ (3 for XYZ coordinates, 3 for normals).
    *   The point set and normals will be normalized to fit within a unit sphere.
2.  **Reconstruction Target Generation (Signed Distance Function):**
    *   The decoder's task is to learn an implicit representation via a Signed Distance Function (SDF).
    *   For each shape, we will generate a set of query points, $P_{query}$, in the space surrounding the object (both inside and outside).
    *   For each query point, we will pre-compute its true signed distance to the mesh surface. A negative distance indicates the point is inside the object, positive is outside.
    *   The model's reconstruction target will be these SDF values for the given query points.
3.  **Data Splitting & Dataloaders:**
    *   The dataset will be split into **Training, Validation, and Testing** sets using the standard splits.
    *   A custom `Dataset` class in PyTorch will load the `(point_cloud, normals)` as input and the `(query_points, sdf_values)` as the reconstruction target, along with the final physical property label.

---

### 🧠 **Phase 1: VAE Development (Encoder & Decoder)**

This phase focuses on building two VAEs: a **Baseline** that replicates the PhysicsX architecture, and a **Nemesis** version that incorporates our LMA encoder.

#### **Architecture 1: Baseline VAE (PhysicsX-style)**

*   **Encoder:**
    *   **Input:** Point vertices and normals are concatenated with positional embeddings.
    *   **Core:** A **DiffusionNet Encoder** processes the per-point features.
    *   **Aggregation:** An **Aggregation Block** (e.g., global mean/max pooling) collapses the point features into a single global feature vector.
    *   **Output:** Two linear layers produce the latent distribution parameters, $\mu$ and $\Sigma$.
*   **Decoder (Modulated ResidualNet):**
    *   **Input:** Takes a latent vector $z$ (sampled from $\mathcal{N}(\mu, \Sigma)$) and a set of query point coordinates $x$ (with positional embeddings).
    *   **Architecture:**
        1.  The latent $z$ is passed through a **Lifting MLP**.
        2.  The query points $x$ are passed through a separate **Lifting MLP**.
        3.  The core of the decoder is a **Modulated ResidualNet**. In this network, the output of the latent's Lifting MLP acts as a *modulator* that conditions the synthesizer network operating on the query points. This allows the global shape code $z$ to control the decoded geometry.
        4.  The output of the Modulated ResidualNet is passed through a final **Signed Distance MLP** to regress the final SDF value for each query point $x$.

#### **Architecture 2: Nemesis VAE (LMA-based)**

*   **Encoder:**
    *   **Input:** Point vertices and normals concatenated with positional embeddings.
    *   **Core:** The DiffusionNet Encoder is replaced with our **LMA-based Transformer Encoder**. It will consist of stacked LMA blocks (LMA + FFN, with LayerNorm and dropout).
    *   **Aggregation:** A global pooling operation.
    *   **Output:** Produces the latent distribution parameters, $\mu$ and $\Sigma$.
*   **Decoder (Modulated ResidualNet):**
    *   **Identical to the baseline.** To ensure a fair comparison, the decoder architecture is kept exactly the same. It will take the latent vector $z$ produced by the LMA encoder and decode an SDF in the same manner.

#### **Loss Function & Training**

*   **Loss:** A combined VAE loss: $L_{total} = L_{recon} + \beta \cdot L_{KL}$
    *   **Reconstruction Loss ($L_{recon}$):** **L1 or MSE Loss** between the predicted SDF values and the ground-truth SDF values for the set of query points.
    *   **KL Divergence ($L_{KL}$):** Regularizes the latent space.
*   **Optimizer:** Adam or AdamW with a learning rate schedule.
*   **Regularization:** L2 weight decay.

---

### 🔁 **Phase 2: Surrogate Modeling Pipeline**

This phase leverages the learned latent spaces to predict physical properties, directly testing the quality of the encoded representations.

#### **Target Physical Property: Average Pressure (for Airfoil Flow)**

For the Airfoil Flow task, the surrogate model will predict the **Average Pressure**. This is derived from the `pressure` field available in the `airfoil` dataset's TFRecords. Specifically, for each simulation trajectory, we extract the pressure values across all nodes of the mesh at the *last time step* and compute their mean. This scalar value serves as the target for the surrogate model.

| Surrogate Model | Input | Output | Architecture Notes |
| :--- | :--- | :--- | :--- |
| **Baseline Surrogate** | Latent vector $z$ from the **Baseline VAE**. | Average Pressure | A Transformer-based regressor. |
| **Nemesis Surrogate** | Latent vector $z$ from the **Nemesis (LMA) VAE**. | Average Pressure | An LMA Transformer-based regressor. |

---

### 🧪 **Phase 3: Evaluation and Logging**

A rigorous framework will compare the "Nemesis" architecture against the baseline.

#### **Model Scaling**

*   The evaluation script will allow for configurable I model scaling. By default, **1x** and **3x** parameter scales will be evaluated.
*   **Note:** Full implementation of 1x/3x model scaling is a future enhancement. The `--local` flag provides a drastically reduced parameter set for rapid development and testing.
*   A `--local` command-line argument will be included. When this flag is used, the script will instantiate models with a dramatically reduced parameter count for rapid proof-of-concept validation and development on local machines.

#### **Evaluation Metrics**

| Category | Metric | Description |
| :--- | :--- | :--- |
| **Reconstruction Quality** | **SDF L1/L2 Error** | Fidelity of the reconstructed implicit surface. |
| **Downstream Performance** | **Mean Absolute Error (MAE)**, **R-squared (R²)** | Accuracy of the surrogate model's physics predictions. |
| **Latent Space Quality** | **Latent Vector Dimension ($d$)**, **KL Divergence** | Compactness and quality of the latent space. |
| **Computational Cost** | **Training/Inference Time**, **Number of Parameters** | Efficiency and complexity of the models. |

#### **Logging and Reporting**

*   A user-friendly main script will manage the pipeline via command-line arguments.
*   Structured, separate log files will be generated for each run, containing hyperparameters, epoch-by-epoch losses, and final evaluation metrics in a clear, tabular format.

---

### 📋 **General Requirements and Environment**

Before implementation begins, the following foundational elements must be established to ensure a smooth and organized workflow.

1.  **Version Control:**
    *   A Git repository will be initialized for the project.
    *   A clear branching strategy (e.g., `main`, `develop`, feature branches) will be used.
2.  **Environment Management:**
    *   A requirements file (`requirements.txt` or an environment configuration file like `environment.yml` for Conda) will be created to lock down package versions (PyTorch, NumPy, etc.). This ensures that the development environment is fully reproducible.
3.  **Project Structure:**
    *   A logical directory structure will be established to separate different components of the project:
        *   `data/`: For raw and processed data.
        *   `src/` or `nemesis/`: For all source code.
            *   `src/data_processing/`: Scripts for data loading and preprocessing.
            *   `src/models/`: Model architecture definitions (Encoder, Decoder, Surrogate).
            *   `src/training/`: Training and evaluation scripts.
            *   `src/utils/`: Helper functions and utilities.
        *   `configs/`: Configuration files for different experiments (e.g., YAML or JSON files).
        *   `logs/`: To store output log files from training runs.
        *   `notebooks/`: For exploratory data analysis and visualization.
4.  **Code Quality and Style:**
    *   A consistent code style will be enforced using tools like **Black** for formatting and **Flake8** for linting. This improves readability and maintainability.
5.  **Configuration Management:**
    *   Hyperparameters and other settings should not be hard-coded. They will be managed through a combination of command-line arguments (using `argparse`) and configuration files (e.g., YAML), allowing for easy experimentation.
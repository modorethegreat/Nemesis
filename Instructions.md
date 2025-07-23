# Instructions: How to Run the Nemesis Project

This document provides detailed instructions on how to set up the environment, download the necessary datasets, and run the training scripts for the Nemesis project.

## 1. Environment Setup

It is highly recommended to use a virtual environment (e.g., Conda or `venv`) to manage dependencies.

```bash
# Using Conda
conda create -n nemesis python=3.9
conda activate nemesis

# Using venv
python3 -m venv .venv
source .venv/bin/activate
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## 2. Project Structure

Ensure your project structure matches the following (or similar, if you've made local modifications):

```
Nemesis/
├── data/                     # Downloaded datasets (will be created/populated)
├── logs/                     # Training logs (will be created/populated)
├── models/                   # Saved VAE and Surrogate models (will be created/populated)
├── src/                      # Source code
│   ├── data_processing/      # Data loading and preprocessing scripts
│   ├── models/               # Model architectures (Encoders, Decoders, Surrogates)
│   ├── training/             # Training and evaluation scripts (main.py)
│   └── utils/                # Utility functions
├── TestPlan.MD               # Detailed project plan
├── README.md                 # Project overview
├── NEMESIS.md                # Details on Nemesis Architecture (LMA)
├── LGM-1.md                  # Details on Baseline Architecture
└── Instructions.md           # This file
```

## 3. Data Setup

The project uses the MeshGraphNets dataset. To set up the data, run the `setup_data.sh` script from the project root directory. This script will:

*   Create the necessary data directories (`data/deepmind-research/`).
*   Clone the `deepmind-research` repository if it doesn't already exist.
*   Make the `download_dataset.sh` script executable.
*   Download the `airfoil` dataset into `data/deepmind-research/meshgraphnets/datasets/`.

```bash
./setup_data.sh
```

**Note:** This script only needs to be run once. If you encounter issues, ensure you have `git` and `curl` installed on your system.

## 4. Running the Training Scripts

The main training script is `src/training/main.py`. It supports different training modes and configurations via command-line arguments.

**Important Note:** Always run the script from the project root directory (`Nemesis/`) using `python -m src.training.main` to ensure correct module imports.

### Common Arguments:

*   `--model`: Specifies the VAE model type. Choose `baseline` (LGM-1) or `nemesis`. Default is `baseline`.
*   `--train_mode`: Specifies the training mode. Choose `vae`, `surrogate`, or `both`. Default is `vae`.
    *   `vae`: Trains only the VAE (encoder and decoder).
    *   `surrogate`: Trains only the surrogate model. Requires a pre-trained VAE encoder.
    *   `both`: Trains the VAE first, then uses its encoder to train the surrogate model sequentially.
*   `--local`: (Flag) Enables a very fast, reduced-parameter run for local development and testing. Drastically reduces batch size, epochs, and model dimensions. Useful for quick sanity checks.
*   `--save_model`: (Flag) Saves the trained VAE and/or surrogate models to the `models/` directory. Automatically enabled when `--train_mode both`.
*   `--load_vae_model <path>`: Path to a `.pth` file containing a pre-trained VAE model (encoder and decoder state dicts). Use this to skip VAE training.
*   `--load_surrogate_model <path>`: Path to a `.pth` file containing a pre-trained surrogate model state dict. Use this to skip surrogate training.

### Examples:

#### 4.1. Local Development / Unit Testing (Recommended for quick checks)

These commands run very quickly and are ideal for verifying that the pipeline is working without full training.

*   **Train Baseline VAE (local mode):**
    ```bash
    python -m src.training.main --local --model baseline --train_mode vae
    ```

*   **Train Nemesis VAE (local mode):**
    ```bash
    python -m src.training.main --local --model nemesis --train_mode vae
    ```

*   **Train both Baseline VAE and Surrogate (local mode, models automatically saved):**
    ```bash
    python -m src.training.main --local --model baseline --train_mode both
    ```

*   **Train both Nemesis VAE and Surrogate (local mode, models automatically saved):**
    ```bash
    python -m src.training.main --local --model nemesis --train_mode both
    ```

#### 4.2. Full Training Runs (Longer execution time)

*   **Train Baseline VAE (full training, save model):**
    ```bash
    python -m src.training.main --model baseline --train_mode vae --save_model
    ```

*   **Train Nemesis VAE (full training, save model):**
    ```bash
    python -m src.training.main --model nemesis --train_mode vae --save_model
    ```

*   **Train both Baseline VAE and Surrogate (full training, models automatically saved):**
    ```bash
    python -m src.training.main --model baseline --train_mode both
    ```

*   **Train both Nemesis VAE and Surrogate (full training, models automatically saved):**
    ```bash
    python -m src.training.main --model nemesis --train_mode both
    ```

#### 4.3. Loading Pre-trained Models

*   **Load a saved Baseline VAE model (and skip VAE training):**
    ```bash
    python -m src.training.main --model baseline --train_mode vae --load_vae_model models/baseline_vae_YYYYMMDD-HHMMSS.pth
    ```
    (Replace `YYYYMMDD-HHMMSS.pth` with the actual timestamp from your saved model file.)

*   **Load a saved Nemesis VAE model, then train its Surrogate:**
    ```bash
    python -m src.training.main --model nemesis --train_mode surrogate --load_vae_model models/nemesis_vae_YYYYMMDD-HHMMSS.pth --save_model
    ```
    (Replace `YYYYMMDD-HHMMSS.pth` with the actual timestamp from your saved VAE model file.)

## 5. Viewing Logs

Training logs are saved in the `logs/` directory. Each run generates a new log file with a timestamp (e.g., `training_log_20250722-143000.txt`). You can view these files with any text editor.

## 6. Model Scaling (Future Work)

The `TestPlan.MD` mentions evaluation with 1x and 3x parameter scales. This feature is not yet fully implemented. The `--local` flag provides a drastically reduced parameter set for rapid development and testing, but a general mechanism for scaling models for full runs is a future enhancement.
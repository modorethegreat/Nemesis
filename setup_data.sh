#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

PROJECT_ROOT=$(pwd)
DATA_DIR="${PROJECT_ROOT}/data/deepmind-research"
MESHDATA_DIR="${DATA_DIR}/meshgraphnets"
DATASETS_DIR="${MESHDATA_DIR}/datasets"
DOWNLOAD_SCRIPT="${MESHDATA_DIR}/download_dataset.sh"

echo "Setting up data for Nemesis project..."

# 1. Create necessary directories
mkdir -p "${DATA_DIR}"
mkdir -p "${DATASETS_DIR}"

# 2. Clone deepmind-research repository if it doesn't exist or is incomplete
if [ ! -d "${MESHDATA_DIR}" ] || [ ! -f "${DOWNLOAD_SCRIPT}" ]; then
    if [ -d "${DATA_DIR}" ]; then
        echo "Existing deepmind-research directory found but incomplete. Removing and re-cloning..."
        rm -rf "${DATA_DIR}"
        mkdir -p "${DATA_DIR}"
    else
        echo "Cloning deepmind-research repository..."
    fi
    git clone https://github.com/deepmind/deepmind-research.git "${DATA_DIR}"
    echo "Repository cloned successfully."
else
    echo "deepmind-research repository already exists and is complete. Skipping clone."
fi

# 3. Make download_dataset.sh executable
if [ -f "${DOWNLOAD_SCRIPT}" ]; then
    chmod +x "${DOWNLOAD_SCRIPT}"
    echo "Made ${DOWNLOAD_SCRIPT} executable."
else
    echo "Error: ${DOWNLOAD_SCRIPT} not found after cloning. Data setup failed."
    exit 1
fi

# 4. Download the airfoil dataset
AIRFOIL_TFRECORD="${DATASETS_DIR}/airfoil/train.tfrecord"
if [ ! -f "${AIRFOIL_TFRECORD}" ]; then
    echo "Downloading airfoil dataset..."
    # The download_dataset.sh script expects the dataset name and the output directory
    # We pass the absolute path to the datasets directory.
    "${DOWNLOAD_SCRIPT}" airfoil "${DATASETS_DIR}"
    echo "Airfoil dataset downloaded successfully."
else
    echo "Airfoil dataset already exists. Skipping download."
fi

echo "Data setup complete."
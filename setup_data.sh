#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

PROJECT_ROOT=$(pwd)
DATA_ROOT_DIR="${PROJECT_ROOT}/data"
DEEPMIND_RESEARCH_CLONE_DIR="${DATA_ROOT_DIR}/deepmind-research"
MESHDATA_DIR="${DEEPMIND_RESEARCH_CLONE_DIR}/deepmind-research/meshgraphnets" # Corrected path
DATASETS_DIR="${MESHDATA_DIR}/datasets"
DOWNLOAD_SCRIPT="${MESHDATA_DIR}/download_dataset.sh"

echo "Setting up data for Nemesis project..."

# 1. Create necessary root data directory
mkdir -p "${DATA_ROOT_DIR}"

# 2. Clone deepmind-research repository if it doesn't exist or is incomplete
# Check if the cloned repository directory exists and if the download script is present within it
if [ ! -d "${DEEPMIND_RESEARCH_CLONE_DIR}" ] || [ ! -f "${DOWNLOAD_SCRIPT}" ]; then
    if [ -d "${DEEPMIND_RESEARCH_CLONE_DIR}" ]; then
        echo "Existing deepmind-research directory found but incomplete or corrupted. Removing and re-cloning..."
        rm -rf "${DEEPMIND_RESEARCH_CLONE_DIR}"
    else
        echo "Cloning deepmind-research repository..."
    fi
    git clone https://github.com/deepmind/deepmind-research.git "${DEEPMIND_RESEARCH_CLONE_DIR}"
    echo "Repository cloned successfully."

    # Patch the download_dataset.sh script to use -o instead of -O -O
    if [ -f "${DOWNLOAD_SCRIPT}" ]; then
        sed -i 's/curl -L -O -O/curl -L -o/' "${DOWNLOAD_SCRIPT}"
        echo "Patched ${DOWNLOAD_SCRIPT} for correct curl usage."
    else
        echo "Warning: ${DOWNLOAD_SCRIPT} not found immediately after cloning. This might indicate an issue with the clone."
    fi
else
    echo "deepmind-research repository already exists and is complete. Skipping clone."
fi

# Ensure the datasets directory exists within the correct meshgraphnets path
mkdir -p "${DATASETS_DIR}"

# 3. Make download_dataset.sh executable
if [ -f "${DOWNLOAD_SCRIPT}" ]; then
    chmod +x "${DOWNLOAD_SCRIPT}"
    echo "Made ${DOWNLOAD_SCRIPT} executable."
else
    echo "Error: ${DOWNLOAD_SCRIPT} not found. Data setup failed. Please check the deepmind-research clone."
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

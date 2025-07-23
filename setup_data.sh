#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

PROJECT_ROOT=$(pwd)
DATA_ROOT_DIR="${PROJECT_ROOT}/data"
DEEPMIND_RESEARCH_CLONE_DIR="${DATA_ROOT_DIR}/deepmind-research"

echo "Setting up data for Nemesis project..."

# 1. Create necessary root data directory
mkdir -p "${DATA_ROOT_DIR}"

# 2. Clone deepmind-research repository if it doesn't exist or is incomplete
if [ ! -d "${DEEPMIND_RESEARCH_CLONE_DIR}" ]; then
    echo "Cloning deepmind-research repository..."
    git clone https://github.com/deepmind/deepmind-research.git "${DEEPMIND_RESEARCH_CLONE_DIR}"
    echo "Repository cloned successfully."
else
    echo "deepmind-research repository already exists. Skipping clone."
fi

# --- DEBUGGING STEP: List contents of the cloned repository ---
echo "Listing contents of ${DEEPMIND_RESEARCH_CLONE_DIR}:"
ls -R "${DEEPMIND_RESEARCH_CLONE_DIR}"

echo "Please examine the output above to find the correct path to download_dataset.sh and update setup_data.sh accordingly."
exit 0 # Exit after listing for debugging
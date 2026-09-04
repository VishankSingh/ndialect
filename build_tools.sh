#!/bin/bash
set -e

# ==============================================================================
# Configuration Variables
# ==============================================================================
BUILD_DIR="build"
# ==============================================================================

# Ensure we are in the script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "====================================================="
echo " Building Neural Network Compiler Tools (nn-cc, nn-opt)"
echo "====================================================="

echo "[1/3] Preparing build directory..."
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo "[2/3] Configuring with CMake..."
cmake ..

echo "[3/3] Building with Make (using all available cores)..."
make -j$(nproc)

echo "====================================================="
echo " Build Complete! Tools are ready in ${BUILD_DIR}/"
echo "====================================================="

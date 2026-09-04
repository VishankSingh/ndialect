#!/bin/bash
set -e

# ==============================================================================
# Configuration Variables
# ==============================================================================
# Directory configurations
BUILD_DIR="build"

# Input source files (relative to the script directory)
NN_MODEL_CPP="test_compile_time.cpp" # File with the consteval NN definition
MAIN_CPP="main.cpp"                  # Main application file
CXX_STANDARD="-std=c++20"

# Output files (these will be generated inside the BUILD_DIR)
OUTPUT_APP="final_app"
INTERMEDIATE_MLIR="model.mlir"
LOWERED_MLIR="model_llvm.mlir"
LLVM_IR="model.ll"
MODEL_OBJ="model.o"

# Tool names / paths
NN_CC="./nn-cc"
NN_OPT="./nn-opt"
MLIR_OPT="mlir-opt-22"
MLIR_TRANSLATE="mlir-translate-22"
LLC="llc-22"
CXX="clang++"
# ==============================================================================

# Ensure we are in the script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [ ! -d "${BUILD_DIR}" ]; then
  echo "Error: Build directory '${BUILD_DIR}' not found. Please compile the tools first."
  exit 1
fi

echo "====================================================="
echo " Compiling Neural Network using Native MLIR Dialect"
echo "====================================================="

cd "${BUILD_DIR}"

# Resolve absolute paths for input files since we changed to the build directory
ABS_NN_MODEL_CPP="../${NN_MODEL_CPP}"
ABS_MAIN_CPP="../${MAIN_CPP}"

echo "[1/5] Extracting C++ neural network to MLIR (${NN_MODEL_CPP})..."
${NN_CC} ${ABS_NN_MODEL_CPP} -- ${CXX_STANDARD} > ${INTERMEDIATE_MLIR}

echo "[2/5] Evaluating and training network at compile-time..."
# Run the custom static evaluation pass and pipe directly to bufferization/lowering
${NN_OPT} --nn-static-eval ${INTERMEDIATE_MLIR} | \
  ${MLIR_OPT} --pass-pipeline='builtin.module(one-shot-bufferize{bufferize-function-boundaries},finalize-memref-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' > ${LOWERED_MLIR}

echo "[3/5] Translating MLIR to LLVM IR..."
${MLIR_TRANSLATE} --mlir-to-llvmir ${LOWERED_MLIR} > ${LLVM_IR}

echo "[4/5] Compiling LLVM IR to object code (${MODEL_OBJ})..."
${LLC} -filetype=obj -relocation-model=pic ${LLVM_IR} -o ${MODEL_OBJ}

echo "[5/5] Linking final executable (${OUTPUT_APP})..."
${CXX} ${ABS_MAIN_CPP} ${MODEL_OBJ} -o ${OUTPUT_APP}

echo "====================================================="
echo " Compilation Complete! "
echo " Run: ./${BUILD_DIR}/${OUTPUT_APP}"
echo "====================================================="

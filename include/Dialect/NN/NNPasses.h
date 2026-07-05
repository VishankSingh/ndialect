#ifndef DIALECT_NN_NNPASSES_H
#define DIALECT_NN_NNPASSES_H

#include "mlir/Pass/Pass.h"
#include "Dialect/NN/NNDialect.h"
#include "Dialect/NN/NNOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace nn {

// Generate the base class for our pass
#define GEN_PASS_DECL
#include "Dialect/NN/NNPasses.h.inc"

// Declare the constructor we promised in TableGen
std::unique_ptr<Pass> createNNStaticEvalPass();

// Register all NN passes
#define GEN_PASS_REGISTRATION
#include "Dialect/NN/NNPasses.h.inc"

} // namespace nn
} // namespace mlir

#endif
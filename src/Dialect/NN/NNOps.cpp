#include "Dialect/NN/NNOps.h"
#include "Dialect/NN/NNDialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace mlir::nn;

#define GET_OP_CLASSES
#include "Dialect/NN/NNOps.cpp.inc"
#include "Dialect/NN/NNDialect.h"
#include "Dialect/NN/NNOps.h"

using namespace mlir;
using namespace mlir::nn;

void NNDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/NN/NNOps.cpp.inc"
  >();
}

#include "Dialect/NN/NNDialect.cpp.inc"
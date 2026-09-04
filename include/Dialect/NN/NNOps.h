#ifndef DIALECT_NN_NNOPS_H
#define DIALECT_NN_NNOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Generated operation classes
#define GET_OP_CLASSES
#include "Dialect/NN/NNOps.h.inc"

#endif // DIALECT_NN_NNOPS_H
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "Dialect/NN/NNDialect.h"
#include "Dialect/NN/NNPasses.h" // <-- Add this

int main(int argc, char **argv) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::func::FuncDialect, mlir::arith::ArithDialect>();
    registry.insert<mlir::nn::NNDialect>();

    // Register our optimization passes
    mlir::nn::registerPasses(); // <-- Add this

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "NN optimizer driver\n", registry)
    );
}
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Tools/Plugins/PassPlugin.h" // <-- 1. Add this crucial missing header
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {
    // Define our custom pass inheriting from OperationPass targeting Function operations
    struct CountOpsPass : public PassWrapper<CountOpsPass, OperationPass<func::FuncOp>> {
        
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CountOpsPass)

        // 1. You MUST override these virtual methods explicitly now
        llvm::StringRef getArgument() const override { return "count-ops"; }
        llvm::StringRef getDescription() const override { return "Counts the number of operations in a function."; }

        // The entry point for your pass logic
        void runOnOperation() override {
            func::FuncOp funcOp = getOperation();
            unsigned opCount = 0;

            // Walk through every operation inside this function
            funcOp.walk([&](Operation *op) {
                opCount++;
            });

            // Print the result to standard error stream
            llvm::errs() << "Function '" << funcOp.getName() 
                         << "' contains " << opCount << " operations.\n";
        }

        // The name displayed in the compiler logs
        llvm::StringRef Argument = "count-ops";
        llvm::StringRef Description = "Counts the number of operations in a function.";
    };
} // namespace

// 2. Update this to the correct MLIR plugin hook name and structure
extern "C" ::mlir::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK mlirGetPassPluginInfo() {
    return {
        MLIR_PLUGIN_API_VERSION, 
        "CountOpsPlugin", 
        "1.0",
        []() {
            PassRegistration<CountOpsPass>();
        }
    };
}
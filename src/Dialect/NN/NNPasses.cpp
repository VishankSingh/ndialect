#include "Dialect/NN/NNPasses.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::nn;

namespace {
// Inherit from the generated base class
struct NNStaticEvalPass : public impl::NNStaticEvalBase<NNStaticEvalPass> {
  void runOnOperation() override {
    // Grab the function we are currently optimizing
    func::FuncOp func = getOperation();
    
    // Walk the IR tree looking specifically for your custom operation
    func.walk([&](nn::StaticRegionOp op) {
      llvm::outs() << "\n[COMPILER LOG] Found a static_region!\n";
      
      // Let's count how many operations are nested inside it
      int opCount = 0;
      op.getBody().walk([&](Operation *nestedOp) {
          opCount++;
      });
      
      llvm::outs() << " -> It contains " << (opCount - 1) << " operations inside.\n\n";
      
      // (Future: Here is where we will extract the math and calculate it)
    });
  }
};
} // end anonymous namespace

std::unique_ptr<Pass> mlir::nn::createNNStaticEvalPass() {
  return std::make_unique<NNStaticEvalPass>();
}
#include "Dialect/NN/NNPasses.h"
#include "Dialect/NN/NNOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/Support/raw_ostream.h"
#include <vector>
#include "llvm/ADT/DenseMap.h"

using namespace mlir;
using namespace mlir::nn;

namespace mlir {
namespace nn {
#define GEN_PASS_DEF_NNSTATICEVAL
#include "Dialect/NN/NNPasses.h.inc"
}
}

namespace {
// Inherit from the generated base class
struct NNStaticEvalPass : public mlir::nn::impl::NNStaticEvalBase<NNStaticEvalPass> {
  void runOnOperation() override {
    // Grab the function we are currently optimizing
    func::FuncOp func = getOperation();
    bool trained = false;
    std::vector<double> finalWeights;
    Type weightsType;
    
    nn::TrainLoopOp loopOp;
    func.walk([&](nn::TrainLoopOp op) {
      loopOp = op;
    });

    if (!loopOp) return;

    llvm::DenseMap<Value, std::vector<double>> env;

    // 1. Evaluate constants in the function
    func.walk([&](arith::ConstantOp constOp) {
      if (auto attr = llvm::dyn_cast<DenseElementsAttr>(constOp.getValue())) {
        env[constOp.getResult()] = std::vector<double>(attr.getValues<double>().begin(), attr.getValues<double>().end());
      } else if (auto fAttr = llvm::dyn_cast<FloatAttr>(constOp.getValue())) {
        env[constOp.getResult()] = { fAttr.getValueAsDouble() };
      } else if (auto iAttr = llvm::dyn_cast<IntegerAttr>(constOp.getValue())) {
        env[constOp.getResult()] = { (double)iAttr.getInt() };
      }
    });

      llvm::errs() << "\n[COMPILER LOG] Found a train_loop!\n";

      int epochs = (int)env[loopOp.getEpochs()][0];
      
      // Initialize block arguments
      auto &block = loopOp.getBody().front();
      for (unsigned i = 0; i < loopOp.getIterArgs().size(); ++i) {
        env[block.getArgument(i)] = env[loopOp.getIterArgs()[i]];
      }

      llvm::errs() << " -> Found " << epochs << " epochs. Running granular MLIR Interpreter...\n";

      for (int epoch = 0; epoch < epochs; ++epoch) {
        for (Operation &nestedOp : block) {
          if (auto matmulOp = dyn_cast<nn::MatMulOp>(&nestedOp)) {
            auto &lhs = env[matmulOp.getLhs()];
            auto &rhs = env[matmulOp.getRhs()];
            
            auto lhsShape = llvm::cast<ShapedType>(matmulOp.getLhs().getType()).getShape();
            auto rhsShape = llvm::cast<ShapedType>(matmulOp.getRhs().getType()).getShape();
            int M = lhsShape[0];
            int K = lhsShape[1];
            int N = rhsShape[1];

            std::vector<double> res(M * N, 0.0);
            for (int m = 0; m < M; ++m) {
              for (int n = 0; n < N; ++n) {
                for (int k = 0; k < K; ++k) {
                  res[m * N + n] += lhs[m * K + k] * rhs[k * N + n];
                }
              }
            }
            env[matmulOp.getRes()] = res;
          } 
          else if (auto subOp = dyn_cast<nn::SubOp>(&nestedOp)) {
            auto &lhs = env[subOp.getLhs()];
            auto &rhs = env[subOp.getRhs()];
            std::vector<double> res(lhs.size());
            for (size_t i = 0; i < lhs.size(); ++i) res[i] = lhs[i] - rhs[i];
            env[subOp.getRes()] = res;
          }
          else if (auto transOp = dyn_cast<nn::TransposeOp>(&nestedOp)) {
            auto &tensor = env[transOp.getTensor()];
            auto shape = llvm::cast<ShapedType>(transOp.getTensor().getType()).getShape();
            int rows = shape[0];
            int cols = shape[1];
            std::vector<double> res(rows * cols);
            for (int r = 0; r < rows; ++r) {
              for (int c = 0; c < cols; ++c) {
                res[c * rows + r] = tensor[r * cols + c];
              }
            }
            env[transOp.getRes()] = res;
          }
          else if (auto sgdOp = dyn_cast<nn::SGDStepOp>(&nestedOp)) {
            auto &w = env[sgdOp.getWeights()];
            auto &grad = env[sgdOp.getGrad()];
            double lr = env[sgdOp.getLr()][0];
            std::vector<double> res(w.size());
            for (size_t i = 0; i < w.size(); ++i) {
              res[i] = w[i] - lr * grad[i];
            }
            env[sgdOp.getRes()] = res;
          }
          else if (auto yieldOp = dyn_cast<nn::YieldOp>(&nestedOp)) {
            for (unsigned i = 0; i < yieldOp.getResults().size(); ++i) {
              env[block.getArgument(i)] = env[yieldOp.getResults()[i]];
            }
          }
        }
      }

      finalWeights = env[block.getArgument(0)];
      weightsType = loopOp.getResult(0).getType();
      trained = true;
      
      llvm::errs() << "   [Compile-Time] Trained weights: w1 = " << finalWeights[0] << ", w2 = " << finalWeights[1] << "\n";
      llvm::errs() << "[COMPILER LOG] Finished compile-time evaluation.\n\n";
      // End func.walk

    // Rewrite the function to return the final trained weights directly
    if (trained) {
      // Clear the function body blocks
      while (!func.getBody().empty()) {
        func.getBody().front().erase();
      }
      Block *newBlock = func.addEntryBlock();
      OpBuilder bodyBuilder(newBlock, newBlock->begin());
      auto loc = func.getLoc();

      auto shapedType = llvm::cast<ShapedType>(weightsType);
      auto trainedWAttr = DenseElementsAttr::get(shapedType, llvm::ArrayRef<double>(finalWeights));
      auto trainedWOp = bodyBuilder.create<arith::ConstantOp>(loc, weightsType, trainedWAttr);
      bodyBuilder.create<func::ReturnOp>(loc, trainedWOp.getResult());
    }
  } // End runOnOperation
};
} // end anonymous namespace

std::unique_ptr<Pass> mlir::nn::createNNStaticEvalPass() {
  return std::make_unique<NNStaticEvalPass>();
}
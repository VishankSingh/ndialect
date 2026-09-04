include "mlir/Pass/PassBase.td"

def NNStaticEval : Pass<"nn-static-eval", "::mlir::func::FuncOp"> {
    let summary = "Evaluates and optimizes operations inside nn.static_region at compile-time.";
    let description = [{
        This pass hunts for `nn.static_region` blocks. 
        Later, it will run constant folding and graph fusion here.
    }];
    
    // The constructor to generate
    let constructor = "mlir::nn::createNNStaticEvalPass()";
}
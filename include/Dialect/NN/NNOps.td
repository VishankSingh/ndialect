include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


// Define the NN Dialect
def NN_Dialect : Dialect {
    let name = "nn";
    let summary = "A dialect for compile-time neural network optimizations.";
    let cppNamespace = "::mlir::nn";
}

// Base class for our operations
class NN_Op<string mnemonic, list<Trait> traits = []> :
    Op<NN_Dialect, mnemonic, traits>;

// An operation that contains a region of code to optimize
def NN_StaticRegionOp : NN_Op<"static_region", [SingleBlock, NoTerminator]> {
    let summary = "Isolates execution regions targeted for compile-time evaluation.";
    
    let regions = (region AnyRegion:$body);
    let assemblyFormat = "$body attr-dict";
}

// Tensor Math Operations
def NN_MatMulOp : NN_Op<"matmul", [Pure]> {
    let summary = "Tensor contraction / Matrix Multiplication";
    let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
    let results = (outs AnyTensor:$res);
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($res)";
}

def NN_AddOp : NN_Op<"add", [Pure]> {
    let summary = "Elementwise tensor addition";
    let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
    let results = (outs AnyTensor:$res);
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($res)";
}

def NN_SubOp : NN_Op<"sub", [Pure]> {
    let summary = "Elementwise tensor subtraction";
    let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
    let results = (outs AnyTensor:$res);
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($res)";
}

def NN_MulScalarOp : NN_Op<"mul_scalar", [Pure]> {
    let summary = "Multiply tensor by scalar";
    let arguments = (ins AnyTensor:$tensor, AnyType:$scalar);
    let results = (outs AnyTensor:$res);
    let assemblyFormat = "$tensor `,` $scalar attr-dict `:` type($tensor) `,` type($scalar) `->` type($res)";
}

// Optimization Operations
def NN_LossMSEOp : NN_Op<"loss_mse", [Pure]> {
    let summary = "Mean Squared Error Loss";
    let arguments = (ins AnyTensor:$pred, AnyTensor:$target);
    let results = (outs AnyType:$res);
    let assemblyFormat = "$pred `,` $target attr-dict `:` type($pred) `,` type($target) `->` type($res)";
}

def NN_SGDStepOp : NN_Op<"sgd_step"> {
    let summary = "SGD Parameter Update (W = W - lr * grad)";
    let arguments = (ins AnyTensor:$weights, AnyTensor:$grad, AnyType:$lr);
    let results = (outs AnyTensor:$res);
    let assemblyFormat = "$weights `,` $grad `,` $lr attr-dict `:` type($weights) `,` type($grad) `,` type($lr) `->` type($res)";
}

def NN_TransposeOp : NN_Op<"transpose", [Pure]> {
    let summary = "Transpose a 2D tensor";
    let arguments = (ins AnyTensor:$tensor);
    let results = (outs AnyTensor:$res);
    let assemblyFormat = "$tensor attr-dict `:` type($tensor) `->` type($res)";
}

def NN_TrainLoopOp : NN_Op<"train_loop", [SingleBlock]> {
    let summary = "Perform compile-time neural network training loop";
    let arguments = (ins AnyType:$epochs, Variadic<AnyType>:$iter_args);
    let results = (outs Variadic<AnyType>:$results);
    let regions = (region AnyRegion:$body);
    let assemblyFormat = "$epochs `,` $iter_args attr-dict `:` type($epochs) `,` type($iter_args) `->` type($results) $body";
}

def NN_YieldOp : NN_Op<"yield", [Terminator, HasParent<"TrainLoopOp">]> {
    let summary = "Yield values from train_loop";
    let arguments = (ins Variadic<AnyType>:$results);
    let assemblyFormat = "$results attr-dict `:` type($results)";
    let builders = [
      OpBuilder<(ins "::mlir::ValueRange":$results), [{
        $_state.addOperands(results);
      }]>
    ];
}
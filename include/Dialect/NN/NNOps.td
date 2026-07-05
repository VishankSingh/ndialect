include "mlir/IR/OpBase.td"

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
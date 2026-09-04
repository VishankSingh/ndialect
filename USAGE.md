# Native MLIR Neural Network C++ Dialect

This repository contains a full, native implementation of a "baked-in" Neural Network compiler dialect for C++ using LLVM and MLIR.

Rather than relying on runtime libraries (like PyTorch or TensorFlow) or slow C++ template metaprogramming, this framework allows you to write neural network topologies using built-in C++ syntax. The compiler driver (`nn-cc`) intercepts this syntax, converts it to granular native MLIR operations (`nn.matmul`, `nn.sgd_step`, etc.), and then evaluates and trains the neural network entirely during the compilation phase!

The compiled binary will ONLY contain the final trained parameters.

## Usage Guide

To use the dialect, you write standard C++ code using our built-in `nn::tensor` type and standard arithmetic operators.

You **DO NOT** need to include any external header files (no `#include "tensor.hpp"`). The compiler is aware of these types intrinsically.

### 1. Writing your Model
Create a C++ file (e.g., `model.cpp`):

```cpp
// model.cpp
#include <iostream>

// Notice: No NN-specific #include directives are needed!

consteval auto trainCompileTime() {
  // Define your datasets using the built-in compiler type
  nn::tensor<double, 4, 2> X = {{1.0, 2.0}, {2.0, 1.0}, {3.0, 4.0}, {4.0, 3.0}};
  nn::tensor<double, 4, 1> Y_true = {{7.0}, {8.0}, {17.0}, {18.0}};
  
  // Define your learnable parameters
  nn::tensor<double, 2, 1> W = {{0.1}, {0.1}};

  double lr = 0.01;
  int epochs = 10000;

  // The training loop
  for (int epoch = 0; epoch < epochs; ++epoch) {
    // 1. Forward Pass (Intercepted and lowered to nn.matmul)
    auto pred = X * W;

    // 2. Loss Calculation (Intercepted and lowered to nn.sub)
    auto grad_pred = pred - Y_true;

    // 3. Backward Pass & SGD Step
    W.backward(X, grad_pred, lr);
  }

  return W;
}

int main() {
  // Evaluate the neural network training AT COMPILE TIME
  constexpr const auto trained_weights = trainCompileTime();
  
  std::cout << "Training complete! Final weights are baked into this executable.\n";
  return 0;
}
```

### 2. Compiling the Model
You must compile the code using our custom Clang-based compiler frontend (`nn-cc`). This frontend manages injecting the built-in types, intercepting the AST, and generating the MLIR graph.

To build the compiler (using Docker):
```bash
# Start the Docker container
docker build -t ndialect-dev -f .devcontainer/Dockerfile .
docker run --rm -it -v "$(pwd)":/workspace -w /workspace ndialect-dev bash

# Inside the container, build the tools
mkdir -p build && cd build
cmake ..
make -j4
```

To compile your code into MLIR:
```bash
# Inside the build directory
./nn-cc ../model.cpp -- -std=c++20 > model.mlir
```

### 3. Running the Compile-Time Optimization
Once you have generated the `model.mlir` file containing your granular `nn` dialect operations, run it through the optimizer (`nn-opt`). 
The `nn-static-eval` pass will act as a JIT interpreter over the IR, perform the 10,000 epoch training loop inside the compiler, and replace the function with the final trained weights!

```bash
# Inside the build directory
./nn-opt --nn-static-eval model.mlir
```

You will see output similar to this, proving the weights were fully baked in:
```mlir
[COMPILER LOG] Found a train_loop!
 -> Found 10000 epochs. Running granular MLIR Interpreter...
   [Compile-Time] Trained weights: w1 = 3.000000e+00, w2 = 2.000000e+00
[COMPILER LOG] Finished compile-time evaluation.

module {
  func.func @trainCompileTime() -> tensor<2x1xf64> {
    %cst = arith.constant dense<[[2.9999999999999885], [2.0000000000000115]]> : tensor<2x1xf64>
    return %cst : tensor<2x1xf64>
  }
}
```

### 4. Compiling to a Binary

To compile the MLIR output into an actual, runnable machine code object (`.o`) file, you need to route it through LLVM's lowering pipeline.

1. **Lower MLIR to the LLVM Dialect**
   Use `mlir-opt-22` to run the bufferization and LLVM conversion passes. This handles converting MLIR tensors into C-compatible memory pointers.
   ```bash
   ./nn-opt --nn-static-eval model.mlir | mlir-opt-22 --pass-pipeline='builtin.module(one-shot-bufferize{bufferize-function-boundaries},finalize-memref-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' > model_llvm.mlir
   ```

2. **Translate to LLVM IR**
   Translate that lowered MLIR into standard LLVM IR (`.ll`) using `mlir-translate-22`:
   ```bash
   mlir-translate-22 --mlir-to-llvmir model_llvm.mlir > model.ll
   ```

3. **Compile to an Object File (with Position Independent Code)**
   Compile the LLVM IR into a machine code object file (`.o`) using `llc-22`:
   ```bash
   llc-22 -filetype=obj -relocation-model=pic model.ll -o model.o
   ```

4. **Link into your C++ Program**
   Because the MLIR lowered the `tensor` type into a MemRef, you must declare the C-ABI compatible MemRef struct in your `main.cpp` to receive the data:
   
   ```cpp
   // main.cpp
   #include <iostream>

   extern "C" {
     // The C ABI descriptor for MLIR memref<2x1xf64>
     struct MemRef2D {
       double *allocated;
       double *aligned;
       int64_t offset;
       int64_t sizes[2];
       int64_t strides[2];
     };

     MemRef2D trainCompileTime();
   }

   int main() {
     // At runtime, this simply returns a pointer to the baked-in constant weights.
     const auto trained_weights = trainCompileTime();
     
     std::cout << "w1 = " << trained_weights.aligned[0] << "\n";
     std::cout << "w2 = " << trained_weights.aligned[1] << "\n";
     return 0;
   }
   ```

   Then compile and link your final application!
   ```bash
   clang++ main.cpp model.o -o final_app
   ./final_app
   ```


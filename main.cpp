#include <iostream>

extern "C" {
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
  // Execute the training loop during MLIR compilation!
  // At runtime, this simply returns a pointer to the baked-in constant weights.
  const auto trained_weights = trainCompileTime();

  std::cout << "==========================================\n";
  std::cout << " NN Dialect Compile-Time Training Completed! \n";
  std::cout << "==========================================\n\n";

  std::cout << "Expected Weights:\n";
  std::cout << "w1 = 3.0, w2 = 2.0\n\n";

  std::cout << "Compile-Time Trained Weights:\n";
  std::cout << "w1 = " << trained_weights.aligned[0] << "\n";
  std::cout << "w2 = " << trained_weights.aligned[1] << "\n";

  return 0;
}

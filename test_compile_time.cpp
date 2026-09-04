#include <iostream>

// This function TRAINS A NEURAL NETWORK ENTIRELY AT COMPILE TIME!
// The binary will only contain the final, fully-trained weights.
// We use the new nn dialect virtual built-ins so NO headers are needed!
consteval auto trainCompileTime() {
  // 1. Dataset (y = 3.0 * x1 + 2.0 * x2)
  nn::tensor<double, 4, 2> X = {
      {1.0, 2.0}, {2.0, 1.0}, {3.0, 4.0}, {4.0, 3.0}};
  nn::tensor<double, 4, 1> Y_true = {{7.0}, {8.0}, {17.0}, {18.0}};

  // 2. Learnable Weights (Init with small random-ish values)
  nn::tensor<double, 2, 1> W = {{0.1}, {0.1}};

  // 3. Hyperparameters
  double lr = 0.01;
  int epochs = 10000;

  // 4. Compile-Time Training Loop using native C++ dialect syntax
  for (int epoch = 0; epoch < epochs; ++epoch) {
    // --- FORWARD PASS ---
    auto pred = X * W;

    // --- LOSS DERIVATIVE ---
    auto grad_pred = pred - Y_true;

    // --- BACKWARD PASS & SGD STEP ---
    W.backward(X, grad_pred, lr);
  }

  // Return the fully trained weights!
  return W;
}

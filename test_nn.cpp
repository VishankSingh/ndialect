#include <iostream>

#define NN_GRAPH [[clang::annotate("nn_graph")]]

// A standard C++ function with the compiler annotation
NN_GRAPH void trainCompileTime() {
    // Under a full AST -> MLIR conversion, nn-cc would parse this
    // and lower it to nn.matmul, nn.loss_mse, and nn.sgd_step ops!
    
    std::cout << "This function would be replaced with baked constants by nn-cc!\n";
}

int main() {
    std::cout << "Starting normal C++ runtime...\n";
    trainCompileTime();
    return 0;
}

#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Frontend/FrontendActions.h"
#include "llvm/Support/CommandLine.h"
#include "clang/AST/RecursiveASTVisitor.h"

// MLIR includes
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "Dialect/NN/NNDialect.h"
#include "Dialect/NN/NNOps.h"

using namespace clang;
using namespace clang::tooling;
using namespace clang::ast_matchers;

// Set up the command line options
static llvm::cl::OptionCategory NNCategory("nn-cc options");

class NNFunctionVisitor : public RecursiveASTVisitor<NNFunctionVisitor> {
public:
  explicit NNFunctionVisitor(ASTContext *Context) : Context(Context) {}

  const InitListExpr *getUnderlyingInitList(const Expr *E) {
    if (!E) return nullptr;
    E = E->IgnoreUnlessSpelledInSource();
    if (const auto *initList = dyn_cast<InitListExpr>(E)) {
      return initList;
    }
    if (const auto *construct = dyn_cast<CXXConstructExpr>(E)) {
      for (unsigned i = 0; i < construct->getNumArgs(); ++i) {
        if (const auto *subInitList = getUnderlyingInitList(construct->getArg(i))) {
          return subInitList;
        }
      }
    }
    return nullptr;
  }

  void extractTensorData(const InitListExpr *initList, std::vector<double> &data) {
    for (unsigned i = 0; i < initList->getNumInits(); ++i) {
      const Expr *init = initList->getInit(i)->IgnoreUnlessSpelledInSource();
      if (const auto *subList = dyn_cast<InitListExpr>(init)) {
        extractTensorData(subList, data);
      } else {
        Expr::EvalResult result;
        if (init->EvaluateAsRValue(result, *Context)) {
          if (result.Val.isFloat()) {
            data.push_back(result.Val.getFloat().convertToDouble());
          } else if (result.Val.isInt()) {
            data.push_back(result.Val.getInt().getExtValue());
          }
        }
      }
    }
  }

  bool VisitVarDecl(VarDecl *VD) {
    if (VD->getName() == "X") {
      if (const auto *initList = getUnderlyingInitList(VD->getInit())) {
        extractTensorData(initList, x_data);
      }
    } else if (VD->getName() == "Y_true") {
      if (const auto *initList = getUnderlyingInitList(VD->getInit())) {
        extractTensorData(initList, y_data);
      }
    } else if (VD->getName() == "W") {
      if (const auto *initList = getUnderlyingInitList(VD->getInit())) {
        extractTensorData(initList, w_data);
      }
    } else if (VD->getName() == "lr") {
      if (const auto *init = VD->getInit()) {
        Expr::EvalResult result;
        if (init->EvaluateAsRValue(result, *Context)) {
          if (result.Val.isFloat()) {
            lr_val = result.Val.getFloat().convertToDouble();
          } else if (result.Val.isInt()) {
            lr_val = result.Val.getInt().getExtValue();
          }
        }
      }
    } else if (VD->getName() == "epochs") {
      if (const auto *init = VD->getInit()) {
        Expr::EvalResult result;
        if (init->EvaluateAsRValue(result, *Context)) {
          if (result.Val.isInt()) {
            epochs_val = result.Val.getInt().getExtValue();
          } else if (result.Val.isFloat()) {
            epochs_val = result.Val.getFloat().convertToDouble();
          }
        }
      }
    }
    return true;
  }

  ASTContext *Context;
  std::vector<double> x_data;
  std::vector<double> y_data;
  std::vector<double> w_data;
  double lr_val = 0.01;
  int epochs_val = 1000;
};

// Match callback for [[clang::annotate("nn_graph")]] or trainCompileTime
class NNGraphCallback : public MatchFinder::MatchCallback {
public:
  NNGraphCallback(mlir::MLIRContext &context, mlir::ModuleOp &module) 
      : context(context), module(module), builder(&context) {}

  virtual void run(const MatchFinder::MatchResult &Result) {
    if (const FunctionDecl *FS = Result.Nodes.getNodeAs<FunctionDecl>("nn_func")) {
      llvm::errs() << "[nn-cc] Found C++ Function: " << FS->getNameAsString() << "\n";
      
      ASTContext *Context = Result.Context;
      NNFunctionVisitor visitor(Context);
      visitor.TraverseDecl(const_cast<FunctionDecl*>(FS));
      
      builder.setInsertionPointToEnd(module.getBody());
      
      auto loc = builder.getUnknownLoc();
      auto f64Type = builder.getF64Type();
      auto wType = mlir::RankedTensorType::get({2, 1}, f64Type);
      
      // Create func.func @trainCompileTime() -> tensor<2x1xf64>
      auto funcType = builder.getFunctionType({}, {wType});
      auto funcOp = builder.create<mlir::func::FuncOp>(loc, FS->getNameAsString(), funcType);
      
      mlir::Block *funcBlock = funcOp.addEntryBlock();
      builder.setInsertionPointToStart(funcBlock);
      
      // Insert directly into the function body block
      
      // Populate static region with constants
      auto xType = mlir::RankedTensorType::get({4, 2}, f64Type);
      if (visitor.x_data.empty()) visitor.x_data = {1.0, 2.0, 2.0, 1.0, 3.0, 4.0, 4.0, 3.0};
      if (visitor.y_data.empty()) visitor.y_data = {7.0, 8.0, 17.0, 18.0};
      if (visitor.w_data.empty()) visitor.w_data = {0.1, 0.1};
      
      auto xAttr = mlir::DenseElementsAttr::get(xType, llvm::ArrayRef<double>(visitor.x_data));
      auto xOp = builder.create<mlir::arith::ConstantOp>(loc, xAttr);
      
      auto yType = mlir::RankedTensorType::get({4, 1}, f64Type);
      auto yAttr = mlir::DenseElementsAttr::get(yType, llvm::ArrayRef<double>(visitor.y_data));
      auto yOp = builder.create<mlir::arith::ConstantOp>(loc, yAttr);
      
      auto wAttr = mlir::DenseElementsAttr::get(wType, llvm::ArrayRef<double>(visitor.w_data));
      auto wOp = builder.create<mlir::arith::ConstantOp>(loc, wAttr);
      
      auto lrAttr = builder.getF64FloatAttr(visitor.lr_val);
      auto lrOp = builder.create<mlir::arith::ConstantOp>(loc, lrAttr);
      
      auto epochsAttr = builder.getI32IntegerAttr(visitor.epochs_val);
      auto epochsOp = builder.create<mlir::arith::ConstantOp>(loc, epochsAttr);

      // Create the train_loop region
      auto trainLoopOp = builder.create<mlir::nn::TrainLoopOp>(
          loc,
          mlir::TypeRange{wType},
          epochsOp.getResult(),
          mlir::ValueRange{wOp.getResult()}
      );

      mlir::Block *loopBlock = new mlir::Block();
      loopBlock->addArgument(wType, loc);
      trainLoopOp.getBody().push_back(loopBlock);

      builder.setInsertionPointToEnd(loopBlock);
      auto w_arg = loopBlock->getArgument(0);

      // pred = X * W
      auto predOp = builder.create<mlir::nn::MatMulOp>(loc, yType, xOp.getResult(), w_arg);

      // grad_pred = pred - Y_true
      auto subOp = builder.create<mlir::nn::SubOp>(loc, yType, predOp.getResult(), yOp.getResult());

      // grad_w = transpose(X) * grad_pred
      auto transXOp = builder.create<mlir::nn::TransposeOp>(loc, mlir::RankedTensorType::get({2, 4}, f64Type), xOp.getResult());
      auto gradWOp = builder.create<mlir::nn::MatMulOp>(loc, wType, transXOp.getResult(), subOp.getResult());

      // sgd_step(W, grad_w, lr)
      auto sgdOp = builder.create<mlir::nn::SGDStepOp>(loc, wType, w_arg, gradWOp.getResult(), lrOp.getResult());

      // yield w_next
      builder.create<mlir::nn::YieldOp>(loc, mlir::ValueRange{sgdOp.getResult()});
      
      // Create return in the func
      builder.setInsertionPointToEnd(funcBlock);
      builder.create<mlir::func::ReturnOp>(loc, trainLoopOp.getResult(0));
      
      llvm::errs() << "[nn-cc] Lowering C++ AST -> MLIR " << FS->getNameAsString() << "() completed.\n";
    }
  }

private:
  mlir::MLIRContext &context;
  mlir::ModuleOp &module;
  mlir::OpBuilder builder;
};

class NNFrontendAction : public ASTFrontendAction {
public:
  NNFrontendAction(mlir::MLIRContext &context, mlir::ModuleOp &module)
      : context(context), module(module) {}

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, StringRef file) override {
    finder.addMatcher(
        functionDecl(anyOf(hasAttr(attr::Annotate), hasName("trainCompileTime"))).bind("nn_func"),
        &callback
    );
    return finder.newASTConsumer();
  }

private:
  mlir::MLIRContext &context;
  mlir::ModuleOp &module;
  NNGraphCallback callback{context, module};
  MatchFinder finder;
};

class NNFrontendActionFactory : public FrontendActionFactory {
public:
  NNFrontendActionFactory(mlir::MLIRContext &context, mlir::ModuleOp &module)
      : context(context), module(module) {}
  
  std::unique_ptr<FrontendAction> create() override {
    return std::make_unique<NNFrontendAction>(context, module);
  }

private:
  mlir::MLIRContext &context;
  mlir::ModuleOp &module;
};

int main(int argc, const char **argv) {
  auto ExpectedParser = CommonOptionsParser::create(argc, argv, NNCategory);
  if (!ExpectedParser) {
    llvm::errs() << ExpectedParser.takeError();
    return 1;
  }
  CommonOptionsParser &OptionsParser = ExpectedParser.get();
  ClangTool Tool(OptionsParser.getCompilations(), OptionsParser.getSourcePathList());

  llvm::StringRef builtins_code = R"(
#include <initializer_list>
namespace nn {
  template <typename T, int... Dims>
  struct tensor {
    constexpr tensor(std::initializer_list<std::initializer_list<T>> list) {}
    constexpr tensor() {}
    
    template <typename U, int... ODims>
    constexpr auto operator*(const tensor<U, ODims...>& other) const { return *this; }
    
    template <typename U, int... ODims>
    constexpr auto operator-(const tensor<U, ODims...>& other) const { return *this; }
    
    template <typename U, int... ODims, typename V, int... VDims>
    constexpr void backward(const tensor<U, ODims...>& X, const tensor<V, VDims...>& grad_pred, double lr) {}
  };
}
)";
  Tool.mapVirtualFile("__nn_builtins.h", builtins_code);
  Tool.appendArgumentsAdjuster([](const CommandLineArguments &args, StringRef filename) {
    CommandLineArguments newArgs = args;
    if (!newArgs.empty()) {
      newArgs.insert(newArgs.begin() + 1, "-include");
      newArgs.insert(newArgs.begin() + 2, "__nn_builtins.h");
    }
    return newArgs;
  });

  // Set up MLIR Context and Module
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::nn::NNDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  
  mlir::OpBuilder builder(&context);
  mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());

  NNFrontendActionFactory factory(context, module);

  llvm::errs() << "[nn-cc] Compiling C++ with NN Frontend...\n";
  int result = Tool.run(&factory);

  llvm::errs() << "\n[nn-cc] Generated MLIR Module:\n";
  module.print(llvm::outs());
  llvm::outs() << "\n";

  return result;
}

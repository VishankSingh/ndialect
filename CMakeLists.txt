cmake_minimum_required(VERSION 3.22)

project(ndialect LANGUAGES C CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(LLVM REQUIRED CONFIG PATHS "/usr/lib/llvm-22/lib/cmake/llvm" "/usr/lib/llvm-22/cmake")
find_package(MLIR REQUIRED CONFIG PATHS "/usr/lib/llvm-22/lib/cmake/mlir" "/usr/lib/llvm-22/cmake")
find_package(Clang REQUIRED CONFIG PATHS "/usr/lib/llvm-22/lib/cmake/clang" "/usr/lib/llvm-22/cmake")

message(STATUS "Using LLVM Config: ${LLVM_DIR}")
message(STATUS "Using MLIR Config: ${MLIR_DIR}")
message(STATUS "Using Clang Config: ${Clang_DIR}")

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

include_directories(
  ${LLVM_INCLUDE_DIRS} 
  ${MLIR_INCLUDE_DIRS} 
  ${CMAKE_CURRENT_SOURCE_DIR}/include 
  ${CMAKE_CURRENT_BINARY_DIR}/include
)
link_directories(${LLVM_LIBRARY_DIRS})
add_definitions(${LLVM_DEFINITIONS})

include(TableGen)
include(AddLLVM)
include(AddMLIR)

set(LLVM_TARGET_DEFINITIONS include/Dialect/NN/NNOps.td)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/Dialect/NN)
mlir_tablegen(include/Dialect/NN/NNDialect.h.inc -gen-dialect-decls)
mlir_tablegen(include/Dialect/NN/NNDialect.cpp.inc -gen-dialect-defs)
mlir_tablegen(include/Dialect/NN/NNOps.h.inc -gen-op-decls)
mlir_tablegen(include/Dialect/NN/NNOps.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS include/Dialect/NN/NNPasses.td)
mlir_tablegen(include/Dialect/NN/NNPasses.h.inc -gen-pass-decls -name NN)

add_custom_target(NNTableGenDependentTarget DEPENDS
  include/Dialect/NN/NNDialect.h.inc
  include/Dialect/NN/NNDialect.cpp.inc
  include/Dialect/NN/NNOps.h.inc
  include/Dialect/NN/NNOps.cpp.inc
  include/Dialect/NN/NNPasses.h.inc
)

add_llvm_library(NNDialect
  src/Dialect/NN/NNDialect.cpp
  src/Dialect/NN/NNOps.cpp
  src/Dialect/NN/NNPasses.cpp
  PARTIAL_SOURCES_INTENDED
  DEPENDS NNTableGenDependentTarget
  LINK_LIBS MLIRIR MLIRSideEffectInterfaces MLIRPass
)

add_executable(nn-opt tools/nn-opt/nn-opt.cpp)
target_link_libraries(nn-opt PRIVATE
  NNDialect
  MLIRIR
  MLIROptLib
  MLIRFuncDialect
  MLIRArithDialect
)

# Clang frontend driver
add_executable(nn-cc tools/nn-cc/nn-cc.cpp)
target_link_libraries(nn-cc PRIVATE
  NNDialect
  MLIRIR
  MLIRFuncDialect
  MLIRArithDialect
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangSerialization
  clangTooling
)
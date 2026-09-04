FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget lsb-release software-properties-common gnupg cmake ninja-build git python3 \
    && rm -rf /var/lib/apt/lists/*

RUN wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc && \
    add-apt-repository "deb http://apt.llvm.org/noble/ llvm-toolchain-noble-22 main" && \
    apt-get update && apt-get install -y \
    clang-22 \
    lld-22 \
    clangd-22 \
    clang-format-22 \
    clang-tidy-22 \
    libllvm22 \
    llvm-22-dev \
    llvm-22-tools \
    libmlir-22 \
    libmlir-22-dev \
    mlir-22-tools \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    libclang-22-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/clang-22 /usr/bin/clang && \
    ln -s /usr/bin/clang++-22 /usr/bin/clang++ && \
    ln -s /usr/bin/clangd-22 /usr/bin/clangd && \
    ln -s /usr/bin/clang-format-22 /usr/bin/clang-format && \
    ln -s /usr/bin/clang-tidy-22 /usr/bin/clang-tidy && \
    ln -s /usr/bin/mlir-opt-22 /usr/bin/mlir-opt && \
    ln -s /usr/bin/mlir-tblgen-22 /usr/bin/mlir-tblgen

WORKDIR /workspace

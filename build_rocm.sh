#!/usr/bin/env bash
# ROCm/HIP build for oneflow. The GPU arch is auto-detected from the build host; set
# CMAKE_HIP_ARCHITECTURES (e.g. gfx90a) to choose it explicitly.
# BUILD_CUDA stays ON (defines WITH_CUDA, the GPU backend == kCUDA reached
# through HIP); USE_HIP routes the toolkit/library/.cu-language plumbing to ROCm.
# Usage: ./build_rocm.sh [config|tp|oneflow|all]   (default: all)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BUILD="${HERE}/build"
ARCH_ARGS=()
if [ -n "${CMAKE_HIP_ARCHITECTURES:-}" ]; then
  ARCH_ARGS=(-DCMAKE_HIP_ARCHITECTURES="${CMAKE_HIP_ARCHITECTURES}")
fi
JOBS="${JOBS:-16}"
export CCACHE_DIR="${CCACHE_DIR:-${HOME}/.ccache}"

# The whole monolith builds with the ROCm clang (C, C++, and HIP), not g++. oneflow's
# core type header data_type.h and ep/cuda/cuda_stream.h pull in the HIP half/bf16 and
# library handle headers (hip_fp16/hip_bf16/hipblaslt/hipsolver/miopen) into thousands of
# host .cpp TUs; those headers use clang builtins (__builtin_elementwise_*) and only parse
# under clang. Using one clang toolchain for both the host .cpp and the .cu device code
# (the way libtorch_hip builds) makes those headers parse everywhere and guarantees host
# and device agree on the half/bf16 ABI.
ROCM_CLANG="${ROCM_CLANG:-/opt/rocm/lib/llvm/bin/clang++}"
ROCM_CLANG_C="${ROCM_CLANG_C:-/opt/rocm/lib/llvm/bin/clang}"

configure() {
  cmake -S "${HERE}" -B "${BUILD}" -G Ninja \
    -DUSE_HIP=ON "${ARCH_ARGS[@]}" \
    -DBUILD_CUDA=ON \
    -DCMAKE_C_COMPILER="${ROCM_CLANG_C}" -DCMAKE_CXX_COMPILER="${ROCM_CLANG}" \
    -DWITH_CUTLASS=OFF -DWITH_MLIR=OFF -DWITH_MLIR_CUDA_CODEGEN=OFF \
    -DBUILD_RDMA=OFF -DBUILD_CPP_API=OFF -DBUILD_HWLOC=OFF \
    -DWITH_ONEDNN=ON \
    -DBUILD_TESTING=OFF -DTREAT_WARNINGS_AS_ERRORS=OFF \
    -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_HIP_COMPILER_LAUNCHER=ccache \
    -DPython3_EXECUTABLE="$(which python3)"
}

case "${1:-all}" in
  config) configure ;;
  tp)      cmake --build "${BUILD}" -j"${JOBS}" --target prepare_oneflow_third_party ;;
  oneflow) cmake --build "${BUILD}" -j"${JOBS}" --target oneflow ;;
  all)     configure && cmake --build "${BUILD}" -j"${JOBS}" ;;
  *) echo "usage: $0 [config|tp|oneflow|all]"; exit 2 ;;
esac

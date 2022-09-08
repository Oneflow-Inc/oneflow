/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/api/python/of_api_registry.h"
#ifdef WITH_CUDA
#include <cuda.h>
#endif

namespace oneflow {

ONEFLOW_API_PYBIND11_MODULE("flags", m) {
  m.def("with_cuda", []() {
#ifdef WITH_CUDA
    return true;
#else
    return false;
#endif  // WITH_CUDA
  });

  m.def("cuda_version", []() {
#ifdef WITH_CUDA
    return CUDA_VERSION;
#else
    return 0;
#endif  // WITH_CUDA
  });

  m.def("use_cxx11_abi", []() {
#if _GLIBCXX_USE_CXX11_ABI == 1
    return true;
#else
    return false;
#endif  // _GLIBCXX_USE_CXX11_ABI
  });

  m.def("with_mlir", []() {
#ifdef WITH_MLIR
    return true;
#else
    return false;
#endif  // WITH_MLIR
  });

  m.def("with_mlir_cuda_codegen", []() {
#ifdef WITH_MLIR_CUDA_CODEGEN
    return true;
#else
    return false;
#endif  // WITH_MLIR_CUDA_CODEGEN
  });

  m.def("with_rdma", []() {
#ifdef WITH_RDMA
    return true;
#else
    return false;
#endif  // WITH_RDMA
  });

  m.def("has_rpc_backend_grpc", []() {
#ifdef RPC_BACKEND_GRPC
    return true;
#else
    return false;
#endif  // RPC_BACKEND_GRPC
  });

  m.def("has_rpc_backend_local", []() {
#ifdef RPC_BACKEND_LOCAL
    return true;
#else
    return false;
#endif  // RPC_BACKEND_LOCAL
  });

#define STRINGIFY(x) STRINGIFY_(x)
#define STRINGIFY_(x) #x
  m.def("cmake_build_type", []() {
#ifdef ONEFLOW_CMAKE_BUILD_TYPE
    return std::string(STRINGIFY(ONEFLOW_CMAKE_BUILD_TYPE));
#else
    return std::string("Undefined");
#endif  // ONEFLOW_CMAKE_BUILD_TYPE
  });
#undef STRINGIFY
#undef STRINGIFY_
}

}  // namespace oneflow

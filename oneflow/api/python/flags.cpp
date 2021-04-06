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

namespace oneflow {

ONEFLOW_API_PYBIND11_MODULE("flags", m) {
  m.def("with_cuda", []() {
#ifdef WITH_CUDA
    return true;
#else
    return false;
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

  m.def("with_xla", []() {
#ifdef WITH_XLA
    return true;
#else
    return false;
#endif  // WITH_XLA
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
}

}  // namespace oneflow

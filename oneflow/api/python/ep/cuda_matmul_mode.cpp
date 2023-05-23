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

#include <memory>
#include <pybind11/pybind11.h>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/ep/cuda/cuda_matmul_mode.h"

namespace py = pybind11;

namespace oneflow {

namespace ep {

ONEFLOW_API_PYBIND11_MODULE("ep", m) {
  m.def("is_matmul_allow_tf32", &CudaMatmulMode::is_matmul_allow_tf32);
  m.def("set_matmul_allow_tf32", &CudaMatmulMode::set_matmul_allow_tf32);
  m.def("is_matmul_allow_fp16_reduced_precision_reduction",
        &CudaMatmulMode::is_matmul_allow_fp16_reduced_precision_reduction);
  m.def("set_matmul_allow_fp16_reduced_precision_reduction",
        &CudaMatmulMode::set_matmul_allow_fp16_reduced_precision_reduction);
}

}  // namespace ep

}  // namespace oneflow

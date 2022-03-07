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
#include <pybind11/pybind11.h>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/framework/tensor_pool.h"
#include "oneflow/core/vm/dtr_cuda_allocator.h"

namespace py = pybind11;

using namespace oneflow;

ONEFLOW_API_PYBIND11_MODULE("dtr", m) {
  m.def("allocated_memory", []() -> size_t {
    return Global<vm::DtrCudaAllocator>::Get()->allocated_memory();
  });
  m.def("display_all_pieces", []() -> void {
    return Global<vm::DtrCudaAllocator>::Get()->DisplayAllPieces();
  });
  m.def("display", []() -> void {
    Global<one::DTRTensorPool>::Get()->display().GetOrThrow();
  });
}

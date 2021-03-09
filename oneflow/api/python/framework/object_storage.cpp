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
#include "oneflow/core/framework/object_storage.h"

namespace py = pybind11;

namespace oneflow {

namespace {

bool PyHasSharedOpKernelObject4ParallelConfSymbol(
    const std::shared_ptr<ParallelDesc>& parallel_conf_sym) {
  return HasSharedOpKernelObject4ParallelConfSymbol(parallel_conf_sym).GetOrThrow();
}

std::shared_ptr<compatible_py::Object> PyGetSharedOpKernelObject4ParallelConfSymbol(
    const std::shared_ptr<ParallelDesc>& parallel_conf_sym) {
  return GetOpKernelObject4ParallelConfSymbol(parallel_conf_sym).GetPtrOrThrow();
}

void PySetSharedOpKernelObject4ParallelConfSymbol(
    const std::shared_ptr<ParallelDesc>& parallel_conf_sym,
    const std::shared_ptr<compatible_py::Object>& shared_opkernel_object) {
  return SetSharedOpKernelObject4ParallelConfSymbol(parallel_conf_sym, shared_opkernel_object)
      .GetOrThrow();
}

}  // namespace

ONEFLOW_API_PYBIND11_MODULE("", m) {
  m.def("HasSharedOpKernelObject4ParallelConfSymbol",
        &PyHasSharedOpKernelObject4ParallelConfSymbol);
  m.def("GetSharedOpKernelObject4ParallelConfSymbol",
        &PyGetSharedOpKernelObject4ParallelConfSymbol);
  m.def("SetSharedOpKernelObject4ParallelConfSymbol",
        &PySetSharedOpKernelObject4ParallelConfSymbol);
}

}  // namespace oneflow

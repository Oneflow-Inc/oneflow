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
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/job/sbp_parallel.cfg.h"

namespace py = pybind11;

namespace oneflow {

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<Symbol<cfg::ParallelDistribution>, std::shared_ptr<Symbol<cfg::ParallelDistribution>>>(
      m, "ParallelDistribution")
      .def(py::init([](const std::shared_ptr<cfg::ParallelDistribution>& parallel_distrition) {
        return Symbol<cfg::ParallelDistribution>(*parallel_distrition);
      }))
      .def("__str__",
           [](const Symbol<cfg::ParallelDistribution>& x) {
             const auto& parallel_distrition = *x;
             return x.DebugString();
           })
      .def("__repr__", [](const Symbol<cfg::ParallelDistribution>& x) {
        const auto& parallel_distrition = *x;
        return x.DebugString();
      });
}

}  // namespace oneflow

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

#include <vector>
#include <pybind11/pybind11.h>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/autograd/autograd_engine.h"

namespace py = pybind11;

namespace oneflow {

namespace {

struct FunctionNodeUtil final {
  static std::string ToString(const one::FunctionNode& func_node) {
    std::stringstream ss;
    ss << "<";
    ss << func_node.GetOpName();
    ss << ">";
    return ss.str();
  }
};

}  // namespace

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<one::FunctionNode, std::shared_ptr<one::FunctionNode>>(m, "Size")
      .def("__str__", &FunctionNodeUtil::ToString)
      .def("__repr__", &FunctionNodeUtil::ToString)
      .def("next_functions",
           [](const one::FunctionNode& func_node) { return func_node.GetNextFunctions(); });
}

}  // namespace oneflow

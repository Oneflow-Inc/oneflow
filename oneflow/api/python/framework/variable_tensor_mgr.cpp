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
#include <tuple>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/common/global.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/variable_tensor_mgr.h"

namespace py = pybind11;

namespace oneflow {

ONEFLOW_API_PYBIND11_MODULE("", m) {
  m.def("FillVariableTensorMgr",
        [](const std::vector<std::string>& variable_op_names,
           const std::vector<std::shared_ptr<one::Tensor>>& variable_tensors) {
          auto mgr = Global<VariableTensorMgr>::Get();
          mgr->Fill(variable_op_names, variable_tensors).GetOrThrow();
        });
  m.def("DumpVariableTensorMgr",
        []() -> std::tuple<std::vector<std::string>, std::vector<std::shared_ptr<one::Tensor>>> {
          auto mgr = Global<VariableTensorMgr>::Get();
          return mgr->Dump();
        });
}

}  // namespace oneflow

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
#include "oneflow/core/common/container_util.h"
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/framework/py_distribute.h"

namespace py = pybind11;

namespace oneflow {

namespace compatible_py {

ONEFLOW_API_PYBIND11_MODULE("distribute", m) {
  py::class_<Distribute, std::shared_ptr<Distribute>>(m, "Distribute");
  py::class_<AutoDistribute, Distribute, std::shared_ptr<AutoDistribute>>(m, "AutoDistribute");
  py::class_<BroadcastDistribute, Distribute, std::shared_ptr<BroadcastDistribute>>(
      m, "BroadcastDistribute");
  py::class_<SplitDistribute, Distribute, std::shared_ptr<SplitDistribute>>(m, "SplitDistribute")
      .def_property_readonly("axis", &SplitDistribute::axis);
  m.def("auto", &GlobalAutoDistribute);
  m.def("broadcast", &GlobalBroadcastDistribute);
  m.def("split", [](int axis) { return GlobalSplitDistribute(axis).GetPtrOrThrow(); });
}

}  // namespace compatible_py

}  // namespace oneflow

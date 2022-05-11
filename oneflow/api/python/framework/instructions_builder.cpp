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
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <functional>
#include "oneflow/api/python/framework/size.h"
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/tensor.h"

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(
    std::unordered_map<std::string, std::shared_ptr<::oneflow::compatible_py::BlobObject>>);

namespace oneflow {

namespace {

Maybe<void> DeprecatedPhysicalRun(const std::function<void(InstructionsBuilder*)>& Build) {
  return PhysicalRun([&](InstructionsBuilder* instruction_builder) -> Maybe<void> {
    Build(instruction_builder);
    return Maybe<void>::Ok();
  });
}

}  // namespace

ONEFLOW_API_PYBIND11_MODULE("deprecated", m) {
  py::class_<InstructionsBuilder, std::shared_ptr<InstructionsBuilder>>(m, "InstructionsBuilder")
      .def("BuildInitialScope", &InstructionsBuilder::BuildInitialScope,
           py::arg("session_id").none(false), py::arg("job_conf").none(false),
           py::arg("device_tag").none(false), py::arg("machine_device_ids").none(false),
           py::arg("hierarchy").none(true), py::arg("is_mirrored").none(false))
      .def("BuildInitialScopeWithPlacement", &InstructionsBuilder::BuildInitialScopeWithPlacement,
           py::arg("session_id").none(false), py::arg("job_conf").none(false),
           py::arg("placement").none(false), py::arg("is_mirrored").none(false))
      .def("BuildScopeWithNewParallelDesc", &InstructionsBuilder::BuildScopeWithNewParallelDesc,
           py::arg("scope").none(false), py::arg("device_tag").none(false),
           py::arg("machine_device_ids").none(false), py::arg("hierarchy").none(true))
      .def("BuildScopeWithNewParallelConf", &InstructionsBuilder::BuildScopeWithNewParallelConf)
      .def("BuildScopeWithNewIsMirrored", &InstructionsBuilder::BuildScopeWithNewIsMirrored)
      .def("BuildScopeWithNewScopeName", &InstructionsBuilder::BuildScopeWithNewScopeName)
      .def("BuildScopeByProtoSetter", &InstructionsBuilder::BuildScopeByProtoSetter)
      .def("BuildScopeByProtoStrSetter", &InstructionsBuilder::BuildScopeByProtoStrSetter);

  m.def("PhysicalRun", &DeprecatedPhysicalRun, py::call_guard<py::gil_scoped_release>());
}

}  // namespace oneflow

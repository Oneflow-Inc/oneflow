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

std::shared_ptr<Scope> BuildInitialScope(InstructionsBuilder* x, int64_t session_id,
                                         const std::shared_ptr<cfg::JobConfigProto>& job_conf,
                                         const std::string& device_tag,
                                         const std::vector<std::string>& machine_device_ids,
                                         const std::shared_ptr<Shape>& hierarchy,
                                         bool is_mirrored) {
  return x
      ->BuildInitialScope(session_id, job_conf, device_tag, machine_device_ids, hierarchy,
                          is_mirrored)
      .GetPtrOrThrow();
}

std::shared_ptr<Scope> BuildScopeWithNewParallelDesc(
    InstructionsBuilder* x, const std::shared_ptr<Scope>& scope, const std::string& device_tag,
    const std::vector<std::string>& machine_device_ids, const std::shared_ptr<Shape>& hierarchy) {
  return x->BuildScopeWithNewParallelDesc(scope, device_tag, machine_device_ids, hierarchy)
      .GetPtrOrThrow();
}

std::shared_ptr<Scope> BuildScopeWithNewParallelConf(
    InstructionsBuilder* x, const std::shared_ptr<Scope>& scope,
    const std::shared_ptr<cfg::ParallelConf>& parallel_conf) {
  return x->BuildScopeWithNewParallelConf(scope, parallel_conf).GetPtrOrThrow();
}

std::shared_ptr<Scope> BuildScopeWithNewIsMirrored(InstructionsBuilder* x,
                                                   const std::shared_ptr<Scope>& scope,
                                                   bool is_mirrored) {
  return x->BuildScopeWithNewIsMirrored(scope, is_mirrored).GetPtrOrThrow();
}

std::shared_ptr<Scope> BuildScopeWithNewScopeName(InstructionsBuilder* x,
                                                  const std::shared_ptr<Scope>& scope,
                                                  std::string scope_name) {
  return x->BuildScopeWithNewScopeName(scope, scope_name).GetPtrOrThrow();
}

std::shared_ptr<Scope> BuildScopeByProtoSetter(
    InstructionsBuilder* x, const std::shared_ptr<Scope>& scope,
    const std::function<void(const std::shared_ptr<cfg::ScopeProto>&)>& Setter) {
  return x->BuildScopeByProtoSetter(scope, Setter).GetPtrOrThrow();
}

Maybe<void> DeprecatedPhysicalRun(const std::function<void(InstructionsBuilder*)>& Build) {
  return PhysicalRun([&](InstructionsBuilder* instruction_builder) -> Maybe<void> {
    Build(instruction_builder);
    return Maybe<void>::Ok();
  });
}

}  // namespace

ONEFLOW_API_PYBIND11_MODULE("deprecated", m) {
  py::class_<InstructionsBuilder, std::shared_ptr<InstructionsBuilder>>(m, "InstructionsBuilder")
      .def("BuildInitialScope", &BuildInitialScope, py::arg("session_id").none(false),
           py::arg("job_conf").none(false), py::arg("device_tag").none(false),
           py::arg("machine_device_ids").none(false), py::arg("hierarchy").none(true),
           py::arg("is_mirrored").none(false))
      .def("BuildScopeWithNewParallelDesc", &BuildScopeWithNewParallelDesc,
           py::arg("scope").none(false), py::arg("device_tag").none(false),
           py::arg("machine_device_ids").none(false), py::arg("hierarchy").none(true))
      .def("BuildScopeWithNewParallelConf", &BuildScopeWithNewParallelConf)
      .def("BuildScopeWithNewIsMirrored", &BuildScopeWithNewIsMirrored)
      .def("BuildScopeWithNewScopeName", &BuildScopeWithNewScopeName)
      .def("BuildScopeByProtoSetter", &BuildScopeByProtoSetter);

  m.def(
      "PhysicalRun",
      [](const std::function<void(InstructionsBuilder*)>& Build) {
        return DeprecatedPhysicalRun(Build).GetOrThrow();
      },
      py::call_guard<py::gil_scoped_release>());
}

}  // namespace oneflow

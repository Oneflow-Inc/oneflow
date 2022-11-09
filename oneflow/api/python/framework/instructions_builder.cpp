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
      .def(
          "BuildInitialScope",
          [](const std::shared_ptr<InstructionsBuilder>& builder, int64_t session_id,
             const std::string& job_conf_str, const std::string& device_tag,
             const std::vector<std::string>& machine_device_ids,
             const std::shared_ptr<Shape>& hierarchy, bool is_local) -> Maybe<Scope> {
            JobConfigProto job_conf;
            CHECK_OR_RETURN(TxtString2PbMessage(job_conf_str, &job_conf))
                << Error::RuntimeError() << "job conf parse failed";
            return builder->BuildInitialScope(session_id, job_conf, device_tag, machine_device_ids,
                                              hierarchy, is_local);
          },
          py::arg("session_id").none(false), py::arg("job_conf_str").none(false),
          py::arg("device_tag").none(false), py::arg("machine_device_ids").none(false),
          py::arg("hierarchy").none(true), py::arg("is_local").none(false))
      .def(
          "BuildInitialScopeWithPlacement",
          [](const std::shared_ptr<InstructionsBuilder>& builder, int64_t session_id,
             const std::string& job_conf_str, Symbol<ParallelDesc> placement,
             bool is_local) -> Maybe<Scope> {
            JobConfigProto job_conf;
            CHECK_OR_RETURN(TxtString2PbMessage(job_conf_str, &job_conf))
                << Error::RuntimeError() << "job conf parse failed";
            return builder->BuildInitialScopeWithPlacement(session_id, job_conf, placement,
                                                           is_local);
          },
          py::arg("session_id").none(false), py::arg("job_conf_str").none(false),
          py::arg("placement").none(false), py::arg("is_local").none(false))
      .def("BuildScopeWithNewParallelDesc", &InstructionsBuilder::BuildScopeWithNewParallelDesc,
           py::arg("scope").none(false), py::arg("device_tag").none(false),
           py::arg("machine_device_ids").none(false), py::arg("hierarchy").none(true))
      .def("BuildScopeWithNewParallelConf",
           [](const std::shared_ptr<InstructionsBuilder>& builder,
              const std::shared_ptr<Scope>& scope,
              const std::string& parallel_conf_str) -> Maybe<Scope> {
             ParallelConf parallel_conf;
             CHECK_OR_RETURN(TxtString2PbMessage(parallel_conf_str, &parallel_conf))
                 << Error::RuntimeError() << "parallel conf parse failed";
             return builder->BuildScopeWithNewParallelConf(scope, parallel_conf);
           })
      .def("BuildScopeWithNewIsLocal", &InstructionsBuilder::BuildScopeWithNewIsLocal)
      .def("BuildScopeWithNewScopeName", &InstructionsBuilder::BuildScopeWithNewScopeName)
      .def("BuildScopeByProtoStrSetter", &InstructionsBuilder::BuildScopeByProtoStrSetter);

  m.def("PhysicalRun", &DeprecatedPhysicalRun, py::call_guard<py::gil_scoped_release>());
}

}  // namespace oneflow

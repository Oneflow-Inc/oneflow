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
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/framework/session_util.h"

namespace py = pybind11;

namespace oneflow {

class PySession : public Session {
 public:
  using Session::Session;

  std::pair<std::shared_ptr<one::Tensor>, std::shared_ptr<one::Tensor>>
  TryGetVariableBlobOfJobFromStash(const std::string& job_name,
                                   const std::string& variable_name) const override {
    using P = std::pair<std::shared_ptr<one::Tensor>, std::shared_ptr<one::Tensor>>;
    PYBIND11_OVERRIDE(P, Session, TryGetVariableBlobOfJobFromStash, job_name, variable_name);
  }

  std::string GetJobNameScopePrefix(const std::string& job_name) const override {
    PYBIND11_OVERRIDE(std::string, Session, GetJobNameScopePrefix, job_name);
  }

  bool IsMirroredStrategyEnabled() const override {
    PYBIND11_OVERRIDE(bool, Session, IsMirroredStrategyEnabled);
  }
  bool IsConsistentStrategyEnabled() const override {
    PYBIND11_OVERRIDE(bool, Session, IsConsistentStrategyEnabled);
  }
};

ONEFLOW_API_PYBIND11_MODULE("deprecated", m) {
  py::class_<Session, PySession, std::shared_ptr<Session>>(m, "Session")
      .def(py::init([](int64_t id) {
        return std::make_shared<PySession>(id, std::make_shared<vm::cfg::InstructionListProto>(),
                                           std::make_shared<eager::cfg::EagerSymbolList>());
      }))
      .def_property_readonly("id_", &Session::id)
      .def("snapshot_mgr_", &Session::snapshot_mgr)
      .def("TryGetVariableBlobOfJobFromStash", &Session::TryGetVariableBlobOfJobFromStash)
      .def("GetJobNameScopePrefix", &Session::GetJobNameScopePrefix)
      .def("IsMirroredStrategyEnabled", &Session::IsMirroredStrategyEnabled)
      .def("IsConsistentStrategyEnabled", &Session::IsConsistentStrategyEnabled);

  m.def("GetDefaultSessionId", []() { return GetDefaultSessionId().GetOrThrow(); });
  m.def("RegsiterSession", [](int64_t id, const std::shared_ptr<Session>& sess) {
    RegsiterSession(id, sess).GetOrThrow();
  });
  m.def("GetDefaultSession", []() { return GetDefaultSession().GetPtrOrThrow(); });
  m.def("ClearSessionById", [](int64_t id) { return ClearSessionById(id).GetOrThrow(); });
}

}  // namespace oneflow

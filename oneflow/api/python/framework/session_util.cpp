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
#include "oneflow/core/framework/session_util.h"

namespace py = pybind11;

namespace oneflow {

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<Session, std::shared_ptr<Session>>(m, "Session")
      .def_property_readonly("id", &Session::id)
      .def("push_mirrored_strategy_enabled",
           [](Session* sess, const bool& val) {
             sess->PushMirroredStrategyEnabled(val).GetOrThrow();
           })
      .def("pop_mirrored_strategy_enabled",
           [](Session* sess) { sess->PopMirroredStrategyEnabled().GetOrThrow(); })
      .def("is_mirrored_strategy_enabled",
           [](const Session* sess) { return sess->IsMirroredStrategyEnabled().GetOrThrow(); })
      .def("is_consistent_strategy_enabled",
           [](const Session* sess) { return sess->IsConsistentStrategyEnabled().GetOrThrow(); })
      .def("is_mirrored_strategy_enabled_stack_size",
           [](const Session* sess) { return sess->is_mirrored_strategy_enabled_stack()->size(); });

  m.def("GetDefaultSessionId", []() { return GetDefaultSessionId().GetOrThrow(); });
  m.def("RegsiterSession", [](int64_t id) { return RegsiterSession(id).GetPtrOrThrow(); });

  m.def("GetDefaultSession", []() { return GetDefaultSession().GetPtrOrThrow(); });
  m.def("ClearSessionById", [](int64_t id) { return ClearSessionById(id).GetOrThrow(); });
}

}  // namespace oneflow

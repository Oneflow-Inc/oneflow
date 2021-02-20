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
      .def("instruction_list", &Session::instruction_list)
      .def("eager_symbol_list", &Session::eager_symbol_list);

  m.def("GetDefaultSessionId", []() {
    int64_t default_sess_id = *GetDefaultSessionId().GetOrThrow();
    return default_sess_id;
  });
  m.def("SetDefaultSessionId", [](int64_t val) { return SetDefaultSessionId(val).GetOrThrow(); });
  m.def("RegsiterSession", [](int64_t id) { return RegsiterSession(id).GetPtrOrThrow(); });

  m.def("GetDefaultSession", []() { return GetDefaultSession().GetPtrOrThrow(); });
  m.def("ClearDefaultSession", []() { return ClearDefaultSession().GetOrThrow(); });

  m.def("ClearSessionById", [](int64_t id) { return ClearSessionById(id).GetOrThrow(); });
  m.def("ClearAllSession", []() { return ClearAllSession().GetOrThrow(); });
}

}  // namespace oneflow

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
#include <string>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/job/session.h"

namespace py = pybind11;

ONEFLOW_API_PYBIND11_MODULE("", m) {
  using namespace oneflow;
  m.def("NewSessionId", &NewSessionId);
  py::class_<LogicalConfigProtoContext>(m, "LogicalConfigProtoContext")
      .def(py::init<const std::string&>());
  ;
}

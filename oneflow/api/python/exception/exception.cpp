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
#include "oneflow/core/common/exception.h"
#include "oneflow/core/common/error.h"
#include "oneflow/api/python/of_api_registry.h"

namespace py = pybind11;

namespace oneflow {

ONEFLOW_API_PYBIND11_MODULE("exception", m) {
  m.def("GetThreadLocalLastError", &ThreadLocalError);
  py::register_exception<oneflow::Exception>(m, "Exception");
  py::register_exception<oneflow::RuntimeException>(m, "RuntimeError", PyExc_RuntimeError);
  py::register_exception<oneflow::TypeException>(m, "TypeError", PyExc_TypeError);
  py::register_exception<oneflow::IndexException>(m, "IndexError", PyExc_IndexError);
  py::register_exception<oneflow::NotImplementedException>(m, "NotImplementedError",
                                                           PyExc_NotImplementedError);
}

}  // namespace oneflow

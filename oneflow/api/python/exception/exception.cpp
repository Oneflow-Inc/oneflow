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
#define REGISTER_EXCEPTION(cls) \
  py::register_exception<oneflow::OF_PP_CAT(cls, Exception)>(m, OF_PP_STRINGIZE(cls) "Exception");

  OF_PP_FOR_EACH_TUPLE(REGISTER_EXCEPTION, EXCEPTION_SEQ)

#undef REGISTER_EXCEPTION
}

}  // namespace oneflow

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

#include "oneflow/api/python/functional/common.h"

namespace oneflow {
namespace one {
namespace functional {

bool PyTensorCheck(PyObject* object) {
  auto obj = py::reinterpret_borrow<py::object>(object);
  return detail::isinstance<std::shared_ptr<one::Tensor>>(obj);
}

const char* PyStringAsString(PyObject* object) {
  return PyBytes_AsString(PyUnicode_AsEncodedString(object, "utf-8", "~E~"));
}

}  // namespace functional
}  // namespace one
}  // namespace oneflow

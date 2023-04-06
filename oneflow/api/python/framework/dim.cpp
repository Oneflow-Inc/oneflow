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
#include <pybind11/operators.h>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/common/dim.h"

namespace py = pybind11;

namespace oneflow {

#define DEFINE_OPERATOR(op)                           \
  .def(py::self op py::self)                          \
      .def(py::self op char())                        \
      .def(py::self op static_cast<unsigned char>(0)) \
      .def(py::self op int())                         \
      .def(py::self op static_cast<unsigned int>(0))  \
      .def(py::self op long())                        \
      .def(py::self op 0UL)                           \
      .def(py::self op 0LL)                           \
      .def(py::self op 0ULL)                          \
      .def(py::self op float())                       \
      .def(py::self op double())

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<Dim>(m, "Dim")
      .def(py::init([](int value) { return Dim(value); }))
      .def_static("unknown", &Dim::Unknown)
      .def("__str__",
           [](Dim dim) {
             std::stringstream ss;
             ss << dim;
             return ss.str();
           })
      .def("__repr__",
           [](Dim dim) {
             std::stringstream ss;
             ss << dim;
             return ss.str();
           })
      // clang-format off
      DEFINE_OPERATOR(==)
      DEFINE_OPERATOR(!=)
      DEFINE_OPERATOR(<)
      DEFINE_OPERATOR(<=)
      DEFINE_OPERATOR(>)
      DEFINE_OPERATOR(>=)
      // clang-format on
      .def(py::hash(py::self));
}

}  // namespace oneflow

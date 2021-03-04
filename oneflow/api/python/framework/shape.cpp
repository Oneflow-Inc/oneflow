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
#include <vector>
#include <pybind11/pybind11.h>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/common/shape.h"

namespace py = pybind11;

namespace oneflow {

namespace {

Maybe<Shape> CreateShape(const py::tuple& py_shape) {
  DimVector shape_dims;
  CHECK_OR_RETURN(py::isinstance<py::tuple>(py_shape));
  for (const auto& dim : py_shape) { shape_dims.emplace_back(dim.cast<int64_t>()); }
  return std::make_shared<Shape>(shape_dims);
}

}  // namespace

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<Shape, std::shared_ptr<Shape>>(m, "Size")
      .def(
          py::init([](const py::tuple& py_shape) { return CreateShape(py_shape).GetPtrOrThrow(); }))
      .def("__str__", &Shape::ToString)
      .def("__repr__", &Shape::ToString);
}

}  // namespace oneflow

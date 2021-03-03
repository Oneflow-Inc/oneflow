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
struct ShapeExportUtil final {
  static Maybe<Shape> MakeShape(const py::tuple& py_shape) {
    DimVector shape_dims;
    CHECK_OR_RETURN(py::isinstance<py::tuple>(py_shape))
        << Error::ValueError("Input shape must be tuple.");
    for (const auto& dim : py_shape) { shape_dims.emplace_back(dim.cast<int64_t>()); }
    return std::make_shared<Shape>(shape_dims);
  }

  static std::shared_ptr<Shape> ApiMakeShape(const py::tuple& py_shape) {
    return MakeShape(py_shape).GetPtrOrThrow();
  }

  static std::string ToString(const Shape& shape) {
    std::stringstream ss;
    int32_t idx = 0;
    ss << "flow.Size([";
    for (int64_t dim : shape.dim_vec()) {
      ss << dim;
      if (++idx != dim_vec.size()) { ss << ", "; }
    }
    ss << "])";
    return ss.str();
  }

};

}  // namespace

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<Shape, std::shared_ptr<Shape>>(m, "Size")
      .def(py::init(&ShapeExportUtil::ApiMakeShape))
      .def("__str__", &ShapeExportUtil::ToString)
      .def("__repr__", &ShapeExportUtil::ToString)
      .def("__getitem__", [](const Shape& shape, int idx) { return shape.At(idx); })
      .def(
          "__iter__",
          [](const Shape& shape) {
            return py::make_iterator(shape.dim_vec().begin(), shape.dim_vec().end());
          },
          py::keep_alive<0, 1>())
      .def("__len__", [](const Shape& shape) { return shape.NumAxes(); });
}

}  // namespace oneflow

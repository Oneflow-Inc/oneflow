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
#include "oneflow/core/framework/dtype.h"

namespace py = pybind11;

namespace oneflow {

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<DType, std::shared_ptr<DType>>(m, "dtype")
      .def_property_readonly("is_signed", &DType::is_signed)
      .def_property_readonly("is_complex", &DType::is_complex)
      .def_property_readonly("is_floating_point", &DType::is_floating_point)
      .def("__str__", &DType::name)
      .def("__repr__", &DType::name);

  m.attr("char") = DType::Char().GetPtrOrThrow();
  m.attr("float16") = DType::Float16().GetPtrOrThrow();
  m.attr("float") = DType::Float().GetPtrOrThrow();

  m.attr("float32") = DType::Float().GetPtrOrThrow();
  m.attr("double") = DType::Double().GetPtrOrThrow();
  m.attr("float64") = DType::Double().GetPtrOrThrow();

  m.attr("int8") = DType::Int8().GetPtrOrThrow();
  m.attr("int32") = DType::Int32().GetPtrOrThrow();
  m.attr("int64") = DType::Int64().GetPtrOrThrow();

  m.attr("uint8") = DType::UInt8().GetPtrOrThrow();
  m.attr("record") = DType::OFRecord().GetPtrOrThrow();
  m.attr("tensor_buffer") = DType::TensorBuffer().GetPtrOrThrow();
}

}  // namespace oneflow

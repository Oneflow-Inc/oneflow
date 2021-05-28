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

namespace {

struct DTypeExportUtil final {
  static bool is_signed(const DType& dtype) { return dtype.is_signed().GetOrThrow(); }
  static bool is_complex(const DType& dtype) { return dtype.is_complex().GetOrThrow(); }
  static bool is_floating_point(const DType& dtype) {
    return dtype.is_floating_point().GetOrThrow();
  }
  static const std::string& name(const DType& dtype) { return dtype.name().GetOrThrow(); }
  static size_t bytes(const DType& dtype) { return dtype.bytes().GetOrThrow(); }
};

}  // namespace

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<DType, std::shared_ptr<DType>>(m, "dtype")
      .def_property_readonly("is_signed", &DTypeExportUtil::is_signed)
      .def_property_readonly("is_complex", &DTypeExportUtil::is_complex)
      .def_property_readonly("is_floating_point", &DTypeExportUtil::is_floating_point)
      .def("__str__", &DTypeExportUtil::name)
      .def("__repr__", &DTypeExportUtil::name)
      .def_property_readonly("bytes", &DTypeExportUtil::bytes);

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

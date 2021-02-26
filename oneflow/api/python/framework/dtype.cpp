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

ONEFLOW_API_PYBIND11_MODULE("dtype", m) {
  py::class_<DType, std::shared_ptr<DType>>(m, "DType")
      .def_property_readonly("is_signed", &DType::is_signed)
      .def_property_readonly("is_complex", &DType::is_complex)
      .def_property_readonly("is_floating_point", &DType::is_floating_point)
      .def("__str__", &DType::ToString)
      .def("__repr__", &DType::ToString);

  m.attr("char") = Char();
  m.attr("float16") = Float16();
  m.attr("float") = Float();

  m.attr("float32") = Float();
  m.attr("double") = Double();
  m.attr("float64") = Double();

  m.attr("int8") = Int8();
  m.attr("int32") = Int32();
  m.attr("int64") = Int64();

  m.attr("uint8") = UInt8();
  m.attr("record") = RecordDType();
  m.attr("tensor_buffer") = TensorBufferDType();
}

}  // namespace oneflow

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
      // type enum of Protobuf at python side
      .def_property_readonly("oneflow_proto_dtype",
                             [](const std::shared_ptr<DType>& x) {
                               return static_cast<int>(x->oneflow_proto_dtype());
                             })
      .def_property_readonly_static("char", [](py::object) { return DType::Char(); })
      .def_property_readonly_static("float16", [](py::object) { return DType::Float16(); })
      .def_property_readonly_static("float", [](py::object) { return DType::Float(); })
      .def_property_readonly_static("float32", [](py::object) { return DType::Float(); })
      .def_property_readonly_static("double", [](py::object) { return DType::Double(); })
      .def_property_readonly_static("float64", [](py::object) { return DType::Double(); })
      .def_property_readonly_static("int8", [](py::object) { return DType::Int8(); })
      .def_property_readonly_static("int32", [](py::object) { return DType::Int32(); })
      .def_property_readonly_static("int64", [](py::object) { return DType::Int64(); })
      .def_property_readonly_static("uint8", [](py::object) { return DType::UInt8(); })
      .def_property_readonly_static("record", [](py::object) { return DType::OFRecordDType(); })
      .def_property_readonly_static("tensor_buffer",
                                    [](py::object) { return DType::TensorBufferDType(); })
      .def_static(
          "GetDTypeByDataType",
          [](int data_type) { return DType::GetDTypeByDataType(static_cast<DataType>(data_type)); })
      .def("__str__", &DType::name)
      .def("__repr__", &DType::name);
}

}  // namespace oneflow

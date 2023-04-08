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
#include "oneflow/api/python/framework/tensortype.h"
#include "oneflow/api/python/functional/common.h"
#include "oneflow/core/framework/dtype.h"

namespace py = pybind11;

namespace oneflow {

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<Symbol<DType>, std::shared_ptr<Symbol<DType>>>(m, "dtype")
      .def_property_readonly("is_signed", [](const Symbol<DType>& d) { return d->is_signed(); })
      .def_property_readonly("is_complex", [](const Symbol<DType>& d) { return d->is_complex(); })
      .def_property_readonly("is_floating_point",
                             [](const Symbol<DType>& d) { return d->is_floating_point(); })
      .def("__str__", [](const Symbol<DType>& d) { return d->name(); })
      .def("__repr__", [](const Symbol<DType>& d) { return d->name(); })
      .def(py::self == py::self)
      .def(py::hash(py::self))
      .def(py::pickle(
          [](const Symbol<DType>& dtype) {  // __getstate__
            return static_cast<int>(dtype->data_type());
          },
          [](int t) {  // __setstate__
            return CHECK_JUST(DType::Get(DataType(t)));
          }))
      .def_property_readonly("bytes", [](const Symbol<DType>& dtype) { return dtype->bytes(); })
      .def("get", [](const int data_type_enum) {
        return CHECK_JUST(DType::Get(static_cast<DataType>(data_type_enum)));
      });

  m.attr("bool") = &CHECK_JUST(DType::Get(DataType::kBool));
  m.attr("char") = &CHECK_JUST(DType::Get(DataType::kChar));
  m.attr("float16") = &CHECK_JUST(DType::Get(DataType::kFloat16));
  m.attr("float") = &CHECK_JUST(DType::Get(DataType::kFloat));
  m.attr("float32") = &CHECK_JUST(DType::Get(DataType::kFloat));
  m.attr("double") = &CHECK_JUST(DType::Get(DataType::kDouble));
  m.attr("float64") = &CHECK_JUST(DType::Get(DataType::kDouble));
  m.attr("int8") = &CHECK_JUST(DType::Get(DataType::kInt8));
  m.attr("int32") = &CHECK_JUST(DType::Get(DataType::kInt32));
  m.attr("int64") = &CHECK_JUST(DType::Get(DataType::kInt64));
  m.attr("uint8") = &CHECK_JUST(DType::Get(DataType::kUInt8));
  m.attr("record") = &CHECK_JUST(DType::Get(DataType::kOFRecord));
  m.attr("tensor_buffer") = &CHECK_JUST(DType::Get(DataType::kTensorBuffer));
  m.attr("bfloat16") = &CHECK_JUST(DType::Get(DataType::kBFloat16));
  m.attr("uint16") = &CHECK_JUST(DType::Get(DataType::kUInt16));
  m.attr("uint32") = &CHECK_JUST(DType::Get(DataType::kUInt32));
  m.attr("uint64") = &CHECK_JUST(DType::Get(DataType::kUInt64));
  m.attr("uint128") = &CHECK_JUST(DType::Get(DataType::kUInt128));
  m.attr("int16") = &CHECK_JUST(DType::Get(DataType::kInt16));
  m.attr("int128") = &CHECK_JUST(DType::Get(DataType::kInt128));
  m.attr("complex32") = &CHECK_JUST(DType::Get(DataType::kComplex32));
  m.attr("chalf") = &CHECK_JUST(DType::Get(DataType::kComplex32));
  m.attr("complex64") = &CHECK_JUST(DType::Get(DataType::kComplex64));
  m.attr("cfloat") = &CHECK_JUST(DType::Get(DataType::kComplex64));
  m.attr("complex128") = &CHECK_JUST(DType::Get(DataType::kComplex128));
  m.attr("cdouble") = &CHECK_JUST(DType::Get(DataType::kComplex128));

  py::options options;
  options.disable_function_signatures();
  m.def("get_default_dtype", []() { return GetDefaultDType(); });
  m.def("set_default_dtype",
        [](const Symbol<DType>& dtype) { SetDefaultDType(dtype).GetOrThrow(); });
  m.def("set_default_tensor_type", [](const py::object& tensor_type) {
    if (one::PyTensorType_Check(tensor_type.ptr())) {
      CHECK_JUST(SetDefaultDType(one::PyTensorType_UnpackDType(tensor_type.ptr())));
    } else {
      throw py::type_error("invalid type object");
    }
  });
}

}  // namespace oneflow

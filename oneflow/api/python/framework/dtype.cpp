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
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/common/symbol.h"
namespace py = pybind11;

namespace oneflow {

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<Symbol<DType>, std::shared_ptr<Symbol<DType>>>(m, "dtype")
      .def_property_readonly("is_signed", [](const Symbol<DType>& d) { return d->is_signed(); })
      .def_property_readonly("is_complex", [](const Symbol<DType>& d) { return d->is_complex(); })
      .def_property_readonly("is_floating_point", [](const Symbol<DType>& d) { return d->is_floating_point(); })
      .def("__str__", [](const Symbol<DType>& d) { return d->name(); })
      .def("__repr__", [](const Symbol<DType>& d) { return d->name(); })
      .def(py::self == py::self)
      .def(py::hash(py::self))
      .def_property_readonly("bytes",
                             [](const Symbol<DType>& dtype) { return dtype->bytes().GetOrThrow(); });

  m.attr("char") = [](const Symbol<DType>& d) { return d->Char().get(); };
  m.attr("float16") = [](const Symbol<DType>& d) { return d->Float16().get(); };
  m.attr("float") = [](const Symbol<DType>& d) { return d->Float().get(); };

  m.attr("float32") = [](const Symbol<DType>& d) { return d->Float().get(); };
  m.attr("double") = [](const Symbol<DType>& d) { return d->Double().get(); };
  m.attr("float64") = [](const Symbol<DType>& d) { return d->Double().get(); };

  m.attr("int8") = [](const Symbol<DType>& d) { return d->Int8().get(); };
  m.attr("int32") = [](const Symbol<DType>& d) { return d->Int32().get(); };
  m.attr("int64") = [](const Symbol<DType>& d) { return d->Int64().get(); };

  m.attr("uint8") = [](const Symbol<DType>& d) { return d->UInt8().get(); };
  m.attr("record") = [](const Symbol<DType>& d) { return d->OFRecord().get(); };
  m.attr("tensor_buffer") = [](const Symbol<DType>& d) { return d->TensorBuffer().get(); };
}

}  // namespace oneflow

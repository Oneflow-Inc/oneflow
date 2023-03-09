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
#include "oneflow/core/framework/layout.h"

namespace py = pybind11;

namespace oneflow {

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<Symbol<Layout>>(m, "layout")
      .def("__str__", [](Symbol<Layout> d) { return d->name(); })
      .def("__repr__", [](Symbol<Layout> d) { return d->name(); })
      .def(py::self == py::self)
      .def(py::hash(py::self))
      .def(py::pickle(
          [](Symbol<Layout> layout) {  // __getstate__
            return static_cast<int>(layout->layout_type());
          },
          [](int t) {  // __setstate__
            return Layout::Get(LayoutType(t));
          }))
      .def("get", [](const int layout_type_enum) {
        return Layout::Get(static_cast<LayoutType>(layout_type_enum));
      });

  m.attr("strided") = Layout::Get(LayoutType::kStrided);
}

}  // namespace oneflow

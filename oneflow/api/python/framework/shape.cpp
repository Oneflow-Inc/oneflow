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

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<Shape, std::shared_ptr<Shape>>(m, "Shape")
      .def("element_count", &Shape::elem_cnt)
      .def("as_list",
           [](const std::shared_ptr<Shape>& x) {
             py::list ret;
             for (auto& dim : x->dim_vec()) { ret.append((int64_t)dim); }
             return ret;
           })
      .def("as_tuple", [](const std::shared_ptr<Shape>& x) {
        py::tuple ret(x->NumAxes());
        for (int i = 0; i < x->NumAxes(); ++i) { ret[i] = x->At(i); }
        return ret;
      });
}

}  // namespace oneflow

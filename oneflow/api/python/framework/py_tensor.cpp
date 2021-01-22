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
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/job/placement.cfg.h"
#include "oneflow/core/autograd/autograd.h"

namespace py = pybind11;

namespace oneflow {

namespace one {

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
      .def(py::init([](const py::tuple& py_shape, int dtype) {
        DimVector shape_dims;
        CHECK(py::isinstance<py::tuple>(py_shape));
        for (auto dim : py_shape) { shape_dims.emplace_back(dim.cast<int64_t>()); }
        std::shared_ptr<Shape> shape = std::make_shared<Shape>(shape_dims);
        std::shared_ptr<cfg::ParallelConf> parallel_conf = std::make_shared<cfg::ParallelConf>();
        return std::make_shared<Tensor>(shape, static_cast<DataType>(dtype), parallel_conf);
      }))
      .def_property_readonly("parallel_conf", &Tensor::parallel_conf)
      .def_property_readonly("shape", &Tensor::shape)
      .def("get_dtype", [](std::shared_ptr<Tensor>& x) { return static_cast<int>(x->dtype()); })
      .def("storage", &Tensor::storage)
      .def_property_readonly("is_leaf", &Tensor::is_leaf)
      .def_property_readonly("ndim", &Tensor::dim)
      .def_property_readonly("grad", &Tensor::grad)
      .def("size", &Tensor::shape)
      .def("backward", &Tensor::Backward);
}

}  // namespace one

}  // namespace oneflow

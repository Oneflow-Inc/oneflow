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

class PyTensor : public Tensor {
 public:
  using Tensor::Tensor;
};

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<Tensor, PyTensor, std::shared_ptr<Tensor>>(m, "Tensor")
      .def(py::init([](std::shared_ptr<Shape> shape, cfg::DataType dtype,
                       std::shared_ptr<cfg::ParallelConf> parallel_conf) {
        return std::make_shared<PyTensor>(shape, dtype, parallel_conf);
      }))
      .def(py::init([](){
        return std::make_shared<PyTensor>();
      }))
      .def_property_readonly("parallel_conf", &Tensor::parallel_conf)
      .def_property_readonly("shape", &Tensor::shape)
      .def_property_readonly("dtype", &Tensor::dtype)
      .def_property_readonly("storage", &Tensor::storage)
      .def_property_readonly("defined", &Tensor::defined)
      .def_property_readonly("has_storage", &Tensor::has_storage)
      .def_property_readonly("required_grad", &Tensor::requires_grad)
      .def_property_readonly("is_leaf", &Tensor::is_leaf)
      .def_property_readonly("dim", &Tensor::dim)
      .def_property_readonly("grad", &Tensor::grad)
      .def("Backward", &Tensor::Backward)
      .def("SetFuncNode", &Tensor::SetFuncNode);
}

}  // namespace one

}  // namespace oneflow

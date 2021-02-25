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
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/py_distribute.h"
#include "oneflow/core/job/placement.cfg.h"
#include "oneflow/core/job/global_for.h"

namespace py = pybind11;

namespace oneflow {

namespace one {

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<MirroredTensor, std::shared_ptr<MirroredTensor>>(m, "MirroredTensor")
      .def(py::init([](const py::tuple& py_shape, int dtype,
                       const std::shared_ptr<Device>& device) {
        DimVector shape_dims;
        CHECK(py::isinstance<py::tuple>(py_shape));
        for (auto dim : py_shape) { shape_dims.emplace_back(dim.cast<int64_t>()); }
        std::shared_ptr<Shape> shape = std::make_shared<Shape>(shape_dims);
        std::shared_ptr<MirroredTensorImpl> impl;
        if (*Global<bool, EagerExecution>::Get()) {
          impl = std::static_pointer_cast<MirroredTensorImpl>(
              std::make_shared<EagerMirroredTensorImpl>(shape, static_cast<DataType>(dtype),
                                                        device));
        } else {
          impl =
              std::static_pointer_cast<MirroredTensorImpl>(std::make_shared<LazyMirroredTensorImpl>(
                  shape, static_cast<DataType>(dtype), device));
        }
        return std::make_shared<MirroredTensor>(impl);
      }))
      .def_property_readonly("shape",
                             [](std::shared_ptr<MirroredTensor>& x) {
                               return std::const_pointer_cast<Shape>(x->shape());
                             })
      .def_property_readonly("device",
                             [](std::shared_ptr<MirroredTensor>& x) {
                               return std::const_pointer_cast<Device>(x->device());
                             })
      .def_property_readonly("data", []() {})
      .def("get_dtype",
           [](std::shared_ptr<MirroredTensor>& x) { return static_cast<int>(x->dtype()); })
      .def("size", &MirroredTensor::shape)
      .def("storage", []() {});

  py::class_<ConsistentTensor, std::shared_ptr<ConsistentTensor>>(m, "ConsistentTensor")
      .def(py::init([](const std::shared_ptr<Shape>& shape, DataType dtype,
                       const std::shared_ptr<compatible_py::Distribute>& distribute,
                       std::shared_ptr<ParallelDesc>& parallel_desc) {
        std::shared_ptr<ConsistentTensorImpl> impl;
        if (*Global<bool, EagerExecution>::Get()) {
          impl = std::static_pointer_cast<ConsistentTensorImpl>(
              std::make_shared<EagerConsistentTensorImpl>(shape, static_cast<DataType>(dtype),
                                                          distribute, parallel_desc));
        } else {
          impl = std::static_pointer_cast<ConsistentTensorImpl>(
              std::make_shared<LazyConsistentTensorImpl>(shape, static_cast<DataType>(dtype),
                                                         distribute, parallel_desc));
        }
        return std::make_shared<ConsistentTensor>(impl);
      }))
      .def_property_readonly("shape",
                             [](std::shared_ptr<ConsistentTensor>& x) {
                               return std::const_pointer_cast<Shape>(x->shape());
                             })
      .def("get_dtype",
           [](std::shared_ptr<ConsistentTensor>& x) { return static_cast<int>(x->dtype()); })
      .def("size", &ConsistentTensor::shape);
}

}  // namespace one

}  // namespace oneflow

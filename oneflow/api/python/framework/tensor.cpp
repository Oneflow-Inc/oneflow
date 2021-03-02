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
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/autograd/autograd_engine.h"

namespace py = pybind11;

namespace oneflow {

namespace one {

namespace {

template<typename T>
struct TensorExportUtil final {};

template<>
struct TensorExportUtil<FacadeTensor> final {
  static Maybe<FacadeTensor> MakeTensor(
      const std::shared_ptr<const Shape>& shape, const std::shared_ptr<const DType>& dtype,
      const std::shared_ptr<const Device>& device,
      const std::shared_ptr<const ParallelDesc>& parallel_desc,
      const std::shared_ptr<const compatible_py::Distribute>& distribute, bool is_lazy,
      bool requires_grad, bool is_leaf, bool retain_grad, bool is_determined, bool is_consistent) {
    std::shared_ptr<Tensor> tensor;
    if (is_determined) {
      if (is_consistent) {
        tensor = std::static_pointer_cast<Tensor>(ConsistentTensor::MakeTensor(
            shape, dtype, distribute, parallel_desc, is_lazy, requires_grad, is_leaf, retain_grad));
      } else {
        tensor = std::static_pointer_cast<Tensor>(MirroredTensor::MakeTensor(
            shape, dtype, device, is_lazy, requires_grad, is_leaf, retain_grad));
      }
    } else {
      tensor = std::make_shared<UndeterminedTensor>(shape, dtype, is_leaf, retain_grad);
    }
    return std::make_shared<FacadeTensor>(tensor);
  }

  static std::shared_ptr<FacadeTensor> ApiMakeTensor(
      const std::shared_ptr<const Shape>& shape, const std::shared_ptr<const DType>& dtype,
      const std::shared_ptr<const Device>& device,
      const std::shared_ptr<const ParallelDesc>& parallel_desc,
      const std::shared_ptr<const compatible_py::Distribute>& distribute, bool is_lazy,
      bool requires_grad, bool is_leaf, bool retain_grad, bool is_determined, bool is_consistent) {
    return MakeTensor(shape, dtype, device, parallel_desc, distribute, is_lazy, requires_grad,
                      is_leaf, retain_grad, is_determined, is_consistent)
        .GetPtrOrThrow();
  }
};

template<typename T>
void ExportTensor(py::module& m, const char* name) {
  py::class_<T, std::shared_ptr<T>>(m, name)
      .def(py::init(&TensorExportUtil<T>::ApiMakeTensor))
      // Properties of pytorch
      .def_property_readonly("shape",
                             [](const T& tensor) { return tensor.shape().GetPtrOrThrow(); })
      .def_property_readonly("device",
                             [](const T& tensor) { return tensor.device().GetPtrOrThrow(); })
      .def_property_readonly("ndim", [](const T& tensor) { return tensor.ndim().GetOrThrow(); })
      .def_property_readonly("is_cuda",
                             [](const T& tensor) { return tensor.is_cuda().GetOrThrow(); })
      .def_property_readonly("dtype",
                             [](const T& tensor) { return tensor.device().GetPtrOrThrow(); })
      .def_property_readonly("data", []() { TODO(); })
      .def_property_readonly("grad",
                             [](const T& tensor) { return tensor.acc_grad().GetPtrOrThrow(); })
      .def_property_readonly("grad_fn",
                             [](const T& tensor) { return tensor.grad_fn_node().GetPtrOrThrow(); })
      .def_property_readonly("requires_grad",
                             [](const T& tensor) { return tensor.requires_grad().GetOrThrow(); })
      .def_property_readonly("is_leaf",
                             [](const T& tensor) { return tensor.is_leaf().GetOrThrow(); })
      // Methods of pytorch
      .def("size", [](const T& tensor) { return tensor.shape().GetOrThrow(); })
      .def("dim", [](const T& tensor, int index) { return tensor.dim(index).GetOrThrow(); })
      .def("ndimension", [](const T& tensor) { return tensor.ndim().GetOrThrow(); })
      .def("get_device", [](const T& tensor) { return tensor.device().GetPtrOrThrow(); })
      .def("nelement", [](const T& tensor) { return tensor.nelement().GetOrThrow(); })
      .def("data_ptr", []() { TODO(); })
      .def("element_size", []() { TODO(); })
      .def("numpy", []() { TODO(); })
      .def("tolist", []() { TODO(); })
      .def("backward", []() { TODO(); })
      .def("__str__", []() { TODO(); })
      .def("__repr__", []() { TODO(); })
      .def("__array__", []() { TODO(); })
      .def("__sizeof__", []() { TODO(); })
      // OneFlow tensor properties other than pytorch tensor
      .def_property_readonly("placement",
                             [](const T& tensor) { return tensor.parallel_desc().GetPtrOrThrow(); })
      .def_property_readonly("is_lazy", [](const T& tensor) { return tensor.is_lazy(); })
      .def_property_readonly("is_consistent",
                             [](const T& tensor) { return tensor.is_consistent().GetOrThrow(); });
}

}  // namespace

ONEFLOW_API_PYBIND11_MODULE("", m) { ExportTensor<FacadeTensor>(m, "Tensor"); }

}  // namespace one

}  // namespace oneflow

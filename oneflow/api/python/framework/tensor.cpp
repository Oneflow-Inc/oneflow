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
struct TensorExportUtil<MirroredTensor> final {
  static Maybe<MirroredTensor> MakeTensor(const py::tuple& py_shape,
                                          const std::shared_ptr<const DType>& dtype,
                                          const std::shared_ptr<const Device>& device, bool is_lazy,
                                          bool requires_grad, bool is_leaf, bool retain_grad) {
    DimVector shape_dims;
    CHECK_OR_RETURN(py::isinstance<py::tuple>(py_shape))
        << Error::ValueError("Input shape must be tuple.");
    for (const auto& dim : py_shape) { shape_dims.emplace_back(dim.cast<int64_t>()); }
    std::shared_ptr<Shape> shape = std::make_shared<Shape>(shape_dims);
    return MirroredTensor::MakeTensor(shape, dtype, device, is_lazy, requires_grad, is_leaf,
                                      retain_grad);
  }

  static std::shared_ptr<MirroredTensor> ApiMakeTensor(const py::tuple& py_shape,
                                                       const std::shared_ptr<const DType>& dtype,
                                                       const std::shared_ptr<const Device>& device,
                                                       bool is_lazy, bool requires_grad,
                                                       bool is_leaf, bool retain_grad) {
    return MakeTensor(py_shape, dtype, device, is_lazy, requires_grad, is_leaf, retain_grad)
        .GetPtrOrThrow();
  }
};

template<>
struct TensorExportUtil<ConsistentTensor> final {
  static Maybe<ConsistentTensor> MakeTensor(
      const py::tuple& py_shape, const std::shared_ptr<const DType>& dtype,
      const std::shared_ptr<const compatible_py::Distribute>& distribute,
      const std::shared_ptr<const ParallelDesc>& parallel_desc, bool is_lazy, bool requires_grad,
      bool is_leaf, bool retain_grad) {
    DimVector shape_dims;
    CHECK_OR_RETURN(py::isinstance<py::tuple>(py_shape))
        << Error::ValueError("Input shape must be tuple.");
    for (const auto& dim : py_shape) { shape_dims.emplace_back(dim.cast<int64_t>()); }
    std::shared_ptr<Shape> shape = std::make_shared<Shape>(shape_dims);
    return ConsistentTensor::MakeTensor(shape, dtype, distribute, parallel_desc, is_lazy,
                                        requires_grad, is_leaf, retain_grad);
  }

  static std::shared_ptr<ConsistentTensor> ApiMakeTensor(
      const py::tuple& py_shape, const std::shared_ptr<const DType>& dtype,
      const std::shared_ptr<const compatible_py::Distribute>& distribute,
      const std::shared_ptr<const ParallelDesc>& parallel_desc, bool is_lazy, bool requires_grad,
      bool is_leaf, bool retain_grad) {
    return MakeTensor(py_shape, dtype, distribute, parallel_desc, is_lazy, requires_grad, is_leaf,
                      retain_grad)
        .GetPtrOrThrow();
  }
};

template<typename T>
void ExportTensor(py::module& m, const char* name) {
  py::class_<T, std::shared_ptr<T>>(m, name)
      .def(py::init(&TensorExportUtil<T>::ApiMakeTensor))
      // Properties of pytorch
      .def_property_readonly("shape", &T::shape)
      .def_property_readonly("device",
                             [](const T& tensor) { return tensor.device().GetPtrOrThrow(); })
      .def_property_readonly("ndim", &T::ndim)
      .def_property_readonly("is_cuda",
                             [](const T& tensor) { return tensor.is_cuda().GetOrThrow(); })
      .def_property_readonly("dtype", &T::dtype)
      .def_property_readonly("data", []() { TODO(); })
      .def_property_readonly("grad", &T::acc_grad)
      .def_property_readonly("grad_fn", &T::grad_fn_node)
      .def_property_readonly("requires_grad", &T::requires_grad)
      .def_property_readonly("is_leaf", &T::is_leaf)
      // Methods of pytorch
      .def("size", &T::shape)
      .def("dim", &T::dim)
      .def("ndimension", &T::ndim)
      .def("get_device", &T::device)
      .def("nelement", &T::nelement)
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

ONEFLOW_API_PYBIND11_MODULE("", m) {
  ExportTensor<MirroredTensor>(m, "LocalTensor");
  ExportTensor<ConsistentTensor>(m, "ConsistentTensor");
}

}  // namespace one

}  // namespace oneflow

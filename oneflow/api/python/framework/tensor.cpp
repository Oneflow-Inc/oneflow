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

  static std::shared_ptr<FacadeTensor> MakeFacadeTensor(const py::tuple& py_shape,
                                                        const std::shared_ptr<const DType>& dtype,
                                                        const std::shared_ptr<const Device>& device,
                                                        bool is_lazy, bool requires_grad,
                                                        bool is_leaf, bool retain_grad) {
    return std::make_shared<FacadeTensor>(
        MakeTensor(py_shape, dtype, device, is_lazy, requires_grad, is_leaf, retain_grad)
            .GetPtrOrThrow());
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

  static std::shared_ptr<FacadeTensor> MakeFacadeTensor(
      const py::tuple& py_shape, const std::shared_ptr<const DType>& dtype,
      const std::shared_ptr<const compatible_py::Distribute>& distribute,
      const std::shared_ptr<const ParallelDesc>& parallel_desc, bool is_lazy, bool requires_grad,
      bool is_leaf, bool retain_grad) {
    return std::make_shared<FacadeTensor>(MakeTensor(py_shape, dtype, distribute, parallel_desc,
                                                     is_lazy, requires_grad, is_leaf, retain_grad)
                                              .GetPtrOrThrow());
  }
};

template<>
struct TensorExportUtil<UndeterminedTensor> final {
  static Maybe<UndeterminedTensor> MakeTensor(const py::tuple& py_shape,
                                              const std::shared_ptr<const DType>& dtype,
                                              bool is_lazy, bool requires_grad, bool is_leaf,
                                              bool retain_grad) {
    DimVector shape_dims;
    CHECK_OR_RETURN(py::isinstance<py::tuple>(py_shape))
        << Error::ValueError("Input shape must be tuple.");
    for (const auto& dim : py_shape) { shape_dims.emplace_back(dim.cast<int64_t>()); }
    std::shared_ptr<Shape> shape = std::make_shared<Shape>(shape_dims);
    return std::make_shared<UndeterminedTensor>(shape, dtype, is_lazy, requires_grad, is_leaf,
                                                retain_grad);
  }

  static std::shared_ptr<FacadeTensor> MakeFacadeTensor(
      const py::tuple& py_shape, const std::shared_ptr<const DType>& dtype,
      const std::shared_ptr<const Device>& device,
      const std::shared_ptr<const ParallelDesc>& parallel_desc,
      const std::shared_ptr<const compatible_py::Distribute>& distribute, bool is_lazy,
      bool requires_grad, bool is_leaf, bool retain_grad) {
    return std::make_shared<FacadeTensor>(
        MakeTensor(py_shape, dtype, is_lazy, requires_grad, is_leaf, retain_grad).GetPtrOrThrow());
  }
};

}  // namespace

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<FacadeTensor, std::shared_ptr<FacadeTensor>>(m, "Tensor")
      .def(py::init(&TensorExportUtil<MirroredTensor>::MakeFacadeTensor))
      .def(py::init(&TensorExportUtil<ConsistentTensor>::MakeFacadeTensor))
      .def(py::init(&TensorExportUtil<UndeterminedTensor>::MakeFacadeTensor))
      // Properties of pytorch
      .def_property_readonly("shape",
                             [](const std::shared_ptr<FacadeTensor>& tensor) {
                               return tensor->shape().GetPtrOrThrow();
                             })
      .def_property_readonly("device",
                             [](const std::shared_ptr<FacadeTensor>& tensor) {
                               return tensor->device().GetPtrOrThrow();
                             })
      .def_property_readonly(
          "ndim", [](std::shared_ptr<FacadeTensor>& tensor) { return tensor->ndim().GetOrThrow(); })
      .def_property_readonly(
          "is_cuda",
          [](std::shared_ptr<FacadeTensor>& tensor) { return tensor->is_cuda().GetOrThrow(); })
      .def_property_readonly("dtype",
                             [](const std::shared_ptr<FacadeTensor>& tensor) {
                               return tensor->device().GetPtrOrThrow();
                             })
      .def_property_readonly("data", []() { TODO(); })
      .def_property_readonly(
          "grad",
          [](std::shared_ptr<FacadeTensor>& tensor) { return tensor->acc_grad().GetPtrOrThrow(); })
      .def_property_readonly("grad_fn",
                             [](std::shared_ptr<FacadeTensor>& tensor) {
                               return tensor->grad_fn_node().GetPtrOrThrow();
                             })
      .def_property_readonly("requires_grad",
                             [](const std::shared_ptr<FacadeTensor>& tensor) {
                               return tensor->requires_grad().GetOrThrow();
                             })
      .def_property_readonly("is_leaf",
                             [](const std::shared_ptr<FacadeTensor>& tensor) {
                               return tensor->is_leaf().GetOrThrow();
                             })
      // Methods of pytorch
      .def("size",
           [](const std::shared_ptr<FacadeTensor>& tensor) {
             return tensor->shape().GetPtrOrThrow();
           })
      .def("dim", [](std::shared_ptr<FacadeTensor>& tensor,
                     int index) { return tensor->dim(index).GetOrThrow(); })
      .def("ndimension",
           [](std::shared_ptr<FacadeTensor>& tensor) { return tensor->ndim().GetOrThrow(); })
      .def("get_device",
           [](const std::shared_ptr<FacadeTensor>& tensor) {
             return tensor->device().GetPtrOrThrow();
           })
      .def("nelement",
           [](std::shared_ptr<FacadeTensor>& tensor) { return tensor->nelement().GetOrThrow(); })
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
                             [](const std::shared_ptr<FacadeTensor>& tensor) {
                               return tensor->parallel_desc().GetPtrOrThrow();
                             })
      .def_property_readonly(
          "is_lazy", [](const std::shared_ptr<FacadeTensor>& tensor) { return tensor->is_lazy(); })
      .def_property_readonly("is_consistent", [](const std::shared_ptr<FacadeTensor>& tensor) {
        return tensor->is_consistent().GetOrThrow();
      });
}

}  // namespace one

}  // namespace oneflow

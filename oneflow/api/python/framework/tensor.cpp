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
  static std::shared_ptr<MirroredTensor> MakeTensor(const std::shared_ptr<const Shape>& shape,
                                                    const std::shared_ptr<const DType>& dtype,
                                                    const std::shared_ptr<const Device>& device,
                                                    bool is_lazy, bool requires_grad, bool is_leaf,
                                                    bool retain_grad) {
    return MirroredTensor::MakeTensor(shape, dtype, device, is_lazy, requires_grad, is_leaf,
                                      retain_grad);
  }
};

template<>
struct TensorExportUtil<ConsistentTensor> final {
  static std::shared_ptr<ConsistentTensor> MakeTensor(
      const std::shared_ptr<const Shape>& shape, const std::shared_ptr<const DType>& dtype,
      const std::shared_ptr<const compatible_py::Distribute>& distribute,
      const std::shared_ptr<const ParallelDesc>& parallel_desc, bool is_lazy, bool requires_grad,
      bool is_leaf, bool retain_grad) {
    return ConsistentTensor::MakeTensor(shape, dtype, distribute, parallel_desc, is_lazy,
                                        requires_grad, is_leaf, retain_grad);
  }
};

template<typename T>
void ExportTensor(py::module& m, const char* name) {
  py::class_<T, Tensor, std::shared_ptr<T>>(m, name)
      .def(py::init(&TensorExportUtil<T>::MakeTensor))
      // Properties of pytorch
      .def_property_readonly("shape", &T::shape)
      .def_property_readonly("device", &T::device)
      .def_property_readonly("is_cuda", &T::is_cuda)
      .def_property_readonly("dtype", &T::dtype)
      .def_property_readonly("data", []() { TODO(); })
      .def_property_readonly("grad", [](const T& t) { return t.api_acc_grad().GetPtrOrThrow(); })
      .def_property_readonly("grad_fn", &T::grad_fn_node)
      .def_property_readonly("requires_grad", &T::requires_grad)
      .def_property_readonly("is_leaf", &T::is_leaf)
      // Methods of pytorch
      .def("data_ptr", []() { TODO(); })
      .def("numpy", []() { TODO(); })
      .def("tolist", []() { TODO(); })
      .def("retain_grad", [](T& t) { t.set_retain_grad(true); })
      .def("__str__", []() { TODO(); })
      .def("__repr__", []() { TODO(); })
      // OneFlow tensor properties other than pytorch tensor
      .def_property_readonly("placement", &T::parallel_desc)
      .def_property_readonly("is_lazy", &T::is_lazy)
      .def_property_readonly("is_consistent", &T::is_consistent)
      .def_property_readonly("_blob_object", &T::blob_object)
      // OneFlow tensor methods other than pytorch tensor
      .def("_set_blob_object", &T::set_blob_object);
}

}  // namespace

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor");

  ExportTensor<MirroredTensor>(m, "LocalTensor");
  ExportTensor<ConsistentTensor>(m, "ConsistentTensor");
}

}  // namespace one

}  // namespace oneflow

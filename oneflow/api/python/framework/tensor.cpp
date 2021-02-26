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

namespace {

template<typename T>
std::shared_ptr<T> MakeTensor(const py::tuple& py_shape, int dtype,
                              const std::shared_ptr<Device>& device,
                              const std::shared_ptr<compatible_py::Distribute>& distribute,
                              std::shared_ptr<ParallelDesc>& parallel_desc) {
  UNIMPLEMENTED();
}

template<>
std::shared_ptr<MirroredTensor> MakeTensor<MirroredTensor>(const py::tuple& py_shape, int dtype,
                              const std::shared_ptr<Device>& device,
                              const std::shared_ptr<compatible_py::Distribute>& distribute,
                              std::shared_ptr<ParallelDesc>& parallel_desc) {
  DimVector shape_dims;
  CHECK(py::isinstance<py::tuple>(py_shape));
  for (auto dim : py_shape) { shape_dims.emplace_back(dim.cast<int64_t>()); }
  std::shared_ptr<Shape> shape = std::make_shared<Shape>(shape_dims);
  std::shared_ptr<MirroredTensorImpl> impl;
  if (*Global<bool, EagerExecution>::Get()) {
    impl = std::make_shared<EagerMirroredTensorImpl>(shape, static_cast<DataType>(dtype), device);
  } else {
    impl = std::make_shared<LazyMirroredTensorImpl>(shape, static_cast<DataType>(dtype), device);
  }
  return std::make_shared<MirroredTensor>(impl);
}

template<>
std::shared_ptr<ConsistentTensor> MakeTensor<ConsistentTensor>(const py::tuple& py_shape, int dtype,
                              const std::shared_ptr<Device>& device,
                              const std::shared_ptr<compatible_py::Distribute>& distribute,
                              std::shared_ptr<ParallelDesc>& parallel_desc) {
  DimVector shape_dims;
  CHECK(py::isinstance<py::tuple>(py_shape));
  for (auto dim : py_shape) { shape_dims.emplace_back(dim.cast<int64_t>()); }
  std::shared_ptr<Shape> shape = std::make_shared<Shape>(shape_dims);
  std::shared_ptr<ConsistentTensorImpl> impl;
  if (*Global<bool, EagerExecution>::Get()) {
    impl = std::make_shared<EagerConsistentTensorImpl>(shape, static_cast<DataType>(dtype),
                                                       distribute, parallel_desc);
  } else {
    impl = std::make_shared<LazyConsistentTensorImpl>(shape, static_cast<DataType>(dtype),
                                                      distribute, parallel_desc);
  }
  return std::make_shared<ConsistentTensor>(impl);
}

template<typename T>
void ExportTensor(py::module& m, const char* name) {
  py::class_<T, std::shared_ptr<T>>(m, name)
      .def(py::init(&MakeTensor<T>))
      .def_property_readonly("shape", &T::shape)
      .def_property_readonly("device", &T::device)
      .def_property_readonly("ndim", &T::ndim)
      .def_property_readonly("is_cuda", &T::is_cuda)
      .def_property_readonly("dtype", &T::dtype)
      .def_property_readonly("data", &T::data)
      .def_property_readonly("grad", &T::grad)
      .def_property_readonly("grad_fn", &T::grad_fn)
      .def_property_readonly("requires_grad", &T::requires_grad)
      .def_property_readonly("is_leaf", &T::is_leaf)
      .def("size", &T::shape)
      .def("dim", &T::dim)
      .def("ndimension", &T::ndimension)
      .def("get_device", &T::get_device)
      .def("nelement", &T::nelement)
      .def("data_ptr", &T::data_ptr)
      .def("element_size", &T::element_size)
      .def("numpy", &T::numpy)
      .def("tolist", &T::tolist)
      .def("backward", &T::backward)
      .def("__str__", &T::ToString)
      .def("__repr__", &T::ToString)
      .def("__array__", &T::ToArray)
      .def("__sizeof__", &T::SizeOf);
}

}  // namespace

ONEFLOW_API_PYBIND11_MODULE("", m) {
  ExportTensor<MirroredTensor>(m, "LocalTensor");
  ExportTensor<ConsistentTensor>(m, "MirroredTensor");
}

}  // namespace one

}  // namespace oneflow

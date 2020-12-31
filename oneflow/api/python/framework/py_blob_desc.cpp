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
#include "oneflow/core/framework/py_blob_desc.h"

namespace py = pybind11;

namespace oneflow {

namespace compatible_py {

class PyBlobDesc : public BlobDesc {
 public:
  using BlobDesc::BlobDesc;
  std::shared_ptr<BlobDesc> Clone() const override {
    PYBIND11_OVERRIDE_PURE(std::shared_ptr<BlobDesc>, BlobDesc, Clone, );
  }

  std::shared_ptr<BlobDesc> with_distribute(
      const std::shared_ptr<Distribute>& distribute) const override {
    PYBIND11_OVERRIDE_PURE(std::shared_ptr<BlobDesc>, BlobDesc, with_distribute, distribute);
  }
};

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<BlobDesc, PyBlobDesc, std::shared_ptr<BlobDesc>>(m, "BlobDesc")
      .def(py::init(
          [](std::shared_ptr<cfg::LogicalBlobId> lbi, std::shared_ptr<Distribute> distribute) {
            return std::make_shared<PyBlobDesc>(lbi, distribute);
          }))
      .def_property_readonly("lbi", &BlobDesc::lbi)
      .def_property_readonly("logical_blob_name", &BlobDesc::logical_blob_name)
      .def_property_readonly("op_name", &BlobDesc::op_name)
      .def_property_readonly("blob_name", &BlobDesc::blob_name)
      .def_property_readonly("shape", &BlobDesc::shape)
      .def_property_readonly("dtype", &BlobDesc::dtype)
      .def_property_readonly("batch_axis", &BlobDesc::batch_axis)
      .def_property_readonly("is_dynamic", &BlobDesc::is_dynamic)
      .def_property_readonly("is_tensor_list", &BlobDesc::is_tensor_list)
      .def_property_readonly("parallel_conf", &BlobDesc::parallel_conf)
      .def_property_readonly("distribute", &BlobDesc::distribute)
      .def_property_readonly("unique_name", &BlobDesc::unique_name)
      .def("Clone", &BlobDesc::Clone)
      .def("set_distribute", &BlobDesc::set_distribute)
      .def("with_distribute", &BlobDesc::with_distribute)
      .def("with_split_distribute",
           [](const std::shared_ptr<BlobDesc>& blob_desc, int64_t axis) {
             return blob_desc->with_split_distribute(axis).GetPtrOrThrow();
           })
      .def("with_broadcast_distribute", &BlobDesc::with_broadcast_distribute);
}

}  // namespace compatible_py

}  // namespace oneflow

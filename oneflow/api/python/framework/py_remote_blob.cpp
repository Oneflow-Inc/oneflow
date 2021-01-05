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
#include "oneflow/core/framework/py_remote_blob.h"

namespace py = pybind11;

namespace oneflow {

namespace compatible_py {

std::shared_ptr<BlobDesc> ConsistentBlob::Clone() const {
  PYBIND11_OVERRIDE_PURE(std::shared_ptr<BlobDesc>, ConsistentBlob, Clone, );
}

std::string LazyConsistentBlob::get_shape_log_warning() const {
  PYBIND11_OVERRIDE_PURE(std::string, LazyConsistentBlob, get_shape_log_warning, );
}

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<BlobDesc, std::shared_ptr<BlobDesc>>(m, "BlobDesc")
      .def(py::init(
          [](std::shared_ptr<cfg::LogicalBlobId> lbi, std::shared_ptr<Distribute> distribute) {
            return std::make_shared<BlobDesc>(lbi, distribute);
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

  py::class_<ConsistentBlob, BlobDesc, std::shared_ptr<ConsistentBlob>>(m, "ConsistentBlob")
      .def(py::init([](std::shared_ptr<cfg::LogicalBlobId> lbi, std::string job_name,
                       std::shared_ptr<Distribute> distribute) {
        return std::make_shared<ConsistentBlob>(lbi, job_name, distribute);
      }))
      .def_property_readonly("lbi", &ConsistentBlob::lbi)
      .def_property_readonly("logical_blob_name", &ConsistentBlob::logical_blob_name)
      .def_property_readonly("op_name", &ConsistentBlob::op_name)
      .def_property_readonly("blob_name", &ConsistentBlob::blob_name)
      .def_property_readonly("shape", &ConsistentBlob::shape)
      .def_property_readonly("dtype", &ConsistentBlob::dtype)
      .def_property_readonly("batch_axis", &ConsistentBlob::batch_axis)
      .def_property_readonly("is_dynamic", &ConsistentBlob::is_dynamic)
      .def_property_readonly("is_tensor_list", &ConsistentBlob::is_tensor_list)
      .def_property_readonly("parallel_conf", &ConsistentBlob::parallel_conf)
      .def_property_readonly("distribute", &ConsistentBlob::distribute)
      .def_property_readonly("unique_name", &ConsistentBlob::unique_name)
      .def_property_readonly("job_name", &ConsistentBlob::job_name)
      .def_property_readonly("parallel_size", &ConsistentBlob::parallel_size)
      .def("set_job_name", &ConsistentBlob::set_job_name)
      .def("with_distribute", &ConsistentBlob::with_distribute);

  py::class_<LazyConsistentBlob, ConsistentBlob, std::shared_ptr<LazyConsistentBlob>>(
      m, "LazyConsistentBlob")
      .def(py::init([](std::shared_ptr<cfg::LogicalBlobId> lbi, std::string job_name,
                       std::shared_ptr<Distribute> distribute) {
        return std::make_shared<LazyConsistentBlob>(lbi, job_name, distribute);
      }))
      .def_property_readonly("shape", &LazyConsistentBlob::shape)
      .def_property_readonly(
          "dtype", [](const std::shared_ptr<LazyConsistentBlob>& x) { return int(x->dtype()); })
      .def_property_readonly("batch_axis", &LazyConsistentBlob::batch_axis)
      .def_property_readonly("split_axis", &LazyConsistentBlob::batch_axis)
      .def_property_readonly("is_dynamic", &LazyConsistentBlob::is_dynamic)
      .def_property_readonly("is_tensor_list", &LazyConsistentBlob::is_tensor_list)
      .def_property_readonly("parallel_conf", &LazyConsistentBlob::parallel_conf)
      .def("IdenticalTo", &LazyConsistentBlob::IdenticalTo)
      .def("with_gradient_distribute", &LazyConsistentBlob::with_gradient_distribute)
      .def("get_shape_log_warning", &LazyConsistentBlob::get_shape_log_warning);
}

}  // namespace compatible_py

}  // namespace oneflow

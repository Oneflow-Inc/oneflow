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
#include <pybind11/functional.h>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/framework/object.h"
#include "oneflow/core/framework/opkernel_object.h"

namespace py = pybind11;

namespace oneflow {

namespace compatible_py {

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<Object, std::shared_ptr<Object>>(m, "Object")
      .def(py::init(
          [](int64_t object_id, const std::shared_ptr<ParallelDesc>& parallel_desc_symbol) {
            return std::make_shared<Object>(object_id, parallel_desc_symbol);
          }))
      .def_property_readonly("object_id", &Object::object_id)
      .def_property_readonly("parallel_desc_symbol", &Object::parallel_desc_symbol);

  py::class_<BlobObject, Object, std::shared_ptr<BlobObject>>(m, "BlobObject")
      .def(py::init([](int64_t object_id,
                       const std::shared_ptr<OpArgParallelAttribute>& op_arg_parallel_attr,
                       const std::shared_ptr<OpArgBlobAttribute>& op_arg_blob_attr) {
        return std::make_shared<BlobObject>(object_id, op_arg_parallel_attr, op_arg_blob_attr);
      }))
      .def_property_readonly("object_id", &BlobObject::object_id)
      .def_property_readonly("parallel_desc_symbol", &BlobObject::parallel_desc_symbol)
      .def_property_readonly("op_arg_parallel_attr", &BlobObject::op_arg_parallel_attr)
      .def_property_readonly("op_arg_blob_attr", &BlobObject::op_arg_blob_attr)
      .def("add_releaser", &BlobObject::add_releaser)
      .def("ForceReleaseAll", &BlobObject::ForceReleaseAll);

  py::class_<OpKernelObject, Object, std::shared_ptr<OpKernelObject>>(m, "OpKernelObject")
      .def(py::init([](int64_t object_id, const std::shared_ptr<cfg::OperatorConf>& op_conf,
                       const std::function<void(Object*)>& release) {
        return std::make_shared<OpKernelObject>(object_id, op_conf, release);
      }))
      .def_property_readonly("object_id", &OpKernelObject::object_id)
      .def_property_readonly("parallel_desc_symbol", &OpKernelObject::parallel_desc_symbol)
      .def_property_readonly("op_conf", &OpKernelObject::op_conf)
      .def_property_readonly("scope_symbol", &OpKernelObject::scope_symbol);
}

}  // namespace compatible_py

}  // namespace oneflow

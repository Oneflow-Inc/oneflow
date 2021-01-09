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
#include <pybind11/operators.h>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/framework/op_arg_util.h"

namespace py = pybind11;

namespace oneflow {

namespace compatible_py {

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<OpArgBlobAttribute, std::shared_ptr<OpArgBlobAttribute>>(m, "OpArgBlobAttribute")
      .def(py::init([](const std::shared_ptr<cfg::OptInt64>& batch_axis,
                       const std::shared_ptr<cfg::BlobDescProto>& blob_desc,
                       const std::string& logical_blob_name) {
        return std::make_shared<OpArgBlobAttribute>(batch_axis, blob_desc, logical_blob_name);
      }))
      .def_property_readonly("batch_axis", &OpArgBlobAttribute::batch_axis)
      .def_property_readonly("blob_desc", &OpArgBlobAttribute::blob_desc)
      .def_property_readonly("logical_blob_name", &OpArgBlobAttribute::logical_blob_name)
      .def_property_readonly("is_tensor_list", &OpArgBlobAttribute::is_tensor_list)
      .def_property_readonly("is_dynamic", &OpArgBlobAttribute::is_dynamic)
      .def_property_readonly("shape",
                             [](const std::shared_ptr<OpArgBlobAttribute>& x) {
                               const auto& x_shape = x->shape();
                               py::tuple ret(x_shape->NumAxes());
                               for (int i = 0; i < x_shape->NumAxes(); ++i) {
                                 ret[i] = x_shape->At(i);
                               }
                               return ret;
                             })
      .def("get_dtype",
           [](const std::shared_ptr<OpArgBlobAttribute>& x) {
             return static_cast<int>(x->get_dtype());
           })
      .def("DumpToInterfaceBlobConf", &OpArgBlobAttribute::DumpToInterfaceBlobConf)
      .def(py::self == py::self);

  py::class_<OpArgParallelAttribute, std::shared_ptr<OpArgParallelAttribute>>(
      m, "OpArgParallelAttribute")
      .def(py::init([](std::shared_ptr<ParallelDesc> parallel_desc,
                       std::shared_ptr<cfg::SbpParallel> sbp_parallel,
                       std::shared_ptr<cfg::OptMirroredParallel> opt_mirrored_parallel) {
        return std::make_shared<OpArgParallelAttribute>(parallel_desc, sbp_parallel,
                                                        opt_mirrored_parallel);
      }))
      .def_property_readonly("parallel_desc_symbol", &OpArgParallelAttribute::parallel_desc_symbol)
      .def_property_readonly("sbp_parallel", &OpArgParallelAttribute::sbp_parallel)
      .def_property_readonly("opt_mirrored_parallel",
                             &OpArgParallelAttribute::opt_mirrored_parallel)
      .def("is_mirrored", &OpArgParallelAttribute::is_mirrored)
      .def("_Hash", &OpArgParallelAttribute::_Hash)
      .def("Assign", &OpArgParallelAttribute::Assign)
      .def("DumpToInterfaceBlobConf", &OpArgParallelAttribute::DumpToInterfaceBlobConf)
      .def("__str__", &OpArgParallelAttribute::ToString)
      .def("__repr__", &OpArgParallelAttribute::ToString)
      .def(py::self == py::self)
      .def(py::hash(py::self));
}

}  // namespace compatible_py

}  // namespace oneflow

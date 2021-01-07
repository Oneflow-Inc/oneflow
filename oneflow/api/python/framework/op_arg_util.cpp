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
      .def("__str__", &OpArgParallelAttribute::ToString)
      .def("__repr__", &OpArgParallelAttribute::ToString)
      .def(py::self == py::self)
      .def(py::hash(py::self));
}

}  // namespace compatible_py

}  // namespace oneflow

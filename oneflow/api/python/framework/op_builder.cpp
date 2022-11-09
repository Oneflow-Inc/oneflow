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
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/throw.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_builder.h"

namespace py = pybind11;

namespace oneflow {

namespace one {

ONEFLOW_API_PYBIND11_MODULE("one", m) {
  py::class_<one::OpBuilder, std::shared_ptr<one::OpBuilder>>(m, "OpBuilder")
      .def(py::init<const std::string&>())
      .def(py::init<const std::string&, const std::string&>())
      .def("input", &OpBuilder::MaybeInput)
      .def("output", &OpBuilder::MaybeOutput)
      .def("attr",
           [](const std::shared_ptr<one::OpBuilder>& x, const std::string& attr_name,
              const std::string& attr_val_str) -> Maybe<OpBuilder&> {
             AttrValue attr_val;
             if (!TxtString2PbMessage(attr_val_str, &attr_val)) {
               THROW(RuntimeError) << "attr val parse failed.\n" << attr_val_str;
             }
             return x->MaybeAttr(attr_name, attr_val);
           })
      .def("build", &OpBuilder::Build);
}

}  // namespace one

}  // namespace oneflow

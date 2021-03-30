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
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_builder.h"

namespace py = pybind11;

namespace oneflow {

std::shared_ptr<one::OpBuilder> OpBuilder_Input(const std::shared_ptr<one::OpBuilder>& builder,
                                                const std::string& input_name,
                                                const int input_num) {
  builder->MaybeInput(input_name, input_num).GetOrThrow();
  return builder;
}

std::shared_ptr<one::OpBuilder> OpBuilder_Output(const std::shared_ptr<one::OpBuilder>& builder,
                                                 const std::string& output_name,
                                                 const int output_num) {
  builder->MaybeOutput(output_name, output_num).GetOrThrow();
  return builder;
}

std::shared_ptr<one::OpBuilder> OpBuilder_Attr(const std::shared_ptr<one::OpBuilder>& builder,
                                               const std::string& attr_name,
                                               const cfg::AttrValue& attr_value) {
  builder->MaybeAttr(attr_name, attr_value).GetOrThrow();
  return builder;
}

ONEFLOW_API_PYBIND11_MODULE("one", m) {
  py::class_<one::OpBuilder, std::shared_ptr<one::OpBuilder>>(m, "OpBuilder")
      .def(py::init<const std::string&>())
      .def(py::init<const std::string&, const std::string&>())
      .def("input", &OpBuilder_Input)
      .def("output", &OpBuilder_Output)
      .def("attr", &OpBuilder_Attr)
      .def("build", [](const std::shared_ptr<one::OpBuilder>& builder) {
        return builder->Build().GetPtrOrThrow();
      });
}

}  // namespace oneflow

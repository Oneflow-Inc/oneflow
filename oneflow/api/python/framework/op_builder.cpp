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

one::OpBuilder& OpBuilder_Name(const std::shared_ptr<one::OpBuilder>& builder,
                               const std::string& op_name) {
  return builder->MaybeName(op_name).GetOrThrow();
}

one::OpBuilder& OpBuilder_Op(const std::shared_ptr<one::OpBuilder>& builder,
                             const std::string& op_type_name) {
  return builder->MaybeOp(op_type_name).GetOrThrow();
}

one::OpBuilder& OpBuilder_Input(const std::shared_ptr<one::OpBuilder>& builder,
                                const std::string& input_name, const int input_num) {
  return builder->MaybeInput(input_name, input_num).GetOrThrow();
}

one::OpBuilder& OpBuilder_Output(const std::shared_ptr<one::OpBuilder>& builder,
                                 const std::string& output_name, const int output_num) {
  return builder->MaybeOutput(output_name, output_num).GetOrThrow();
}

one::OpBuilder& OpBuilder_Attr(const std::shared_ptr<one::OpBuilder>& builder,
                               const std::string& attr_name,
                               const std::string& serialized_attr_value) {
  AttrValue attr_value;
  TxtString2PbMessage(serialized_attr_value, &attr_value);
  return builder->MaybeAttr(attr_name, attr_value).GetOrThrow();
}

ONEFLOW_API_PYBIND11_MODULE("one", m) {
  py::class_<one::OpBuilder, std::shared_ptr<one::OpBuilder>>(m, "OpBuilder")
      .def(py::init<>())
      .def(py::init<const std::string&>())
      .def("Name", &OpBuilder_Name)
      .def("Op", &OpBuilder_Op)
      .def("Input", &OpBuilder_Input)
      .def("Output", &OpBuilder_Output)
      .def("Attr", &OpBuilder_Attr)
      .def("Build", [](const std::shared_ptr<one::OpBuilder>& builder) {
        return builder->Build().GetPtrOrThrow();
      });
}

}  // namespace oneflow

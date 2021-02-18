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

namespace py = pybind11;

namespace oneflow {

std::vector<std::shared_ptr<one::Tensor>> Interpret(
    const std::shared_ptr<one::OpExpr>& op,
    const std::vector<std::shared_ptr<one::Tensor>>& inputs) {
  // TODO(): Execute the op by Autograd.
  return std::vector<std::shared_ptr<one::Tensor>>{};
}

ONEFLOW_API_PYBIND11_MODULE("one", m) {
  py::class_<one::OpExpr, std::shared_ptr<one::OpExpr>>(m, "OpExpr")
      .def("__call__", [](const std::shared_ptr<one::OpExpr>& op_expr,
                          const std::vector<std::shared_ptr<one::Tensor>>& inputs) {
        return Interpret(op_expr, inputs);
      });

  py::class_<one::BuiltinOpExpr, one::OpExpr, std::shared_ptr<one::BuiltinOpExpr>>(m,
                                                                                   "BuiltinOpExpr")
      .def_property_readonly("name", &one::BuiltinOpExpr::op_name);

  py::class_<one::UserOpExpr, one::BuiltinOpExpr, std::shared_ptr<one::UserOpExpr>>(m, "UserOpExpr")
      .def(py::init<>())
      .def(py::init([](const std::string& op_name, const std::string& serialized_proto) {
        UserOpConf proto;
        TxtString2PbMessage(serialized_proto, &proto);
        return std::make_shared<one::UserOpExpr>(op_name, std::move(proto));
      }))
      .def_property_readonly("type", &one::UserOpExpr::type)
      .def_property_readonly(
          "proto", [](const one::UserOpExpr& op) { return PbMessage2TxtString(op.proto()); })
      .def_property_readonly("indexed_input_names", &one::UserOpExpr::indexed_input_names);
}

}  // namespace oneflow

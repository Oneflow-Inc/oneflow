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
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter.h"
#include "oneflow/core/framework/op_interpreter_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"

namespace py = pybind11;

namespace oneflow {

namespace {

Maybe<one::TensorTuple> Interpret(const std::shared_ptr<one::OpExpr>& op,
                                  const one::TensorTuple& inputs) {
  CHECK_EQ_OR_RETURN(op->input_num(), inputs.size())
      << "The operation requires " << op->input_num() << " inputs, but " << inputs.size()
      << " is given.";
  one::OpExprInterpState state;
  auto outputs = std::make_shared<one::TensorTuple>(op->output_num());
  auto interperter = JUST(one::OpInterpUtil::GetInterpreter());
  JUST(interperter->Apply(*op.get(), &state, inputs, outputs.get()));
  return outputs;
}

Maybe<std::vector<std::shared_ptr<one::Tensor>>> Interpret(
    const std::shared_ptr<one::OpExpr>& op,
    const std::vector<std::shared_ptr<one::Tensor>>& inputs) {
  one::TensorTuple input_list(inputs.size());
  for (int i = 0; i < inputs.size(); ++i) { input_list[i] = inputs[i]; }
  const auto& outputs = JUST(Interpret(op, input_list));
  return static_cast<std::shared_ptr<std::vector<std::shared_ptr<one::Tensor>>>>(outputs);
}

}  // namespace

ONEFLOW_API_PYBIND11_MODULE("one", m) {
  py::class_<one::OpExpr, std::shared_ptr<one::OpExpr>>(m, "OpExpr")
      .def("apply",
           [](const std::shared_ptr<one::OpExpr>& op_expr,
              const std::vector<std::shared_ptr<one::Tensor>>& inputs) {
             return Interpret(op_expr, inputs).GetOrThrow();
           })
      .def("apply", [](const std::shared_ptr<one::OpExpr>& op_expr,
                       const std::shared_ptr<one::TensorTuple>& inputs) {
        return Interpret(op_expr, *inputs).GetPtrOrThrow();
      });

  py::class_<one::BuiltinOpExpr, one::OpExpr, std::shared_ptr<one::BuiltinOpExpr>>(m,
                                                                                   "BuiltinOpExpr")
      .def_property_readonly("name", &one::BuiltinOpExpr::op_name);

  py::class_<one::UserOpExpr, one::BuiltinOpExpr, std::shared_ptr<one::UserOpExpr>>(m, "UserOpExpr")
      .def(py::init<>())
      .def_property_readonly("type", &one::UserOpExpr::type)
      .def_property_readonly(
          "proto", [](const one::UserOpExpr& op) { return PbMessage2TxtString(op.proto()); })
      .def_property_readonly("indexed_ibns", &one::UserOpExpr::indexed_ibns)
      .def_property_readonly("indexed_obns", &one::UserOpExpr::indexed_obns);
}

}  // namespace oneflow

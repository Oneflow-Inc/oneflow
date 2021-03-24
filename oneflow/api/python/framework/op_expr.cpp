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
#include "oneflow/core/framework/user_op_conf.cfg.h"

namespace py = pybind11;

namespace oneflow {

namespace {

Maybe<one::TensorTuple> Interpret(const one::OpExpr& op, const one::TensorTuple& inputs) {
  CHECK_EQ_OR_RETURN(op.input_num(), inputs.size())
      << "The operation requires " << op.input_num() << " inputs, but " << inputs.size()
      << " is given.";
  auto outputs = std::make_shared<one::TensorTuple>(op.output_num());
  auto interperter = JUST(one::OpInterpUtil::GetInterpreter());
  JUST(interperter->Apply(op, inputs, outputs.get()));
  return outputs;
}

Maybe<std::vector<std::shared_ptr<one::Tensor>>> Interpret(
    const one::OpExpr& op, const std::vector<std::shared_ptr<one::Tensor>>& inputs) {
  one::TensorTuple input_list(inputs.size());
  for (int i = 0; i < inputs.size(); ++i) { input_list[i] = inputs[i]; }
  const auto& outputs = JUST(Interpret(op, input_list));
  return static_cast<std::shared_ptr<std::vector<std::shared_ptr<one::Tensor>>>>(outputs);
}

}  // namespace

ONEFLOW_API_PYBIND11_MODULE("one", m) {
  py::class_<one::OpExpr, std::shared_ptr<one::OpExpr>>(m, "OpExpr")
      .def_property_readonly("type", &one::OpExpr::type)
      .def_property_readonly("input_num", &one::OpExpr::input_num)
      .def_property_readonly("output_num", &one::OpExpr::output_num)
      .def("apply",
           [](const one::OpExpr& op_expr, const std::vector<std::shared_ptr<one::Tensor>>& inputs) {
             return Interpret(op_expr, inputs).GetOrThrow();
           })
      .def("apply", [](const one::OpExpr& op_expr, const one::TensorTuple& inputs) {
        return Interpret(op_expr, inputs).GetPtrOrThrow();
      });

  py::class_<one::BuiltinOpExpr, one::OpExpr, std::shared_ptr<one::BuiltinOpExpr>>(m,
                                                                                   "BuiltinOpExpr")
      .def_property_readonly("name", &one::BuiltinOpExpr::op_name)
      .def_property_readonly("indexed_ibns", &one::BuiltinOpExpr::indexed_ibns)
      .def_property_readonly("indexed_obns", &one::BuiltinOpExpr::indexed_obns);

#define PYBIND_EXPORT_OP_EXPR(_op_type) PYBIND_EXPORT_OP_EXPR_IMPL(_op_type##Expr, _op_type##Conf)

#define PYBIND_EXPORT_OP_EXPR_IMPL(_op_expr_type, _op_conf_type)                      \
  {                                                                                   \
    using T = one::_op_expr_type;                                                     \
    py::class_<T, one::BuiltinOpExpr, std::shared_ptr<T>>(m, #_op_expr_type)          \
        .def(py::init([](const std::string& op_name,                                  \
                         const std::shared_ptr<cfg::_op_conf_type>& op_conf,          \
                         const std::vector<std::string>& indexed_ibns,                \
                         const std::vector<std::string>& indexed_obns) {              \
          _op_conf_type proto_op_conf;                                                \
          op_conf->ToProto(&proto_op_conf);                                           \
          return std::make_shared<T>(op_name, std::move(proto_op_conf), indexed_ibns, \
                                     indexed_obns);                                   \
        }))                                                                           \
        .def_property_readonly("proto", [](const T& op) {                             \
          return std::make_shared<cfg::_op_conf_type>(op.proto());                    \
        });                                                                           \
  }

  PYBIND_EXPORT_OP_EXPR(UserOp);
  PYBIND_EXPORT_OP_EXPR(VariableOp);

#undef PYBIND_EXPORT_OP_EXPR
}

}  // namespace oneflow

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
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/user_op_conf.cfg.h"

namespace py = pybind11;

namespace oneflow {

namespace {

Maybe<one::TensorTuple> Interpret(const one::OpExpr& op, const one::TensorTuple& inputs,
                                  const AttrMap& attrs) {
  CHECK_EQ_OR_RETURN(op.input_size(), inputs.size())
      << "The operation requires " << op.input_size() << " inputs, but " << inputs.size()
      << " is given.";
  return JUST(one::OpInterpUtil::Dispatch<one::TensorTuple>(op, inputs, attrs));
}

Maybe<one::TensorTuple> Interpret(const one::OpExpr& op,
                                  const std::vector<std::shared_ptr<one::Tensor>>& inputs,
                                  const AttrMap& attrs) {
  one::TensorTuple input_list(inputs.size());
  for (int i = 0; i < inputs.size(); ++i) { input_list[i] = inputs[i]; }
  return JUST(Interpret(op, input_list, attrs));
}

Maybe<one::TensorTuple> Interpret(const one::OpExpr& op, const Symbol<ParallelDesc>& placement,
                                  const std::vector<Symbol<cfg::SbpParallel>>& sbp_tuple,
                                  const AttrMap& attrs) {
  CHECK_EQ_OR_RETURN(op.input_size(), 0)
      << " the op :  " << op.op_type_name()
      << " is NOT source op with input_size = " << op.input_size();
  const auto& nd_sbp = JUST(GetNdSbp(sbp_tuple));
  return JUST(one::OpInterpUtil::Dispatch<one::TensorTuple>(
      op, {}, one::OpExprInterpContext(attrs, placement, nd_sbp)));
}

Maybe<one::TensorTuple> Interpret(const one::OpExpr& op, const Symbol<Device>& device,
                                  const AttrMap& attrs) {
  CHECK_EQ_OR_RETURN(op.input_size(), 0)
      << " the op :  " << op.op_type_name()
      << " is NOT source op with input_size = " << op.input_size();
  return JUST(one::OpInterpUtil::Dispatch<one::TensorTuple>(
      op, {}, one::OpExprInterpContext(attrs, device)));
}

template<typename OpT, typename ConfT,
         typename std::enable_if<std::is_base_of<one::BuiltinOpExpr, OpT>::value>::type* = nullptr>
py::class_<OpT, one::BuiltinOpExpr, std::shared_ptr<OpT>> PybindExportOpExpr(
    py::module& m, const char* op_type_name) {
  using ProtoConfT = decltype(std::declval<OpT>().proto());
  return py::class_<OpT, one::BuiltinOpExpr, std::shared_ptr<OpT>>(m, op_type_name)
      .def(py::init([](const std::string& op_name, const std::shared_ptr<ConfT>& op_conf,
                       const std::vector<std::string>& indexed_ibns,
                       const std::vector<std::string>& indexed_obns) {
        typename std::decay<ProtoConfT>::type proto_op_conf;
        op_conf->ToProto(&proto_op_conf);
        return OpT::New(op_name, std::move(proto_op_conf), indexed_ibns, indexed_obns)
            .GetPtrOrThrow();
      }))
      .def_property_readonly("proto",
                             [](const OpT& op) { return std::make_shared<ConfT>(op.proto()); });
}

}  // namespace

ONEFLOW_API_PYBIND11_MODULE("one", m) {
  py::class_<one::OpExpr, std::shared_ptr<one::OpExpr>>(m, "OpExpr")
      .def_property_readonly("op_type_name", &one::OpExpr::op_type_name)
      .def_property_readonly("input_size", &one::OpExpr::input_size)
      .def_property_readonly("output_size", &one::OpExpr::output_size)
      .def("apply",
           [](const one::OpExpr& op_expr, const std::vector<std::shared_ptr<one::Tensor>>& inputs,
              const MutableCfgAttrMap& attrs) {
             return Interpret(op_expr, inputs, attrs).GetPtrOrThrow();
           })
      .def("apply",
           [](const one::OpExpr& op_expr, const one::TensorTuple& inputs,
              const MutableCfgAttrMap& attrs) {
             return Interpret(op_expr, inputs, attrs).GetPtrOrThrow();
           })
      .def("apply",
           [](const one::OpExpr& op_expr, const Symbol<ParallelDesc>& placement,
              const std::vector<Symbol<cfg::SbpParallel>>& sbp_tuple,
              const MutableCfgAttrMap& attrs) {
             return Interpret(op_expr, placement, sbp_tuple, attrs).GetPtrOrThrow();
           })
      .def("apply", [](const one::OpExpr& op_expr, const Symbol<Device>& device,
                       const MutableCfgAttrMap& attrs) {
        return Interpret(op_expr, device, attrs).GetPtrOrThrow();
      });

  py::class_<one::BuiltinOpExpr, one::OpExpr, std::shared_ptr<one::BuiltinOpExpr>>(m,
                                                                                   "BuiltinOpExpr")
      .def_property_readonly("name", &one::BuiltinOpExpr::op_name)
      .def_property_readonly("indexed_ibns", &one::BuiltinOpExpr::indexed_ibns)
      .def_property_readonly("indexed_obns", &one::BuiltinOpExpr::indexed_obns);

  auto py_user_op_class = PybindExportOpExpr<one::UserOpExpr, cfg::UserOpConf>(m, "UserOpExpr");
  py_user_op_class.def_property_readonly(
      "op_type_name", [](const one::UserOpExpr& op) { return op.proto().op_type_name(); });
  PybindExportOpExpr<one::VariableOpExpr, cfg::VariableOpConf>(m, "VariableOpExpr");
  // NOTE(chengcheng): export for Lazy nn.Graph Feed/Fetch EagerTensor to/from LazyTensor.
  PybindExportOpExpr<one::FeedInputOpExpr, cfg::FeedInputOpConf>(m, "FeedInputOpExpr");
  PybindExportOpExpr<one::FeedVariableOpExpr, cfg::FeedVariableOpConf>(m, "FeedVariableOpExpr");
  PybindExportOpExpr<one::FetchOutputOpExpr, cfg::FetchOutputOpConf>(m, "FetchOutputOpExpr");
  PybindExportOpExpr<one::ImageDecoderRandomCropResizeOpExpr,
                     cfg::ImageDecoderRandomCropResizeOpConf>(m,
                                                              "ImageDecoderRandomCropResizeOpExpr");
}

}  // namespace oneflow

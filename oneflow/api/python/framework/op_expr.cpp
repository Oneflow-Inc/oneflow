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
#include "oneflow/core/common/throw.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"

namespace py = pybind11;

namespace oneflow {

namespace {

template<typename OpT, typename ConfT,
         typename std::enable_if<std::is_base_of<one::BuiltinOpExpr, OpT>::value>::type* = nullptr>
py::class_<OpT, one::BuiltinOpExpr, std::shared_ptr<OpT>> PybindExportOpExpr(
    py::module& m, const char* op_type_name) {
  return py::class_<OpT, one::BuiltinOpExpr, std::shared_ptr<OpT>>(m, op_type_name)
      .def(py::init([](const std::string& op_name, const std::string& op_conf_str,
                       const std::vector<std::string>& indexed_ibns,
                       const std::vector<std::string>& indexed_obns) {
        ConfT proto_op_conf;
        if (!TxtString2PbMessage(op_conf_str, &proto_op_conf)) {
          THROW(RuntimeError) << "op conf parse failed.\n" << op_conf_str;
        }
        return OpT::New(op_name, std::move(proto_op_conf), indexed_ibns, indexed_obns)
            .GetPtrOrThrow();
      }));
}

}  // namespace

ONEFLOW_API_PYBIND11_MODULE("one", m) {
  py::class_<one::OpExpr, std::shared_ptr<one::OpExpr>>(m, "OpExpr")
      .def_property_readonly("op_type_name", &one::OpExpr::op_type_name)
      .def_property_readonly("input_size", &one::OpExpr::input_size)
      .def_property_readonly("output_size", &one::OpExpr::output_size);

  py::class_<one::BuiltinOpExpr, one::OpExpr, std::shared_ptr<one::BuiltinOpExpr>>(m,
                                                                                   "BuiltinOpExpr")
      .def_property_readonly("name", &one::BuiltinOpExpr::op_name)
      .def_property_readonly("indexed_ibns", &one::BuiltinOpExpr::indexed_ibns)
      .def_property_readonly("indexed_obns", &one::BuiltinOpExpr::indexed_obns);

  auto py_user_op_class = PybindExportOpExpr<one::UserOpExpr, UserOpConf>(m, "UserOpExpr");
  py_user_op_class.def_property_readonly(
      "op_type_name", [](const one::UserOpExpr& op) { return op.proto().op_type_name(); });
  PybindExportOpExpr<one::VariableOpExpr, VariableOpConf>(m, "VariableOpExpr");
  // NOTE(chengcheng): export for Lazy nn.Graph Feed/Fetch EagerTensor to/from LazyTensor.
  PybindExportOpExpr<one::FeedInputOpExpr, FeedInputOpConf>(m, "FeedInputOpExpr");
  PybindExportOpExpr<one::FeedVariableOpExpr, FeedVariableOpConf>(m, "FeedVariableOpExpr");
  PybindExportOpExpr<one::FetchOutputOpExpr, FetchOutputOpConf>(m, "FetchOutputOpExpr");
  PybindExportOpExpr<one::ImageDecoderRandomCropResizeOpExpr, ImageDecoderRandomCropResizeOpConf>(
      m, "ImageDecoderRandomCropResizeOpExpr");
}

}  // namespace oneflow

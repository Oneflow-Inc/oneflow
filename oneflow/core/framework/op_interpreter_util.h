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
#ifndef ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_UTIL_H_
#define ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_UTIL_H_

#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/op_arg_util.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/scope_util.h"
#include "oneflow/core/framework/session_util.h"

namespace oneflow {
namespace one {

using TensorList = std::vector<std::shared_ptr<Tensor>>;

class OpInterpUtil {
 public:
  static std::shared_ptr<OperatorConf> GenBuiltinOpConf(const BuiltinOpExpr* op_expr);

  static std::shared_ptr<cfg::OpAttribute> AddBuiltinOpAndInferOpAttribute(
      const OperatorConf& op_conf, const bool is_mirrored_strategy_enabled);

  static std::shared_ptr<cfg::OpAttribute> AddBuiltinOpAndInferOpAttribute(
      const BuiltinOpExpr* op_expr, const std::shared_ptr<Scope>& scope,
      const bool is_mirrored_strategy_enabled);

  static void InitVariableOutputBlob(const std::shared_ptr<Session>& session,
                                     const std::shared_ptr<Tensor>& output,
                                     const OpAttribute& op_attribute);

 private:
  static std::shared_ptr<OperatorConf> GenModelInitOpConf(const OperatorConf& variable_conf);
  static std::shared_ptr<OperatorConf> GenModelIOPathInputOpConf();
  static std::shared_ptr<OperatorConf> GenModelLoadOpConf(const OperatorConf& variable_conf,
                                                          const OperatorConf& path_input_op_conf);

  using Bn2BlobObjectMap = HashMap<std::string, std::shared_ptr<compatible_py::BlobObject>>;

  static std::shared_ptr<cfg::OpAttribute> InferOpAttribute(
      const OperatorConf& op_conf, const std::shared_ptr<Scope>& scope,
      const Bn2BlobObjectMap& ibn2blob_object);

  static std::function<void(const std::shared_ptr<InstructionsBuilder>&)>
  BuildModelInitOrIOPathInputInstruction(const OperatorConf& op_conf,
                                         const std::shared_ptr<Bn2BlobObjectMap>& bn2blob_object);

  static std::function<void(const std::shared_ptr<InstructionsBuilder>&)> BuildFeedPathInstruction(
      const std::string& path, const std::shared_ptr<Bn2BlobObjectMap>& bn2blob_object);

  static std::shared_ptr<compatible_py::BlobObject> EagerRunModelInit(const OperatorConf& op_conf);

  static std::shared_ptr<compatible_py::BlobObject> EagerRunModelLoad(
      const OperatorConf& op_conf, const std::string& snapshot_path);

  static void Assign(const std::shared_ptr<compatible_py::BlobObject>& target_blob_object,
                     const std::shared_ptr<compatible_py::BlobObject>& blob_object);
};

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_UTIL_H_

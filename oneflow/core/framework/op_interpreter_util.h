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
#include "oneflow/core/framework/op_interpreter.h"
#include "oneflow/core/framework/scope_util.h"
#include "oneflow/core/framework/session_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"

namespace oneflow {
namespace one {

class OpInterpUtil {
 public:
  static Maybe<OpExprInterpreter> GetOrCreateInterpreter();

  static Maybe<OperatorConf> GenBuiltinOpConf(const BuiltinOpExpr* op_expr);

  static Maybe<cfg::OpAttribute> AddBuiltinOpAndInferOpAttribute(
      const OperatorConf& op_conf, const bool is_mirrored_strategy_enabled);

  static Maybe<cfg::OpAttribute> AddBuiltinOpAndInferOpAttribute(
      const BuiltinOpExpr* op_expr, const std::shared_ptr<Scope>& scope,
      const bool is_mirrored_strategy_enabled);

  static Maybe<cfg::OpAttribute> InferOpAttribute(const BuiltinOpExpr* op_expr,
                                                  const std::shared_ptr<Scope>& scope,
                                                  const TensorTuple& inputs);

  static Maybe<compatible_py::BlobObject> GetTensorBlobObject(
      const std::shared_ptr<Tensor>& tensor);

  static Maybe<void> InitVariableOutputBlob(const std::shared_ptr<Session>& session,
                                            const std::shared_ptr<Tensor>& output,
                                            const OpAttribute& op_attribute);

  using Bn2BlobObjectMap = HashMap<std::string, std::shared_ptr<compatible_py::BlobObject>>;

  static Maybe<Bn2BlobObjectMap> MakeBn2BlobObjectMap(const std::vector<std::string>& indexed_ibns,
                                                      const TensorTuple& inputs);

 private:
  static Maybe<OperatorConf> GenModelInitOpConf(const OperatorConf& variable_conf);
  static Maybe<OperatorConf> GenModelIOPathInputOpConf();
  static Maybe<OperatorConf> GenModelLoadOpConf(const OperatorConf& variable_conf,
                                                const OperatorConf& path_input_op_conf);

  static Maybe<cfg::OpAttribute> InferOpAttribute(const OperatorConf& op_conf,
                                                  const std::shared_ptr<Scope>& scope,
                                                  const Bn2BlobObjectMap& ibn2blob_object);

  static Maybe<std::function<void(const std::shared_ptr<InstructionsBuilder>&)>>
  BuildModelInitOrIOPathInputInstruction(const OperatorConf& op_conf,
                                         const std::shared_ptr<Bn2BlobObjectMap>& bn2blob_object);

  static Maybe<std::function<void(const std::shared_ptr<InstructionsBuilder>&)>>
  BuildFeedPathInstruction(const std::string& path,
                           const std::shared_ptr<Bn2BlobObjectMap>& bn2blob_object);

  static Maybe<compatible_py::BlobObject> EagerRunModelInit(const OperatorConf& op_conf);

  static Maybe<compatible_py::BlobObject> EagerRunModelLoad(const OperatorConf& op_conf,
                                                            const std::string& snapshot_path);

  static Maybe<void> Assign(const std::shared_ptr<compatible_py::BlobObject>& target_blob_object,
                            const std::shared_ptr<compatible_py::BlobObject>& blob_object);
};

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_UTIL_H_

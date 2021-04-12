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
  static Maybe<AutogradInterpreter> GetInterpreter();

  static Maybe<OperatorConf> GenBuiltinOpConf(const BuiltinOpExpr& op_expr);

  static Maybe<cfg::OpAttribute> AddOpAndInferOpAttribute(const OperatorConf& op_conf,
                                                          const bool is_mirrored_strategy_enabled);

  static Maybe<cfg::OpAttribute> InferOpAttribute(const BuiltinOpExpr& op_expr,
                                                  const TensorTuple& inputs);

  static Maybe<HashMap<std::string, std::shared_ptr<compatible_py::BlobObject>>>
  MakeBn2BlobObjectMap(const std::vector<std::string>& indexed_ibns, const TensorTuple& inputs);

  static Maybe<compatible_py::BlobObject> GetTensorBlobObject(
      const std::shared_ptr<Tensor>& tensor);

  static Maybe<Tensor> BuildTensor(
      const std::shared_ptr<compatible_py::OpArgBlobAttribute>& blob_attr,
      const std::shared_ptr<compatible_py::OpArgParallelAttribute>& parallel_attr,
      const bool is_lazy);

  static Maybe<Tensor> BuildTensorFromBlobObject(
      const std::shared_ptr<compatible_py::BlobObject>& blob_object);
};

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_UTIL_H_

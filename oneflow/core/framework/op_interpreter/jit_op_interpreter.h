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
#ifndef ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_JIT_OP_INTERPRETER_H_
#define ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_JIT_OP_INTERPRETER_H_

#include "oneflow/core/framework/op_interpreter.h"
#include "oneflow/ir/include/OneFlow/JIT.h"
#include "mlir/IR/Builders.h"

namespace oneflow {

namespace one {

using namespace mlir;

class JitInterpreter : public OpExprInterpreter {
 public:
  JitInterpreter()
      : context_(new MLIRContext()),
        module_(ir::CreateJitModule(context_)),
        importer_(context_, *module_) {}
  ~JitInterpreter() = default;

  Maybe<void> Apply(const OpExpr& op_expr, const TensorTuple& inputs, TensorTuple* outputs,
                    const AttrMap& attrs) const {
    return Apply(op_expr, inputs, outputs, OpExprInterpContext(attrs));
  }
  Maybe<void> Apply(const OpExpr& op_expr, const TensorTuple& inputs, TensorTuple* outputs,
                    const OpExprInterpContext& ctx) const override;

 private:
  DECLARE_NORMAL_APPLY_FUNC(UserOp);  // note(BBuf) jit deal with user op only, now.
  MLIRContext* context_;
  OwningOpRef<ModuleOp> module_;
  ir::JitImporter importer_;
};

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_JIT_OP_INTERPRETER_H_

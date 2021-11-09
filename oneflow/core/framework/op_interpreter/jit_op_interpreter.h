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

#ifdef WITH_MLIR

#include "oneflow/core/framework/op_interpreter.h"
#include "oneflow/ir/include/OneFlow/JIT.h"
#include "mlir/IR/Builders.h"
#include "oneflow/core/framework/op_expr.h"

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
  void Interrupt();
  ir::JitImporter& GetImporter() { return importer_; }
  void CacheExpr(Operation&, std::shared_ptr<one::UserOpExpr>);
  llvm::Optional<std::shared_ptr<one::UserOpExpr>> GetExpr(Operation*);
  void Start() { start_time_ = std::chrono::steady_clock::now(); }
  void MlirTraceEnd() { mlir_end_time_ = std::chrono::steady_clock::now(); }
  void End() { end_time_ = std::chrono::steady_clock::now(); }
  float MlirTraceOverhead() {
    const float mlir_trace_time =
        std::chrono::duration_cast<std::chrono::microseconds>(mlir_end_time_ - start_time_).count();
    const float jit_time =
        std::chrono::duration_cast<std::chrono::microseconds>(end_time_ - start_time_).count();
    return mlir_trace_time / jit_time;
  }

 private:
  DECLARE_NORMAL_APPLY_FUNC(UserOp);  // note(BBuf) jit deal with user op only, now.
  mutable llvm::DenseMap<llvm::hash_code, std::shared_ptr<one::UserOpExpr>> cached_user_op_exprs_;
  mutable MLIRContext* context_;
  mutable OwningOpRef<ModuleOp> module_;
  mutable ir::JitImporter importer_;
  mutable std::chrono::steady_clock::time_point start_time_;
  mutable std::chrono::steady_clock::time_point mlir_end_time_;
  mutable std::chrono::steady_clock::time_point end_time_;
};

}  // namespace one
}  // namespace oneflow

#endif  // WITH_MLIR

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_JIT_OP_INTERPRETER_H_

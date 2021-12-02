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

#ifndef ONEFLOW_IR_ONEFLOW_JIT_INCLUDE_ONEFLOW_JIT_OP_INTERPRETER_H_
#define ONEFLOW_IR_ONEFLOW_JIT_INCLUDE_ONEFLOW_JIT_OP_INTERPRETER_H_

#ifdef WITH_MLIR

#include "oneflow/core/framework/op_interpreter.h"
#include "oneflow/ir/oneflow-jit/include/OneFlow/JIT.h"
#include "mlir/IR/Builders.h"
#include "oneflow/core/framework/op_expr.h"

namespace oneflow {

namespace one {

using namespace mlir;

class JitFunctionContext {
 public:
  JitFunctionContext() = default;
  JitFunctionContext(const std::string& func_name,
                     const std::vector<std::shared_ptr<one::Tensor>>& arg_tensors)
      : func_name_(func_name), forward_args_(arg_tensors){};
  JitFunctionContext(JitFunctionContext&&) = default;
  JitFunctionContext(const JitFunctionContext&) = default;
  JitFunctionContext& operator=(JitFunctionContext&&) = default;
  JitFunctionContext& operator=(const JitFunctionContext&) = default;
  ~JitFunctionContext() = default;

  std::string GetFuncName() { return func_name_; }
  const std::string& GetJitFuncName() { return func_name_; }
  const std::vector<std::shared_ptr<one::Tensor>>& GetJitForwardArgs() { return forward_args_; }

 private:
  std::string func_name_;
  std::vector<std::shared_ptr<one::Tensor>> forward_args_;
};

class JitInterpreter : public OpExprInterpreter {
 public:
  JitInterpreter()
      : context_(new MLIRContext()),
        module_(ir::CreateJitModule(context_)),
        importer_(context_, *module_),
        current_importer_(&importer_) {}
  ~JitInterpreter() = default;

  static std::shared_ptr<JitInterpreter> Get();

  Maybe<void> Apply(const OpExpr& op_expr, const TensorTuple& inputs, TensorTuple* outputs,
                    const AttrMap& attrs) const {
    return Apply(op_expr, inputs, outputs, OpExprInterpContext(attrs));
  }
  Maybe<void> Apply(const OpExpr& op_expr, const TensorTuple& inputs, TensorTuple* outputs,
                    const OpExprInterpContext& ctx) const override;
  void Interrupt();
  ir::JitImporter& GetImporter() const { return *current_importer_; }
  void CacheExpr(Operation&, std::shared_ptr<one::UserOpExpr>);
  llvm::Optional<std::shared_ptr<one::UserOpExpr>> GetExpr(Operation*);
  void Start() { trace_start_time_ = std::chrono::steady_clock::now(); }
  void MlirTraceEnd() { trace_end_time_ = std::chrono::steady_clock::now(); }
  void End() { dispatch_end_time_ = std::chrono::steady_clock::now(); }
  float TraceOverhead() {
    const float mlir_trace_time =
        std::chrono::duration_cast<std::chrono::microseconds>(trace_end_time_ - trace_start_time_)
            .count();
    const float jit_time = std::chrono::duration_cast<std::chrono::microseconds>(
                               dispatch_end_time_ - trace_start_time_)
                               .count();
    return mlir_trace_time / jit_time;
  }
  void Trace(ir::JitImporter& importer, const std::string& func_name,
             const std::vector<std::shared_ptr<one::Tensor>>& arg_tensors,
             const std::function<std::vector<std::shared_ptr<one::Tensor>>(void)>& forward_func);
  void DispatchModule(ModuleOp module, const std::string& func_name,
                      const std::vector<std::shared_ptr<one::Tensor>>& arg_tensors);
  const std::string& GetJitFuncName() const { return jit_function_context_.GetJitFuncName(); }
  const std::vector<std::shared_ptr<one::Tensor>>& GetJitForwardArgs() const {
    return jit_function_context_.GetJitForwardArgs();
  }

 private:
  DECLARE_NORMAL_APPLY_FUNC(UserOp);  // note(BBuf) jit deal with user op only, now.
  mutable llvm::DenseMap<llvm::hash_code, std::shared_ptr<one::UserOpExpr>> cached_user_op_exprs_;
  mutable MLIRContext* context_;
  mutable OwningOpRef<ModuleOp> module_;
  mutable ir::JitImporter importer_;
  ir::JitImporter* current_importer_;
  mutable std::chrono::steady_clock::time_point trace_start_time_;
  mutable std::chrono::steady_clock::time_point trace_end_time_;
  mutable std::chrono::steady_clock::time_point dispatch_end_time_;
  mutable JitFunctionContext jit_function_context_;
};

}  // namespace one
}  // namespace oneflow

#endif  // WITH_MLIR

#endif  // ONEFLOW_IR_ONEFLOW_JIT_INCLUDE_ONEFLOW_JIT_OP_INTERPRETER_H_

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
#ifndef ONEFLOW_IR_INCLUDE_ONEFLOW_OKL_KERNEL_WRAPPERCONTEXT_H_
#define ONEFLOW_IR_INCLUDE_ONEFLOW_OKL_KERNEL_WRAPPERCONTEXT_H_

#include "mlir/IR/BuiltinAttributes.h"
#include "OneFlow/OKL/Kernel/InitContext.h"
#include "OneFlow/OKL/Kernel/RegContext.h"
#include "OneFlow/OKL/Kernel/ComputeContext.h"
#include "oneflow/core/framework/op_kernel.h"

namespace oneflow {
namespace okl {

class CompileTimeWrapperContext {
 public:
  explicit CompileTimeWrapperContext(mlir::Operation* op)
      : reg_ctx_(std::make_shared<const RegContext>(op)) {}

  CompileTimeWrapperContext(CompileTimeWrapperContext&&) = default;

  RegContext const* GetRegContext() const { return reg_ctx_.get(); }

 private:
  std::shared_ptr<const RegContext> reg_ctx_;
};

class RunTimeWrapperContext {
 public:
  RunTimeWrapperContext(mlir::Operation* op, user_op::KernelComputeContext* ctx)
      : compile_time_wrapper_ctx_(op),
        compute_ctx_(std::make_unique<ComputeContext>(GetRegContext(), ctx)),
        init_ctx_(std::make_unique<InitContext>(GetRegContext(), ctx)),
        kernel_state_(GetRegContext()->GetKernel()->CreateOpKernelState(init_ctx_.get())),
        kernel_cache_(GetRegContext()->GetKernel()->InitOpKernelCache(init_ctx_.get())) {}

  void Run() {
    GetRegContext()->GetKernel()->Compute(compute_ctx_.get(), kernel_state_.get(),
                                          kernel_cache_.get());
  }

  RegContext const* GetRegContext() const { return compile_time_wrapper_ctx_.GetRegContext(); }

 private:
  CompileTimeWrapperContext compile_time_wrapper_ctx_;
  std::unique_ptr<ComputeContext> compute_ctx_;
  std::unique_ptr<InitContext> init_ctx_;

  std::shared_ptr<user_op::OpKernelState> kernel_state_;
  std::shared_ptr<user_op::OpKernelCache> kernel_cache_;
};

}  // namespace okl
}  // namespace oneflow

#endif  // ONEFLOW_IR_INCLUDE_ONEFLOW_OKL_KERNEL_WRAPPERCONTEXT_H_

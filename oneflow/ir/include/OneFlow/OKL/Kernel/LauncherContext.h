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
#ifndef ONEFLOW_IR_INCLUDE_ONEFLOW_OKL_KERNEL_LAUNCHER_CONTEXT_H_
#define ONEFLOW_IR_INCLUDE_ONEFLOW_OKL_KERNEL_LAUNCHER_CONTEXT_H_

#include "oneflow/core/framework/op_kernel.h"
#include "OneFlow/OKL/OKLOps.h"
#include "OneFlow/OKL/Kernel/RegContext.h"
#include "OneFlow/OKL/Kernel/WrapperContext.h"
#include "mlir/IR/Operation.h"

namespace oneflow {
namespace okl {

class LauncherContext final {
 public:
  // compile the mlir to ctx
  explicit LauncherContext(mlir::ModuleOp module);
  // infer ctx with okl info
  bool Infer() { return inferred_; }
  bool Infer(user_op::KernelComputeContext* compute_context);
  // launch kernel with index
  void Launch(int index);

 private:
  bool inferred_ = false;

  std::vector<CompileTimeWrapperContext> compile_ctx_vec_;
  std::vector<RunTimeWrapperContext> run_ctx_vec_;
};

}  // namespace okl
}  // namespace oneflow

#endif  // ONEFLOW_IR_INCLUDE_ONEFLOW_OKL_KERNEL_LAUNCHER_CONTEXT_H_

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
#ifndef ONEFLOW_IR_ONEFLOW_EXTENSION_INCLUDE_ONEFLOW_KERNEL_LAUNCH_LAUNCHER_CONTEXT_H_
#define ONEFLOW_IR_ONEFLOW_EXTENSION_INCLUDE_ONEFLOW_KERNEL_LAUNCH_LAUNCHER_CONTEXT_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "oneflow/core/framework/op_kernel.h"
#include "OneFlow/OKL/OKLOps.h"
#include "oneflow/ir/oneflow-extension/include/OneFlow/kernel_launch/RegContext.h"
#include "oneflow/ir/oneflow-extension/include/OneFlow/kernel_launch/RunContext.h"

namespace oneflow {
namespace okl {

class LauncherContext final {
  using RegContextResource = std::shared_ptr<RegContext>;
  using RunContextResource = std::shared_ptr<RunContext>;

 public:
  explicit LauncherContext(user_op::KernelComputeContext* compute_context, mlir::ModuleOp module);

  void* FetchKernel(int index);
  void* FetchRunCtx(int index);

 private:
  std::vector<const oneflow::user_op::OpKernel*> kernel_vec_;
  std::vector<RegContextResource> reg_ctx_vec_;
  std::vector<RunContextResource> run_ctx_vec_;
  ::mlir::ModuleOp module_;
};

}  // namespace okl
}  // namespace oneflow

#endif  // ONEFLOW_IR_ONEFLOW_EXTENSION_INCLUDE_ONEFLOW_KERNEL_LAUNCH_LAUNCHER_CONTEXT_H_

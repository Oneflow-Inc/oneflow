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

  static int GetOpIndex(mlir::Operation* op, int index) {
    return op->getOperand(index)
        .getDefiningOp()
        ->getAttr("index")
        .dyn_cast<mlir::IntegerAttr>()
        .getInt();
  };

 public:
  explicit LauncherContext(user_op::KernelComputeContext* compute_context, mlir::ModuleOp module)
      : module_(module) {
    auto func = module.lookupSymbol("okl_init_context");
    auto context = func->getContext();

    auto& ops = func->getRegion(0).front();

    for (auto& op : ops) {
      auto index = 0;
      auto op_name = op.getName().getStringRef();
      if (op_name == ::mlir::okl::BuildKernelOp::getOperationName()) {
        index = kernel_vec_.size();

        auto reg_ctx = reg_ctx_vec_[GetOpIndex(&op, 0)];
        auto kernel = CHECK_JUST(user_op::UserOpRegistryMgr::Get().GetOpKernelRegistryResult(
                                     reg_ctx->GetOp()->getName().stripDialect().str(), *reg_ctx))
                          ->create_fn();
        kernel_vec_.push_back(kernel);

        kernel->IsKernelLaunchSynchronized();
        kernel_vec_[index]->IsKernelLaunchSynchronized();
      } else if (op_name == mlir::okl::BuildRegContextOp::getOperationName()) {
        index = reg_ctx_vec_.size();

        auto& reg_op = op.getRegion(0).front().front();
        reg_ctx_vec_.emplace_back(std::make_shared<RegContext>(&reg_op));
      } else if (op_name == mlir::okl::BuildRunContextOp::getOperationName()) {
        index = run_ctx_vec_.size();

        auto reg_ctx = reg_ctx_vec_[GetOpIndex(&op, 0)];
        run_ctx_vec_.emplace_back(
            std::make_shared<RunContext>(std::move(reg_ctx), compute_context));
      } else if (op_name == mlir::func::ReturnOp::getOperationName()) {
        return;
      } else {
        op.emitError("Fail to parse this op in okl init context");
      }
      op.setAttr("index", mlir::IntegerAttr::get(mlir::IntegerType::get(context, 32), index));
    }
  }
  void* FetchKernel(int index) { return (void*)kernel_vec_[index]; }

  void* FetchRunCtx(int index) { return run_ctx_vec_[index].get(); }

 private:
  std::vector<const oneflow::user_op::OpKernel*> kernel_vec_;
  std::vector<RegContextResource> reg_ctx_vec_;
  std::vector<RunContextResource> run_ctx_vec_;
  ::mlir::ModuleOp module_;
};

}  // namespace okl
}  // namespace oneflow

#endif  // ONEFLOW_IR_ONEFLOW_EXTENSION_INCLUDE_ONEFLOW_KERNEL_LAUNCH_LAUNCHER_CONTEXT_H_

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
#ifndef ONEFLOW_IR_ONEFLOW_EXTENSION_INCLUDE_ONEFLOW_KERNEL_LAUNCH_KERNEL_MANAGER_H_
#define ONEFLOW_IR_ONEFLOW_EXTENSION_INCLUDE_ONEFLOW_KERNEL_LAUNCH_KERNEL_MANAGER_H_

#include <memory>
#include "OneFlow/kernel_launch/InferMisc/InitContext.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "oneflow/core/framework/op_kernel.h"
#include "OneFlow/OKL/OKLOps.h"
#include "OneFlow/kernel_launch/RegContext.h"
#include "OneFlow/kernel_launch/RunContext.h"
#include "llvm/ADT/TypeSwitch.h"

namespace oneflow {
namespace okl {

// which element is ensured before computing.
struct KernelReSource {
  std::shared_ptr<RegContext> reg_ctx_;
  std::shared_ptr<InitContext> init_ctx_;
  std::shared_ptr<user_op::OpKernelState> kernel_state_;
  std::shared_ptr<user_op::OpKernelCache> kernel_cache_;
};

// TODO: make this as the only core resource manager in an okl kernel.
class ResourcesManager final {
 public:
  explicit ResourcesManager(mlir::ModuleOp module) : is_init_all_(false) {
    auto func = module.lookupSymbol("okl_init_context");
    auto& ops = func->getRegion(0).front();

    for (auto& op : ops) {
      if (auto reg_ctx = llvm::dyn_cast_or_null<mlir::okl::BuildRegContextOp>(&op)) {
        mlir::Operation* reg_op = nullptr;
        for (auto& op_it : op.getRegion(0).front().getOperations()) {
          if (op_it.getDialect()->getNamespace() == "oneflow") {
            reg_op = &op_it;
            break;
          }
        }
        if (!reg_op) { llvm_unreachable("Failed to find reg_op in okl.build_reg_context_op"); }
        kernel_resources_.push_back(KernelReSource{std::make_shared<RegContext>(reg_op)});
      }
    }
  }

  ResourcesManager(mlir::ModuleOp module, user_op::KernelInitContext* ctx)
      : ResourcesManager(module) {
    InitAll(ctx);
  }

  const KernelReSource* FetchKernelResource(int index) const { return &kernel_resources_[index]; }

  void InitAll(user_op::KernelInitContext* ctx) {
    for (auto elem : kernel_resources_) {
      elem.init_ctx_ =
          std::make_shared<InitContext>(elem.reg_ctx_.get(), ctx->stream(), ctx->parallel_ctx());
      elem.kernel_state_ = elem.reg_ctx_->GetKernel()->CreateOpKernelState(elem.init_ctx_.get());
      elem.kernel_cache_ = elem.reg_ctx_->GetKernel()->InitOpKernelCache(elem.init_ctx_.get());
    }
    is_init_all_ = true;
  }

  bool IsCudaGraphSupported(user_op::KernelInitContext* ctx) {
    if (!is_init_all_) { InitAll(ctx); }

    for (const auto& elem : kernel_resources_) {
      auto* kernel = const_cast<user_op::OpKernel*>(elem.reg_ctx_->GetKernel());
      auto* cuda_graph_support = dynamic_cast<user_op::CudaGraphSupport*>(kernel);
      if (!cuda_graph_support) { return false; }
      InitContext init_ctx(elem.reg_ctx_.get(), ctx->stream(), ctx->parallel_ctx());
      if (!cuda_graph_support->IsCudaGraphSupported(&init_ctx, elem.kernel_state_.get())) {
        return false;
      }
    }
    return true;
  }

 private:
  std::vector<KernelReSource> kernel_resources_;
  bool is_init_all_;
};

}  // namespace okl
}  // namespace oneflow

#endif  // ONEFLOW_IR_ONEFLOW_EXTENSION_INCLUDE_ONEFLOW_KERNEL_LAUNCH_KERNEL_MANAGER_H_
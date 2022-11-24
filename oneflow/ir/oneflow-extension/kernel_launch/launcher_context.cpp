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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "oneflow/core/framework/op_kernel.h"
#include "OneFlow/OKL/OKLOps.h"
#include "OneFlow/kernel_launch/RegContext.h"
#include "OneFlow/kernel_launch/RunContext.h"
#include "OneFlow/kernel_launch/LauncherContext.h"
#include "llvm/ADT/TypeSwitch.h"

namespace oneflow {
namespace okl {

static int GetOpIndex(mlir::Operation* op, int index) {
  return op->getOperand(index)
      .getDefiningOp()
      ->getAttr("index")
      .dyn_cast<mlir::IntegerAttr>()
      .getInt();
};

LauncherContext::LauncherContext(user_op::KernelComputeContext* compute_context,
                                 mlir::ModuleOp module)
    : module_(module) {
  auto func = module.lookupSymbol("okl_init_context");
  auto context = func->getContext();

  auto& ops = func->getRegion(0).front();

  for (auto& op : ops) {
    auto index = 0;
    llvm::TypeSwitch<mlir::Operation*>(&op)
        .Case([&](mlir::okl::BuildKernelOp elem) {
          index = kernel_vec_.size();

          auto reg_ctx = reg_ctx_vec_[GetOpIndex(&op, 0)];
          kernel_vec_.push_back(reg_ctx->GetKernel());
          op.setAttr("index", mlir::IntegerAttr::get(mlir::IntegerType::get(context, 32), index));
        })
        .Case([&](mlir::okl::BuildRegContextOp elem) {
          index = reg_ctx_vec_.size();

          mlir::Operation* reg_op = nullptr;
          for (auto& op_it : op.getRegion(0).front().getOperations()) {
            if (op_it.getDialect()->getNamespace() == "oneflow") {
              reg_op = &op_it;
              break;
            }
          }

          if (!reg_op) { LOG(FATAL) << "Failed to find reg_op in okl.build_reg_context_op"; }

          reg_ctx_vec_.emplace_back(std::make_shared<RegContext>(reg_op));
          op.setAttr("index", mlir::IntegerAttr::get(mlir::IntegerType::get(context, 32), index));
        })
        .Case([&](mlir::okl::BuildRunContextOp elem) {
          index = run_ctx_vec_.size();

          auto reg_ctx = reg_ctx_vec_[GetOpIndex(&op, 0)];
          run_ctx_vec_.emplace_back(
              std::make_shared<RunContext>(std::move(reg_ctx), compute_context));
          op.setAttr("index", mlir::IntegerAttr::get(mlir::IntegerType::get(context, 32), index));
        })
        .Case([&](mlir::func::ReturnOp elem) {})
        .Default([&](mlir::Operation* elem) {
          op.emitError("Fail to parse this op in okl init context");
          exit(1);
        });
  }
}
void* LauncherContext::FetchKernel(int index) { return (void*)kernel_vec_[index]; }

void* LauncherContext::FetchRunCtx(int index) { return run_ctx_vec_[index].get(); }

}  // namespace okl
}  // namespace oneflow

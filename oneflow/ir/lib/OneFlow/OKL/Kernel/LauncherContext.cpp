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
#include "OneFlow/OKL/Kernel/WrapperContext.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "oneflow/core/framework/op_kernel.h"
#include "OneFlow/OKL/OKLOps.h"
#include "OneFlow/OKL/Conversion/SplitIntoFuncs.h"
#include "OneFlow/OKL/Kernel/RegContext.h"
#include "OneFlow/OKL/Kernel/ComputeContext.h"
#include "OneFlow/OKL/Kernel/LauncherContext.h"
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

LauncherContext::LauncherContext(mlir::ModuleOp module) {
  auto func = module.lookupSymbol(mlir::okl::function::CREATE_FUNC_NAME);
  auto context = func->getContext();

  auto& ops = func->getRegion(0).front();

  for (auto& op : ops) {
    llvm::TypeSwitch<mlir::Operation*>(&op)
        .Case([&](mlir::okl::BuildRegContextOp elem) {
          auto index = compile_ctx_vec_.size();

          mlir::Operation* reg_op = nullptr;
          for (auto& op_it : op.getRegion(0).front().getOperations()) {
            if (op_it.getDialect()->getNamespace() == "oneflow") {
              reg_op = &op_it;
              break;
            }
          }

          if (!reg_op) { LOG(FATAL) << "Failed to find reg_op in okl.build_reg_context_op"; }
          auto&& compile_ctx = CompileTimeWrapperContext(reg_op);
          compile_ctx_vec_.emplace_back(compile_ctx);
          op.setAttr("index", mlir::IntegerAttr::get(mlir::IntegerType::get(context, 32), index));
        })
        .Case([&](mlir::func::ReturnOp elem) {})
        .Default([&](mlir::Operation* elem) {
          LOG(FATAL) << "Fail to parse this op in okl init context";
        });
  }
}
bool LauncherContext::Infer(user_op::KernelComputeContext* compute_context) {
  // if this context has been inferred before, it won't be rebuilt later
  if (inferred_) { return inferred_; }

  for (auto& elem : compile_ctx_vec_) {
    auto&& run_ctx = RunTimeWrapperContext(elem.GetRegContext()->GetOp(), compute_context);
    run_ctx_vec_.emplace_back(run_ctx);
  }
  inferred_ = compile_ctx_vec_.size() == run_ctx_vec_.size();
  return inferred_;
}
void LauncherContext::Launch(int index) { TODO(); }

}  // namespace okl
}  // namespace oneflow

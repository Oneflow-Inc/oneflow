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
#include "OneFlow/OKM/passes.h"
#include "OneFlow/Passes.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "oneflow/core/framework/op_kernel.h"
#include "OneFlow/OKL/OKLOps.h"
#include "OneFlow/OKL/Kernel/RegContext.h"
#include "OneFlow/OKL/Kernel/ComputeContext.h"
#include "OneFlow/OKL/Kernel/LauncherContext.h"
#include "llvm/ADT/TypeSwitch.h"

namespace oneflow {
namespace okl {

LauncherContext::LauncherContext(mlir::ModuleOp module) {
  mlir::Operation* func;
  module->walk([&](mlir::func::FuncOp op) {
    if (op.getSymName().startswith(mlir::okm::func_name::OKL_GRAPH_NAME)) { func = op; }
  });
  if (!func) { LOG(FATAL) << "Not Found okl_func in mlir ir"; }
  auto& ops = func->getRegion(0).front();

  for (auto& op : ops) {
    llvm::TypeSwitch<mlir::Operation*>(&op)
        .Case([&](mlir::okl::WrapperKernelOp elem) {
          mlir::Operation* reg_op = nullptr;
          for (auto& op_it : op.getRegion(0).front().getOperations()) {
            if (op_it.getDialect()->getNamespace() == "oneflow") {
              reg_op = &op_it;
              break;
            }
          }

          if (!reg_op) { LOG(FATAL) << "Failed to find reg_op in okl.build_reg_context_op"; }
          compile_ctx_vec_.emplace_back(reg_op);
        })
        .Case([&](mlir::func::ReturnOp elem) {})
        .Default([&](mlir::Operation* elem) {
          elem->dump();
          LOG(FATAL) << "Fail to parse this op in okl init context";
        });
  }
}

bool LauncherContext::Infer(user_op::KernelComputeContext* compute_context) {
  // if this context has been inferred before, it won't be rebuilt later
  if (inferred_) { return inferred_; }

  for (auto& elem : compile_ctx_vec_) {
    run_ctx_vec_.emplace_back(elem.GetRegContext()->GetOp(), compute_context);
  }
  inferred_ = compile_ctx_vec_.size() == run_ctx_vec_.size();
  return inferred_;
}

void LauncherContext::Launch(int index) {
  if (!inferred_) { LOG(FATAL) << "Not infer yet when launch kernels"; }
  run_ctx_vec_[index].Run();
}

}  // namespace okl
}  // namespace oneflow

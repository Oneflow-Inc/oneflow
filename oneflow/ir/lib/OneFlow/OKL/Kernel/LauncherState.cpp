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

#include "OneFlow/OKL/Conversion/Conversion.h"
#include "OneFlow/OKM/Conversion/Conversion.h"
#include "OneFlow/Passes.h"
#include "OneFlow/OKM/passes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "oneflow/core/framework/op_kernel.h"
#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/OKL/OKLDialect.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "OneFlow/OKL/Kernel/JITEngine.h"
#include "OneFlow/OKL/Kernel/LauncherContext.h"
#include "OneFlow/OKL/Kernel/LauncherState.h"

namespace oneflow {
namespace okl {

namespace {

mlir::OwningOpRef<mlir::ModuleOp> GetModule(user_op::KernelInitContext* ctx,
                                            mlir::MLIRContext* mlir_ctx) {
  auto module =
      mlir::parseSourceString<mlir::ModuleOp>(ctx->Attr<std::string>("mlir_assembly"), mlir_ctx);
  if (!module) { LOG(FATAL) << "Fail to load mlir assembly"; }
  // lower oneflow wrap ops into okl dialect
  if (failed(mlir::okm::LowerWrapOpsToOKL(*module))) {
    LOG(FATAL) << "Fail lowering kernel launch Module to okm and okl ir";
  }
  return module;
}

JITEngine GetEngine(mlir::ModuleOp module) {
  if (failed(mlir::okl::LowerOKLComputeToLLVM(module))) {
    LOG(FATAL) << "Fail lowering okl compute Module to llvm ir";
  }
  return JITEngine(module);
}

}  // namespace

LauncherState::LauncherState(user_op::KernelInitContext* ctx)
    : mlir_ctx_(GetRegistry()),
      module_(GetModule(ctx, &mlir_ctx_)),
      launcher_context_(module_->clone()),
      engine_(GetEngine(module_->clone())) {}

bool LauncherState::IsCudaGraphSupported(user_op::KernelInitContext* ctx) {
  const auto tag_name = mlir::okl::cuda_graph_support::TAG_NAME;
  if (const auto func = module_->lookupSymbol(mlir::okm::func_name::OKL_GRAPH_NAME)) {
    if (const auto is_supported = func->getAttr(tag_name).dyn_cast_or_null<mlir::BoolAttr>()) {
      return is_supported.getValue();
    }
  }
  return false;
}

void LauncherState::DoCompute(user_op::KernelComputeContext* ctx) {
  launcher_context_.Infer(ctx);
  engine_.Run(mlir::okm::func_name::OKL_GRAPH_NAME, &launcher_context_);
}

}  // namespace okl
}  // namespace oneflow

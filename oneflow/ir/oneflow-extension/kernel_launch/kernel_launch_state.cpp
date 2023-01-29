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
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/IR/DialectRegistry.h"
#include "oneflow/core/framework/op_kernel.h"
#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/OKL/OKLDialect.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "OneFlow/kernel_launch/JITEngine.h"
#include "OneFlow/kernel_launch/LauncherContext.h"
#include "OneFlow/kernel_launch/KernelLaunchState.h"

namespace oneflow {
namespace okl {

KernelLaunchState::KernelLaunchState(user_op::KernelInitContext* ctx) : mlir_ctx_(GetRegistry()) {
  // get raw module from ctx attr
  module_ =
      mlir::parseSourceString<mlir::ModuleOp>(ctx->Attr<std::string>("mlir_assembly"), &mlir_ctx_);
  if (!module_) { LOG(FATAL) << "Fail to load mlir assembly"; }
  // lower oneflow wrap ops into okl dialect
  if (failed(mlir::okl::LowerWrapOpsToOKL(*module_))) {
    LOG(ERROR) << "Fail lowering kernel launch Module to okl ir";
    exit(1);
  }
};

void KernelLaunchState::DoCompute(user_op::KernelComputeContext* ctx) {
  if (!launcher_context_) { LazyInitLauncher(ctx); }
  engine_->Run("okl_compute", launcher_context_.get());
}

void KernelLaunchState::LazyInitLauncher(user_op::KernelComputeContext* ctx) {
  launcher_context_ = std::make_shared<LauncherContext>(ctx, module_->clone());

  if (failed(mlir::okl::LowerOKLComputeToLLVM(*module_))) {
    LOG(ERROR) << "Fail lowering okl compute Module to llvm ir";
    exit(1);
  }

  engine_ = std::make_shared<JITEngine>(*module_);
}

}  // namespace okl
}  // namespace oneflow

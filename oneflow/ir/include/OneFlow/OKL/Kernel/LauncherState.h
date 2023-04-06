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
#ifndef ONEFLOW_IR_INCLUDE_ONEFLOW_OKL_KERNEL_OP_KERNEL_STATE_H_
#define ONEFLOW_IR_INCLUDE_ONEFLOW_OKL_KERNEL_OP_KERNEL_STATE_H_

#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/OKL/OKLDialect.h"
#include "OneFlow/OKL/Kernel/JITEngine.h"
#include "OneFlow/OKL/Kernel/LauncherContext.h"
#include "OneFlow/OKL/Conversion/Conversion.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

namespace oneflow {
namespace okl {

inline mlir::DialectRegistry GetRegistry() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::oneflow::OneFlowDialect, mlir::okl::OKLDialect, mlir::func::FuncDialect,
                  mlir::arith::ArithmeticDialect, mlir::LLVM::LLVMDialect>();
  mlir::registerLLVMDialectTranslation(registry);
  return registry;
}

class LauncherState final : public user_op::OpKernelState {
 public:
  explicit LauncherState(user_op::KernelInitContext* ctx);
  ~LauncherState() = default;

  void DoCompute(user_op::KernelComputeContext* ctx);
  bool IsCudaGraphSupported(user_op::KernelInitContext* ctx);

 private:
  // manage module(compile)
  mlir::MLIRContext mlir_ctx_;
  mlir::OwningOpRef<mlir::ModuleOp> module_;

  // manage context
  LauncherContext launcher_context_;

  // manage engine(runtime)
  JITEngine engine_;
};

}  // namespace okl
}  // namespace oneflow

#endif  // ONEFLOW_IR_INCLUDE_ONEFLOW_OKL_KERNEL_OP_KERNEL_STATE_H_

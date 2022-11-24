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
#ifndef ONEFLOW_IR_ONEFLOW_EXTENSION_INCLUDE_ONEFLOW_KERNEL_LAUNCH_OP_KERNEL_STATE_H_
#define ONEFLOW_IR_ONEFLOW_EXTENSION_INCLUDE_ONEFLOW_KERNEL_LAUNCH_OP_KERNEL_STATE_H_

#include <memory>
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

namespace oneflow {
namespace okl {

class KernelLaunchState final : public user_op::OpKernelState {
 public:
  static mlir::DialectRegistry GetRegistry() {
    mlir::DialectRegistry registry;
    registry.insert<mlir::oneflow::OneFlowDialect, mlir::okl::OKLDialect, mlir::func::FuncDialect,
                    mlir::arith::ArithmeticDialect, mlir::LLVM::LLVMDialect>();
    mlir::registerLLVMDialectTranslation(registry);
    return registry;
  }

  explicit KernelLaunchState(user_op::KernelInitContext* ctx);
  ~KernelLaunchState() = default;

  void DoCompute(user_op::KernelComputeContext* ctx);

 private:
  mlir::MLIRContext mlir_ctx_;
  mlir::OwningOpRef<mlir::ModuleOp> module_;
  std::shared_ptr<LauncherContext> launcher_context_{};
  std::shared_ptr<JITEngine> engine_{};
  void LazyInitLauncher(user_op::KernelComputeContext* ctx);
};

}  // namespace okl
}  // namespace oneflow

#endif  // ONEFLOW_IR_ONEFLOW_EXTENSION_INCLUDE_ONEFLOW_KERNEL_LAUNCH_OP_KERNEL_STATE_H_

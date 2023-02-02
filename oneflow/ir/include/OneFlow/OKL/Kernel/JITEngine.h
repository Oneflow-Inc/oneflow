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
#ifndef ONEFLOW_IR_ONEFLOW_EXTENSION_INCLUDE_ONEFLOW_JITENGINE_H_
#define ONEFLOW_IR_ONEFLOW_EXTENSION_INCLUDE_ONEFLOW_JITENGINE_H_

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"
#include "oneflow/core/framework/op_kernel.h"
#include "OneFlow/OKL/Kernel/LauncherContext.h"

extern "C" {
void okl_llvm_func(void* launcher, int64_t index);
}  // extern "C"

namespace oneflow {
namespace okl {

using LLVMLaunchArgs = std::tuple<LauncherContext*, int>;

class JITEngine {
 public:
  explicit JITEngine(mlir::ModuleOp module);

  void Run(const std::string& name, LauncherContext* launcher) const {
    auto error = engine_->invoke(name, launcher);
    CHECK(!error) << "fail to invoke jit engine, error: " << llvm::toString(std::move(error));
  }

 private:
  std::unique_ptr<mlir::ExecutionEngine> engine_;
};

namespace llvm_func {
#define C_FUNC_NAME(func) #func

const auto LLVM_FUNC = C_FUNC_NAME(okl_llvm_func);

#undef C_FUNC_NAME
}  // namespace llvm_func

}  // namespace okl
}  // namespace oneflow

#endif  // ONEFLOW_IR_ONEFLOW_EXTENSION_INCLUDE_ONEFLOW_JITENGINE_H_

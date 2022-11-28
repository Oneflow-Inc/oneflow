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

namespace oneflow {
using TypeKernelLaunchArgs =
    std::tuple<oneflow::user_op::KernelComputeContext*, const oneflow::user_op::OpKernel*>;
}  // namespace oneflow

class JIT_Engine {
 public:
  explicit JIT_Engine(mlir::ModuleOp module);
  template<typename ArgsT, class... Args>
  void Run(const std::string& name, Args... args) const {
    using Tuple = std::tuple<Args...>;
    static_assert(std::is_same<ArgsT, Tuple>::value, "args of jit function don't match");
    auto error = engine_->invoke(name, args...);
    CHECK(!error) << "fail to invoke jit engine, error: " << llvm::toString(std::move(error));
  }

 private:
  std::unique_ptr<mlir::ExecutionEngine> engine_;
};
#endif  // ONEFLOW_IR_ONEFLOW_EXTENSION_INCLUDE_ONEFLOW_JITENGINE_H_

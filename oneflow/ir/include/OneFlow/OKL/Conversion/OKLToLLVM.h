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
#ifndef ONEFLOW_IR_INCLUDE_ONEFLOW_OKL_CONVERSION_OKLTOLLVM_H_
#define ONEFLOW_IR_INCLUDE_ONEFLOW_OKL_CONVERSION_OKLTOLLVM_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace okl {

// lower !okl.launcher_ctx to !llvm.ptr<i8>
std::unique_ptr<mlir::Pass> createLowerLauncherToLLVMPtrPass();

// lower okl ops to llvm.call @{callee in liboneflow.so}
std::unique_ptr<mlir::Pass> createLowerOKLToLLVMCallPass();

// tag {okl.cuda_graph_support} according to its wrapped ops
std::unique_ptr<mlir::Pass> createTagCudaGraphSupportPass();

namespace cuda_graph_support {

static const auto TAG_NAME = "cuda_graph_support";

}  // namespace cuda_graph_support
}  // namespace okl
}  // namespace mlir

#endif  // ONEFLOW_IR_INCLUDE_ONEFLOW_OKL_CONVERSION_OKLTOLLVM_H_

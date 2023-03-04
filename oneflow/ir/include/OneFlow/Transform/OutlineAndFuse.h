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
#ifndef ONEFLOW_IR_INCLUDE_ONEFLOW_TRANSFORM_OUTLINEJITFUNCTION_H_
#define ONEFLOW_IR_INCLUDE_ONEFLOW_TRANSFORM_OUTLINEJITFUNCTION_H_

#include "mlir/Pass/Pass.h"

namespace mlir {

namespace oneflow {

namespace wrap_mode {
const std::string SIMPLE = "simple";
const std::string CUDA_GRAPH = "cuda_graph";
}  // namespace wrap_mode

std::unique_ptr<mlir::Pass> createWrapOpsToKernelLaunchPass();
std::unique_ptr<mlir::Pass> createOutlineJitFunctionPass();
std::unique_ptr<mlir::Pass> createFuseIntoExistingOpPass();
std::unique_ptr<mlir::Pass> createGroupMatMul();
std::unique_ptr<mlir::Pass> createFuseForwardOps();
std::unique_ptr<mlir::Pass> createFuseOpsWithBackwardImpl();
std::unique_ptr<mlir::Pass> createFuseNormalizationOps();

}  // namespace oneflow

}  // namespace mlir

#endif  // ONEFLOW_IR_INCLUDE_ONEFLOW_TRANSFORM_OUTLINEJITFUNCTION_H_

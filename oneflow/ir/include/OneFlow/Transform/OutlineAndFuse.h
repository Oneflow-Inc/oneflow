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

namespace wrap_options {
namespace mode {
const std::string SIMPLE = "simple";
const std::string CUDA_GRAPH = "cuda_graph";
}  // namespace mode

namespace tensor {
// the option : tensor=normal will map all resources to wrap_func operands and results, 
// it will use large memory space in runtime, and it is security in any case.
const std::string NORMAL = "normal";
// the option : tensor=trim will only map the necessary resources in use-def flow among
// wrap_funcs and no-wrap-ops, it will infer memory reuse in order to smaller memory space.
const std::string TRIM = "trim";
}  // namespace mode

}  // namespace wrap_options

std::unique_ptr<mlir::Pass> createLowerToOKLPass();
std::unique_ptr<mlir::Pass> createWrapOpsToKernelLaunchPass();
std::unique_ptr<mlir::Pass> createExtractKernelLaunchTensorPass();
std::unique_ptr<mlir::Pass> createTrimReturnAsVoidPass();
std::unique_ptr<mlir::Pass> createOutlineJitFunctionPass();
std::unique_ptr<mlir::Pass> createFuseIntoExistingOpPass();
std::unique_ptr<mlir::Pass> createGroupMatMul();
std::unique_ptr<mlir::Pass> createFuseForwardOps();
std::unique_ptr<mlir::Pass> createFuseNormalizationOps();

}  // namespace oneflow

}  // namespace mlir

#endif  // ONEFLOW_IR_INCLUDE_ONEFLOW_TRANSFORM_OUTLINEJITFUNCTION_H_

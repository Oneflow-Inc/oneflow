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
#ifndef ONEFLOW_IR_INCLUDE_ONEFLOW_PASSES_H_
#define ONEFLOW_IR_INCLUDE_ONEFLOW_PASSES_H_

#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "OneFlow/Conversion/OneFlowToTosa.h"
#include "OneFlow/Transform/BufferHostRegister.h"
#include "OneFlow/Transform/ConvertInferenceOp.h"
#include "OneFlow/Transform/OutlineAndFuse.h"
#include "OneFlow/Transform/AutoNhwc.h"
#include "OneFlow/Transform/AggregateOps.h"
#include "OneFlow/Transform/CSEWithAttributesIgnored.h"

#ifdef WITH_MLIR_CUDA_CODEGEN
#include "OneFlow/Conversion/PTXToCubin.h"
#endif  // WITH_MLIR_CUDA_CODEGEN

namespace mlir {

#define GEN_PASS_CLASSES
#define GEN_PASS_REGISTRATION
#include "OneFlow/OneFlowPasses.h.inc"

namespace oneflow {

LogicalResult LowerModuleToLLVM(mlir::MLIRContext* context, ModuleOp module);
#ifdef WITH_MLIR_CUDA_CODEGEN
LogicalResult LowerModuleToCUDALLVM(mlir::MLIRContext* context, ModuleOp module);
#endif  // WITH_MLIR_CUDA_CODEGEN
void populateFuserPasses(::mlir::RewritePatternSet& patterns);
void populateWrapOpsToKernelLaunchPatterns(::mlir::RewritePatternSet& patterns,
                                           const std::string& mode);
void populateFuserForExistingOp(::mlir::RewritePatternSet& patterns);
void populateGpuHelperPatterns(::mlir::RewritePatternSet& patterns);
void populateAutoNhwcPatterns(::mlir::RewritePatternSet& patterns);

void populatePreConvertInferenceOp(::mlir::RewritePatternSet& patterns);
void populateConvertInferenceOp(::mlir::RewritePatternSet& patterns);
void populatePostConvertInferenceOp(::mlir::RewritePatternSet& patterns);

namespace okl_func {
const auto OKL_FUNC = "_mlir_okl_subgraph";
}  // namespace okl_func

}  // namespace oneflow

}  // namespace mlir

#endif  // ONEFLOW_IR_INCLUDE_ONEFLOW_PASSES_H_

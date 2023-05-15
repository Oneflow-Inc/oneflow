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
#ifndef ONEFLOW_IR_INCLUDE_ONEFLOW_OKM_PASSES_H_
#define ONEFLOW_IR_INCLUDE_ONEFLOW_OKM_PASSES_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

namespace okm {

namespace func_name {

extern const std::string GRAPH_NAME;
extern const std::string MEM_GRAPH_NAME;
extern const std::string WRAP_GRAPH_NAME;
extern const std::string OPT_GRAPH_NAME;
extern const std::string OKL_GRAPH_NAME;
extern const std::string OKL_POOL_SIZE_TAG;

}  // namespace func_name

std::unique_ptr<mlir::Pass> createExtractOKMTensorPass();
std::unique_ptr<mlir::Pass> createWrapOKMKernelPass();
std::unique_ptr<mlir::Pass> createOptOKMMemrefPass();
std::unique_ptr<mlir::Pass> createConvertOKMToOKLPass();

#define GEN_PASS_CLASSES
#define GEN_PASS_REGISTRATION
#include "OneFlow/OKMPasses.h.inc"

}  // namespace okm

}  // namespace mlir

#endif  // ONEFLOW_IR_INCLUDE_ONEFLOW_OKM_PASSES_H_

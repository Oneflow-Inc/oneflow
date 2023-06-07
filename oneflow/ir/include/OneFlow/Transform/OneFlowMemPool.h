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
#ifndef ONEFLOW_IR_INCLUDE_ONEFLOW_TRANSFORM_ONEFLOW_MEMPOOL_H_
#define ONEFLOW_IR_INCLUDE_ONEFLOW_TRANSFORM_ONEFLOW_MEMPOOL_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace oneflow {

namespace codegen {
namespace mempool {

inline const std::string MEMPOOL_ATTR_NAME = "oneflow.mempool";

}  // namespace mempool
}  // namespace codegen

void applyFoldAlloc(func::FuncOp op);

std::unique_ptr<mlir::Pass> createFoldAllocToSubviewPass();
std::unique_ptr<mlir::Pass> createInsertOneFlowMemPoolPass();

}  // namespace oneflow
}  // namespace mlir

#endif  // ONEFLOW_IR_INCLUDE_ONEFLOW_TRANSFORM_ONEFLOW_MEMPOOL_H_
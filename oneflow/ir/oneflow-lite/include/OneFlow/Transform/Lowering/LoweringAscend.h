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
#ifndef ONEFLOW_IR_ONEFLOW_LITE_INCLUDE_ONEFLOW_TRANSFORM_LOWERING_LOWERINGASCEND_H_
#define ONEFLOW_IR_ONEFLOW_LITE_INCLUDE_ONEFLOW_TRANSFORM_LOWERING_LOWERINGASCEND_H_

#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/OneFlowOps.h"
#include "OneFlow/OneFlowOpTraits.h"
#include "OneFlow/OneFlowLiteUtils.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"

namespace mlir {
namespace oneflow {
namespace lite {

LogicalResult loweringAscend(OpBuilder& builder, Operation* callee, StringRef checkpointDir,
                             llvm::SmallVector<uint8_t, 4>* loweringData);

}  // namespace lite
}  // namespace oneflow
}  // namespace mlir

#endif  // ONEFLOW_IR_ONEFLOW_LITE_INCLUDE_ONEFLOW_TRANSFORM_LOWERING_LOWERINGASCEND_H_

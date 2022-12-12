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
#ifndef ONEFLOW_IR_INCLUDE_ONEFLOW_ONEFLOWOPS_H_
#define ONEFLOW_IR_INCLUDE_ONEFLOW_ONEFLOWOPS_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/IR/PatternMatch.h"
#include "OneFlow/OneFlowSupport.h"
#include "OneFlow/OneFlowInterfaces.h.inc"
#include "OneFlow/OneFlowOpTraits.h"
#include "OneFlow/SBP/SBPAttributes.h"

namespace mlir {

namespace func {
class FuncOp;
}  // namespace func

}  // namespace mlir

#define GET_OP_CLASSES
#include "OneFlow/OneFlowOps.h.inc"
#define GET_OP_CLASSES
#include "OneFlow/OneFlow.gen_ops.h.inc"

namespace mlir {

namespace oneflow {

template<typename T>
inline std::string GetOpTypeName(T op) {
  std::string op_type_name = op->getName().stripDialect().str();
  if (op->template hasTrait<OpTrait::IsAlternative>()) {
    op_type_name =
        op->template getAttrOfType<StringAttr>(OpTrait::IsAlternative<void>::getOpTypeNameAttr())
            .str();
  }
  if (auto alternative_name = dyn_cast<oneflow::HasAlternativeOpTypeName>(op)) {
    op_type_name = alternative_name.getOriginalOpTypeName();
  }
  if (auto user_op = dyn_cast<oneflow::UserOp>(op)) { op_type_name = user_op.op_type_name().str(); }
  return op_type_name;
}
ResultRange GetDataOutputResults(Operation* op);
OperandRange GetDataInputOperands(Operation* op);
llvm::Optional<OperandRange> GetCtrlIntputOperands(Operation* op);
llvm::Optional<OpResult> GetCtrlOutputResult(Operation* op);

ArrayAttr getSI32ArrayAttr(::mlir::PatternRewriter& rewriter, ArrayRef<int32_t> values);

}  // namespace oneflow

}  // namespace mlir

#endif  // ONEFLOW_IR_INCLUDE_ONEFLOW_ONEFLOWOPS_H_

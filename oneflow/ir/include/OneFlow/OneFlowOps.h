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

#include "OneFlow/OneFlowSupport.h"
#include "OneFlow/OneFlowInterfaces.h.inc"
#include "OneFlow/OneFlowOpTraits.h"
#include "OneFlow/OneFlowEnums.h.inc"
#include "OneFlow/OneFlowOpTraits.h"

namespace mlir {

class FuncOp;

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
  return op_type_name;
}

}  // namespace mlir

#define GET_OP_CLASSES
#include "OneFlow/OneFlowOps.h.inc"
#define GET_OP_CLASSES
#include "OneFlow/OneFlow.gen_ops.h.inc"

#endif  // ONEFLOW_IR_INCLUDE_ONEFLOW_ONEFLOWOPS_H_

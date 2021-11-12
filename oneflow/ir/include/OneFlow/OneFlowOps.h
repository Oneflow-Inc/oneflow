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
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#include "OneFlow/OneFlowEnums.h.inc"

namespace mlir {

class FuncOp;

namespace OpTrait {

namespace impl {
OpFoldResult foldIdempotentOfIdenticalPlacement(Operation* op);
OpFoldResult foldInvolutionOfIdenticalPlacement(Operation* op);
}  // namespace impl

template<typename ConcreteType>
class IsIdempotentOfIdenticalPlacement
    : public TraitBase<ConcreteType, IsIdempotentOfIdenticalPlacement> {
 public:
  static LogicalResult verifyTrait(Operation* op) {
    static_assert(ConcreteType::template hasTrait<OneResult>(),
                  "expected operation to produce one result");
    static_assert(ConcreteType::template hasTrait<OneOperand>(),
                  "expected operation to take one operand");
    static_assert(ConcreteType::template hasTrait<SameOperandsAndResultType>(),
                  "expected operation to preserve type");
    return impl::verifyIsIdempotent(op);
  }

  static OpFoldResult foldTrait(Operation* op, ArrayRef<Attribute> operands) {
    assert(op->hasAttr("device_name"));
    return impl::foldIdempotentOfIdenticalPlacement(op);
  }
};

template<typename ConcreteType>
class IsInvolutionOfIdenticalPlacement
    : public TraitBase<ConcreteType, IsInvolutionOfIdenticalPlacement> {
 public:
  static LogicalResult verifyTrait(Operation* op) {
    static_assert(ConcreteType::template hasTrait<OneResult>(),
                  "expected operation to produce one result");
    static_assert(ConcreteType::template hasTrait<OneOperand>(),
                  "expected operation to take one operand");
    static_assert(ConcreteType::template hasTrait<SameOperandsAndResultType>(),
                  "expected operation to preserve type");
    return impl::verifyIsInvolution(op);
  }

  static OpFoldResult foldTrait(Operation* op, ArrayRef<Attribute> operands) {
    assert(op->hasAttr("device_name"));
    return impl::foldInvolutionOfIdenticalPlacement(op);
  }
};

}  // namespace OpTrait

}  // namespace mlir

#define GET_OP_CLASSES
#include "OneFlow/OneFlowOps.h.inc"

#endif  // ONEFLOW_IR_INCLUDE_ONEFLOW_ONEFLOWOPS_H_

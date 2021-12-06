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

#include "OneFlow/OneFlowSupport.h"
#include "OneFlow/OneFlowInterfaces.h.inc"
#include "OneFlow/OneFlowEnums.h.inc"

namespace mlir {

class FuncOp;

namespace OpTrait {

namespace impl {

OpFoldResult foldIdempotentOfIdenticalPlacement(Operation* op);
OpFoldResult foldInvolutionOfIdenticalPlacement(Operation* op);
LogicalResult VerifyIsOpConfCompatible(Operation* op);
LogicalResult VerifyIsImportCompatible(Operation* op);

}  // namespace impl

template<typename ConcreteType>
class IsOpConfCompatible : public TraitBase<ConcreteType, IsOpConfCompatible> {
 public:
  static StringRef getOpNameAttr() { return "op_name"; }
  static StringRef getDeviceTagAttr() { return "device_tag"; }
  static StringRef getDeviceNameAttr() { return "device_name"; }
  static StringRef getScopeSymbolIDAttr() { return "scope_symbol_id"; }
  static StringRef getHierarchyAttr() { return "hierarchy"; }
  static LogicalResult verifyTrait(Operation* op) { return impl::VerifyIsOpConfCompatible(op); }
};

template<typename ConcreteType>
class IsImportCompatible : public TraitBase<ConcreteType, IsImportCompatible> {
 public:
  static StringRef getOutputLBNsAttr() { return "output_lbns"; }
  static LogicalResult verifyTrait(Operation* op) { return impl::VerifyIsImportCompatible(op); }
};

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
    static_assert(ConcreteType::template hasTrait<OpTrait::IsOpConfCompatible>(),
                  "expected operation to be op conf compatible");
    return impl::verifyIsIdempotent(op);
  }

  static OpFoldResult foldTrait(Operation* op, ArrayRef<Attribute> operands) {
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
    static_assert(ConcreteType::template hasTrait<OpTrait::IsOpConfCompatible>(),
                  "expected operation to be op conf compatible");
    return impl::verifyIsInvolution(op);
  }

  static OpFoldResult foldTrait(Operation* op, ArrayRef<Attribute> operands) {
    return impl::foldInvolutionOfIdenticalPlacement(op);
  }
};

template<typename ConcreteType>
class IsAlternative : public TraitBase<ConcreteType, IsAlternative> {
 public:
  static StringRef getOpTypeNameAttr() { return "op_type_name"; }
  static LogicalResult verifyTrait(Operation* op) {
    if (op->hasAttrOfType<StringAttr>(getOpTypeNameAttr())) {
      return success();
    } else {
      return op->emitError("expected operation to have attribute: " + getOpTypeNameAttr());
    }
  }
};

}  // namespace OpTrait

template<typename T>
inline std::string GetOpTypeName(T op) {
  std::string op_type_name = op->getName().stripDialect().str();
  if (op->template hasTrait<OpTrait::IsAlternative>()) {
    op_type_name =
        op->template getAttrOfType<StringAttr>(OpTrait::IsAlternative<void>::getOpTypeNameAttr())
            .str();
  }
  if (auto alternative_name = dyn_cast<HasAlternativeOpTypeName>(op)) {
    op_type_name = alternative_name.getOriginalOpTypeName();
  }
  return op_type_name;
}

}  // namespace mlir

#define GET_OP_CLASSES
#include "OneFlow/OneFlowOps.h.inc"
#define GET_OP_CLASSES
#include "OneFlow/OneFlow.Ops.h.inc"

#endif  // ONEFLOW_IR_INCLUDE_ONEFLOW_ONEFLOWOPS_H_

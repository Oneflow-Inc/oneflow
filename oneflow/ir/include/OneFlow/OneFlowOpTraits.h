#ifndef ONEFLOW_IR_INCLUDE_ONEFLOW_ONEFLOWOPTRAITS_H_
#define ONEFLOW_IR_INCLUDE_ONEFLOW_ONEFLOWOPTRAITS_H_

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"

namespace mlir {

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

}  // namespace mlir

#endif  // ONEFLOW_IR_INCLUDE_ONEFLOW_ONEFLOWOPTRAITS_H_

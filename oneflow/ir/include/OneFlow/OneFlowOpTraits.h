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
#ifndef ONEFLOW_IR_INCLUDE_ONEFLOW_ONEFLOWOPTRAITS_H_
#define ONEFLOW_IR_INCLUDE_ONEFLOW_ONEFLOWOPTRAITS_H_

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "oneflow/core/operator/op_conf.pb.h"

namespace mlir {

namespace OpTrait {

namespace impl {

OpFoldResult foldIdempotentOfIdenticalPlacement(Operation* op);
OpFoldResult foldInvolutionOfIdenticalPlacement(Operation* op);
LogicalResult VerifyIsOpConfCompatible(Operation* op);
LogicalResult VerifyIsImportCompatible(Operation* op);

// trait IsOpConfCompatible
LogicalResult saveAttrToOpConf(Operation* op, ::oneflow::OperatorConf* op_conf);
LogicalResult saveAttrsToNamedAttrList(Operation* op, NamedAttrList& named_attr_list);
StringAttr getOpName(Operation* op);
StringAttr getDeviceTag(Operation* op);
ArrayAttr getDeviceName(Operation* op);
IntegerAttr getScopeSymbolID(Operation* op);
ArrayAttr getHierarchy(Operation* op);

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
  static LogicalResult dump_attr(Operation* op, ::oneflow::OperatorConf* op_conf) {
    return impl::saveAttrToOpConf(op, op_conf);
  }
  static LogicalResult saveToNamedAttrList(Operation* op, NamedAttrList& named_attr_list) {
    return impl::saveAttrsToNamedAttrList(op, named_attr_list);
  }
  static StringAttr getOpName(Operation* op) { return impl::getOpName(op); }
  static StringAttr getDeviceTag(Operation* op) { return impl::getDeviceTag(op); }
  static ArrayAttr getDeviceName(Operation* op) { return impl::getDeviceName(op); }
  static IntegerAttr getScopeSymbolID(Operation* op) { return impl::getScopeSymbolID(op); }
  static ArrayAttr getHierarchy(Operation* op) { return impl::getHierarchy(op); }
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

template<typename ConcreteType>
class TensorSource : public TraitBase<ConcreteType, TensorSource> {
 public:
  static StringRef getShapeAttrName() { return "shape"; }
  static StringRef getDataTypeAttrName() { return "data_type"; }
  static StringRef getIsDynamicAttrName() { return "is_dynamic"; }
  static StringRef getNdSbpAttrName() { return "nd_sbp"; }
  static StringRef getSbpAttrName() { return "parallel"; }

  static LogicalResult verifyTrait(Operation* op) {
    if (!op->hasAttrOfType<ArrayAttr>(getShapeAttrName())) {
      return op->emitError("expected operation to have attribute: " + getShapeAttrName());
    }
    if (!op->hasAttrOfType<IntegerAttr>(getDataTypeAttrName())) {
      return op->emitError("expected operation to have attribute: " + getDataTypeAttrName());
    }
    return success();
  }
};

template<typename ConcreteType>
class OnlyExistsInIR : public TraitBase<ConcreteType, OnlyExistsInIR> {};

template<typename ConcreteType>
class IsElementwise : public TraitBase<ConcreteType, IsElementwise> {};

}  // namespace OpTrait

}  // namespace mlir

#endif  // ONEFLOW_IR_INCLUDE_ONEFLOW_ONEFLOWOPTRAITS_H_

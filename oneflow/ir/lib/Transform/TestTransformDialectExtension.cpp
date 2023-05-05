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
//===- TestTransformDialectExtension.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines an extension of the MLIR Transform dialect for testing
// purposes.
//
//===----------------------------------------------------------------------===//

#include "Transform/TestTransformDialectExtension.h"
#include "Transform/TestTransformStateExtension.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {
/// Simple transform op defined outside of the dialect. Just emits a remark when
/// applied. This op is defined in C++ to test that C++ definitions also work
/// for op injection into the Transform dialect.
class TestTransformOp : public Op<TestTransformOp, transform::TransformOpInterface::Trait,
                                  MemoryEffectOpInterface::Trait> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestTransformOp)

  using Op::Op;

  static ArrayRef<StringRef> getAttributeNames() { return {}; }

  static constexpr llvm::StringLiteral getOperationName() {
    return llvm::StringLiteral("transform.test_transform_op");
  }

  DiagnosedSilenceableFailure apply(transform::TransformResults& results,
                                    transform::TransformState& state) {
    InFlightDiagnostic remark = emitRemark() << "applying transformation";
    if (Attribute message = getMessage()) remark << " " << message;

    return DiagnosedSilenceableFailure::success();
  }

  Attribute getMessage() { return getOperation()->getAttr("message"); }

  static ParseResult parse(OpAsmParser& parser, OperationState& state) {
    StringAttr message;
    OptionalParseResult result = parser.parseOptionalAttribute(message);
    if (!result.has_value()) return success();

    if (result.value().succeeded()) state.addAttribute("message", message);
    return result.value();
  }

  void print(OpAsmPrinter& printer) {
    if (getMessage()) printer << " " << getMessage();
  }

  // No side effects.
  void getEffects(SmallVectorImpl<MemoryEffects::EffectInstance>& effects) {}
};

/// A test op to exercise the verifier of the PossibleTopLevelTransformOpTrait
/// in cases where it is attached to ops that do not comply with the trait
/// requirements. This op cannot be defined in ODS because ODS generates strict
/// verifiers that overalp with those in the trait and run earlier.
class TestTransformUnrestrictedOpNoInterface
    : public Op<TestTransformUnrestrictedOpNoInterface, transform::PossibleTopLevelTransformOpTrait,
                transform::TransformOpInterface::Trait, MemoryEffectOpInterface::Trait> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestTransformUnrestrictedOpNoInterface)

  using Op::Op;

  static ArrayRef<StringRef> getAttributeNames() { return {}; }

  static constexpr llvm::StringLiteral getOperationName() {
    return llvm::StringLiteral("transform.test_transform_unrestricted_op_no_interface");
  }

  DiagnosedSilenceableFailure apply(transform::TransformResults& results,
                                    transform::TransformState& state) {
    return DiagnosedSilenceableFailure::success();
  }

  // No side effects.
  void getEffects(SmallVectorImpl<MemoryEffects::EffectInstance>& effects) {}
};
}  // namespace

DiagnosedSilenceableFailure mlir::transform::TestProduceSelfHandleOrForwardOperandOp::apply(
    transform::TransformResults& results, transform::TransformState& state) {
  if (getOperation()->getNumOperands() != 0) {
    results.set(getResult().cast<OpResult>(), getOperation()->getOperand(0).getDefiningOp());
  } else {
    results.set(getResult().cast<OpResult>(), getOperation());
  }
  return DiagnosedSilenceableFailure::success();
}

void mlir::transform::TestProduceSelfHandleOrForwardOperandOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance>& effects) {
  if (getOperand()) transform::onlyReadsHandle(getOperand(), effects);
  transform::producesHandle(getRes(), effects);
}

DiagnosedSilenceableFailure mlir::transform::TestProduceValueHandleToSelfOperand::apply(
    transform::TransformResults& results, transform::TransformState& state) {
  results.setValues(getOut().cast<OpResult>(), getIn());
  return DiagnosedSilenceableFailure::success();
}

void mlir::transform::TestProduceValueHandleToSelfOperand::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance>& effects) {
  transform::onlyReadsHandle(getIn(), effects);
  transform::producesHandle(getOut(), effects);
  transform::onlyReadsPayload(effects);
}

DiagnosedSilenceableFailure mlir::transform::TestProduceValueHandleToResult::applyToOne(
    Operation* target, transform::ApplyToEachResultList& results,
    transform::TransformState& state) {
  if (target->getNumResults() <= getNumber())
    return emitSilenceableError() << "payload has no result #" << getNumber();
  results.push_back(target->getResult(getNumber()));
  return DiagnosedSilenceableFailure::success();
}

void mlir::transform::TestProduceValueHandleToResult::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance>& effects) {
  transform::onlyReadsHandle(getIn(), effects);
  transform::producesHandle(getOut(), effects);
  transform::onlyReadsPayload(effects);
}

DiagnosedSilenceableFailure
mlir::transform::TestProduceValueHandleToArgumentOfParentBlock::applyToOne(
    Operation* target, transform::ApplyToEachResultList& results,
    transform::TransformState& state) {
  if (!target->getBlock()) return emitSilenceableError() << "payload has no parent block";
  if (target->getBlock()->getNumArguments() <= getNumber())
    return emitSilenceableError() << "parent of the payload has no argument #" << getNumber();
  results.push_back(target->getBlock()->getArgument(getNumber()));
  return DiagnosedSilenceableFailure::success();
}

void mlir::transform::TestProduceValueHandleToArgumentOfParentBlock::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance>& effects) {
  transform::onlyReadsHandle(getIn(), effects);
  transform::producesHandle(getOut(), effects);
  transform::onlyReadsPayload(effects);
}

DiagnosedSilenceableFailure mlir::transform::TestConsumeOperand::apply(
    transform::TransformResults& results, transform::TransformState& state) {
  return DiagnosedSilenceableFailure::success();
}

void mlir::transform::TestConsumeOperand::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance>& effects) {
  transform::consumesHandle(getOperand(), effects);
  if (getSecondOperand()) transform::consumesHandle(getSecondOperand(), effects);
  transform::modifiesPayload(effects);
}

DiagnosedSilenceableFailure mlir::transform::TestConsumeOperandOfOpKindOrFail::apply(
    transform::TransformResults& results, transform::TransformState& state) {
  ArrayRef<Operation*> payload = state.getPayloadOps(getOperand());
  assert(payload.size() == 1 && "expected a single target op");
  if (payload[0]->getName().getStringRef() != getOpKind()) {
    return emitSilenceableError()
           << "op expected the operand to be associated a payload op of kind " << getOpKind()
           << " got " << payload[0]->getName().getStringRef();
  }

  emitRemark() << "succeeded";
  return DiagnosedSilenceableFailure::success();
}

void mlir::transform::TestConsumeOperandOfOpKindOrFail::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance>& effects) {
  transform::consumesHandle(getOperand(), effects);
  transform::modifiesPayload(effects);
}

DiagnosedSilenceableFailure mlir::transform::TestSucceedIfOperandOfOpKind::matchOperation(
    Operation* op, transform::TransformResults& results, transform::TransformState& state) {
  if (op->getName().getStringRef() != getOpKind()) {
    return emitSilenceableError()
           << "op expected the operand to be associated with a payload op of "
              "kind "
           << getOpKind() << " got " << op->getName().getStringRef();
  }
  return DiagnosedSilenceableFailure::success();
}

void mlir::transform::TestSucceedIfOperandOfOpKind::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance>& effects) {
  transform::onlyReadsHandle(getOperand(), effects);
  transform::onlyReadsPayload(effects);
}

DiagnosedSilenceableFailure mlir::transform::TestPrintRemarkAtOperandOp::apply(
    transform::TransformResults& results, transform::TransformState& state) {
  ArrayRef<Operation*> payload = state.getPayloadOps(getOperand());
  for (Operation* op : payload) op->emitRemark() << getMessage();

  return DiagnosedSilenceableFailure::success();
}

void mlir::transform::TestPrintRemarkAtOperandOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance>& effects) {
  transform::onlyReadsHandle(getOperand(), effects);
  transform::onlyReadsPayload(effects);
}

DiagnosedSilenceableFailure mlir::transform::TestPrintRemarkAtOperandValue::apply(
    transform::TransformResults& results, transform::TransformState& state) {
  ArrayRef<Value> values = state.getPayloadValues(getIn());
  for (Value value : values) {
    std::string note;
    llvm::raw_string_ostream os(note);
    if (auto arg = value.dyn_cast<BlockArgument>()) {
      os << "a block argument #" << arg.getArgNumber() << " in block #"
         << std::distance(arg.getOwner()->getParent()->begin(), arg.getOwner()->getIterator())
         << " in region #" << arg.getOwner()->getParent()->getRegionNumber();
    } else {
      os << "an op result #" << value.cast<OpResult>().getResultNumber();
    }
    InFlightDiagnostic diag = ::emitRemark(value.getLoc()) << getMessage();
    diag.attachNote() << "value handle points to " << os.str();
  }
  return DiagnosedSilenceableFailure::success();
}

void mlir::transform::TestPrintRemarkAtOperandValue::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance>& effects) {
  transform::onlyReadsHandle(getIn(), effects);
  transform::onlyReadsPayload(effects);
}

DiagnosedSilenceableFailure mlir::transform::TestAddTestExtensionOp::apply(
    transform::TransformResults& results, transform::TransformState& state) {
  state.addExtension<TestTransformStateExtension>(getMessageAttr());
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure mlir::transform::TestCheckIfTestExtensionPresentOp::apply(
    transform::TransformResults& results, transform::TransformState& state) {
  auto* extension = state.getExtension<TestTransformStateExtension>();
  if (!extension) {
    emitRemark() << "extension absent";
    return DiagnosedSilenceableFailure::success();
  }

  InFlightDiagnostic diag = emitRemark() << "extension present, " << extension->getMessage();
  for (Operation* payload : state.getPayloadOps(getOperand())) {
    diag.attachNote(payload->getLoc()) << "associated payload op";
#ifndef NDEBUG
    SmallVector<Value> handles;
    assert(succeeded(state.getHandlesForPayloadOp(payload, handles)));
    assert(llvm::is_contained(handles, getOperand())
           && "inconsistent mapping between transform IR handles and payload IR "
              "operations");
#endif  // NDEBUG
  }

  return DiagnosedSilenceableFailure::success();
}

void mlir::transform::TestCheckIfTestExtensionPresentOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance>& effects) {
  transform::onlyReadsHandle(getOperand(), effects);
  transform::onlyReadsPayload(effects);
}

DiagnosedSilenceableFailure mlir::transform::TestRemapOperandPayloadToSelfOp::apply(
    transform::TransformResults& results, transform::TransformState& state) {
  auto* extension = state.getExtension<TestTransformStateExtension>();
  if (!extension) return emitDefiniteFailure("TestTransformStateExtension missing");

  if (failed(extension->updateMapping(state.getPayloadOps(getOperand()).front(), getOperation())))
    return DiagnosedSilenceableFailure::definiteFailure();
  if (getNumResults() > 0) results.set(getResult(0).cast<OpResult>(), getOperation());
  return DiagnosedSilenceableFailure::success();
}

void mlir::transform::TestRemapOperandPayloadToSelfOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance>& effects) {
  transform::onlyReadsHandle(getOperand(), effects);
  transform::producesHandle(getOut(), effects);
  transform::onlyReadsPayload(effects);
}

DiagnosedSilenceableFailure mlir::transform::TestRemoveTestExtensionOp::apply(
    transform::TransformResults& results, transform::TransformState& state) {
  state.removeExtension<TestTransformStateExtension>();
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure mlir::transform::TestReversePayloadOpsOp::apply(
    transform::TransformResults& results, transform::TransformState& state) {
  ArrayRef<Operation*> payloadOps = state.getPayloadOps(getTarget());
  auto reversedOps = llvm::to_vector(llvm::reverse(payloadOps));
  results.set(getResult().cast<OpResult>(), reversedOps);
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure mlir::transform::TestTransformOpWithRegions::apply(
    transform::TransformResults& results, transform::TransformState& state) {
  return DiagnosedSilenceableFailure::success();
}

void mlir::transform::TestTransformOpWithRegions::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance>& effects) {}

DiagnosedSilenceableFailure mlir::transform::TestBranchingTransformOpTerminator::apply(
    transform::TransformResults& results, transform::TransformState& state) {
  return DiagnosedSilenceableFailure::success();
}

void mlir::transform::TestBranchingTransformOpTerminator::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance>& effects) {}

DiagnosedSilenceableFailure mlir::transform::TestEmitRemarkAndEraseOperandOp::apply(
    transform::TransformResults& results, transform::TransformState& state) {
  emitRemark() << getRemark();
  for (Operation* op : state.getPayloadOps(getTarget())) op->erase();

  if (getFailAfterErase()) return emitSilenceableError() << "silenceable error";
  return DiagnosedSilenceableFailure::success();
}

void mlir::transform::TestEmitRemarkAndEraseOperandOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance>& effects) {
  transform::consumesHandle(getTarget(), effects);
  transform::modifiesPayload(effects);
}

DiagnosedSilenceableFailure mlir::transform::TestWrongNumberOfResultsOp::applyToOne(
    Operation* target, transform::ApplyToEachResultList& results,
    transform::TransformState& state) {
  OperationState opState(target->getLoc(), "foo");
  results.push_back(OpBuilder(target).create(opState));
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure mlir::transform::TestWrongNumberOfMultiResultsOp::applyToOne(
    Operation* target, transform::ApplyToEachResultList& results,
    transform::TransformState& state) {
  static int count = 0;
  if (count++ == 0) {
    OperationState opState(target->getLoc(), "foo");
    results.push_back(OpBuilder(target).create(opState));
  }
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure mlir::transform::TestCorrectNumberOfMultiResultsOp::applyToOne(
    Operation* target, transform::ApplyToEachResultList& results,
    transform::TransformState& state) {
  OperationState opState(target->getLoc(), "foo");
  results.push_back(OpBuilder(target).create(opState));
  results.push_back(OpBuilder(target).create(opState));
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure mlir::transform::TestMixedNullAndNonNullResultsOp::applyToOne(
    Operation* target, transform::ApplyToEachResultList& results,
    transform::TransformState& state) {
  OperationState opState(target->getLoc(), "foo");
  results.push_back(nullptr);
  results.push_back(OpBuilder(target).create(opState));
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure mlir::transform::TestMixedSuccessAndSilenceableOp::applyToOne(
    Operation* target, transform::ApplyToEachResultList& results,
    transform::TransformState& state) {
  if (target->hasAttr("target_me")) return DiagnosedSilenceableFailure::success();
  return emitDefaultSilenceableFailure(target);
}

DiagnosedSilenceableFailure mlir::transform::TestPrintNumberOfAssociatedPayloadIROps::apply(
    transform::TransformResults& results, transform::TransformState& state) {
  if (!getHandle()) emitRemark() << 0;
  emitRemark() << state.getPayloadOps(getHandle()).size();
  return DiagnosedSilenceableFailure::success();
}

void mlir::transform::TestPrintNumberOfAssociatedPayloadIROps::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance>& effects) {
  transform::onlyReadsHandle(getHandle(), effects);
}

DiagnosedSilenceableFailure mlir::transform::TestCopyPayloadOp::apply(
    transform::TransformResults& results, transform::TransformState& state) {
  results.set(getCopy().cast<OpResult>(), state.getPayloadOps(getHandle()));
  return DiagnosedSilenceableFailure::success();
}

void mlir::transform::TestCopyPayloadOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance>& effects) {
  transform::onlyReadsHandle(getHandle(), effects);
  transform::producesHandle(getCopy(), effects);
  transform::onlyReadsPayload(effects);
}

DiagnosedSilenceableFailure mlir::transform::TestDialectOpType::checkPayload(
    Location loc, ArrayRef<Operation*> payload) const {
  if (payload.empty()) return DiagnosedSilenceableFailure::success();

  for (Operation* op : payload) {
    if (op->getName().getDialectNamespace() != "test") {
      return emitSilenceableError(loc) << "expected the payload operation to "
                                          "belong to the 'test' dialect";
    }
  }

  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure mlir::transform::TestDialectParamType::checkPayload(
    Location loc, ArrayRef<Attribute> payload) const {
  for (Attribute attr : payload) {
    auto integerAttr = attr.dyn_cast<IntegerAttr>();
    if (integerAttr && integerAttr.getType().isSignlessInteger(32)) continue;
    return emitSilenceableError(loc) << "expected the parameter to be a i32 integer attribute";
  }

  return DiagnosedSilenceableFailure::success();
}

void mlir::transform::TestReportNumberOfTrackedHandlesNestedUnder::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance>& effects) {
  transform::onlyReadsHandle(getTarget(), effects);
}

DiagnosedSilenceableFailure mlir::transform::TestReportNumberOfTrackedHandlesNestedUnder::apply(
    transform::TransformResults& results, transform::TransformState& state) {
  int64_t count = 0;
  for (Operation* op : state.getPayloadOps(getTarget())) {
    op->walk([&](Operation* nested) {
      SmallVector<Value> handles;
      (void)state.getHandlesForPayloadOp(nested, handles);
      count += handles.size();
    });
  }
  emitRemark() << count << " handles nested under";
  return DiagnosedSilenceableFailure::success();
}

void mlir::transform::TestPrintParamOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance>& effects) {
  transform::onlyReadsHandle(getParam(), effects);
  if (getAnchor()) transform::onlyReadsHandle(getAnchor(), effects);
  transform::onlyReadsPayload(effects);
}

DiagnosedSilenceableFailure mlir::transform::TestPrintParamOp::apply(
    transform::TransformResults& results, transform::TransformState& state) {
  std::string str;
  llvm::raw_string_ostream os(str);
  if (getMessage()) os << *getMessage() << " ";
  llvm::interleaveComma(state.getParams(getParam()), os);
  if (!getAnchor()) {
    emitRemark() << os.str();
    return DiagnosedSilenceableFailure::success();
  }
  for (Operation* payload : state.getPayloadOps(getAnchor()))
    ::mlir::emitRemark(payload->getLoc()) << os.str();
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure mlir::transform::TestAddToParamOp::apply(
    transform::TransformResults& results, transform::TransformState& state) {
  SmallVector<uint32_t> values(/*Size=*/1, /*Value=*/0);
  if (Value param = getParam()) {
    values =
        llvm::to_vector(llvm::map_range(state.getParams(param), [](Attribute attr) -> uint32_t {
          return attr.cast<IntegerAttr>().getValue().getLimitedValue(UINT32_MAX);
        }));
  }

  Builder builder(getContext());
  SmallVector<Attribute> result =
      llvm::to_vector(llvm::map_range(values, [this, &builder](uint32_t value) -> Attribute {
        return builder.getI32IntegerAttr(value + getAddendum());
      }));
  results.setParams(getResult().cast<OpResult>(), result);
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure mlir::transform::TestProduceParamWithNumberOfTestOps::apply(
    transform::TransformResults& results, transform::TransformState& state) {
  Builder builder(getContext());
  SmallVector<Attribute> result = llvm::to_vector(llvm::map_range(
      state.getPayloadOps(getHandle()), [&builder](Operation* payload) -> Attribute {
        int32_t count = 0;
        payload->walk([&count](Operation* op) {
          if (op->getName().getDialectNamespace() == "test") ++count;
        });
        return builder.getI32IntegerAttr(count);
      }));
  results.setParams(getResult().cast<OpResult>(), result);
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure mlir::transform::TestProduceIntegerParamWithTypeOp::apply(
    transform::TransformResults& results, transform::TransformState& state) {
  Attribute zero = IntegerAttr::get(getType(), 0);
  results.setParams(getResult().cast<OpResult>(), zero);
  return DiagnosedSilenceableFailure::success();
}

LogicalResult mlir::transform::TestProduceIntegerParamWithTypeOp::verify() {
  if (!getType().isa<IntegerType>()) { return emitOpError() << "expects an integer type"; }
  return success();
}

void mlir::transform::TestProduceTransformParamOrForwardOperandOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance>& effects) {
  transform::onlyReadsHandle(getIn(), effects);
  transform::producesHandle(getOut(), effects);
  transform::producesHandle(getParam(), effects);
}

DiagnosedSilenceableFailure
mlir::transform::TestProduceTransformParamOrForwardOperandOp::applyToOne(
    Operation* target, ::transform::ApplyToEachResultList& results,
    ::transform::TransformState& state) {
  Builder builder(getContext());
  if (getFirstResultIsParam()) {
    results.push_back(builder.getI64IntegerAttr(0));
  } else if (getFirstResultIsNull()) {
    results.push_back(nullptr);
  } else {
    results.push_back(state.getPayloadOps(getIn()).front());
  }

  if (getSecondResultIsHandle()) {
    results.push_back(state.getPayloadOps(getIn()).front());
  } else {
    results.push_back(builder.getI64IntegerAttr(42));
  }

  return DiagnosedSilenceableFailure::success();
}

void mlir::transform::TestProduceNullPayloadOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance>& effects) {
  transform::producesHandle(getOut(), effects);
}

DiagnosedSilenceableFailure mlir::transform::TestProduceNullPayloadOp::apply(
    transform::TransformResults& results, transform::TransformState& state) {
  SmallVector<Operation*, 1> null({nullptr});
  results.set(getOut().cast<OpResult>(), null);
  return DiagnosedSilenceableFailure::success();
}

void mlir::transform::TestProduceNullParamOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance>& effects) {
  transform::producesHandle(getOut(), effects);
}

DiagnosedSilenceableFailure mlir::transform::TestProduceNullParamOp::apply(
    transform::TransformResults& results, transform::TransformState& state) {
  results.setParams(getOut().cast<OpResult>(), Attribute());
  return DiagnosedSilenceableFailure::success();
}

void mlir::transform::TestProduceNullValueOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance>& effects) {
  transform::producesHandle(getOut(), effects);
}

DiagnosedSilenceableFailure mlir::transform::TestProduceNullValueOp::apply(
    transform::TransformResults& results, transform::TransformState& state) {
  results.setValues(getOut().cast<OpResult>(), Value());
  return DiagnosedSilenceableFailure::success();
}

void mlir::transform::TestRequiredMemoryEffectsOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance>& effects) {
  if (getHasOperandEffect()) transform::consumesHandle(getIn(), effects);

  if (getHasResultEffect())
    transform::producesHandle(getOut(), effects);
  else
    transform::onlyReadsHandle(getOut(), effects);

  if (getModifiesPayload()) transform::modifiesPayload(effects);
}

DiagnosedSilenceableFailure mlir::transform::TestRequiredMemoryEffectsOp::apply(
    transform::TransformResults& results, transform::TransformState& state) {
  results.set(getOut().cast<OpResult>(), state.getPayloadOps(getIn()));
  return DiagnosedSilenceableFailure::success();
}

namespace {
/// Test extension of the Transform dialect. Registers additional ops and
/// declares PDL as dependent dialect since the additional ops are using PDL
/// types for operands and results.
class TestTransformDialectExtension
    : public transform::TransformDialectExtension<TestTransformDialectExtension> {
 public:
  using Base::Base;

  void init() {
    declareDependentDialect<pdl::PDLDialect>();
    registerTransformOps<TestTransformOp, TestTransformUnrestrictedOpNoInterface,
#define GET_OP_LIST
#include "Transform/TestTransformDialectExtension.cpp.inc"
                         >();
    registerTypes<
#define GET_TYPEDEF_LIST
#include "Transform/TestTransformDialectExtensionTypes.cpp.inc"
        >();
  }
};
}  // namespace

// These are automatically generated by ODS but are not used as the Transform
// dialect uses a different dispatch mechanism to support dialect extensions.
LLVM_ATTRIBUTE_UNUSED static OptionalParseResult generatedTypeParser(AsmParser& parser,
                                                                     StringRef* mnemonic,
                                                                     Type& value);
LLVM_ATTRIBUTE_UNUSED static LogicalResult generatedTypePrinter(Type def, AsmPrinter& printer);

#define GET_TYPEDEF_CLASSES
#include "Transform/TestTransformDialectExtensionTypes.cpp.inc"

#define GET_OP_CLASSES
#include "Transform/TestTransformDialectExtension.cpp.inc"

void ::transform::registerTestTransformDialectExtension(DialectRegistry& registry) {
  registry.addExtensions<TestTransformDialectExtension>();
}

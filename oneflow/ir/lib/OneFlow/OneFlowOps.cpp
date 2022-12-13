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
#include "OneFlow/OneFlowOps.h"
#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/OneFlowSupport.h"
#include "OneFlow/SBP/SBPAttributes.h"
#include "OneFlow/Transform/TransposeHelpers.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/just.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/shape_vec.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/functional/functional_api.yaml.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/framework/tensor_util.h"
#include "oneflow/core/job/lazy_mode.h"
#include "oneflow/core/vm/vm_util.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"

#include "llvm/Support/Casting.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace mlir {

namespace oneflow {

OperandRange UserOp::dataInputOperands() { return data_input(); }
OperandRange UserOp::ctrlInputOperands() { return ctrl_inputs(); }
ResultRange UserOp::dataOutputResults() { return data_output(); }
Value UserOp::ctrlOutputResult() { return ctrl_output(); }

OperandRange SystemOp::dataInputOperands() { return data_input(); }
OperandRange SystemOp::ctrlInputOperands() { return ctrl_inputs(); }
ResultRange SystemOp::dataOutputResults() { return data_output(); }
Value SystemOp::ctrlOutputResult() { return ctrl_output(); }

OperandRange VariableOp::dataInputOperands() { return {operand_begin(), operand_begin()}; }
OperandRange VariableOp::ctrlInputOperands() { return ctrl_inputs(); }
ResultRange VariableOp::dataOutputResults() { return output().dyn_cast<OpResult>(); }
Value VariableOp::ctrlOutputResult() { return ctrl_output(); }

OperandRange InputOp::dataInputOperands() { return getODSOperands(0); }
OperandRange InputOp::ctrlInputOperands() { return ctrl_inputs(); }
ResultRange InputOp::dataOutputResults() { return output().dyn_cast<OpResult>(); }
Value InputOp::ctrlOutputResult() { return ctrl_output(); }

OperandRange OutputOp::dataInputOperands() { return getODSOperands(0); }
OperandRange OutputOp::ctrlInputOperands() { return ctrl_inputs(); }
ResultRange OutputOp::dataOutputResults() { return output().dyn_cast<OpResult>(); }
Value OutputOp::ctrlOutputResult() { return ctrl_output(); }

static ParseResult parseConstantOp(OpAsmParser& parser, OperationState& result) {
  mlir::DenseElementsAttr value;
  if (parser.parseOptionalAttrDict(result.attributes)
      || parser.parseAttribute(value, "value", result.attributes)) {
    return failure();
  }
  result.addTypes(value.getType());
  return success();
}

ArrayAttr getSI32ArrayAttr(::mlir::PatternRewriter& rewriter, ArrayRef<int32_t> values) {
  auto attrs = llvm::to_vector<8>(llvm::map_range(
      values, [&](int32_t v) -> Attribute { return rewriter.getSI32IntegerAttr(v); }));
  return rewriter.getArrayAttr(attrs);
}

namespace {

LogicalResult TrimRedundantCtrl(Operation* op, PatternRewriter& rewriter) {
  auto ctrl_out = GetCtrlOutputResult(op);
  auto data_outputs = GetDataOutputResults(op);
  if (ctrl_out && ctrl_out.getValue().use_empty()) {
    const int32_t num_data_outputs = data_outputs.size();
    NamedAttrList attributes(op->getAttrs());
    if (op->hasTrait<OpTrait::AttrSizedResultSegments>()) {
      attributes.erase(OpTrait::AttrSizedResultSegments<void>::getResultSegmentSizeAttr());
      attributes.push_back(
          rewriter.getNamedAttr(OpTrait::AttrSizedResultSegments<void>::getResultSegmentSizeAttr(),
                                rewriter.getI32VectorAttr({num_data_outputs, 0})));
    }
    OperationState state(op->getLoc(), op->getName(), op->getOperands(), data_outputs.getTypes(),
                         attributes);
    auto created = rewriter.create(state);
    for (auto data_output : data_outputs) {
      data_output.replaceAllUsesWith(created->getOpResult(data_output.getResultNumber()));
    }
    op->erase();
    return success();
  }
  return failure();
}

bool IsCtrlOutTrimmed(UserOp& op) { return !op.ctrl_output(); }

bool IsCtrlInAbsent(UserOp& op) {
  if (!op->hasAttrOfType<DenseIntElementsAttr>(
          OpTrait::AttrSizedOperandSegments<void>::getOperandSegmentSizeAttr()))
    op.dump();
  return op.ctrl_inputs().empty();
}

}  // namespace

template<typename T>
static void getValuesFromIntArrayAttribute(ArrayAttr attr, SmallVector<T>& arrayValues) {
  for (Attribute val : attr.getValue()) {
    arrayValues.push_back(val.cast<IntegerAttr>().getValue().getSExtValue());
  }
}

struct ConcreteUserOps : public OpRewritePattern<UserOp> {
  explicit ConcreteUserOps(MLIRContext* context)
      : OpRewritePattern<UserOp>(context, /*benefit=*/1) {}
  LogicalResult matchAndRewrite(UserOp op, PatternRewriter& rewriter) const override {
    if (succeeded(TrimRedundantCtrl(op, rewriter))) { return success(); }
    // In principle, a concrete user op has no ctrl input/output. Some benefits:
    // 1. simplify things
    // 2. make conversion and code gen more doable
    // 3. enable the reuse of established MLIR infra like built-in traits
    if (IsCtrlOutTrimmed(op) && IsCtrlInAbsent(op)) {
      NamedAttrList attributes(op->getAttrDictionary());
      attributes.erase(op.input_sizesAttrName());
      attributes.erase(op.output_sizesAttrName());
      attributes.erase(op.output_lbnsAttrName());
      attributes.erase(OpTrait::AttrSizedOperandSegments<void>::getOperandSegmentSizeAttr());
      attributes.erase(OpTrait::AttrSizedResultSegments<void>::getResultSegmentSizeAttr());
      llvm::SmallVector<int32_t> input_sizes, output_sizes;
      getValuesFromIntArrayAttribute(op.input_sizes(), input_sizes);
      getValuesFromIntArrayAttribute(op.output_sizes(), output_sizes);
      if (!input_sizes.empty()) {
        attributes.push_back(rewriter.getNamedAttr(
            OpTrait::AttrSizedOperandSegments<void>::getOperandSegmentSizeAttr(),
            rewriter.getI32VectorAttr(input_sizes)));
      }
      if (!output_sizes.empty()) {
        attributes.push_back(rewriter.getNamedAttr(
            OpTrait::AttrSizedResultSegments<void>::getResultSegmentSizeAttr(),
            rewriter.getI32VectorAttr(output_sizes)));
      }
      OperationState state(op->getLoc(), OneFlowDialect::getDialectNamespace().str() + "."
                                             + op.op_type_name().str());
      state.addAttributes(attributes);
      state.addOperands(op.getODSOperands(0) /* data in */);
      state.addTypes(op.getODSResults(0 /* data out */).getTypes());
      if (auto created = rewriter.create(state)) {
        if (created->hasTrait<OpTrait::AttrSizedOperandSegments>() == false) {
          created->removeAttr(OpTrait::AttrSizedOperandSegments<void>::getOperandSegmentSizeAttr());
        }
        if (created->hasTrait<OpTrait::AttrSizedResultSegments>() == false) {
          created->removeAttr(OpTrait::AttrSizedResultSegments<void>::getResultSegmentSizeAttr());
        }
        if (created->hasTrait<OpTrait::IsAlternative>() == false) {
          created->removeAttr(OpTrait::IsAlternative<void>::getOpTypeNameAttr());
        }
        rewriter.replaceOp(op, created->getResults());
      } else {
        op->emitError("Fail to convert opaque user op to concrete op when creating: "
                      + op.op_type_name());
        op->dump();
        return failure();
      }
    }
    return success();
  }
};

void UserOp::getCanonicalizationPatterns(RewritePatternSet& results, MLIRContext* context) {
  results.insert<ConcreteUserOps>(context);
}

struct ConcreteSystemOps : public OpRewritePattern<SystemOp> {
  explicit ConcreteSystemOps(MLIRContext* context)
      : OpRewritePattern<SystemOp>(context, /*benefit=*/1) {}
  LogicalResult matchAndRewrite(oneflow::SystemOp op, PatternRewriter& rewriter) const override {
    return TrimRedundantCtrl(op, rewriter);
  }
};

void SystemOp::getCanonicalizationPatterns(RewritePatternSet& results, MLIRContext* context) {
  results.insert<ConcreteSystemOps>(context);
}

struct ConvertAddOpWithArity : public OpRewritePattern<AddNOp> {
  explicit ConvertAddOpWithArity(MLIRContext* context)
      : OpRewritePattern<AddNOp>(context, /*benefit=*/1) {}
  LogicalResult matchAndRewrite(AddNOp op, PatternRewriter& rewriter) const override {
    const auto arity = op.in().size();
    if (arity == 2) {
      NamedAttrList attributes = op->getAttrs();
      attributes.set(OpTrait::IsAlternative<void>::getOpTypeNameAttr(),
                     rewriter.getStringAttr("add_n"));
      if (auto created_op = rewriter.replaceOpWithNewOp<Add2Op>(op, op->getResultTypes(),
                                                                op.getOperands(), attributes)) {
        return success();
      } else {
        op->emitError("Fail to convert add op with arity: ") << arity;
        op->dump();
        return failure();
      }
    }
    return failure();
  }
};

void AddNOp::getCanonicalizationPatterns(RewritePatternSet& results, MLIRContext* context) {
  results.insert<ConvertAddOpWithArity>(context);
}

template<typename OpType>
struct ConcreteSystemOpPattern : public OpRewritePattern<OpType> {
  explicit ConcreteSystemOpPattern(MLIRContext* context)
      : OpRewritePattern<OpType>(context, /*benefit=*/1) {}
  LogicalResult matchAndRewrite(OpType op, PatternRewriter& rewriter) const override {
    if (op.ctrl_output() && op.ctrl_output().use_empty()) {
      NamedAttrList attributes(op->getAttrDictionary());
      if (auto created = rewriter.create<OpType>(op->getLoc(), op.output().getType(),
                                                 op->getOperands(), attributes)) {
        op.output().replaceAllUsesWith(
            created->getResult(op.output().template cast<OpResult>().getResultNumber()));
        op->erase();
        return success();
      }
    }
    return failure();
  }
};

void VariableOp::getCanonicalizationPatterns(RewritePatternSet& results, MLIRContext* context) {
  results.insert<ConcreteSystemOpPattern<VariableOp>>(context);
}

void InputOp::getCanonicalizationPatterns(RewritePatternSet& results, MLIRContext* context) {
  results.insert<ConcreteSystemOpPattern<InputOp>>(context);
}

void OutputOp::getCanonicalizationPatterns(RewritePatternSet& results, MLIRContext* context) {
  results.insert<ConcreteSystemOpPattern<OutputOp>>(context);
}

std::string Add2Op::getOriginalOpTypeName() { return "add_n"; }
std::string NormalizationInferenceOp::getOriginalOpTypeName() { return "normalization"; }

void Job::build(OpBuilder& builder, OperationState& state, StringRef name, FunctionType type) {
  state.addAttribute(SymbolTable::getSymbolAttrName(), builder.getStringAttr(name));
  state.addAttribute(getTypeAttrName(), TypeAttr::get(type));
  state.addRegion();
}

ParseResult Job::parse(OpAsmParser& parser, OperationState& result) {
  auto buildFuncType = [](Builder& builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
                          function_interface_impl::VariadicFlag,
                          std::string&) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(parser, result, /*allowVariadic=*/false,
                                                  buildFuncType);
}

void Job::print(OpAsmPrinter& p) {
  function_interface_impl::printFunctionOp(p, *this, /*isVariadic=*/false);
}

LogicalResult Job::verify() {
  // If this function is external there is nothing to do.
  if (isExternal()) return success();

  // Verify that the argument list of the function and the arg list of the entry
  // block line up.  The trait already verified that the number of arguments is
  // the same between the signature and the block.
  auto fnInputTypes = getFunctionType().getInputs();
  Block& entryBlock = front();
  for (unsigned i = 0, e = entryBlock.getNumArguments(); i != e; ++i)
    if (fnInputTypes[i] != entryBlock.getArgument(i).getType())
      return emitOpError("type of entry block argument #")
             << i << '(' << entryBlock.getArgument(i).getType()
             << ") must match the type of the corresponding argument in "
             << "function signature(" << fnInputTypes[i] << ')';

  return success();
}

LogicalResult ReturnOp::verify() {
  auto job = cast<Job>((*this)->getParentOp());

  // The operand number and types must match the function signature.
  const auto& results = job.getFunctionType().getResults();
  if (getNumOperands() != results.size())
    return emitOpError("has ") << getNumOperands() << " operands, but enclosing function (@"
                               << job.getName() << ") returns " << results.size();

  for (unsigned i = 0, e = results.size(); i != e; ++i)
    if (getOperand(i).getType() != results[i])
      return emitError() << "type of return operand " << i << " (" << getOperand(i).getType()
                         << ") doesn't match function result type (" << results[i] << ")"
                         << " in function @" << job.getName();

  return success();
}

struct NormalizationInferencePattern : public OpRewritePattern<NormalizationOp> {
  explicit NormalizationInferencePattern(MLIRContext* context)
      : OpRewritePattern<NormalizationOp>(context, /*benefit=*/1) {}
  LogicalResult matchAndRewrite(oneflow::NormalizationOp op,
                                PatternRewriter& rewriter) const override {
    if (op.mean() || op.inv_variance()) return failure();
    if (auto created_op = rewriter.replaceOpWithNewOp<NormalizationInferenceOp>(
            op, op->getResultTypes(), op.getOperands(), op->getAttrs())) {
      return success();
    }
    op.emitError("Failed to create inference bn op");
    return failure();
  }
};

void NormalizationOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                                  MLIRContext* context) {
  results.insert<NormalizationInferencePattern>(context);
}

ResultRange GetDataOutputResults(Operation* op) {
  if (auto cec = dyn_cast<ControlEdgeCompatible>(op)) {
    return cec.dataOutputResults();
  } else {
    return op->getResults();
  }
}

OperandRange GetDataInputOperands(Operation* op) {
  if (auto cec = dyn_cast<ControlEdgeCompatible>(op)) {
    return cec.dataInputOperands();
  } else {
    return op->getOperands();
  }
}

llvm::Optional<OperandRange> GetCtrlIntputOperands(Operation* op) {
  if (auto cec = dyn_cast<ControlEdgeCompatible>(op)) {
    return cec.ctrlInputOperands();
  } else {
    return llvm::None;
  }
}

llvm::Optional<OpResult> GetCtrlOutputResult(Operation* op) {
  if (auto cec = dyn_cast<ControlEdgeCompatible>(op)) {
    if (auto ctrl_out = cec.ctrlOutputResult()) { return ctrl_out.cast<OpResult>(); }
  }
  return llvm::None;
}

}  // namespace oneflow

}  // namespace mlir

#include "OneFlow/OneFlowEnums.cpp.inc"

#define GET_OP_CLASSES
#include "OneFlow/OneFlowOps.cpp.inc"
#include "OneFlow/OneFlowInterfaces.cpp.inc"

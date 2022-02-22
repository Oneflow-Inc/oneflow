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
#include "OneFlow/Passes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include <iostream>
#include <string>

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

namespace {

template<typename OpType>
LogicalResult TrimRedundantCtrl(OpType& op, PatternRewriter& rewriter) {
  if (op.ctrl_output() && op.ctrl_output().use_empty()) {
    const int32_t num_data_outputs =
        *(op.result_segment_sizes().template getValues<uint32_t>()).begin();
    NamedAttrList attributes(op->getAttrDictionary());
    attributes.erase(OpTrait::AttrSizedResultSegments<void>::getResultSegmentSizeAttr());
    attributes.append(OpTrait::AttrSizedResultSegments<void>::getResultSegmentSizeAttr(),
                      rewriter.getI32VectorAttr({num_data_outputs, 0}));
    if (auto created =
            rewriter.create<OpType>(op->getLoc(), op.getODSResults(0 /* data out */).getTypes(),
                                    op->getOperands(), attributes)) {
      for (auto out : op.data_output()) {
        out.replaceAllUsesWith(created->getResult(out.getResultNumber()));
      }
      op->erase();
      return success();
    }
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
      if (auto created = rewriter.createOperation(state)) {
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
      attributes.push_back(rewriter.getNamedAttr(OpTrait::IsAlternative<void>::getOpTypeNameAttr(),
                                                 rewriter.getStringAttr("add_n")));
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

void NormalizationAddReluOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                   Value x, Value addend, Value moving_mean, Value moving_variance,
                                   Value gamma, Value beta, StringRef op_name, StringRef device_tag,
                                   ArrayAttr device_name, IntegerAttr scope_symbol_id,
                                   ArrayAttr hierarchy, DenseElementsAttr operand_segment_sizes,
                                   DenseElementsAttr result_segment_sizes, IntegerAttr axis,
                                   FloatAttr epsilon, BoolAttr training, FloatAttr momentum) {
  odsState.addOperands(x);
  if (addend) odsState.addOperands(addend);
  if (moving_mean) odsState.addOperands(moving_mean);
  if (moving_variance) odsState.addOperands(moving_variance);
  odsState.addOperands(gamma);
  odsState.addOperands(beta);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name),
                        odsBuilder.getI32VectorAttr({1, (addend ? 1 : 0), (moving_mean ? 1 : 0),
                                                     (moving_variance ? 1 : 0), 1, 1}));

  odsState.addAttribute(op_nameAttrName(odsState.name), odsBuilder.getStringAttr(op_name));
  odsState.addAttribute(device_tagAttrName(odsState.name), odsBuilder.getStringAttr(device_tag));
  odsState.addAttribute(device_nameAttrName(odsState.name), device_name);
  if (scope_symbol_id) {
    odsState.addAttribute(scope_symbol_idAttrName(odsState.name), scope_symbol_id);
  }
  if (hierarchy) { odsState.addAttribute(hierarchyAttrName(odsState.name), hierarchy); }
  // TODO: remove the workaround if normalization_add_relu supports infererence mode
  odsState.addAttribute(result_segment_sizesAttrName(odsState.name),
                        odsBuilder.getI32VectorAttr({1, 1, 1, 1}));
  odsState.addAttribute(axisAttrName(odsState.name), axis);
  odsState.addAttribute(epsilonAttrName(odsState.name), epsilon);
  odsState.addAttribute(trainingAttrName(odsState.name), training);
  odsState.addAttribute(momentumAttrName(odsState.name), momentum);
  auto y = x.getType();
  odsState.addTypes(y);
  // TODO: add real type infer, or get types from user of x and moving_mean, if it is a bn
  /*reserve_space */ odsState.addTypes(x.getType());
  /*mean */ odsState.addTypes(x.getType());
  /*inv_variance */ odsState.addTypes(x.getType());
}

std::string Add2Op::getOriginalOpTypeName() { return "add_n"; }

void Job::build(OpBuilder& builder, OperationState& state, StringRef name, FunctionType type) {
  state.addAttribute(SymbolTable::getSymbolAttrName(), builder.getStringAttr(name));
  state.addAttribute(getTypeAttrName(), TypeAttr::get(type));
  state.addRegion();
}

static ParseResult parseJob(OpAsmParser& parser, OperationState& result) {
  auto buildFuncType = [](Builder& builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
                          function_interface_impl::VariadicFlag,
                          std::string&) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(parser, result, /*allowVariadic=*/false,
                                                  buildFuncType);
}

static void print(Job op, OpAsmPrinter& p) {
  FunctionType fnType = op.getType();
  function_interface_impl::printFunctionOp(p, op, fnType.getInputs(), /*isVariadic=*/false,
                                           fnType.getResults());
}

static LogicalResult verify(Job op) {
  // If this function is external there is nothing to do.
  if (op.isExternal()) return success();

  // Verify that the argument list of the function and the arg list of the entry
  // block line up.  The trait already verified that the number of arguments is
  // the same between the signature and the block.
  auto fnInputTypes = op.getType().getInputs();
  Block& entryBlock = op.front();
  for (unsigned i = 0, e = entryBlock.getNumArguments(); i != e; ++i)
    if (fnInputTypes[i] != entryBlock.getArgument(i).getType())
      return op.emitOpError("type of entry block argument #")
             << i << '(' << entryBlock.getArgument(i).getType()
             << ") must match the type of the corresponding argument in "
             << "function signature(" << fnInputTypes[i] << ')';

  return success();
}

static LogicalResult verify(mlir::oneflow::ReturnOp op) {
  auto job = cast<Job>(op->getParentOp());

  // The operand number and types must match the function signature.
  const auto& results = job.getType().getResults();
  if (op.getNumOperands() != results.size())
    return op.emitOpError("has ") << op.getNumOperands() << " operands, but enclosing function (@"
                                  << job.getName() << ") returns " << results.size();

  for (unsigned i = 0, e = results.size(); i != e; ++i)
    if (op.getOperand(i).getType() != results[i])
      return op.emitError() << "type of return operand " << i << " (" << op.getOperand(i).getType()
                            << ") doesn't match function result type (" << results[i] << ")"
                            << " in function @" << job.getName();

  return success();
}

}  // namespace oneflow

}  // namespace mlir

#include "OneFlow/OneFlowEnums.cpp.inc"

#define GET_OP_CLASSES
#include "OneFlow/OneFlowOps.cpp.inc"
#include "OneFlow/OneFlowInterfaces.cpp.inc"

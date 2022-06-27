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
#include "oneflow/api/common/ofblob.h"
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

void RandomMaskLikeOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState,
                             mlir::Value like, StringRef op_name, StringRef device_tag,
                             ArrayAttr device_name, IntegerAttr scope_symbol_id,
                             ArrayAttr hierarchy, mlir::FloatAttr rate, mlir::IntegerAttr seed) {
  odsState.addOperands(like);
  odsState.addAttribute(op_nameAttrName(odsState.name), odsBuilder.getStringAttr(op_name));
  odsState.addAttribute(device_tagAttrName(odsState.name), odsBuilder.getStringAttr(device_tag));
  odsState.addAttribute(device_nameAttrName(odsState.name), device_name);
  if (scope_symbol_id) {
    odsState.addAttribute(scope_symbol_idAttrName(odsState.name), scope_symbol_id);
  }
  if (hierarchy) { odsState.addAttribute(hierarchyAttrName(odsState.name), hierarchy); }
  odsState.addAttribute(rateAttrName(odsState.name), rate);
  odsState.addAttribute(seedAttrName(odsState.name), seed);
  odsState.addTypes(like.getType());
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

bool Conv2DOp::IsNCHW() { return this->data_format().str() == "channels_first"; }

llvm::DenseSet<Value> Conv2DOp::OperandsToTranspose() { return {this->in(), this->weight()}; }

llvm::DenseSet<Value> Conv2DOp::ResultsToTranspose() { return {this->out()}; }

llvm::SmallVector<Value, 4> Conv2DOp::NchwToNhwc(llvm::SmallVector<Value, 4> value,
                                                 PatternRewriter& rewriter) {
  auto conv_op = *this;
  SmallVector<Value, 4> operands;
  operands.push_back(value[0]);
  operands.push_back(value[1]);
  if (conv_op.bias()) operands.push_back(conv_op.bias());
  if (conv_op.bias_multiplier()) operands.push_back(conv_op.bias_multiplier());
  NamedAttrList attributes = conv_op->getAttrs();
  attributes.set(conv_op.data_formatAttrName(), rewriter.getStringAttr("channels_last"));
  auto res = rewriter
                 .create<oneflow::Conv2DOp>(conv_op.getLoc(), conv_op->getResultTypes(), operands,
                                            attributes)
                 ->getResults();
  llvm::SmallVector<Value, 4> results;
  results.push_back(res[0]);
  return results;
}

bool BiasAddOp::IsNCHW() { return this->axisAttr().getValue().getSExtValue() == 1; }

llvm::DenseSet<Value> BiasAddOp::OperandsToTranspose() { return {this->a()}; }

llvm::DenseSet<Value> BiasAddOp::ResultsToTranspose() { return {this->out()}; }

llvm::SmallVector<Value, 4> BiasAddOp::NchwToNhwc(llvm::SmallVector<Value, 4> value,
                                                  PatternRewriter& rewriter) {
  auto bias_add_op = *this;
  SmallVector<Value, 4> operands;
  operands.push_back(value[0]);
  operands.push_back(bias_add_op.b());
  NamedAttrList attributes = bias_add_op->getAttrs();
  attributes.set(bias_add_op.axisAttrName(), rewriter.getSI32IntegerAttr(3));
  auto res = rewriter
                 .create<oneflow::BiasAddOp>(bias_add_op.getLoc(), bias_add_op->getResultTypes(),
                                             operands, attributes)
                 ->getResults();
  llvm::SmallVector<Value, 4> results;
  results.push_back(res[0]);
  return results;
}

bool NormalizationOp::IsNCHW() { return this->axisAttr().getValue().getSExtValue() == 1; }

llvm::DenseSet<Value> NormalizationOp::OperandsToTranspose() { return {this->x()}; }

llvm::DenseSet<Value> NormalizationOp::ResultsToTranspose() { return {this->y()}; }

llvm::SmallVector<Value, 4> NormalizationOp::NchwToNhwc(llvm::SmallVector<Value, 4> value,
                                                        PatternRewriter& rewriter) {
  auto normalization_op = *this;
  SmallVector<Value, 4> operands;
  operands.push_back(value[0]);
  if (normalization_op.moving_mean()) operands.push_back(normalization_op.moving_mean());
  if (normalization_op.moving_variance()) operands.push_back(normalization_op.moving_variance());
  operands.push_back(normalization_op.gamma());
  operands.push_back(normalization_op.beta());
  if (normalization_op._add_to_output()) operands.push_back(normalization_op._add_to_output());
  NamedAttrList attributes = normalization_op->getAttrs();
  attributes.set(normalization_op.axisAttrName(), rewriter.getSI32IntegerAttr(3));
  auto res =
      rewriter
          .create<oneflow::NormalizationOp>(
              normalization_op.getLoc(), normalization_op->getResultTypes(), operands, attributes)
          ->getResults();
  llvm::SmallVector<Value, 4> results;
  results.push_back(res[0]);
  return results;
}

bool MaxPool2DOp::IsNCHW() { return this->data_format().str() == "channels_first"; }

llvm::DenseSet<Value> MaxPool2DOp::OperandsToTranspose() { return {this->x()}; }

llvm::DenseSet<Value> MaxPool2DOp::ResultsToTranspose() { return {this->y(), this->indice()}; }

llvm::SmallVector<Value, 4> MaxPool2DOp::NchwToNhwc(llvm::SmallVector<Value, 4> value,
                                                    PatternRewriter& rewriter) {
  auto max_pool_2d_op = *this;
  SmallVector<Value, 4> operands;
  operands.push_back(value[0]);
  NamedAttrList attributes = max_pool_2d_op->getAttrs();
  attributes.set(max_pool_2d_op.data_formatAttrName(), rewriter.getStringAttr("channels_last"));
  auto res =
      rewriter
          .create<oneflow::MaxPool2DOp>(max_pool_2d_op.getLoc(), max_pool_2d_op->getResultTypes(),
                                        operands, attributes)
          ->getResults();
  llvm::SmallVector<Value, 4> results;
  results.push_back(res[0]);
  results.push_back(res[1]);
  return results;
}

bool ReluOp::IsNCHW() { return false; }

llvm::DenseSet<Value> ReluOp::OperandsToTranspose() { return {this->x()}; }

llvm::DenseSet<Value> ReluOp::ResultsToTranspose() { return {this->y()}; }

llvm::SmallVector<Value, 4> ReluOp::NchwToNhwc(llvm::SmallVector<Value, 4> value,
                                               PatternRewriter& rewriter) {
  auto relu_op = *this;
  SmallVector<Value, 4> operands{value[0]};
  auto res = rewriter
                 .create<oneflow::ReluOp>(relu_op.getLoc(), relu_op->getResultTypes(), operands,
                                          relu_op->getAttrs())
                 ->getResults();
  return {res[0]};
}

bool Add2Op::IsNCHW() { return false; }

llvm::DenseSet<Value> Add2Op::OperandsToTranspose() { return {this->in0(), this->in1()}; }

llvm::DenseSet<Value> Add2Op::ResultsToTranspose() { return {this->out()}; }

llvm::SmallVector<Value, 4> Add2Op::NchwToNhwc(llvm::SmallVector<Value, 4> value,
                                               PatternRewriter& rewriter) {
  auto add2_op = *this;
  SmallVector<Value, 4> operands{value[0], value[1]};
  auto res = rewriter
                 .create<oneflow::Add2Op>(add2_op.getLoc(), add2_op->getResultTypes(), operands,
                                          add2_op->getAttrs())
                 ->getResults();
  return {res[0]};
}

}  // namespace oneflow

}  // namespace mlir

#include "OneFlow/OneFlowEnums.cpp.inc"

#define GET_OP_CLASSES
#include "OneFlow/OneFlowOps.cpp.inc"
#include "OneFlow/OneFlowInterfaces.cpp.inc"

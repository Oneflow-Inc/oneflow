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
#include <iostream>
#include <string>
#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "oneflow/ir/include/OneFlow/OneFlowSupport.h"

using namespace mlir;
using namespace mlir::oneflow;

::mlir::OperandRange UserOp::dataInputOperands() { return data_input(); }
::mlir::OperandRange UserOp::ctrlInputOperands() { return ctrl_inputs(); }
::mlir::ResultRange UserOp::dataOutputResults() { return data_output(); }
::mlir::Value UserOp::ctrlOutputResult() { return ctrl_output(); }
::mlir::OperandRange SystemOp::dataInputOperands() { return data_input(); }
::mlir::OperandRange SystemOp::ctrlInputOperands() { return ctrl_inputs(); }
::mlir::ResultRange SystemOp::dataOutputResults() { return data_output(); }
::mlir::Value SystemOp::ctrlOutputResult() { return ctrl_output(); }

static mlir::ParseResult parseConstantOp(mlir::OpAsmParser& parser, mlir::OperationState& result) {
  mlir::DenseElementsAttr value;
  if (parser.parseOptionalAttrDict(result.attributes)
      || parser.parseAttribute(value, "value", result.attributes)) {
    return failure();
  }
  result.addTypes(value.getType());
  return success();
}

static mlir::LogicalResult verify(oneflow::ConstantOp op) { return mlir::success(); }

template<typename OpType>
LogicalResult TrimRedundantCtrl(OpType& op, PatternRewriter& rewriter) {
  if (op.ctrl_output() && op.ctrl_output().use_empty()) {
    const int32_t num_data_outputs =
        *(op.result_segment_sizes().template getValues<uint32_t>()).begin();
    NamedAttrList attributes(op->getAttrDictionary());
    attributes.erase(mlir::OpTrait::AttrSizedResultSegments<void>::getResultSegmentSizeAttr());
    attributes.append(mlir::OpTrait::AttrSizedResultSegments<void>::getResultSegmentSizeAttr(),
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

bool IsCtrlOutTrimmed(oneflow::UserOp& op) { return !op.ctrl_output(); }

bool IsCtrlInAbsent(oneflow::UserOp& op) {
  if (!op->hasAttrOfType<::mlir::DenseIntElementsAttr>(
          mlir::OpTrait::AttrSizedOperandSegments<void>::getOperandSegmentSizeAttr()))
    op.dump();
  return op.ctrl_inputs().empty();
}

StringSet<>* GetPrintedOpTypeNames() {
  static llvm::StringSet<> names({});
  return &names;
}

const StringSet<>& GetUnaryOpTypeNames() {
  static llvm::StringSet<> names({"abs", "acos", "ceil", "cosh", "floor", "lgamma", "log_sigmoid",
                                  "reciprocal_no_nan", "rint", "round", "softplus"

  });
  return names;
}

const StringSet<>& GetScalarMathOpTypeNames() {
  static llvm::StringSet<> names(
      {"scalar_add", "scalar_floordiv", "scalar_fmod", "scalar_mul", "scalar_pow"

      });
  return names;
}

const StringSet<>& GetDataOpsTypeNames() {
  static llvm::StringSet<> names({"OFRecordReader", "ofrecord_raw_decoder"

  });
  return names;
}

const StringSet<>& GetLossOpsTypeNames() {
  static llvm::StringSet<> names(
      {"sparse_softmax_cross_entropy", "sparse_softmax_cross_entropy_grad"

      });
  return names;
}

const StringSet<>& GetReduceOpTypeNames() {
  static llvm::StringSet<> names({"reduce_min", "reduce_prod", "reduce_sum", "reduce_max"

  });
  return names;
}

const StringSet<>& GetConvOpTypeNames() {
  static llvm::StringSet<> names(
      {"conv1d", "conv2d", "conv3d", "conv_filter_grad", "conv_data_grad"});
  return names;
}

const StringSet<>& GetPoolOpTypeNames() {
  static llvm::StringSet<> names({"avgpool_1d", "avgpool_2d", "avgpool_3d", "tf_avg_pool_1d",
                                  "tf_avg_pool_2d", "tf_avg_pool_3d", "tf_max_pool_1d",
                                  "tf_max_pool_2d", "tf_max_pool_3d", "tf_max_pool_1d_grad",
                                  "max_pool_2d_grad", "max_pool_3d_grad", "tf_avg_pool_1d_grad",
                                  "tf_avg_pool_2d_grad", "tf_avg_pool_3d_grad", "avgpool_1d_grad",
                                  "avgpool_2d_grad", "avgpool_3d_grad"

  });
  return names;
}

const StringSet<>& GetFloatUnaryOpTypeNames() {
  static llvm::StringSet<> names({"acosh", "asin",     "asinh",      "atan",  "atanh",      "sin",
                                  "cos",   "erf",      "erfc",       "exp",   "expm1",      "log",
                                  "log1p", "negative", "reciprocal", "rsqrt", "sigmoid_v2", "sign",
                                  "sinh",  "sqrt",     "square",     "tan",   "tanh"});
  return names;
}

template<typename T>
static void getValuesFromIntArrayAttribute(ArrayAttr attr, SmallVector<T>& arrayValues) {
  for (Attribute val : attr.getValue()) {
    arrayValues.push_back(val.cast<IntegerAttr>().getValue().getSExtValue());
  }
}

struct ConcreteUserOps : public mlir::OpRewritePattern<oneflow::UserOp> {
  explicit ConcreteUserOps(mlir::MLIRContext* context)
      : OpRewritePattern<oneflow::UserOp>(context, /*benefit=*/1) {}
  mlir::LogicalResult matchAndRewrite(oneflow::UserOp op,
                                      mlir::PatternRewriter& rewriter) const override {
    if (succeeded(TrimRedundantCtrl(op, rewriter))) { return success(); }
    // In principle, a concrete user op has no ctrl input/output. Some benefits:
    // 1. simplify things
    // 2. make conversion and code gen more doable
    // 3. enable the reuse of established MLIR infra like built-in traits
    if (IsCtrlOutTrimmed(op) && IsCtrlInAbsent(op)) {
      NamedAttrList attributes(op->getAttrDictionary());
      attributes.erase(op.input_sizesAttrName());
      attributes.erase(op.output_sizesAttrName());
      attributes.erase(mlir::OpTrait::AttrSizedOperandSegments<void>::getOperandSegmentSizeAttr());
      attributes.erase(mlir::OpTrait::AttrSizedResultSegments<void>::getResultSegmentSizeAttr());
      llvm::SmallVector<int32_t> input_sizes, output_sizes;
      getValuesFromIntArrayAttribute(op.input_sizes(), input_sizes);
      getValuesFromIntArrayAttribute(op.output_sizes(), output_sizes);
      if (!input_sizes.empty()) {
        attributes.push_back(rewriter.getNamedAttr(
            mlir::OpTrait::AttrSizedOperandSegments<void>::getOperandSegmentSizeAttr(),
            rewriter.getI32VectorAttr(input_sizes)));
      }
      if (!output_sizes.empty()) {
        attributes.push_back(rewriter.getNamedAttr(
            mlir::OpTrait::AttrSizedResultSegments<void>::getResultSegmentSizeAttr(),
            rewriter.getI32VectorAttr(output_sizes)));
      }
      OperationState state(op->getLoc(), "oneflow." + op.op_type_name().str());
      state.addAttributes(attributes);
      state.addOperands(op.getODSOperands(0) /* data in */);
      state.addTypes(op.getODSResults(0 /* data out */).getTypes());
      if (auto created = rewriter.createOperation(state)) {
        if (created->hasTrait<mlir::OpTrait::AttrSizedOperandSegments>() == false) {
          created->removeAttr(
              mlir::OpTrait::AttrSizedOperandSegments<void>::getOperandSegmentSizeAttr());
        }
        if (created->hasTrait<mlir::OpTrait::AttrSizedResultSegments>() == false) {
          created->removeAttr(
              mlir::OpTrait::AttrSizedResultSegments<void>::getResultSegmentSizeAttr());
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

void UserOp::getCanonicalizationPatterns(::mlir::RewritePatternSet& results,
                                         ::mlir::MLIRContext* context) {
  results.insert<ConcreteUserOps>(context);
}

struct ConcreteSystemOps : public mlir::OpRewritePattern<oneflow::SystemOp> {
  explicit ConcreteSystemOps(mlir::MLIRContext* context)
      : OpRewritePattern<oneflow::SystemOp>(context, /*benefit=*/1) {}
  mlir::LogicalResult matchAndRewrite(oneflow::SystemOp op,
                                      mlir::PatternRewriter& rewriter) const override {
    return TrimRedundantCtrl(op, rewriter);
  }
};

void SystemOp::getCanonicalizationPatterns(::mlir::RewritePatternSet& results,
                                           ::mlir::MLIRContext* context) {
  results.insert<ConcreteSystemOps>(context);
}

struct ConvertAddOpWithArity : public mlir::OpRewritePattern<oneflow::AddNOp> {
  explicit ConvertAddOpWithArity(mlir::MLIRContext* context)
      : OpRewritePattern<oneflow::AddNOp>(context, /*benefit=*/1) {}
  mlir::LogicalResult matchAndRewrite(oneflow::AddNOp op,
                                      mlir::PatternRewriter& rewriter) const override {
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

void AddNOp::getCanonicalizationPatterns(::mlir::RewritePatternSet& results,
                                         ::mlir::MLIRContext* context) {
  results.insert<ConvertAddOpWithArity>(context);
}

// TODO: merge all ctrl input and output when folding op
bool HaveIdenticalPlacement(mlir::Operation* a, mlir::Operation* b) {
  UserOpAdaptor adaptor_a(a->getOperands(), a->getAttrDictionary());
  UserOpAdaptor adaptor_b(b->getOperands(), b->getAttrDictionary());
  return adaptor_a.device_tag() == adaptor_b.device_tag()
         && adaptor_a.device_name() == adaptor_b.device_name();
}

using namespace OpTrait;
OpFoldResult OpTrait::impl::foldIdempotentOfIdenticalPlacement(Operation* op) {
  auto* argument_op = op->getOperand(0).getDefiningOp();
  if (argument_op && op->getName() == argument_op->getName()
      && HaveIdenticalPlacement(op, argument_op)) {
    return op->getOperand(0);
  }
  return {};
}

OpFoldResult OpTrait::impl::foldInvolutionOfIdenticalPlacement(Operation* op) {
  auto* argument_op = op->getOperand(0).getDefiningOp();
  if (argument_op && op->getName() == argument_op->getName()
      && HaveIdenticalPlacement(op, argument_op)) {
    return argument_op->getOperand(0);
  }
  return {};
}

LogicalResult OpTrait::impl::VerifyIsOpConfCompatible(Operation* op) {
  for (auto attr : {
           IsOpConfCompatible<void>::getOpNameAttr(),
           IsOpConfCompatible<void>::getDeviceTagAttr(),
       }) {
    if (!op->hasAttrOfType<StringAttr>(attr)) {
      return op->emitError("expected operation to have attribute: " + attr);
    }
  }
  if (!op->hasAttrOfType<ArrayAttr>(IsOpConfCompatible<void>::getDeviceNameAttr())) {
    return op->emitError("expected operation to have attribute: "
                         + IsOpConfCompatible<void>::getDeviceNameAttr());
  }
  return success();
}

LogicalResult OpTrait::impl::VerifyIsImportCompatible(Operation* op) {
  if (auto output_lbns =
          op->getAttrOfType<ArrayAttr>(IsImportCompatible<void>::getOutputLBNsAttr())) {
    if (auto cec = dyn_cast<ControlEdgeCompatible>(op)) {
      if (cec.dataOutputResults().size() != output_lbns.size()) {
        return op->emitError("expected number of data output results to be "
                             + std::to_string(output_lbns.size()) + " but got "
                             + std::to_string(cec.dataOutputResults().size()));
      }
    } else {
      return op->emitError("expected to support ControlEdgeCompatible");
    }
  } else {
    return op->emitError("expected operation to have attribute: "
                         + IsImportCompatible<void>::getOutputLBNsAttr());
  }
  return success();
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

#include "OneFlow/OneFlowEnums.cpp.inc"

#define GET_OP_CLASSES
#include "OneFlow/OneFlowOps.cpp.inc"
#include "OneFlow/OneFlowInterfaces.cpp.inc"

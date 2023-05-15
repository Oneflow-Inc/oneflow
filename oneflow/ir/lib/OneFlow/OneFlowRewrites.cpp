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
//===- TestPDLByteCode.cpp - Test PDLL functionality ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OneFlow/UserOpConversion.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "OneFlow/OneFlowPDLLPatterns.h"
#include "OneFlow/OneFlowOps.h"
#include "oneflow/core/framework/random_generator.h"
#include "OneFlow/OneFlowUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
using namespace mlir;

#include "oneflow/ir/lib/OneFlow/PDLL/ForwardOpPatterns.h.inc"

namespace mlir {

namespace oneflow {

namespace {

static std::atomic<int64_t> uniqID{0};

std::string getUniqName(llvm::StringRef name) {
  uniqID += 1;
  return name.str() + "-mlir-gen-" + std::to_string(uniqID);
}

static Operation* CopyUserOpAttrs(PatternRewriter& rewriter, Operation* src, Operation* dst) {
  dst->setAttr(OpTrait::IsOpConfCompatible<void>::getDeviceTagAttr(),
               OpTrait::IsOpConfCompatible<void>::getDeviceTag(src));
  dst->setAttr(OpTrait::IsOpConfCompatible<void>::getDeviceNameAttr(),
               OpTrait::IsOpConfCompatible<void>::getDeviceName(src));
  if (auto hierarchy = OpTrait::IsOpConfCompatible<void>::getHierarchy(src)) {
    dst->setAttr(OpTrait::IsOpConfCompatible<void>::getHierarchyAttr(), hierarchy);
  }
  if (auto scope_symbol_id = OpTrait::IsOpConfCompatible<void>::getScopeSymbolID(src)) {
    dst->setAttr(OpTrait::IsOpConfCompatible<void>::getScopeSymbolIDAttr(), scope_symbol_id);
  }
  dst->setAttr(
      OpTrait::IsOpConfCompatible<void>::getOpNameAttr(),
      rewriter.getStringAttr(getUniqName(OpTrait::IsOpConfCompatible<void>::getOpName(src).str())));
  return dst;
}

static Operation* BuildFusedBiasAddMaskScaleOpWithRate(PatternRewriter& rewriter, Value a, Value b,
                                                       Value mask, Attribute axis, Attribute rate,
                                                       Operation* dropout) {
  auto dropout_op = llvm::dyn_cast<DropoutOp>(dropout);
  assert(dropout_op);
  SmallVector<Value, 4> operands;
  operands.push_back(a);
  operands.push_back(b);
  operands.push_back(mask);
  NamedAttrList attributes;
  attributes.set("axis", axis);
  float scale = 1.0f;
  float rate_float = rate.cast<FloatAttr>().getValueAsDouble();
  if (rate_float < 1.0f) { scale = 1.0f / (1.0f - rate_float); }
  attributes.set("scale", rewriter.getF32FloatAttr(scale));
  return rewriter.create<FusedBiasAddMaskScaleOp>(
      dropout_op->getLoc(), dropout_op.getOut().getType(), operands, attributes);
}

static Operation* CreateConv2dAndErasePad(PatternRewriter& rewriter, Value x, Value weight,
                                          Attribute padding_before, Attribute data_format,
                                          Operation* conv) {
  auto conv_op = llvm::dyn_cast<Conv2DOp>(conv);
  assert(conv_op);
  SmallVector<Value, 4> operands;
  operands.push_back(x);
  operands.push_back(weight);
  NamedAttrList attributes = conv_op->getAttrs();
  llvm::SmallVector<int32_t> padding_before_array;

  attributes.set(OpTrait::IsOpConfCompatible<void>::getOpNameAttr(),
                 rewriter.getStringAttr(OpTrait::IsOpConfCompatible<void>::getOpName(conv).str()
                                        + "-fuse-conv"));

  if (data_format.cast<StringAttr>().str() == "channels_first") {
    for (auto val : padding_before.cast<ArrayAttr>().getValue().take_back(2)) {
      padding_before_array.push_back(val.cast<IntegerAttr>().getValue().getSExtValue());
    }
  } else {
    padding_before_array.push_back(padding_before.cast<ArrayAttr>()
                                       .getValue()[1]
                                       .cast<IntegerAttr>()
                                       .getValue()
                                       .getSExtValue());
    padding_before_array.push_back(padding_before.cast<ArrayAttr>()
                                       .getValue()[2]
                                       .cast<IntegerAttr>()
                                       .getValue()
                                       .getSExtValue());
  }

  attributes.set(conv_op.getPaddingBeforeAttrName(),
                 getSI32ArrayAttr(rewriter, padding_before_array));
  return rewriter.create<Conv2DOp>(conv_op->getLoc(), conv_op.getOut().getType(), operands,
                                   attributes);
}

IntegerAttr getSI64IntegerAttr(::mlir::PatternRewriter& rewriter, int64_t value) {
  return IntegerAttr::get(rewriter.getIntegerType(64, /*isSigned=*/true),
                          APInt(64, value, /*isSigned=*/true));
}

static Attribute GetHeadSizeFromTranpose(PatternRewriter& rewriter, Operation* transpose) {
  auto transpose_op = llvm::dyn_cast<TransposeOp>(transpose);
  CHECK(transpose_op);
  return getSI64IntegerAttr(rewriter,
                            transpose_op.getOutput().getType().cast<ShapedType>().getDimSize(3));
}
NamedAttrList GetUserOpCommonAttrs(MLIRContext* ctx, const std::string& op_name) {
  NamedAttrList attrs;
  attrs.set(OpTrait::IsOpConfCompatible<void>::getOpNameAttr(), StringAttr::get(ctx, op_name));
  attrs.set(OpTrait::IsOpConfCompatible<void>::getDeviceTagAttr(), StringAttr::get(ctx, "cpu"));
  attrs.set(OpTrait::IsOpConfCompatible<void>::getDeviceNameAttr(),
            ArrayAttr::get(ctx, llvm::to_vector<8>(llvm::map_range(ArrayRef<StringRef>({"@0:0"}),
                                                                   [&](StringRef v) -> Attribute {
                                                                     return StringAttr::get(ctx, v);
                                                                   }))));
  return attrs;
}
static Operation* CreateConv2DBatchNorm(PatternRewriter& rewriter, Attribute epsilon,
                                        Operation* conv, Operation* bn) {
  auto conv_op = llvm::dyn_cast<oneflow::Conv2DOp>(conv);
  auto bn_op = llvm::dyn_cast<oneflow::NormalizationInferenceOp>(bn);
  auto ctx = rewriter.getContext();
  NamedAttrList attributes = conv_op->getAttrs();

  attributes.set(OpTrait::AttrSizedOperandSegments<void>::getOperandSegmentSizeAttr(),
                 rewriter.getDenseI32ArrayAttr({1, 1, 1, 0}));

  SmallVector<Value, 4> operands;
  operands.push_back(conv_op.getIn());

  // deal with weight
  auto add_op_attrs = GetUserOpCommonAttrs(ctx, "scalar_add");
  add_op_attrs.set("has_float_operand", BoolAttr::get(ctx, true));

  double epsilon_attr = epsilon.cast<FloatAttr>().getValueAsDouble();
  add_op_attrs.set("float_operand", rewriter.getF64FloatAttr(epsilon_attr));

  auto add_op = rewriter.create<oneflow::ScalarAddOp>(
      conv_op->getLoc(), conv_op.getOut().getType(),
      SmallVector<Value, 4>({bn_op.getMovingVariance()}), add_op_attrs);

  auto sqrt_op = rewriter.create<oneflow::SqrtOp>(conv_op->getLoc(), conv_op.getOut().getType(),
                                                  SmallVector<Value, 4>({add_op.getOut()}),
                                                  GetUserOpCommonAttrs(ctx, "sqrt"));

  auto div_op = rewriter.create<oneflow::BroadcastDivOp>(
      conv_op->getLoc(), conv_op.getOut().getType(),
      SmallVector<Value, 4>({bn_op.getGamma(), sqrt_op.getY()}), GetUserOpCommonAttrs(ctx, "div"));

  auto bn_gamma_variable_op =
      llvm::dyn_cast<oneflow::FrozenVariableOp>(bn_op.getGamma().getDefiningOp());

  CHECK(bn_gamma_variable_op) << "Gamma of batchnorm should be a FrozenVariableOp.";

  auto bn_gamma_shape =
      bn_gamma_variable_op.getValue().getType().cast<mlir::RankedTensorType>().getShape();

  auto conv_weight_variable_op =
      llvm::dyn_cast<oneflow::FrozenVariableOp>(conv_op.getWeight().getDefiningOp());

  CHECK(conv_weight_variable_op) << "Weight of conv2d should be a FrozenVariableOp.";

  auto conv_weight_shape =
      conv_weight_variable_op.getValue().getType().cast<mlir::RankedTensorType>().getShape();

  std::vector<int64_t> bn_gamma_new_shape({bn_gamma_shape.front()});
  for (int i = 1; i < conv_weight_shape.size(); ++i) { bn_gamma_new_shape.emplace_back(1); }
  auto reshape_op_attrs = GetUserOpCommonAttrs(ctx, "reshape");
  reshape_op_attrs.set(
      "shape",
      ArrayAttr::get(ctx, llvm::to_vector<8>(llvm::map_range(
                              ArrayRef<int64_t>(bn_gamma_new_shape), [&](int64_t v) -> Attribute {
                                return getSI64IntegerAttr(rewriter, v);
                              }))));
  auto reshape_op =
      rewriter.create<oneflow::ReshapeOp>(conv_op->getLoc(), conv_op.getOut().getType(),
                                          SmallVector<Value, 4>({div_op.getZ()}), reshape_op_attrs);

  auto mul_op = rewriter.create<oneflow::BroadcastMulOp>(
      conv_op->getLoc(), conv_op.getOut().getType(),
      SmallVector<Value, 4>({conv_op.getWeight(), reshape_op.getOut()}),
      GetUserOpCommonAttrs(ctx, "multiply"));
  operands.push_back(mul_op.getZ());

  // deal with bias
  CHECK(!conv_op.getBias())
      << "Fusing conv2d and batch_norm only supports conv2d without bias now.";

  auto mul_op_bias = rewriter.create<oneflow::BroadcastMulOp>(
      conv_op->getLoc(), conv_op.getOut().getType(),
      SmallVector<Value, 4>({bn_op.getMovingMean(), div_op.getZ()}),
      GetUserOpCommonAttrs(ctx, "multiply_bias"));
  auto sub_op_bias = rewriter.create<oneflow::BroadcastSubOp>(
      conv_op->getLoc(), conv_op.getOut().getType(),
      SmallVector<Value, 4>({bn_op.getBeta(), mul_op_bias.getZ()}),
      GetUserOpCommonAttrs(ctx, "sub_bias"));
  operands.push_back(sub_op_bias.getZ());

  auto new_conv_op = rewriter.create<oneflow::Conv2DOp>(
      conv_op->getLoc(), conv_op.getOut().getType(), operands, attributes);

  return new_conv_op;
}

static LogicalResult IsPaddingCouldBeAssimilatedIntoConv(PatternRewriter& rewriter,
                                                         Attribute padding_before,
                                                         Attribute padding_after,
                                                         Attribute data_format) {
  if (padding_before.cast<ArrayAttr>().size() == 4 && padding_after.cast<ArrayAttr>().size() == 4) {
    if (padding_before.cast<ArrayAttr>().getValue().equals(
            padding_after.cast<ArrayAttr>().getValue())) {
      if (data_format.cast<StringAttr>().str() == "channels_first") {
        return success(padding_before.cast<ArrayAttr>()
                               .getValue()[0]
                               .cast<IntegerAttr>()
                               .getValue()
                               .getSExtValue()
                           == 0
                       && padding_before.cast<ArrayAttr>()
                                  .getValue()[1]
                                  .cast<IntegerAttr>()
                                  .getValue()
                                  .getSExtValue()
                              == 0);
      }
      if (data_format.cast<StringAttr>().str() == "channels_last") {
        return success(padding_before.cast<ArrayAttr>()
                               .getValue()[0]
                               .cast<IntegerAttr>()
                               .getValue()
                               .getSExtValue()
                           == 0
                       && padding_before.cast<ArrayAttr>()
                                  .getValue()[3]
                                  .cast<IntegerAttr>()
                                  .getValue()
                                  .getSExtValue()
                              == 0);
      }
    }
  }
  return failure();
}
static LogicalResult IsNotNestedInJit(PatternRewriter& rewriter, Operation* mul) {
  return success(mul->getParentOfType<oneflow::Job>());
}

static LogicalResult IsScalarTensor(PatternRewriter& rewriter, Value value) {
  if (auto tensor = value.getType().dyn_cast<RankedTensorType>()) {
    return success(tensor.getNumElements() == 1);
  }
  return failure();
}

static float mha_scale_max_diff = 1e-5;

static LogicalResult IsScalarEqualSqrtDim(PatternRewriter& rewriter, Value query_reshape,
                                          Attribute scalar_div_operand) {
  auto query_reshape_shape = query_reshape.getType().dyn_cast<ShapedType>();
  double scalar_div_operand_attr = scalar_div_operand.cast<FloatAttr>().getValueAsDouble();
  return success(
      std::abs(std::sqrt(query_reshape_shape.getShape().back()) - scalar_div_operand_attr)
      < mha_scale_max_diff);
}

static LogicalResult IsScalarEqualSqrtDimReciprocal(PatternRewriter& rewriter, Value query_reshape,
                                                    Attribute scalar_div_operand) {
  auto query_reshape_shape = query_reshape.getType().dyn_cast<ShapedType>();
  double scalar_div_operand_attr = scalar_div_operand.cast<FloatAttr>().getValueAsDouble();
  return success(
      std::abs(std::sqrt(query_reshape_shape.getShape().back()) - (1 / scalar_div_operand_attr))
      < mha_scale_max_diff);
}

static Attribute GetReciprocal(PatternRewriter& rewriter, Attribute a) {
  return rewriter.getF64FloatAttr(1 / a.cast<FloatAttr>().getValueAsDouble());
}

}  // namespace

namespace rewrites {

void populateRewrites(RewritePatternSet& patterns) {
  patterns.getPDLPatterns().registerRewriteFunction("BuildFusedBiasAddMaskScaleOpWithRate",
                                                    BuildFusedBiasAddMaskScaleOpWithRate);
  patterns.getPDLPatterns().registerRewriteFunction("CopyUserOpAttrs", CopyUserOpAttrs);
  patterns.getPDLPatterns().registerRewriteFunction("GetHeadSizeFromTranpose",
                                                    GetHeadSizeFromTranpose);
  patterns.getPDLPatterns().registerRewriteFunction("CreateConv2dAndErasePad",
                                                    CreateConv2dAndErasePad);
  patterns.getPDLPatterns().registerRewriteFunction("CreateConv2DBatchNorm", CreateConv2DBatchNorm);
  patterns.getPDLPatterns().registerRewriteFunction("GetReciprocal", GetReciprocal);
}

mlir::IntegerAttr GetDefaultSeed(::mlir::PatternRewriter& rewriter) {
  const auto gen = CHECK_JUST(::oneflow::one::DefaultAutoGenerator());
  return getSI64IntegerAttr(rewriter, (int64_t)gen->current_seed());
}

}  // namespace rewrites

namespace constraints {

void populateConstraints(RewritePatternSet& patterns) {
  auto& pdll_patterns = patterns.getPDLPatterns();

#define PDLL_REGISTER(NAME) pdll_patterns.registerConstraintFunction(#NAME, NAME);

  PDLL_REGISTER(IsPaddingCouldBeAssimilatedIntoConv);
  PDLL_REGISTER(IsNotNestedInJit);
  PDLL_REGISTER(IsScalarTensor);
  PDLL_REGISTER(IsScalarEqualSqrtDim);
  PDLL_REGISTER(IsScalarEqualSqrtDimReciprocal);

#undef PDLL_REGISTER
}

}  // namespace constraints
}  // namespace oneflow

}  // namespace mlir

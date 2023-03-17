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
#include "mlir/IR/BlockAndValueMapping.h"
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
  NamedAttrList attributes = dropout_op->getAttrs();
  attributes.set("axis", axis);
  attributes.set(OpTrait::IsOpConfCompatible<void>::getOpNameAttr(),
                 rewriter.getStringAttr(OpTrait::IsOpConfCompatible<void>::getOpName(dropout).str()
                                        + "-fuse-bias-add"));
  float scale = 1.0f;
  float rate_float = rate.cast<FloatAttr>().getValueAsDouble();
  if (rate_float < 1.0f) { scale = 1.0f / (1.0f - rate_float); }
  attributes.set("scale", rewriter.getF32FloatAttr(scale));
  attributes.erase(dropout_op.rateAttrName());
  return rewriter.create<FusedBiasAddMaskScaleOp>(dropout_op->getLoc(), dropout_op.out().getType(),
                                                  operands, attributes);
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

  attributes.set(conv_op.padding_beforeAttrName(),
                 getSI32ArrayAttr(rewriter, padding_before_array));
  return rewriter.create<Conv2DOp>(conv_op->getLoc(), conv_op.out().getType(), operands,
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
                            transpose_op.output().getType().cast<ShapedType>().getDimSize(3));
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

  attributes.set("operand_segment_sizes", rewriter.getI32VectorAttr({1, 1, 1, 0}));

  SmallVector<Value, 4> operands;
  operands.push_back(conv_op.in());

  // deal with weight
  auto add_op_attrs = GetUserOpCommonAttrs(ctx, "scalar_add");
  add_op_attrs.set("has_float_operand", BoolAttr::get(ctx, true));

  double epsilon_attr = epsilon.cast<FloatAttr>().getValueAsDouble();
  add_op_attrs.set("float_operand", rewriter.getF64FloatAttr(epsilon_attr));

  auto add_op = rewriter.create<oneflow::ScalarAddOp>(
      conv_op->getLoc(), conv_op.out().getType(), SmallVector<Value, 4>({bn_op.moving_variance()}),
      add_op_attrs);

  auto sqrt_op = rewriter.create<oneflow::SqrtOp>(conv_op->getLoc(), conv_op.out().getType(),
                                                  SmallVector<Value, 4>({add_op.out()}),
                                                  GetUserOpCommonAttrs(ctx, "sqrt"));

  auto div_op = rewriter.create<oneflow::BroadcastDivOp>(
      conv_op->getLoc(), conv_op.out().getType(),
      SmallVector<Value, 4>({bn_op.gamma(), sqrt_op.y()}), GetUserOpCommonAttrs(ctx, "div"));

  auto bn_gamma_variable_op =
      llvm::dyn_cast<oneflow::FrozenVariableOp>(bn_op.gamma().getDefiningOp());

  CHECK(bn_gamma_variable_op) << "Gamma of batchnorm should be a FrozenVariableOp.";

  auto bn_gamma_shape =
      bn_gamma_variable_op.value().getType().cast<mlir::RankedTensorType>().getShape();

  auto conv_weight_variable_op =
      llvm::dyn_cast<oneflow::FrozenVariableOp>(conv_op.weight().getDefiningOp());

  CHECK(conv_weight_variable_op) << "Weight of conv2d should be a FrozenVariableOp.";

  auto conv_weight_shape =
      conv_weight_variable_op.value().getType().cast<mlir::RankedTensorType>().getShape();

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
      rewriter.create<oneflow::ReshapeOp>(conv_op->getLoc(), conv_op.out().getType(),
                                          SmallVector<Value, 4>({div_op.z()}), reshape_op_attrs);

  auto mul_op = rewriter.create<oneflow::BroadcastMulOp>(
      conv_op->getLoc(), conv_op.out().getType(),
      SmallVector<Value, 4>({conv_op.weight(), reshape_op.out()}),
      GetUserOpCommonAttrs(ctx, "multiply"));
  operands.push_back(mul_op.z());

  // deal with bias
  CHECK(!conv_op.bias()) << "Fusing conv2d and batch_norm only supports conv2d without bias now.";

  auto mul_op_bias = rewriter.create<oneflow::BroadcastMulOp>(
      conv_op->getLoc(), conv_op.out().getType(),
      SmallVector<Value, 4>({bn_op.moving_mean(), div_op.z()}),
      GetUserOpCommonAttrs(ctx, "multiply_bias"));
  auto sub_op_bias = rewriter.create<oneflow::BroadcastSubOp>(
      conv_op->getLoc(), conv_op.out().getType(),
      SmallVector<Value, 4>({bn_op.beta(), mul_op_bias.z()}),
      GetUserOpCommonAttrs(ctx, "sub_bias"));
  operands.push_back(sub_op_bias.z());

  auto new_conv_op = rewriter.create<oneflow::Conv2DOp>(conv_op->getLoc(), conv_op.out().getType(),
                                                        operands, attributes);

  return new_conv_op;
}

// TODO: cfg/multi block support
func::FuncOp GetOrInsertFuncOp(::mlir::PatternRewriter& rewriter, mlir::Location loc,
                               StringRef func_name, ValueRange operands, ValueRange results,
                               SmallVector<Operation*, 4> ops) {
  BlockAndValueMapping mapping;
  SmallVector<Type, 4> argument_types;
  argument_types.reserve(operands.size());
  SmallVector<Type, 4> result_types;
  result_types.reserve(results.size());
  for (auto argument : operands) { argument_types.push_back(argument.getType()); }
  for (auto result : results) { result_types.push_back(result.getType()); }
  auto func_type = rewriter.getFunctionType(argument_types, result_types);
  auto first_op = *ops.begin();
  auto parent_func_op = first_op->getParentOfType<oneflow::Job>();
  if (!parent_func_op) {
    emitError(loc) << "null parent oneflow::Job " << *first_op;
    return nullptr;
  }
  auto parent_module_op = parent_func_op->getParentOfType<ModuleOp>();
  if (!parent_module_op) {
    emitError(loc) << "null ModuleOp " << *first_op;
    return nullptr;
  }
  SymbolTable symbol_table(parent_module_op);
  OpBuilder::InsertionGuard guard(rewriter);
  Block::iterator insertPt(parent_func_op->getNextNode());
  rewriter.setInsertionPointToStart(parent_module_op.getBody());
  if (parent_func_op->hasAttr("llvm.emit_c_interface")) {
    emitError(loc) << "parent should not has attr of llvm.emit_c_interface " << *parent_func_op;
    return nullptr;
  }
  auto function = rewriter.create<func::FuncOp>(loc, func_name, func_type);
  function->setAttr("llvm.emit_c_interface", mlir::UnitAttr::get(rewriter.getContext()));
  function.getBody().emplaceBlock();
  for (auto& arg : argument_types) { function.getBody().addArguments(arg, loc); }
  for (auto argument_pair : llvm::zip(operands, function.getBody().getArguments())) {
    mapping.map(std::get<0>(argument_pair), std::get<1>(argument_pair));
  }
  rewriter.setInsertionPointToStart(&function.getBody().front());
  ImplicitLocOpBuilder nb(loc, rewriter);
  for (auto op : ops) { nb.clone(*op, mapping); }
  SmallVector<::mlir::Value, 4> mapped_results;
  for (auto result : results) { mapped_results.push_back(mapping.lookup(result)); }
  rewriter.create<func::ReturnOp>(loc, mapped_results);
  if (symbol_table.lookup(func_name)) {
    emitError(loc) << func_name << " should not be at symbol table of ModuleOp";
    return nullptr;
  }
  return function;
}

NamedAttrList GetJitOpAttributes(PatternRewriter& rewriter, StringRef op_name, int32_t input_size,
                                 int32_t output_size, Operation* op) {
  NamedAttrList attributes;
  attributes.set(OpTrait::IsOpConfCompatible<void>::getDeviceTagAttr(),
                 OpTrait::IsOpConfCompatible<void>::getDeviceTag(op));
  attributes.set(OpTrait::IsOpConfCompatible<void>::getDeviceNameAttr(),
                 OpTrait::IsOpConfCompatible<void>::getDeviceName(op));
  if (auto hierarchy = OpTrait::IsOpConfCompatible<void>::getHierarchy(op)) {
    attributes.set(OpTrait::IsOpConfCompatible<void>::getHierarchyAttr(), hierarchy);
  }
  attributes.set(OpTrait::IsOpConfCompatible<void>::getOpNameAttr(),
                 rewriter.getStringAttr(op_name));
  if (auto scope_symbol_id = OpTrait::IsOpConfCompatible<void>::getScopeSymbolID(op)) {
    attributes.set(OpTrait::IsOpConfCompatible<void>::getScopeSymbolIDAttr(), scope_symbol_id);
  }
  return attributes;
}

template<typename T>
LogicalResult DumpAssembly(::mlir::PatternRewriter& rewriter, T op, StringRef func_name) {
  // TODO: now we only need one JIT engine
  auto parent_func_op = op->template getParentOfType<oneflow::Job>();
  if (!parent_func_op) { return failure(); }
  auto parent_module_op = parent_func_op->template getParentOfType<ModuleOp>();
  if (!parent_module_op) { return failure(); }
  SymbolTable symbol_table(parent_module_op);
  std::string mlir;
  llvm::raw_string_ostream os_mlir(mlir);
  if (auto found = symbol_table.lookup(func_name)) {
    found->print(os_mlir);
  } else {
    parent_module_op->dump();
    return op.emitError("symbol of jit function not found: " + op.op_name());
  }
  op->setAttr("mlir_assembly", rewriter.getStringAttr(mlir));
  return success();
}

static Operation* OutlineMulCast(PatternRewriter& rewriter, Operation* mul, Operation* cast) {
  auto mul_op = llvm::dyn_cast<ScalarMulByTensorOp>(mul);
  if (!mul_op->hasTrait<OpTrait::IsOpConfCompatible>()) {
    mul->emitError("not OpConf compatible");
    exit(1);
  }
  auto cast_op = llvm::dyn_cast<CastOp>(cast);
  // TODO: extract a function to generate op name for jit op from ops being fused
  SmallString<64> op_name_storage;
  auto op_name =
      (cast_op.op_name() + "__FUSE__"
       + mul_op->getAttrOfType<StringAttr>(OpTrait::IsOpConfCompatible<void>::getOpNameAttr())
             .getValue()
             .str())
          .toStringRef(op_name_storage);
  SmallString<16> tempBuffer;
  op_name = SanitizeIdentifier(op_name, tempBuffer);
  SmallVector<::mlir::Value, 2> operands;
  operands.push_back(cast_op.in());
  operands.push_back(mul_op.scalar());
  SmallVector<Value, 1> results;
  results.push_back(mul_op.y());
  NamedAttrList attributes =
      GetJitOpAttributes(rewriter, op_name, operands.size(), results.size(), mul_op);
  SmallVector<Operation*, 4> ops = {cast_op, mul_op};
  auto function = GetOrInsertFuncOp(rewriter, mul_op->getLoc(), op_name, operands, results, ops);
  auto created = rewriter.create<MlirJitOp>(mul_op->getLoc(), function, attributes, operands);
  if (failed(DumpAssembly(rewriter, created, created.op_name()))) { exit(1); }
  cast_op->dropAllUses();
  cast_op.erase();
  return created;
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
  patterns.getPDLPatterns().registerRewriteFunction("OutlineMulCast", OutlineMulCast);
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

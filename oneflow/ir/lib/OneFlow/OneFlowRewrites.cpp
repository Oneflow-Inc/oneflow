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

#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "OneFlow/OneFlowPDLLPatterns.h"
#include "OneFlow/OneFlowOps.h"
#include "oneflow/core/framework/random_generator.h"

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

static LogicalResult IsSingleDevice(PatternRewriter& rewriter, Attribute device_name,
                                    Attribute device_tag) {
  auto devices = device_name.dyn_cast_or_null<ArrayAttr>();
  if (!devices || devices.size() != 1) { return failure(); }
  auto device_str = devices[0].dyn_cast_or_null<StringAttr>();
  if (!device_str) { return failure(); }
  return success(device_str.str().find("-") == std::string::npos);
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

}  // namespace

namespace rewrites {

void populateRewrites(RewritePatternSet& patterns) {
  patterns.getPDLPatterns().registerRewriteFunction("BuildFusedBiasAddMaskScaleOpWithRate",
                                                    BuildFusedBiasAddMaskScaleOpWithRate);
  patterns.getPDLPatterns().registerRewriteFunction("CopyUserOpAttrs", CopyUserOpAttrs);
  patterns.getPDLPatterns().registerRewriteFunction("CreateConv2dAndErasePad",
                                                    CreateConv2dAndErasePad);
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
  PDLL_REGISTER(IsSingleDevice);

#undef PDLL_REGISTER
}

}  // namespace constraints
}  // namespace oneflow

}  // namespace mlir

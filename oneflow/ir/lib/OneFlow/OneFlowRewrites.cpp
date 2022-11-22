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
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
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

IntegerAttr getSI64IntegerAttr(::mlir::PatternRewriter& rewriter, int64_t value) {
  return IntegerAttr::get(rewriter.getIntegerType(64, /*isSigned=*/true),
                          APInt(64, value, /*isSigned=*/true));
}

}  // namespace

namespace rewrites {

void populateRewrites(RewritePatternSet& patterns) {
  patterns.getPDLPatterns().registerRewriteFunction("BuildFusedBiasAddMaskScaleOpWithRate",
                                                    BuildFusedBiasAddMaskScaleOpWithRate);
  patterns.getPDLPatterns().registerRewriteFunction("CopyUserOpAttrs", CopyUserOpAttrs);
}

mlir::IntegerAttr GetDefaultSeed(::mlir::PatternRewriter& rewriter) {
  const auto gen = CHECK_JUST(::oneflow::one::DefaultAutoGenerator());
  return getSI64IntegerAttr(rewriter, (int64_t)gen->current_seed());
}

}  // namespace rewrites

}  // namespace oneflow

}  // namespace mlir

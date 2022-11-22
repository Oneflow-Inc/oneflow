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

#include "oneflow/ir/lib/OneFlow/PDLL/OneFlowPatterns.h.inc"

namespace mlir {

namespace oneflow {

namespace {

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
}

mlir::IntegerAttr GetDefaultSeed(::mlir::PatternRewriter& rewriter) {
  const auto gen = CHECK_JUST(::oneflow::one::DefaultAutoGenerator());
  return getSI64IntegerAttr(rewriter, (int64_t)gen->current_seed());
}

}  // namespace rewrites

}  // namespace oneflow

}  // namespace mlir

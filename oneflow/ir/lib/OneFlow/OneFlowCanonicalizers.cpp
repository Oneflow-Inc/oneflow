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
#include "oneflow/core/framework/random_generator.h"
#include "OneFlow/OneFlowOps.h"
#include "OneFlow/OneFlowPatternUtils.h"

namespace mlir {

namespace oneflow {

namespace {

struct PutSeed : public OpRewritePattern<RandomMaskLikeOp> {
  explicit PutSeed(MLIRContext* context)
      : OpRewritePattern<RandomMaskLikeOp>(context, /*benefit=*/1) {}
  LogicalResult matchAndRewrite(oneflow::RandomMaskLikeOp op,
                                PatternRewriter& rewriter) const override {
    if (op->hasAttr(op.seedAttrName())) {
      return failure();
    } else {
      op->setAttr(op.seedAttrName(), rewrites::GetDefaultSeed(rewriter));
      return success();
    }
  }
};

}  // namespace

void RandomMaskLikeOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                                   MLIRContext* context) {
  results.insert<PutSeed>(context);
}

}  // namespace oneflow

}  // namespace mlir

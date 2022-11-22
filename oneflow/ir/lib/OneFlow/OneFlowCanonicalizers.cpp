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
      op->setAttr(op.seedAttrName(), GetDefaultSeed(rewriter));
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

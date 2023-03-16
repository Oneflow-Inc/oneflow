#include "OneFlow/OneFlowOps.h"
#include "OneFlow/Passes.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

namespace oneflow {

namespace {

struct SoftmaxOpLowering final : public OpConversionPattern<SoftmaxOp> {
 public:
  using OpConversionPattern<SoftmaxOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(SoftmaxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    return failure();
  }
};

struct OneFlowLoweringToLinalgPass
    : public LowerOneFlowToLinalgPassBase<OneFlowLoweringToLinalgPass> {
  void runOnOperation() {
    MLIRContext* context = &getContext();
    ConversionTarget target(*context);
    target
        .addLegalDialect<memref::MemRefDialect, mlir::func::FuncDialect, tosa::TosaDialect,
                         linalg::LinalgDialect, tensor::TensorDialect, arith::ArithmeticDialect>();
    RewritePatternSet patterns(context);
    patterns.add<SoftmaxOpLowering>(context);
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createLowerOneFlowToLinalgPass() {
  return std::make_unique<OneFlowLoweringToLinalgPass>();
}

}  // namespace oneflow
}  // namespace mlir

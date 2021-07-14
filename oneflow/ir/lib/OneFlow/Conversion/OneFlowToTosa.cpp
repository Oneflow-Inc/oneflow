#include "OneFlow/OneFlowOps.h"
#include <iostream>
#include <string>
#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::oneflow;

struct ScalarMulByTensorOpLowering final : public OpConversionPattern<ScalarMulByTensorOp> {
 public:
  using OpConversionPattern<ScalarMulByTensorOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ScalarMulByTensorOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& rewriter) const final {
    auto scalar = op.scalar();
    auto reshaped_scalar =
        rewriter
            .create<tosa::ReshapeOp>(
                op->getLoc(),
                RankedTensorType::get({1, 1}, scalar.getType().cast<TensorType>().getElementType()),
                scalar, rewriter.getI64ArrayAttr({1, 1}))
            .output();
    rewriter.replaceOpWithNewOp<tosa::MulOp>(
        op,
        /* output */ op->getResultTypes().front().cast<TensorType>(),
        /* input1 */ op.x(),
        /* input2 */ reshaped_scalar,
        /* shift */ rewriter.getIntegerAttr(rewriter.getI32Type(), 0));
    return success();
  }
};

struct CastOpLowering final : public OpConversionPattern<CastOp> {
 public:
  using OpConversionPattern<CastOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(CastOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& rewriter) const final {
    rewriter.replaceOpWithNewOp<tosa::CastOp>(op,
                                              /* output */ op.y().getType(),
                                              /* input */ op.x());
    return success();
  }
};

namespace {
struct OneFlowLoweringToTosaPass : public LowerOneFlowToTosaPassBase<OneFlowLoweringToTosaPass> {
  void runOnOperation() override;
};
}  // namespace

std::unique_ptr<Pass> mlir::oneflow::createLowerOneFlowToTosaPass() {
  return std::make_unique<OneFlowLoweringToTosaPass>();
}

void OneFlowLoweringToTosaPass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addLegalDialect<memref::MemRefDialect, StandardOpsDialect, tosa::TosaDialect>();
  target.addIllegalDialect<OneFlowDialect>();
  RewritePatternSet patterns(&getContext());
  // TODO: Add type converter
  patterns.insert<CastOpLowering, ScalarMulByTensorOpLowering>(&getContext());
  if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
    getOperation()->dump();
    signalPassFailure();
  }
}

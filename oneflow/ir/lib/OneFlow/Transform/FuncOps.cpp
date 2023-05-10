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
#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/OneFlowOps.h"
#include "OneFlow/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {

namespace func {
struct FuncConversionToOneFlow final : public OpConversionPattern<FuncOp> {
 public:
  using OpConversionPattern<FuncOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(FuncOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    auto func = rewriter.create<oneflow::Job>(op.getLoc(), op.getName(), op.getFunctionType());
    rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.end());
    rewriter.eraseOp(op);
    return success();
  }
};

struct ReturnConversionToOneFlow final : public OpConversionPattern<ReturnOp> {
 public:
  using OpConversionPattern<ReturnOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(ReturnOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<oneflow::ReturnOp>(op,
                                                   /* operands */ op.getOperands());
    return success();
  }
};
}  // namespace func

namespace oneflow {
struct JobConversionToFunc final : public OpConversionPattern<Job> {
 public:
  using OpConversionPattern<Job>::OpConversionPattern;
  LogicalResult matchAndRewrite(Job op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    auto func = rewriter.create<func::FuncOp>(op.getLoc(), op.getName(), op.getFunctionType());
    rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.end());
    rewriter.eraseOp(op);
    return success();
  }
};

struct ReturnConversionToFunc final : public OpConversionPattern<ReturnOp> {
 public:
  using OpConversionPattern<ReturnOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(ReturnOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op,
                                                /* operands */ op.getOperands());
    return success();
  }
};

namespace {

class OneFlowJobToFuncPass : public OneFlowJobToFuncPassBase<OneFlowJobToFuncPass> {
  void runOnOperation() override {
    Operation* op = getOperation();
    ConversionTarget target(getContext());
    target.addLegalDialect<mlir::func::FuncDialect>();
    RewritePatternSet patterns(&getContext());
    patterns.add<oneflow::JobConversionToFunc, oneflow::ReturnConversionToFunc>(op->getContext());
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
      LOG(ERROR) << "Failed to ofjob to func";
      getOperation()->dump();
    }
  }
};

class FuncToOneFlowJobPass : public FuncToOneFlowJobPassBase<FuncToOneFlowJobPass> {
  void getDependentDialects(::mlir::DialectRegistry& registry) const override {
    registry.insert<oneflow::OneFlowDialect>();
  }
  void runOnOperation() override {
    Operation* op = getOperation();
    ConversionTarget target(getContext());
    target.addLegalDialect<mlir::oneflow::OneFlowDialect>();
    RewritePatternSet patterns(&getContext());
    patterns.add<func::FuncConversionToOneFlow, func::ReturnConversionToOneFlow>(op->getContext());
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
      LOG(ERROR) << "Failed to func to ofjob";
      getOperation()->dump();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createOneFlowJobToFuncPass() {
  return std::make_unique<OneFlowJobToFuncPass>();
}

std::unique_ptr<Pass> createFuncToOneFlowJobPass() {
  return std::make_unique<FuncToOneFlowJobPass>();
}

}  // namespace oneflow

}  // namespace mlir

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
#include "OneFlow/OneFlowOps.h"
#include <cstdint>
#include <iostream>
#include <string>
#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/Passes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "oneflow/ir/llvm_monorepo-src/llvm/lib/Target/NVPTX/NVPTX.h"

#include <limits>

namespace mlir {

namespace oneflow {

struct ScalarMulByTensorOpLowering final : public OpConversionPattern<ScalarMulByTensorOp> {
 public:
  using OpConversionPattern<ScalarMulByTensorOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ScalarMulByTensorOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    Value scalar = op.scalar();
    if (auto scalar_type = scalar.getType().dyn_cast<RankedTensorType>()) {
      auto rank = op.x().getType().dyn_cast<RankedTensorType>().getRank();
      if (scalar_type.getRank() != rank) {
        std::vector<int64_t> perm(rank);
        std::fill(perm.begin(), perm.end(), 1);
        scalar = rewriter
                     .create<tosa::ReshapeOp>(
                         op->getLoc(),
                         RankedTensorType::get(
                             perm, scalar.getType().cast<TensorType>().getElementType()),
                         scalar, rewriter.getI64ArrayAttr(perm))
                     .output();
      }
    }
    rewriter.replaceOpWithNewOp<tosa::MulOp>(
        op,
        /* output */ op->getResultTypes().front().cast<TensorType>(),
        /* input1 */ op.x(),
        /* input2 */ scalar,
        /* shift */ rewriter.getIntegerAttr(rewriter.getI32Type(), 0));
    return success();
  }
};

struct JobLowering final : public OpConversionPattern<Job> {
 public:
  using OpConversionPattern<Job>::OpConversionPattern;
  LogicalResult matchAndRewrite(Job op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    auto func =
        rewriter.create<mlir::func::FuncOp>(op.getLoc(), op.getName(), op.getFunctionType());
    rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.end());
    rewriter.eraseOp(op);
    return success();
  }
};

struct ReturnOpLowering final : public OpConversionPattern<ReturnOp> {
 public:
  using OpConversionPattern<ReturnOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(ReturnOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op,
                                                      /* operands */ op.operands());
    return success();
  }
};

// TODO: shared memory between oneflow and backend
struct InputOpLowering final : public OpConversionPattern<InputOp> {
 public:
  using OpConversionPattern<InputOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(InputOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOp(op, op.input());
    return success();
  }
};

// TODO: shared memory between oneflow and backend
struct OutputOpLowering final : public OpConversionPattern<OutputOp> {
 public:
  using OpConversionPattern<OutputOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(OutputOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOp(op, op.input());
    return success();
  }
};

struct CastOpLowering final : public OpConversionPattern<CastOp> {
 public:
  using OpConversionPattern<CastOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(CastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<tosa::CastOp>(op,
                                              /* output */ op.out().getType(),
                                              /* input */ op.in());
    return success();
  }
};

struct ReluOpLowering final : public OpConversionPattern<ReluOp> {
 public:
  using OpConversionPattern<ReluOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(ReluOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    auto floatMax = std::numeric_limits<float>::max();
    auto intMax = std::numeric_limits<long long>::max();
    rewriter.replaceOpWithNewOp<tosa::ReluNOp>(op,
                                               /* output */ op.y().getType(),
                                               /* input */ op.x(), static_cast<uint64_t>(intMax),
                                               static_cast<::llvm::APFloat>(floatMax));
    return success();
  }
};


struct Conv2DOpLowering final : public OpConversionPattern<Conv2DOp> {
 public:
  using OpConversionPattern<Conv2DOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(Conv2DOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    /* static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type
     * outputType, Value input, Value weight, Value bias, ArrayAttr pad, ArrayAttr stride, ArrayAttr
     * dilation); */

    auto get_pair_int64_from_array = [](ArrayAttr arr) -> std::pair<int64_t, int64_t> {
      return { arr.getValue()[0].cast<IntegerAttr>().getSInt(), arr.getValue()[1].cast<IntegerAttr>().getSInt() };
    };

    auto pad_pair = get_pair_int64_from_array(op.padding_beforeAttr());
    auto pad_values = llvm::ArrayRef<int64_t>({pad_pair.first, pad_pair.second, pad_pair.first, pad_pair.second});
    auto pad = rewriter.getI64ArrayAttr(pad_values);

    auto stride_pair = get_pair_int64_from_array(op.strides());
    auto stride_values = llvm::ArrayRef<int64_t>({stride_pair.first, stride_pair.second});
    auto stride = rewriter.getI64ArrayAttr(stride_values);

    auto dilation_pair = get_pair_int64_from_array(op.dilation_rate());
    auto dilation_values = llvm::ArrayRef<int64_t>({dilation_pair.first, dilation_pair.second});
    auto dilation = rewriter.getI64ArrayAttr(dilation_values);



    rewriter.replaceOpWithNewOp<tosa::Conv2DOp>(op, op.out().getType(), op.in(), op.weight(),
                                                op.bias(), pad, stride, dilation);
    return success();
  }
};

namespace {
struct OneFlowLoweringToTosaPass : public LowerOneFlowToTosaPassBase<OneFlowLoweringToTosaPass> {
  void runOnOperation() override;
};
}  // namespace

std::unique_ptr<Pass> createLowerOneFlowToTosaPass() {
  return std::make_unique<OneFlowLoweringToTosaPass>();
}

void OneFlowLoweringToTosaPass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addLegalDialect<memref::MemRefDialect, mlir::func::FuncDialect, tosa::TosaDialect>();
  target.addIllegalDialect<OneFlowDialect>();
  RewritePatternSet patterns(&getContext());
  patterns.insert<CastOpLowering, ScalarMulByTensorOpLowering>(&getContext());
  patterns.insert<ReluOpLowering, Conv2DOpLowering>(&getContext());
  patterns.insert<JobLowering, ReturnOpLowering>(&getContext());
  patterns.insert<InputOpLowering, OutputOpLowering>(&getContext());

  if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
    getOperation()->dump();
    signalPassFailure();
  }
}

}  // namespace oneflow

}  // namespace mlir

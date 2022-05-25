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
#include <iostream>
#include <string>
#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/Passes.h"
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

struct InputOpLowering final : public OpConversionPattern<InputOp> {
 public:
  using OpConversionPattern<InputOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(InputOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<tosa::CastOp>(op, op.input().getType(), op.input());
    return success();
  }
};

struct OutputOpLowering final : public OpConversionPattern<OutputOp> {
 public:
  using OpConversionPattern<OutputOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(OutputOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<tosa::CastOp>(op, op.input().getType(), op.input());
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

// %y, %mean, %inv_variance = "oneflow.normalization"(%0, %output_2, %output_3, %output_4,
// %output_5) {axis = 1 : si32, device_name = ["@0:0"], device_tag = "cpu", epsilon = 9.99999974E-6
// : f32, hierarchy = [1], momentum = 0.899999976 : f32, op_name = "model.bn1-normalization-2",
// operand_segment_sizes = dense<[1, 1, 1, 1, 1, 0]> : vector<6xi32>, output_lbns =
// ["model.bn1-normalization-2/y_0", "model.bn1-normalization-2/mean_0",
// "model.bn1-normalization-2/inv_variance_0"], result_segment_sizes = dense<1> : vector<3xi32>,
// scope_symbol_id = 4611686018427449343 : i64, training = true} : (tensor<1x64x112x112xf32>,
// tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<1x64x112x112xf32>,
// tensor<64xf32>, tensor<64xf32>)

// let input = (ins
//   OneFlow_Tensor:$x,
//   Optional<OneFlow_Tensor>:$moving_mean,
//   Optional<OneFlow_Tensor>:$moving_variance,
//   OneFlow_Tensor:$gamma,
//   OneFlow_Tensor:$beta,
//   Optional<OneFlow_Tensor>:$_add_to_output
// );
// let output = (outs
//   OneFlow_Tensor:$y,
//   Optional<OneFlow_Tensor>:$mean,
//   Optional<OneFlow_Tensor>:$inv_variance
// );

struct NormalizationOpLowering final : public OpConversionPattern<NormalizationOp> {
 public:
  using OpConversionPattern<NormalizationOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(NormalizationOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    // auto zero = tosa::getConstTensor<float>(rewriter, op, 0, {}).getValue();
    // auto one = tosa::getConstTensor<float>(rewriter, op, 1, {}).getValue();
    // auto loc = op->getLoc();

    // // buildNormalCdf, mean = zero, sigma = one
    // auto outType = x.getType();
    // auto mean = zero;
    // Value xMinusMean = rewriter.create<tosa::SubOp>(loc, outType, x, mean);
    // // rsqrt of 2
    // Value rsqrt2 = tosa::getConstTensor<float>(rewriter, op, 0.70710678, {}).getValue();
    // Value erfArg = rewriter.create<tosa::MulOp>(loc, outType, xMinusMean, rsqrt2,
    //                                             /*shift=*/0);
    // Value erf = approximateErfOp(rewriter, op, erfArg);
    // Value erfPlus1 = rewriter.create<tosa::AddOp>(loc, outType, one, erf);
    // Value oneHalf = tosa::getConstTensor<float>(rewriter, op, 0.5, {}).getValue();
    // Value normalCdf = rewriter.create<tosa::MulOp>(loc, outType, oneHalf, erfPlus1, /*shift=*/0);
    return success();
  }
};

// %0 = "oneflow.conv2d"(%output, %output_0) {data_format = "channels_first", device_name =
// ["@0:0"], device_tag = "cpu", dilation_rate = [1 : si32, 1 : si32], filters = 64 : si32, groups =
// 1 : si32, hierarchy = [1], kernel_size = [7 : si32, 7 : si32], op_name = "model.conv1-conv2d-0",
// operand_segment_sizes = dense<[1, 1, 0, 0]> : vector<4xi32>, output_lbns =
// ["model.conv1-conv2d-0/out_0"], padding_before = [3 : si32, 3 : si32], scope_symbol_id =
// 4611686018427432959 : i64, strides = [2 : si32, 2 : si32]} : (tensor<1x3x224x224xf32>,
// tensor<64x3x7x7xf32>) -> tensor<1x64x112x112xf32>

// let input = (ins
//   OneFlow_Tensor:$in,
//   OneFlow_Tensor:$weight
// );
// let output = (outs
//   OneFlow_Tensor:$out
// );

// static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type
// outputType, Value input, Value weight, Value bias, ArrayAttr pad, ArrayAttr stride, ArrayAttr
// dilation);
struct Conv2DOpLowering final : public OpConversionPattern<Conv2DOp> {
 public:
  using OpConversionPattern<Conv2DOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(Conv2DOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    // oneflow 的卷积操作的输入是5D的输入和5D的输出，tosa的卷积操作是4D的输入和4D的输出，需要把oneflow的卷积操作进行迭代
    auto slice_input = rewriter.create<tosa::SliceOp>(op->getLoc());
    auto slice_conv2d_ouput = rewriter.create<tosa::Conv2DOp>(op->getLoc(), slice_input, op.weight(), op.bias(), op.padding_before(), op.strides(), op.dilation_rate());
    auto output = rewriter.create<tosa::ConcatOp>(op->getLoc());
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
  patterns.insert<ReluOpLowering>(&getContext());
  patterns.insert<JobLowering, ReturnOpLowering>(&getContext());
  patterns.insert<InputOpLowering, OutputOpLowering>(&getContext());
  if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
    getOperation()->dump();
    signalPassFailure();
  }
}

}  // namespace oneflow

}  // namespace mlir

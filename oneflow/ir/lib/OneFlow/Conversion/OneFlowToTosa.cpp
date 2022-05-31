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

struct BroadcastAddOpLowering final : public OpConversionPattern<BroadcastAddOp> {
 public:
  using OpConversionPattern<BroadcastAddOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(BroadcastAddOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<tosa::AddOp>(op, op.z().getType(), op.x(), op.y());
    return success();
  }
};

struct Add2OpLowering final : public OpConversionPattern<Add2Op> {
 public:
  using OpConversionPattern<Add2Op>::OpConversionPattern;
  LogicalResult matchAndRewrite(Add2Op op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<tosa::AddOp>(op, op.out().getType(), op.in0(), op.in1());
    return success();
  }
};

struct Conv2DOpLowering final : public OpConversionPattern<Conv2DOp> {
 public:
  using OpConversionPattern<Conv2DOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(Conv2DOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    auto get_pair_int64_from_array = [](ArrayAttr arr) -> std::pair<int64_t, int64_t> {
      return {arr.getValue()[0].cast<IntegerAttr>().getSInt(),
              arr.getValue()[1].cast<IntegerAttr>().getSInt()};
    };

    auto stride = get_pair_int64_from_array(op.strides());
    auto pad = get_pair_int64_from_array(op.padding_beforeAttr());
    auto dilation = get_pair_int64_from_array(op.dilation_rate());

    auto bias = op.bias();
    if (!bias) {
      auto output_shape = op.out().getType().cast<ShapedType>();
      auto output_channels = output_shape.getDimSize(1);
      auto bias_elem_type = output_shape.getElementType();
      auto type = RankedTensorType::get(output_channels, bias_elem_type);
      bias = rewriter.create<tosa::ConstOp>(
          op.getLoc(), type, DenseElementsAttr::get(type, rewriter.getZeroAttr(bias_elem_type)));
    }
    rewriter.replaceOpWithNewOp<tosa::Conv2DOp>(
        op, op.out().getType(), op.in(), op.weight(), bias,
        /* pad */
        rewriter.getI64ArrayAttr({pad.first, pad.second, pad.first, pad.second}),
        /*  stride  */ rewriter.getI64ArrayAttr({stride.first, stride.second}),
        /* dilation */ rewriter.getI64ArrayAttr({dilation.first, dilation.second}));
    return success();
  }
};

struct MatmulOpLowering final : public OpConversionPattern<MatmulOp> {
 public:
  using OpConversionPattern<MatmulOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(MatmulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    // TODO: more throw for robust in matmul shape rank
    auto preprocess = [&](Value matrix, bool transpose) -> Value {
      auto shape_type = matrix.getType().cast<ShapedType>();
      if (transpose) {
        auto transpose_perms = rewriter.create<tosa::ConstOp>(
            op.getLoc(), RankedTensorType::get({6}, rewriter.getI32Type()),
            rewriter.getI32TensorAttr({0, 1, 2}));
        matrix = rewriter.create<tosa::TransposeOp>(
            op->getLoc(),
            RankedTensorType::get({shape_type.getDimSize(1), shape_type.getDimSize(0)},
                                  shape_type.getElementType()),
            matrix, transpose_perms);
      }
      shape_type = matrix.getType().cast<ShapedType>();
      auto reshape_type = RankedTensorType::get(
          {1, shape_type.getDimSize(0), shape_type.getDimSize(1)}, shape_type.getElementType());
      return rewriter.create<tosa::ReshapeOp>(
          op.getLoc(), reshape_type, matrix,
          rewriter.getI64ArrayAttr({1, shape_type.getDimSize(0), shape_type.getDimSize(1)}));
    };

    auto a = preprocess(op.a(), op.transpose_a());
    auto b = preprocess(op.b(), op.transpose_b());
    auto out_shape_type = op.out().getType().cast<ShapedType>();
    auto out_reshape_type =
        RankedTensorType::get({1, out_shape_type.getDimSize(0), out_shape_type.getDimSize(1)},
                              out_shape_type.getElementType());
    auto matmul = rewriter.create<tosa::MatMulOp>(op.getLoc(), out_reshape_type, a, b);
    rewriter.replaceOpWithNewOp<tosa::ReshapeOp>(
        op, out_shape_type, matmul,
        rewriter.getI64ArrayAttr({out_shape_type.getDimSize(0), out_shape_type.getDimSize(1)}));
    return success();
  }
};

struct NormalizationOpLowering final : public OpConversionPattern<NormalizationOp> {
 public:
  using OpConversionPattern<NormalizationOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(NormalizationOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    auto reshape_dim = [&](Type type, Value value) -> Value {
      RankedTensorType in_type = value.getType().dyn_cast<RankedTensorType>();
      RankedTensorType out_type = type.cast<RankedTensorType>();
      SmallVector<int64_t> new_shape = {in_type.getShape()[0]};
      for (auto i = 2; i < out_type.getRank(); ++i) new_shape.push_back(1);
      auto new_type = RankedTensorType::get(new_shape, out_type.getElementType());
      return rewriter.create<tosa::ReshapeOp>(op->getLoc(), new_type, value,
                                              rewriter.getI64ArrayAttr(new_shape));
    };
    auto out_type = op.y().getType();
    auto epsilon_type = RankedTensorType::get({}, rewriter.getF32Type());
    auto loc = op->getLoc();
    // epsilon   = reshape(epsilon, shape_1)
    auto epsilon = rewriter.create<tosa::ConstOp>(
        loc, epsilon_type, DenseElementsAttr::get(epsilon_type, op.epsilon()));
    //  mean = reshape(mean, shape_0)
    auto mean = reshape_dim(out_type, adaptor.moving_mean());
    //  variance= reshape(variance, shape_0)
    auto variance = reshape_dim(out_type, adaptor.moving_variance());
    // scale = reshape(scale, shape_0)
    auto gamma = reshape_dim(out_type, adaptor.gamma());
    // beta = reshape(beta, shape_0)
    auto beta = reshape_dim(out_type, adaptor.beta());
    // op1 = sub(input, mean)
    auto op1 = rewriter.create<tosa::SubOp>(loc, out_type, op.x(), mean);
    // op2 = add(var, epsilon)
    auto op2 = rewriter.create<tosa::AddOp>(loc, variance.getType(), variance, epsilon);
    // op3 = rsqrt(op2)
    auto op3 = rewriter.create<tosa::RsqrtOp>(loc, variance.getType(), op2);
    // op4 = mul(op1, op3)
    auto op4 = rewriter.create<tosa::MulOp>(loc, out_type, op1, op3, 0);
    // op5 = mul(op4, gamma)
    auto op5 = rewriter.create<tosa::MulOp>(loc, out_type, op4, gamma, 0);
    // op6 = add(op5, beta)
    auto batch_norm = rewriter.create<tosa::AddOp>(loc, out_type, op5, beta);
    // TODO: computer the true mean and variance
    rewriter.replaceOp(op, {batch_norm, op.moving_mean(), op.moving_variance()});
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
  patterns.insert<JobLowering, ReturnOpLowering>(&getContext());
  patterns.insert<InputOpLowering, OutputOpLowering>(&getContext());
  patterns.insert<Add2OpLowering, BroadcastAddOpLowering>(&getContext());
  patterns.insert<ReluOpLowering, Conv2DOpLowering, MatmulOpLowering, NormalizationOpLowering>(
      &getContext());

  if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
    getOperation()->dump();
    signalPassFailure();
  }
}

}  // namespace oneflow

}  // namespace mlir

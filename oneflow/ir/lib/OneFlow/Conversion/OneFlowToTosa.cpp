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
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpImplementation.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/framework/variable_tensor_mgr.h"

#include <limits>

namespace mlir {

namespace oneflow {

Value CreateTranspose(Location& loc, ConversionPatternRewriter& rewriter, Value input,
                      ArrayRef<int32_t> perms) {
  int perms_size = perms.size();
  auto transpose_perms = rewriter.create<tosa::ConstOp>(
      loc, RankedTensorType::get({perms_size}, rewriter.getI32Type()),
      rewriter.getI32TensorAttr(perms));
  const auto shape_type = input.getType().cast<ShapedType>();
  std::vector<int64_t> ranked_type;
  for (const auto& index : perms) ranked_type.push_back(shape_type.getDimSize(index));
  return rewriter.create<tosa::TransposeOp>(
      loc, RankedTensorType::get(ranked_type, shape_type.getElementType()), input, transpose_perms);
};

Value CreateBNOp(Location loc, ConversionPatternRewriter& rewriter, Value output, Value x,
                 Value mean, Value variance, Value epsilon, Value gamma, Value beta) {
  const auto output_type = output.getType();
  // sub_op = sub(input, mean)
  auto sub_op0 = rewriter.create<tosa::SubOp>(loc, output_type, x, mean);
  // add_op0 = add(var, epsilon)
  auto add_op0 = rewriter.create<tosa::AddOp>(loc, variance.getType(), variance, epsilon);
  // rsqrt_op = rsqrt(add_op0)
  auto rsqrt_op = rewriter.create<tosa::RsqrtOp>(loc, variance.getType(), add_op0);
  // op4 = mul(sub_op, rsqrt_op)
  auto mul_op0 = rewriter.create<tosa::MulOp>(loc, output_type, sub_op0, rsqrt_op, 0);
  // op5 = mul(mul_op0, gamma)
  auto mul_op1 = rewriter.create<tosa::MulOp>(loc, output_type, mul_op0, gamma, 0);
  // op6 = add(mul_op1, beta)
  auto batch_norm = rewriter.create<tosa::AddOp>(loc, output_type, mul_op1, beta);
  return batch_norm;
};

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
    // TODO: more choices to passing data between tosa and oneflow
    const auto newValues = op.input();
    const auto is_block_arg = newValues.dyn_cast<BlockArgument>() != nullptr;
    if (!is_block_arg) op->emitError("input is not block arg");
    rewriter.replaceOp(op, newValues);
    return success();
  }
};

struct OutputOpLowering final : public OpConversionPattern<OutputOp> {
 public:
  using OpConversionPattern<OutputOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(OutputOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    // TODO: more choices to passing data between tosa and oneflow
    const auto newValues = op.input();
    rewriter.replaceOp(op, newValues);
    return success();
  }
};

struct VariableOpLowering final : public OpConversionPattern<VariableOp> {
 public:
  using OpConversionPattern<VariableOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(VariableOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    const auto mgr = ::oneflow::Global<::oneflow::VariableTensorMgr>::Get();
    // decide whether call by python or not
    if (!mgr) op->emitError("oneflow variable op doesn't support pure mlir file conversion");

    const auto tensor = mgr->Get(op.op_name().str());
    const auto value = support::TensorToDenseElementsAttr(tensor, rewriter.getContext());
    const auto output = op.output().getType();

    rewriter.replaceOpWithNewOp<tosa::ConstOp>(op, output, value);
    return success();
  }
};

struct CastOpLowering final : public OpConversionPattern<CastOp> {
 public:
  using OpConversionPattern<CastOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(CastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    auto output = op.out().getType();
    auto input = op.in();
    rewriter.replaceOpWithNewOp<tosa::CastOp>(op, output, input);
    return success();
  }
};

struct ReluOpLowering final : public OpConversionPattern<ReluOp> {
 public:
  using OpConversionPattern<ReluOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(ReluOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    const auto floatMax = std::numeric_limits<float>::max();
    const auto intMax = std::numeric_limits<long long>::max();

    const auto output = op.y().getType();
    auto input = op.x();
    auto max_int = static_cast<uint64_t>(intMax);
    auto max_fp = static_cast<::llvm::APFloat>(floatMax);

    rewriter.replaceOpWithNewOp<tosa::ReluNOp>(op, output, input, max_int, max_fp);
    return success();
  }
};

struct BroadcastAddOpLowering final : public OpConversionPattern<BroadcastAddOp> {
 public:
  using OpConversionPattern<BroadcastAddOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(BroadcastAddOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    const auto output = op.z().getType();
    auto input1 = op.x();
    auto input2 = op.y();

    rewriter.replaceOpWithNewOp<tosa::AddOp>(op, output, input1, input2);
    return success();
  }
};

struct Add2OpLowering final : public OpConversionPattern<Add2Op> {
 public:
  using OpConversionPattern<Add2Op>::OpConversionPattern;
  LogicalResult matchAndRewrite(Add2Op op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    const auto output = op.out().getType();
    auto input1 = op.in0();
    auto input2 = op.in1();

    rewriter.replaceOpWithNewOp<tosa::AddOp>(op, output, input1, input2);
    return success();
  }
};

struct AvgPool2DOpLowering final : public OpConversionPattern<AvgPool2DOp> {
 public:
  using OpConversionPattern<AvgPool2DOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(AvgPool2DOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    auto get_pair_int64_from_array = [](ArrayAttr arr) -> std::pair<int64_t, int64_t> {
      return {arr.getValue()[0].cast<IntegerAttr>().getSInt(),
              arr.getValue()[1].cast<IntegerAttr>().getSInt()};
    };

    auto reshape_type = [](ShapedType shape_type, ArrayRef<int32_t> perms) -> RankedTensorType {
      std::vector<int64_t> ranked_type;
      for (auto index : perms) ranked_type.push_back(shape_type.getDimSize(index));
      return RankedTensorType::get(ranked_type, shape_type.getElementType());
    };

    auto stride_pairs = get_pair_int64_from_array(op.stride());
    auto pad_pairs = get_pair_int64_from_array(op.padding());
    auto kernel_pairs = get_pair_int64_from_array(op.kernel_size());

    auto loc = op.getLoc();
    auto perms = {0, 2, 3, 1};

    const auto kernel = rewriter.getI64ArrayAttr({kernel_pairs.first, kernel_pairs.second});
    const auto stride = rewriter.getI64ArrayAttr({stride_pairs.first, stride_pairs.second});
    const auto pad = rewriter.getI64ArrayAttr(
        {pad_pairs.first, pad_pairs.second, pad_pairs.first, pad_pairs.second});

    auto input = CreateTranspose(loc, rewriter, op.x(), perms);
    auto output = reshape_type(op.y().getType().cast<ShapedType>(), perms);

    auto avg_pool2d = rewriter.create<tosa::AvgPool2dOp>(loc, output, input, kernel, stride, pad);

    auto out = CreateTranspose(loc, rewriter, avg_pool2d, {0, 3, 1, 2});
    rewriter.replaceOp(op, {out});
    return success();
  }
};

struct MaxPool2DOpLowering final : public OpConversionPattern<MaxPool2DOp> {
 public:
  using OpConversionPattern<MaxPool2DOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(MaxPool2DOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    auto get_pair_int64_from_array = [](ArrayAttr arr) -> std::pair<int64_t, int64_t> {
      return {arr.getValue()[0].cast<IntegerAttr>().getSInt(),
              arr.getValue()[1].cast<IntegerAttr>().getSInt()};
    };
    auto reshape_type = [](ShapedType shape_type, ArrayRef<int32_t> perms) -> RankedTensorType {
      std::vector<int64_t> ranked_type;
      for (auto index : perms) ranked_type.push_back(shape_type.getDimSize(index));
      return RankedTensorType::get(ranked_type, shape_type.getElementType());
    };
    // TODO: support return indice
    if (op.return_indices()) op->emitError("not support return indices now");
    auto stride_pairs = get_pair_int64_from_array(op.stride());
    auto kernel_pairs = get_pair_int64_from_array(op.kernel_size());
    auto pad_pairs = get_pair_int64_from_array(op.padding());

    auto loc = op.getLoc();
    auto perms = {0, 2, 3, 1};

    const auto kernel = rewriter.getI64ArrayAttr({kernel_pairs.first, kernel_pairs.second});
    const auto stride = rewriter.getI64ArrayAttr({stride_pairs.first, stride_pairs.second});
    const auto pad = rewriter.getI64ArrayAttr(
        {pad_pairs.first, pad_pairs.second, pad_pairs.first, pad_pairs.second});

    auto input = CreateTranspose(loc, rewriter, op.x(), perms);
    auto output = reshape_type(op.y().getType().cast<ShapedType>(), perms);

    auto max_pool2d = rewriter.create<tosa::MaxPool2dOp>(loc, output, input, kernel, stride, pad);

    auto y = CreateTranspose(loc, rewriter, max_pool2d, {0, 3, 1, 2});

    auto indice_output = op.indice().getType();
    auto value = DenseElementsAttr::get(indice_output, rewriter.getZeroAttr(rewriter.getI64Type()));

    auto indice = rewriter.create<tosa::ConstOp>(loc, indice_output, value);
    rewriter.replaceOp(op, {y, indice});
    return success();
  }
};

struct FlattenOpLowering final : public OpConversionPattern<FlattenOp> {
 public:
  using OpConversionPattern<FlattenOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(FlattenOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    const auto start_dim = op.start_dim();
    const auto end_dim = op.end_dim();
    const auto in_type = op.in().getType();

    const auto in_shape = in_type.cast<ShapedType>();
    const auto rank = in_type.dyn_cast<RankedTensorType>().getRank();

    // calculate reshape_vec
    std::vector<int64_t> reshape_vec;
    for (auto dim = 0; dim < start_dim; ++dim) { reshape_vec.push_back(in_shape.getDimSize(dim)); }
    auto last_dim = end_dim < 0 ? rank : end_dim + 1;
    int flatten_size = 1;
    for (auto dim = start_dim; dim < last_dim; ++dim) { flatten_size *= in_shape.getDimSize(dim); }
    reshape_vec.push_back(flatten_size);
    if (end_dim > 0) {
      for (auto dim = end_dim + 1; dim < rank; ++dim) {
        reshape_vec.push_back(in_shape.getDimSize(dim));
      }
    }
    // generate reshape op
    const auto output = RankedTensorType::get(reshape_vec, in_shape.getElementType());
    auto input1 = op.in();
    auto new_shape = rewriter.getI64ArrayAttr(reshape_vec);

    rewriter.replaceOpWithNewOp<tosa::ReshapeOp>(op, output, input1, new_shape);
    return success();
  }
};

struct MatmulOpLowering final : public OpConversionPattern<MatmulOp> {
 public:
  using OpConversionPattern<MatmulOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(MatmulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    // TODO: more throw for robust in matmul shape rank
    auto loc = op.getLoc();

    auto preprocess = [&](Value matrix, bool transpose) -> Value {
      auto shape_type = matrix.getType().cast<ShapedType>();
      if (transpose) { matrix = CreateTranspose(loc, rewriter, matrix, {1, 0}); }

      shape_type = matrix.getType().cast<ShapedType>();
      auto reshape_type = RankedTensorType::get(
          {1, shape_type.getDimSize(0), shape_type.getDimSize(1)}, shape_type.getElementType());

      return rewriter.create<tosa::ReshapeOp>(
          op.getLoc(), reshape_type, matrix,
          rewriter.getI64ArrayAttr({1, shape_type.getDimSize(0), shape_type.getDimSize(1)}));
    };

    auto a = preprocess(op.a(), op.transpose_a());
    auto b = preprocess(op.b(), op.transpose_b());

    const auto out_shape_type = op.out().getType().cast<ShapedType>();
    const auto out_reshape_type =
        RankedTensorType::get({1, out_shape_type.getDimSize(0), out_shape_type.getDimSize(1)},
                              out_shape_type.getElementType());

    auto matmul = rewriter.create<tosa::MatMulOp>(loc, out_reshape_type, a, b);
    const auto new_shape =
        rewriter.getI64ArrayAttr({out_shape_type.getDimSize(0), out_shape_type.getDimSize(1)});

    rewriter.replaceOpWithNewOp<tosa::ReshapeOp>(op, out_shape_type, matmul, new_shape);
    return success();
  }
};

struct NormalizationInferenceOpLowering final
    : public OpConversionPattern<NormalizationInferenceOp> {
 public:
  using OpConversionPattern<NormalizationInferenceOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(NormalizationInferenceOp op, OpAdaptor adaptor,
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

    auto loc = op->getLoc();
    const auto out_type = op.y().getType();

    const auto epsilon_type = RankedTensorType::get({}, rewriter.getF32Type());
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
    auto output = op.y();
    auto x = op.x();

    auto batch_norm =
        oneflow::CreateBNOp(loc, rewriter, output, x, mean, variance, epsilon, gamma, beta);
    rewriter.replaceOp(op, {batch_norm});
    return success();
  }
};

struct NormalizationOpLowering final : public OpConversionPattern<NormalizationOp> {
 public:
  using OpConversionPattern<NormalizationOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(NormalizationOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    auto reshape_dim = [&](Type type, Value value) -> Value {
      const RankedTensorType in_type = value.getType().dyn_cast<RankedTensorType>();
      const RankedTensorType out_type = type.cast<RankedTensorType>();
      SmallVector<int64_t> new_shape = {in_type.getShape()[0]};
      for (auto i = 2; i < out_type.getRank(); ++i) new_shape.push_back(1);
      const auto new_type = RankedTensorType::get(new_shape, out_type.getElementType());
      return rewriter.create<tosa::ReshapeOp>(op->getLoc(), new_type, value,
                                              rewriter.getI64ArrayAttr(new_shape));
    };

    auto loc = op->getLoc();
    const auto out_type = op.y().getType();

    const auto epsilon_type = RankedTensorType::get({}, rewriter.getF32Type());
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
    auto output = op.y();
    auto x = op.x();

    auto batch_norm =
        oneflow::CreateBNOp(loc, rewriter, output, x, mean, variance, epsilon, gamma, beta);
    auto moving_mean = op.moving_mean();
    auto moving_variance = op.moving_variance();

    rewriter.replaceOp(op, {batch_norm, moving_mean, moving_variance});
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
    auto reshape_type = [](ShapedType shape_type, ArrayRef<int32_t> perms) -> RankedTensorType {
      std::vector<int64_t> ranked_type;
      for (auto index : perms) ranked_type.push_back(shape_type.getDimSize(index));
      return RankedTensorType::get(ranked_type, shape_type.getElementType());
    };

    auto stride_pairs = get_pair_int64_from_array(op.strides());
    auto pad_pairs = get_pair_int64_from_array(op.padding_beforeAttr());
    auto dilation_pairs = get_pair_int64_from_array(op.dilation_rate());

    const auto pad = rewriter.getI64ArrayAttr(
        {pad_pairs.first, pad_pairs.second, pad_pairs.first, pad_pairs.second});
    const auto stride = rewriter.getI64ArrayAttr({stride_pairs.first, stride_pairs.second});
    const auto dilation = rewriter.getI64ArrayAttr({dilation_pairs.first, dilation_pairs.second});

    auto bias = op.bias();
    auto loc = op.getLoc();
    if (!bias) {
      const auto output_shape = op.out().getType().cast<ShapedType>();
      const auto output_channels = output_shape.getDimSize(1);
      const auto bias_elem_type = output_shape.getElementType();
      const auto type = RankedTensorType::get(output_channels, bias_elem_type);
      bias = rewriter.create<tosa::ConstOp>(
          op.getLoc(), type, DenseElementsAttr::get(type, rewriter.getZeroAttr(bias_elem_type)));
    }

    auto perms = {0, 2, 3, 1};
    auto in = CreateTranspose(loc, rewriter, op.in(), perms);
    auto weight = CreateTranspose(loc, rewriter, op.weight(), perms);
    const auto output = reshape_type(op.out().getType().cast<ShapedType>(), perms);

    auto conv2d =
        rewriter.create<tosa::Conv2DOp>(loc, output, in, weight, bias, pad, stride, dilation);

    auto res = CreateTranspose(loc, rewriter, conv2d, {0, 3, 1, 2});
    rewriter.replaceOp(op, {res});
    return success();
    getTypeConverter();
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
  MLIRContext* context = &getContext();
  ConversionTarget target(*context);
  target.addLegalDialect<memref::MemRefDialect, mlir::func::FuncDialect, tosa::TosaDialect,
                         tensor::TensorDialect, arith::ArithmeticDialect>();
  target.addIllegalDialect<OneFlowDialect>();

  TypeConverter typeConverter;
  typeConverter.addConversion([](Type type) { return type; });
  RewritePatternSet patterns(context);
  patterns.add<CastOpLowering, ScalarMulByTensorOpLowering, ReluOpLowering, Conv2DOpLowering,
               AvgPool2DOpLowering, FlattenOpLowering, Add2OpLowering, MaxPool2DOpLowering,
               MatmulOpLowering, BroadcastAddOpLowering, JobLowering, ReturnOpLowering,
               VariableOpLowering, InputOpLowering, OutputOpLowering, NormalizationOpLowering,
               NormalizationInferenceOpLowering>(typeConverter, context);
  if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
    getOperation()->dump();
    signalPassFailure();
  }
}

}  // namespace oneflow

}  // namespace mlir

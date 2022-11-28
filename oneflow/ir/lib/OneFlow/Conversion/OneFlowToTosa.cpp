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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/framework/variable_tensor_mgr.h"

#include <limits>

namespace mlir {

namespace oneflow {

Type convertToSignless(MLIRContext* context, Type type) {
  if (auto ranked_tensor = type.dyn_cast<RankedTensorType>()) {
    if (auto intTy = ranked_tensor.getElementType().dyn_cast<IntegerType>()) {
      if (!intTy.isSignless()) {
        return RankedTensorType::get(
            ranked_tensor.getShape(),
            IntegerType::get(context, intTy.getWidth(),
                             mlir::IntegerType::SignednessSemantics::Signless));
      }
    }
  }
  return type;
}

FunctionType convertToSignlessFuncType(MLIRContext* context, FunctionType funcType) {
  llvm::SmallVector<Type, 4> inputs;
  llvm::SmallVector<Type, 4> results;
  for (auto arg : funcType.getInputs()) { inputs.push_back(convertToSignless(context, arg)); }
  for (auto res : funcType.getResults()) { results.push_back(convertToSignless(context, res)); }
  return FunctionType::get(context, inputs, results);
}

bool isSignLessTensorOrOther(Type type) {
  if (auto ranked_tensor = type.dyn_cast<RankedTensorType>()) {
    if (auto intTy = ranked_tensor.getElementType().dyn_cast<IntegerType>()) {
      if (intTy.isUnsigned()) { return false; }
      if (intTy.isSigned()) { return false; }
    }
  }
  return true;
}
bool allSignless(mlir::TypeRange types) {
  for (auto type : types) {
    if (!isSignLessTensorOrOther(type)) { return false; }
  }
  return true;
}

bool allSignless(FunctionType funcType) {
  for (auto arg : funcType.getInputs()) {
    if (!isSignLessTensorOrOther(arg)) { return false; }
  }
  for (auto res : funcType.getResults()) {
    if (!isSignLessTensorOrOther(res)) { return false; }
  }
  return true;
}

Value CreateTransposeValue(Location& loc, ConversionPatternRewriter& rewriter, Value input,
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

RankedTensorType CreateTransposeType(ShapedType output, ArrayRef<int32_t> perms) {
  std::vector<int64_t> ranked_type;
  for (auto index : perms) ranked_type.push_back(output.getDimSize(index));
  return RankedTensorType::get(ranked_type, output.getElementType());
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
    auto func_type = convertToSignlessFuncType(op->getContext(), op.function_type());
    auto func = rewriter.create<mlir::func::FuncOp>(op.getLoc(), op.getName(), func_type);
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
    if (!is_block_arg) { return op->emitError("input is not block arg"); }
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
    const auto mgr = ::oneflow::Singleton<::oneflow::VariableTensorMgr>::Get();
    if (!mgr) { return op->emitError("global variable tensor manager miss"); }

    const auto tensor = CHECK_JUST(mgr->Get(op.op_name().str()));
    if (!tensor) { return op->emitError("tensor is null"); }
    const auto value = support::TensorToDenseElementsAttr(tensor, rewriter.getContext());
    const auto output = op.output().getType();

    rewriter.replaceOpWithNewOp<tosa::ConstOp>(op, output, value);
    return success();
  }
};

struct VariableOpToConstLowering final : public OpConversionPattern<VariableOp> {
 public:
  VariableOpToConstLowering(TypeConverter& typeConverter, MLIRContext* context, int const_val)
      : OpConversionPattern<VariableOp>(typeConverter, context), const_val_(const_val){};

  using OpConversionPattern<VariableOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(VariableOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    const auto output = op.output().getType();
    const auto type = output.cast<ShapedType>().getElementType();

    // TODO: more control about this scope with flag
    if (type.isa<FloatType>()) {
      const auto float_attr = rewriter.getFloatAttr(type, const_val_);
      auto value = DenseElementsAttr::get(output, float_attr);

      rewriter.replaceOpWithNewOp<tosa::ConstOp>(op, output, value);
    } else if (auto integerType = type.dyn_cast<IntegerType>()) {
      const auto int_attr =
          rewriter.getIntegerAttr(type, APInt(type.cast<IntegerType>().getWidth(), const_val_));
      auto value = DenseElementsAttr::get(output, int_attr);

      rewriter.replaceOpWithNewOp<tosa::ConstOp>(op, output, value);
    } else {
      return op->emitError(
          "OneFlow variable op lower to TOSA const op only support integer and float value now");
    }

    return success();
  }

 private:
  int const_val_;
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

    auto stride_pairs = get_pair_int64_from_array(op.stride());
    auto pad_pairs = get_pair_int64_from_array(op.padding());
    auto kernel_pairs = get_pair_int64_from_array(op.kernel_size());

    auto loc = op.getLoc();
    auto perms = {0, 2, 3, 1};

    const auto kernel = rewriter.getI64ArrayAttr({kernel_pairs.first, kernel_pairs.second});
    const auto stride = rewriter.getI64ArrayAttr({stride_pairs.first, stride_pairs.second});
    const auto pad = rewriter.getI64ArrayAttr(
        {pad_pairs.first, pad_pairs.second, pad_pairs.first, pad_pairs.second});

    auto input = CreateTransposeValue(loc, rewriter, op.x(), perms);
    auto output = CreateTransposeType(op.y().getType().cast<ShapedType>(), perms);

    auto avg_pool2d = rewriter.create<tosa::AvgPool2dOp>(loc, output, input, kernel, stride, pad);

    auto out = CreateTransposeValue(loc, rewriter, avg_pool2d, {0, 3, 1, 2});
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
    // TODO: support return indice
    if (op.return_indices()) { return op->emitError("not support return indices now"); }
    auto stride_pairs = get_pair_int64_from_array(op.stride());
    auto kernel_pairs = get_pair_int64_from_array(op.kernel_size());
    auto pad_pairs = get_pair_int64_from_array(op.padding());

    auto loc = op.getLoc();
    auto perms = {0, 2, 3, 1};

    const auto kernel = rewriter.getI64ArrayAttr({kernel_pairs.first, kernel_pairs.second});
    const auto stride = rewriter.getI64ArrayAttr({stride_pairs.first, stride_pairs.second});
    const auto pad = rewriter.getI64ArrayAttr(
        {pad_pairs.first, pad_pairs.second, pad_pairs.first, pad_pairs.second});

    auto input = CreateTransposeValue(loc, rewriter, op.x(), perms);
    auto output = CreateTransposeType(op.y().getType().cast<ShapedType>(), perms);

    auto max_pool2d = rewriter.create<tosa::MaxPool2dOp>(loc, output, input, kernel, stride, pad);
    auto y = CreateTransposeValue(loc, rewriter, max_pool2d, {0, 3, 1, 2});

    auto indice_output = convertToSignless(op->getContext(), op.indice().getType());
    auto value = DenseElementsAttr::get(indice_output, rewriter.getZeroAttr(rewriter.getI64Type()));
    tosa::ConstOp indice = rewriter.create<tosa::ConstOp>(loc, indice_output, value);
    rewriter.replaceOp(op, {y, indice});
    return success();
  }
};

struct ReshapeOpLowering final : public OpConversionPattern<ReshapeOp> {
 public:
  using OpConversionPattern<ReshapeOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(ReshapeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    auto output = op.out().getType();
    auto input = op.in();
    llvm::SmallVector<int64_t> new_shape;
    for (const auto& dim_attr : op.shape()) {
      new_shape.push_back(dim_attr.cast<IntegerAttr>().getSInt());
    }
    rewriter.replaceOpWithNewOp<tosa::ReshapeOp>(op, output, input,
                                                 rewriter.getI64ArrayAttr(new_shape));
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
      if (transpose) { matrix = CreateTransposeValue(loc, rewriter, matrix, {1, 0}); }

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
    auto epsilon = rewriter.create<tosa::ConstOp>(
        loc, epsilon_type, DenseElementsAttr::get(epsilon_type, op.epsilon()));
    auto mean = reshape_dim(out_type, adaptor.moving_mean());
    auto variance = reshape_dim(out_type, adaptor.moving_variance());
    auto gamma = reshape_dim(out_type, adaptor.gamma());
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
    auto in = CreateTransposeValue(loc, rewriter, op.in(), perms);
    auto weight = CreateTransposeValue(loc, rewriter, op.weight(), perms);
    const auto output = CreateTransposeType(op.out().getType().cast<ShapedType>(), perms);

    auto conv2d =
        rewriter.create<tosa::Conv2DOp>(loc, output, in, weight, bias, pad, stride, dilation);

    auto res = CreateTransposeValue(loc, rewriter, conv2d, {0, 3, 1, 2});
    rewriter.replaceOp(op, {res});
    return success();
  }
};

namespace {

struct OneFlowLoweringToTosaPass : public LowerOneFlowToTosaPassBase<OneFlowLoweringToTosaPass> {
  void runOnOperation() override;
};

struct ConvertToSignlessForTosaPass
    : public ConvertToSignlessForTosaPassBase<ConvertToSignlessForTosaPass> {
  void runOnOperation() override;
};

}  // namespace

std::unique_ptr<Pass> createLowerOneFlowToTosaPass() {
  return std::make_unique<OneFlowLoweringToTosaPass>();
}

std::unique_ptr<Pass> createConvertToSignlessForTosaPass() {
  return std::make_unique<ConvertToSignlessForTosaPass>();
}

void OneFlowLoweringToTosaPass::runOnOperation() {
  MLIRContext* context = &getContext();
  ConversionTarget target(*context);
  target.addLegalDialect<memref::MemRefDialect, mlir::func::FuncDialect, tosa::TosaDialect,
                         tensor::TensorDialect, arith::ArithmeticDialect>();
  target.addIllegalDialect<OneFlowDialect>();

  TypeConverter typeConverter;
  typeConverter.addConversion([context](Type type) { return convertToSignless(context, type); });
  typeConverter.addSourceMaterialization(
      [&](OpBuilder& builder, Type resultType, ValueRange inputs, Location loc) -> Optional<Value> {
        CHECK_EQ(inputs.size(), 1) << "expect to materialize a single value";
        return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs).getResult(0);
      });
  typeConverter.addTargetMaterialization(
      [&](OpBuilder& builder, Type resultType, ValueRange inputs, Location loc) -> Optional<Value> {
        CHECK_EQ(inputs.size(), 1) << "expect to materialize a single value";
        return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs).getResult(0);
      });
  RewritePatternSet patterns(context);

  const auto mgr = ::oneflow::Singleton<::oneflow::VariableTensorMgr>::Get();
  // check if the pass is triggered by python based on the presence of variable tensor manger
  if (mgr) {
    patterns.add<VariableOpLowering>(typeConverter, context);
  } else {
    patterns.add<VariableOpToConstLowering>(typeConverter, context, this->variableAsConstant);
  }
  patterns
      .add<CastOpLowering, ScalarMulByTensorOpLowering, ReluOpLowering, Conv2DOpLowering,
           AvgPool2DOpLowering, ReshapeOpLowering, Add2OpLowering, MaxPool2DOpLowering,
           MatmulOpLowering, BroadcastAddOpLowering, JobLowering, ReturnOpLowering, InputOpLowering,
           OutputOpLowering, NormalizationOpLowering, NormalizationInferenceOpLowering>(
          typeConverter, context);
  if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
    signalPassFailure();
    LOG(ERROR) << "Failed to lower OneFlow to Tosa";
    getOperation()->dump();
  }
}

struct ConvertReturnToSignlessPattern : public OpRewritePattern<func::ReturnOp> {
  explicit ConvertReturnToSignlessPattern(::mlir::MLIRContext* context)
      : OpRewritePattern<func::ReturnOp>(context, /*benefit=*/1) {}
  ::mlir::LogicalResult matchAndRewrite(func::ReturnOp op,
                                        ::mlir::PatternRewriter& rewriter) const override {
    // make sure result not converted
    if (allSignless(op.getOperandTypes())) { return failure(); }
    llvm::SmallVector<Type, 1> results;
    for (auto res : op->getOperandTypes()) {
      results.push_back(convertToSignless(op->getContext(), res));
    }
    auto uc = rewriter.create<UnrealizedConversionCastOp>(op->getLoc(), results, op.operands());
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, op->getResultTypes(), uc->getResults(),
                                                op->getAttrs());
    return success();
  }
};

struct ConvertFuncToSignlessPattern : public OpRewritePattern<func::FuncOp> {
  explicit ConvertFuncToSignlessPattern(::mlir::MLIRContext* context)
      : OpRewritePattern<func::FuncOp>(context, /*benefit=*/1) {}
  ::mlir::LogicalResult matchAndRewrite(func::FuncOp op,
                                        ::mlir::PatternRewriter& rewriter) const override {
    if (allSignless(op.getFunctionType())) { return failure(); }
    auto ft = convertToSignlessFuncType(op->getContext(), op.getFunctionType());
    auto func = rewriter.create<mlir::func::FuncOp>(op.getLoc(), op.getName(), ft);
    BlockAndValueMapping bvm;
    op.getRegion().cloneInto(&func.getRegion(), bvm);
    for (auto& block : func.getBody().getBlocks()) {
      for (auto arg : block.getArguments()) {
        arg.setType(convertToSignless(op.getContext(), arg.getType()));
      }
    }
    rewriter.eraseOp(op);
    RewritePatternSet patterns(func->getContext());
    patterns.add<ConvertReturnToSignlessPattern>(func->getContext());
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
    return success();
  }
};

void ConvertToSignlessForTosaPass::runOnOperation() {
  Operation* op = getOperation();
  RewritePatternSet patterns(op->getContext());
  patterns.add<ConvertFuncToSignlessPattern>(op->getContext());
  (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
}

}  // namespace oneflow

}  // namespace mlir

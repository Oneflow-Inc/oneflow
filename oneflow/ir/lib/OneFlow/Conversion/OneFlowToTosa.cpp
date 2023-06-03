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
#include "llvm/Support/Casting.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
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

Value CreateBNOp(Location loc, ConversionPatternRewriter& rewriter, Type output_type, Value x,
                 Value mean, Value variance, Value epsilon, Value gamma, Value beta) {
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
  Value batch_norm = rewriter.create<tosa::AddOp>(loc, output_type, mul_op1, beta);
  return batch_norm;
};

struct ScalarMulByTensorOpLowering final : public OpConversionPattern<ScalarMulByTensorOp> {
 public:
  using OpConversionPattern<ScalarMulByTensorOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ScalarMulByTensorOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    Value scalar = op.getScalar();
    rewriter.replaceOpWithNewOp<tosa::MulOp>(
        op,
        /* output */ op->getResultTypes().front().cast<TensorType>(),
        /* input1 */ op.getX(),
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
    auto func_type = convertToSignlessFuncType(op->getContext(), op.getFunctionType());
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
                                                      /* operands */ op.getOperands());
    return success();
  }
};

struct InputOpLowering final : public OpConversionPattern<InputOp> {
 public:
  using OpConversionPattern<InputOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(InputOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    // TODO: more choices to passing data between tosa and oneflow
    const auto newValues = op.getInput();
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
    const auto newValues = op.getInput();
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

    const auto tensor = CHECK_JUST(mgr->Get(op.getOpName().str()));
    if (!tensor) { return op->emitError("tensor is null"); }
    const auto value = support::TensorToDenseElementsAttr(tensor, rewriter.getContext());
    const auto output = op.getOutput().getType();

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
    const auto output = op.getOutput().getType();
    const auto type = output.cast<ShapedType>().getElementType();

    // TODO: more control about this scope with flag
    if (type.isa<FloatType>()) {
      const auto float_attr = rewriter.getFloatAttr(type, const_val_);
      auto value = DenseElementsAttr::get(output.cast<ShapedType>(), float_attr);

      rewriter.replaceOpWithNewOp<tosa::ConstOp>(op, output, value);
    } else if (auto integerType = type.dyn_cast<IntegerType>()) {
      const auto int_attr =
          rewriter.getIntegerAttr(type, APInt(type.cast<IntegerType>().getWidth(), const_val_));
      auto value = DenseElementsAttr::get(output.cast<ShapedType>(), int_attr);

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
    auto output = op.getOut().getType();
    auto input = op.getIn();
    rewriter.replaceOpWithNewOp<tosa::CastOp>(op, output, input);
    return success();
  }
};

struct ReluOpLowering final : public OpConversionPattern<ReluOp> {
 public:
  using OpConversionPattern<ReluOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(ReluOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    const auto output = op.getY().getType();
    auto input = op.getX();

    auto ranked_output = llvm::dyn_cast_or_null<RankedTensorType>(output);
    auto value =
        DenseElementsAttr::get(output.cast<ShapedType>(),
                               rewriter.getZeroAttr(ranked_output ? ranked_output.getElementType()
                                                                  : rewriter.getI64Type()));
    tosa::ConstOp zeros = rewriter.create<tosa::ConstOp>(op.getLoc(), output, value);
    rewriter.replaceOpWithNewOp<tosa::MaximumOp>(op, output, input, zeros);
    return success();
  }
};

struct BroadcastAddOpLowering final : public OpConversionPattern<BroadcastAddOp> {
 public:
  using OpConversionPattern<BroadcastAddOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(BroadcastAddOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    const auto output = op.getZ().getType();
    auto input1 = op.getX();
    auto input2 = op.getY();

    rewriter.replaceOpWithNewOp<tosa::AddOp>(op, output, input1, input2);
    return success();
  }
};

struct Add2OpLowering final : public OpConversionPattern<Add2Op> {
 public:
  using OpConversionPattern<Add2Op>::OpConversionPattern;
  LogicalResult matchAndRewrite(Add2Op op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    const auto output = op.getOut().getType();
    auto input1 = op.getIn0();
    auto input2 = op.getIn1();

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

    auto stride_pairs = get_pair_int64_from_array(op.getStride());
    auto pad_pairs = get_pair_int64_from_array(op.getPadding());
    auto kernel_pairs = get_pair_int64_from_array(op.getKernelSize());

    auto loc = op.getLoc();
    auto perms = {0, 2, 3, 1};

    const auto kernel = rewriter.getDenseI64ArrayAttr({kernel_pairs.first, kernel_pairs.second});
    const auto stride = rewriter.getDenseI64ArrayAttr({stride_pairs.first, stride_pairs.second});
    const auto pad = rewriter.getDenseI64ArrayAttr(
        {pad_pairs.first, pad_pairs.second, pad_pairs.first, pad_pairs.second});

    auto input = CreateTransposeValue(loc, rewriter, op.getX(), perms);
    auto output = CreateTransposeType(op.getY().getType().cast<ShapedType>(), perms);

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
    if (op.getReturnIndices()) { return op->emitError("not support return indices now"); }
    auto stride_pairs = get_pair_int64_from_array(op.getStride());
    auto kernel_pairs = get_pair_int64_from_array(op.getKernelSize());
    auto pad_pairs = get_pair_int64_from_array(op.getPadding());

    auto loc = op.getLoc();

    const auto kernel = rewriter.getDenseI64ArrayAttr({kernel_pairs.first, kernel_pairs.second});
    const auto stride = rewriter.getDenseI64ArrayAttr({stride_pairs.first, stride_pairs.second});
    const auto pad = rewriter.getDenseI64ArrayAttr(
        {pad_pairs.first, pad_pairs.second, pad_pairs.first, pad_pairs.second});

    auto input = op.getX();
    auto out_type = op.getY().getType().cast<ShapedType>();

    Value y;
    if (op.IsNCHW()) {
      auto perms = {0, 2, 3, 1};
      auto reverse_perms = {0, 3, 1, 2};
      input = CreateTransposeValue(loc, rewriter, input, perms);
      out_type = CreateTransposeType(out_type, perms);
      auto max_pool2d =
          rewriter.create<tosa::MaxPool2dOp>(loc, out_type, input, kernel, stride, pad);
      y = CreateTransposeValue(loc, rewriter, max_pool2d, reverse_perms);
    } else {
      y = rewriter.create<tosa::MaxPool2dOp>(loc, out_type, input, kernel, stride, pad);
    }

    auto indice_output = convertToSignless(op->getContext(), op.getIndice().getType());
    auto value = DenseElementsAttr::get(indice_output.cast<ShapedType>(),
                                        rewriter.getZeroAttr(rewriter.getI64Type()));
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
    auto output = op.getOut().getType();
    auto input = op.getIn();
    llvm::SmallVector<int64_t> new_shape;
    for (const auto& dim_attr : op.getShape()) {
      new_shape.push_back(dim_attr.cast<IntegerAttr>().getSInt());
    }
    rewriter.replaceOpWithNewOp<tosa::ReshapeOp>(op, output, input,
                                                 rewriter.getDenseI64ArrayAttr(new_shape));
    return success();
  }
};

// transpose the last two dims of the tensor. Reshape it to 3D if it is 2D.
Value transposeAndReshapeIfRequired(Location loc, ConversionPatternRewriter& rewriter, Value matrix,
                                    bool transpose) {
  auto shape_type = matrix.getType().cast<ShapedType>();
  CHECK(shape_type.getRank() == 2 || shape_type.getRank() == 3);
  if (transpose) {
    if (shape_type.getRank() == 2) {
      matrix = CreateTransposeValue(loc, rewriter, matrix, {1, 0});
      shape_type = matrix.getType().cast<ShapedType>();
      llvm::SmallVector<int64_t, 4> reshape_dims{1, shape_type.getDimSize(0),
                                                 shape_type.getDimSize(1)};
      auto reshape_type = RankedTensorType::get(reshape_dims, shape_type.getElementType());
      return rewriter.create<tosa::ReshapeOp>(loc, reshape_type, matrix,
                                              rewriter.getDenseI64ArrayAttr(reshape_dims));
    } else if (shape_type.getRank() == 3) {
      return CreateTransposeValue(loc, rewriter, matrix, {0, 2, 1});
    } else {
      return Value{};
    }
  } else if (shape_type.getRank() == 2) {
    llvm::SmallVector<int64_t, 4> reshape_dims{1, shape_type.getDimSize(0),
                                               shape_type.getDimSize(1)};
    auto reshape_type = RankedTensorType::get(reshape_dims, shape_type.getElementType());
    return rewriter.create<tosa::ReshapeOp>(loc, reshape_type, matrix,
                                            rewriter.getDenseI64ArrayAttr(reshape_dims));
  }
  return matrix;
}

// Reshape: 2D -> 3D -> tosa.matmul -> 3D -> 2D
struct MatmulOpLowering final : public OpConversionPattern<MatmulOp> {
 public:
  using OpConversionPattern<MatmulOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(MatmulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    auto a = transposeAndReshapeIfRequired(op->getLoc(), rewriter, op.getA(), op.getTransposeA());
    auto b = transposeAndReshapeIfRequired(op->getLoc(), rewriter, op.getB(), op.getTransposeB());

    const auto out_shape_type = op.getOut().getType().cast<ShapedType>();
    const auto out_reshape_type =
        RankedTensorType::get({1, out_shape_type.getDimSize(0), out_shape_type.getDimSize(1)},
                              out_shape_type.getElementType());

    auto matmul = rewriter.create<tosa::MatMulOp>(op.getLoc(), out_reshape_type, a, b);
    const auto new_shape =
        rewriter.getDenseI64ArrayAttr({out_shape_type.getDimSize(0), out_shape_type.getDimSize(1)});

    rewriter.replaceOpWithNewOp<tosa::ReshapeOp>(op, out_shape_type, matmul, new_shape);
    return success();
  }
};

struct BatchMatmulOpLowering final : public OpConversionPattern<BatchMatmulOp> {
 public:
  using OpConversionPattern<BatchMatmulOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(BatchMatmulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    auto a = transposeAndReshapeIfRequired(op->getLoc(), rewriter, op.getA(), op.getTransposeA());
    auto b = transposeAndReshapeIfRequired(op->getLoc(), rewriter, op.getB(), op.getTransposeB());
    rewriter.replaceOpWithNewOp<tosa::MatMulOp>(op, op.getOut().getType(), a, b);
    return success();
  }
};

struct NormalizationInferenceOpLowering final
    : public OpConversionPattern<NormalizationInferenceOp> {
 public:
  using OpConversionPattern<NormalizationInferenceOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(NormalizationInferenceOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    auto loc = op->getLoc();

    const auto epsilon_type = RankedTensorType::get({}, rewriter.getF32Type());
    auto epsilon = rewriter.create<tosa::ConstOp>(
        loc, epsilon_type, DenseElementsAttr::get(epsilon_type, op.getEpsilon()));
    auto mean = op.getMovingMean();
    auto variance = op.getMovingVariance();
    auto gamma = op.getGamma();
    auto beta = op.getBeta();
    auto output_type = op.getY().getType();
    Value x = op.getX();

    if (op.IsNCHW()) {
      const auto perms = {0, 2, 3, 1};
      x = CreateTransposeValue(loc, rewriter, x, perms);
      output_type = CreateTransposeType(output_type, perms);
    }

    auto batch_norm =
        oneflow::CreateBNOp(loc, rewriter, output_type, x, mean, variance, epsilon, gamma, beta);

    if (op.IsNCHW()) {
      const auto reverse_perms = {0, 3, 1, 2};
      batch_norm = CreateTransposeValue(loc, rewriter, batch_norm, reverse_perms);
    }
    rewriter.replaceOp(op, {batch_norm});
    return success();
  }
};

struct NormalizationOpLowering final : public OpConversionPattern<NormalizationOp> {
 public:
  using OpConversionPattern<NormalizationOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(NormalizationOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    auto loc = op->getLoc();

    const auto epsilon_type = RankedTensorType::get({}, rewriter.getF32Type());
    auto epsilon = rewriter.create<tosa::ConstOp>(
        loc, epsilon_type, DenseElementsAttr::get(epsilon_type, op.getEpsilon()));
    auto mean = op.getMovingMean();
    auto variance = op.getMovingVariance();
    auto gamma = op.getGamma();
    auto beta = op.getBeta();
    auto output_type = op.getY().getType();
    Value x = op.getX();

    if (op.IsNCHW()) {
      const auto perms = {0, 2, 3, 1};
      x = CreateTransposeValue(loc, rewriter, x, perms);
      output_type = CreateTransposeType(output_type, perms);
    }

    auto batch_norm =
        oneflow::CreateBNOp(loc, rewriter, output_type, x, mean, variance, epsilon, gamma, beta);

    if (op.IsNCHW()) {
      const auto reverse_perms = {0, 3, 1, 2};
      batch_norm = CreateTransposeValue(loc, rewriter, batch_norm, reverse_perms);
    }
    auto moving_mean = op.getMovingMean();
    auto moving_variance = op.getMovingVariance();

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

    auto stride_pairs = get_pair_int64_from_array(op.getStrides());
    auto pad_pairs = get_pair_int64_from_array(op.getPaddingBeforeAttr());
    auto dilation_pairs = get_pair_int64_from_array(op.getDilationRate());

    const auto pad = rewriter.getDenseI64ArrayAttr(
        {pad_pairs.first, pad_pairs.second, pad_pairs.first, pad_pairs.second});
    const auto stride = rewriter.getDenseI64ArrayAttr({stride_pairs.first, stride_pairs.second});
    const auto dilation =
        rewriter.getDenseI64ArrayAttr({dilation_pairs.first, dilation_pairs.second});

    auto bias = op.getBias();
    auto loc = op.getLoc();
    if (!bias) {
      const auto output_shape = op.getOut().getType().cast<ShapedType>();
      // support nhwc
      const auto output_channels = output_shape.getDimSize(op.IsNCHW() ? 1 : 3);
      const auto bias_elem_type = output_shape.getElementType();
      const auto type = RankedTensorType::get(output_channels, bias_elem_type);
      bias = rewriter.create<tosa::ConstOp>(
          op.getLoc(), type, DenseElementsAttr::get(type, rewriter.getZeroAttr(bias_elem_type)));
    }

    Value in = op.getIn();
    Value weight = op.getWeight();
    auto out_type = op.getOut().getType().cast<ShapedType>();
    if (out_type.getRank() != 4) {
      LOG(FATAL) << "Failed to lowering oneflow op";
      op->dump();
    }
    // support nhwc
    if (op.IsNCHW()) {
      const auto perms = {0, 2, 3, 1};
      const auto reverse_perms = {0, 3, 1, 2};
      in = CreateTransposeValue(loc, rewriter, in, perms);
      weight = CreateTransposeValue(loc, rewriter, weight, perms);
      out_type = CreateTransposeType(out_type, perms);
      auto conv2d =
          rewriter.create<tosa::Conv2DOp>(loc, out_type, in, weight, bias, pad, stride, dilation);

      auto res = CreateTransposeValue(loc, rewriter, conv2d, reverse_perms);
      rewriter.replaceOp(op, {res});
    } else {
      rewriter.replaceOpWithNewOp<tosa::Conv2DOp>(op, out_type, in, weight, bias, pad, stride,
                                                  dilation);
    }
    return success();
  }
};

struct TransposeOpLowering final : public OpConversionPattern<TransposeOp> {
 public:
  using OpConversionPattern<TransposeOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(TransposeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    llvm::SmallVector<int32_t, 4> perms{};
    for (auto dim : op.getPerm().getAsValueRange<mlir::IntegerAttr>()) {
      perms.push_back(dim.getSExtValue());
    }
    llvm::SmallVector<int64_t, 4> perms_shape(op.getPerm().size(), 1);
    auto perms_op = rewriter.create<tosa::ConstOp>(
        op->getLoc(), RankedTensorType::get(perms_shape, rewriter.getI32Type()),
        rewriter.getI32TensorAttr(perms));
    rewriter.replaceOpWithNewOp<tosa::TransposeOp>(op, op.getOutput().getType(), op.getInput(),
                                                   perms_op.getOutput());
    return success();
  }
};

struct CastInputConversion final : public OpRewritePattern<InputOp> {
 public:
  explicit CastInputConversion(mlir::MLIRContext* context)
      : OpRewritePattern<InputOp>(context, /*benefit=*/0) {}
  mlir::LogicalResult matchAndRewrite(InputOp op, mlir::PatternRewriter& rewriter) const override {
    auto outType = op.getOutput().getType();
    if (isSignLessTensorOrOther(outType)) { return failure(); }
    if (op->hasOneUse()) {
      if (auto cast =
              llvm::dyn_cast<UnrealizedConversionCastOp>(op.getOutput().use_begin()->getOwner())) {
        if (isSignLessTensorOrOther(cast.getResult(0).getType())) { return failure(); }
      }
    }
    InputOp cloned = rewriter.create<InputOp>(op->getLoc(), op.getResultTypes(), op->getOperands(),
                                              op->getAttrs());
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        op, convertToSignless(getContext(), op.getOutput().getType()), cloned.getOutput());
    return success();
  }
};

struct CastVariableConversion final : public OpRewritePattern<VariableOp> {
 public:
  explicit CastVariableConversion(mlir::MLIRContext* context)
      : OpRewritePattern<VariableOp>(context, /*benefit=*/0) {}
  mlir::LogicalResult matchAndRewrite(VariableOp op,
                                      mlir::PatternRewriter& rewriter) const override {
    auto outType = op.getOutput().getType();
    if (isSignLessTensorOrOther(outType)) { return failure(); }
    if (op->hasOneUse()) {
      if (auto cast =
              llvm::dyn_cast<UnrealizedConversionCastOp>(op.getOutput().use_begin()->getOwner())) {
        if (isSignLessTensorOrOther(cast.getResult(0).getType())) { return failure(); }
      }
    }
    if (op.getOutput().getUses().empty()) { return failure(); }
    VariableOp cloned = rewriter.create<VariableOp>(op->getLoc(), op.getResultTypes(),
                                                    op->getOperands(), op->getAttrs());
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        op, convertToSignless(getContext(), op.getOutput().getType()), cloned.getOutput());
    return success();
  }
};

namespace {

class CastOneFlowOpsToSignlessPass
    : public CastOneFlowOpsToSignlessPassBase<CastOneFlowOpsToSignlessPass> {
  void getDependentDialects(::mlir::DialectRegistry& registry) const override {
    registry.insert<oneflow::OneFlowDialect>();
  }
  void runOnOperation() override {
    Operation* op = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<oneflow::CastInputConversion, oneflow::CastVariableConversion>(op->getContext());

    (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
  }
};

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
                         tensor::TensorDialect, arith::ArithDialect, BuiltinDialect>();
  if (fullyConvert) { target.addIllegalDialect<OneFlowDialect>(); }

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

  // check if the pass is triggered by python based on the presence of variable tensor manger
  if (fullyConvert) {
    if (::oneflow::Singleton<::oneflow::VariableTensorMgr>::Get()) {
      patterns.add<VariableOpLowering>(typeConverter, context);
    } else {
      patterns.add<VariableOpToConstLowering>(typeConverter, context, this->variableAsConstant);
    }
  }
  patterns.add<CastOpLowering, ScalarMulByTensorOpLowering, ReluOpLowering, Conv2DOpLowering,
               AvgPool2DOpLowering, ReshapeOpLowering, Add2OpLowering, MaxPool2DOpLowering,
               MatmulOpLowering, BatchMatmulOpLowering, BroadcastAddOpLowering,
               NormalizationOpLowering, NormalizationInferenceOpLowering, TransposeOpLowering>(
      typeConverter, context);
  if (lowerJob) {
    patterns.add<InputOpLowering, OutputOpLowering, JobLowering, ReturnOpLowering>(typeConverter,
                                                                                   context);
  }
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
    auto uc = rewriter.create<UnrealizedConversionCastOp>(op->getLoc(), results, op.getOperands());
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
    IRMapping bvm;
    op.getRegion().cloneInto(&func.getRegion(), bvm);
    for (auto& block : func.getBody().getBlocks()) {
      for (auto arg : block.getArguments()) {
        auto new_type = convertToSignless(op.getContext(), arg.getType());
        arg.setType(new_type);
        for (auto* use : arg.getUsers()) {
          if (auto input = llvm::dyn_cast_or_null<InputOp>(use)) {
            input.getOutput().setType(new_type);
          }
        }
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

std::unique_ptr<Pass> createCastOneFlowOpsToSignlessPass() {
  return std::make_unique<CastOneFlowOpsToSignlessPass>();
}

}  // namespace oneflow

}  // namespace mlir

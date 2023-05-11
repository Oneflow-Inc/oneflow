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
#include "OneFlow/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

namespace oneflow {

namespace {

std::tuple<SmallVector<::mlir::utils::IteratorType>, SmallVector<AffineMap>>
computeIteratorTypesAndIndexingMaps(int64_t inputRank, int64_t dim, OpBuilder& builder,
                                    bool allParallel = false) {
  SmallVector<::mlir::utils::IteratorType> iteratorTypes(inputRank,
                                                         ::mlir::utils::IteratorType::parallel);
  if (!allParallel) iteratorTypes[dim] = ::mlir::utils::IteratorType::reduction;
  auto identityMap = AffineMap::getMultiDimIdentityMap(inputRank, builder.getContext());
  SmallVector<AffineExpr, 2> affineExprs;
  for (int i = 0; i < inputRank; i++) {
    if (i != dim) affineExprs.push_back(mlir::getAffineDimExpr(i, builder.getContext()));
  }
  auto reductionMap = AffineMap::get(inputRank, 0, affineExprs, builder.getContext());
  SmallVector<AffineMap> indexingMaps{identityMap, reductionMap};
  return std::make_tuple(iteratorTypes, indexingMaps);
}

template<typename T>
static Value reduce(Value input, Value output, int64_t dim, Location loc, OpBuilder& builder) {
  auto inputType = input.getType().cast<ShapedType>();
  ArrayRef<int64_t> inputShape = inputType.getShape();
  int64_t inputRank = inputShape.size();
  auto [iteratorTypes, indexingMaps] = computeIteratorTypesAndIndexingMaps(inputRank, dim, builder);
  auto genericOp = builder.create<linalg::GenericOp>(
      loc, output.getType(), input, output, indexingMaps, iteratorTypes,
      [&](OpBuilder& b, Location loc, ValueRange args) {
        Value result = b.create<T>(loc, args[0], args[1]);
        b.create<linalg::YieldOp>(loc, result);
      });
  return genericOp.getResult(0);
}

static Value subtractAndExp(Value input, Value max, Value output, int64_t dim, Location loc,
                            OpBuilder& builder) {
  auto inputType = input.getType().cast<ShapedType>();
  ArrayRef<int64_t> inputShape = inputType.getShape();
  int64_t inputRank = inputShape.size();
  auto [iteratorTypes, indexingMaps] =
      computeIteratorTypesAndIndexingMaps(inputRank, dim, builder, true);
  indexingMaps.push_back(indexingMaps[0]);
  auto genericOp = builder.create<linalg::GenericOp>(
      loc, input.getType(), ValueRange{input, max}, output, indexingMaps, iteratorTypes,
      [&](OpBuilder& b, Location loc, ValueRange args) {
        Value diff = b.create<arith::SubFOp>(loc, args[0], args[1]);
        Value result = b.create<math::ExpOp>(loc, diff);
        b.create<linalg::YieldOp>(loc, result);
      });
  return genericOp.getResult(0);
}

static Value computeSoftmax(Value numerator, Value denominator, Value output, int64_t dim,
                            Location loc, OpBuilder& builder) {
  auto inputType = numerator.getType().cast<ShapedType>();
  ArrayRef<int64_t> inputShape = inputType.getShape();
  int64_t inputRank = inputShape.size();
  auto [iteratorTypes, indexingMaps] =
      computeIteratorTypesAndIndexingMaps(inputRank, dim, builder, true);
  indexingMaps.push_back(indexingMaps[0]);
  auto genericOp = builder.create<linalg::GenericOp>(
      loc, numerator.getType(), ValueRange{numerator, denominator}, output, indexingMaps,
      iteratorTypes, [&](OpBuilder& b, Location loc, ValueRange args) {
        Value result = b.create<arith::DivFOp>(loc, args[0], args[1]);
        b.create<linalg::YieldOp>(loc, result);
      });
  return genericOp.getResult(0);
}

/// Given an N-dimensional tensor x, this op converts
/// softmax(x) to the following sequence of operations:
///
/// 1. Compute the max of x along dimension d. This results
///    in a N-1 dimensional tensor m.
///    m = max(x, dim = d)
///
/// 2. Subtract m from x and exponentiate. This results in
///    a N dimensional tensor z.
///    z = exp(x - m)
///
/// 3. Compute the sum of z along dimension d. This results in
///    a N-1 dimensional tensor l.
///    l = sum(z, dim = d)
///
/// 4. Divide z and l. This gives the N-dimensional softmax.
///    softmax = z / l
///

// Implementation above is from IREE.
// https://github.com/google/iree/blob/b339919814f10589f779b39c3ab7c6575716dab6/llvm-external-projects/iree-dialects/lib/Dialect/LinalgExt/Passes/DecomposeSoftmax.cpp

SmallVector<OpFoldResult> createDimValues(OpBuilder& b, Location loc, Value rankedTensor) {
  auto tensorTy = rankedTensor.getType().cast<RankedTensorType>();
  SmallVector<OpFoldResult> dims;
  for (const auto& en : llvm::enumerate(tensorTy.getShape())) {
    if (ShapedType::isDynamic(en.value())) {
      dims.push_back(b.createOrFold<tensor::DimOp>(loc, rankedTensor, en.index()));
    } else {
      dims.push_back(b.getIndexAttr(en.value()));
    }
  }
  return dims;
}

struct SoftmaxOpLowering final : public OpConversionPattern<SoftmaxOp> {
 public:
  using OpConversionPattern<SoftmaxOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(SoftmaxOp softmaxOp, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(softmaxOp);
    Location loc = softmaxOp.getLoc();
    Value input = softmaxOp.getIn();
    ShapedType inputType = input.getType().cast<ShapedType>();
    Type elementType = inputType.getElementType();
    int64_t reductionDim = inputType.getRank() - 1;
    SmallVector<OpFoldResult> dims = createDimValues(rewriter, loc, input);
    Value outputNd = rewriter.create<tensor::EmptyOp>(loc, dims, elementType);
    dims.erase(dims.begin() + reductionDim);
    // Compute max along dim
    Value output = rewriter.create<tensor::EmptyOp>(loc, dims, elementType);
    Value largeNegative =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(elementType, -1.0e30));
    Value negativeInit =
        rewriter.create<linalg::FillOp>(loc, Value{largeNegative}, output).result();
    Value max = reduce<arith::MaxFOp>(input, negativeInit, reductionDim, loc, rewriter);
    // Subtract max from input and exponentiate
    Value numerator = subtractAndExp(input, max, outputNd, reductionDim, loc, rewriter);
    // Compute sum along dim
    Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(elementType));
    Value zeroInit = rewriter.create<linalg::FillOp>(loc, Value{zero}, output).result();
    Value denominator = reduce<arith::AddFOp>(numerator, zeroInit, reductionDim, loc, rewriter);
    // Compute softmax
    Value result = computeSoftmax(numerator, denominator, outputNd, reductionDim, loc, rewriter);
    rewriter.replaceOp(softmaxOp, {result});
    return success();
  }
};

struct OneFlowLoweringToLinalgPass
    : public LowerOneFlowToLinalgPassBase<OneFlowLoweringToLinalgPass> {
  void runOnOperation() {
    MLIRContext* context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<memref::MemRefDialect, mlir::func::FuncDialect, tosa::TosaDialect,
                           linalg::LinalgDialect, tensor::TensorDialect, arith::ArithDialect,
                           math::MathDialect>();
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

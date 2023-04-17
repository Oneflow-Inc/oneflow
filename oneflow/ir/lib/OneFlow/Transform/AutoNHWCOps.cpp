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
#include "OneFlow/Transform/TransposeHelpers.h"

namespace mlir {

namespace oneflow {

bool Conv2DOp::IsNCHW() { return this->getDataFormat().str() == "channels_first"; }

llvm::DenseSet<Value> Conv2DOp::OperandsToTranspose() {
  if (this->get_addToOutput()) {
    return {this->getIn(), this->getWeight(), this->get_addToOutput()};
  } else {
    return {this->getIn(), this->getWeight()};
  }
}

llvm::DenseSet<Value> Conv2DOp::ResultsToTranspose() { return {this->getOut()}; }

llvm::SmallVector<Value, 4> Conv2DOp::NchwToNhwc(llvm::SmallVector<Value, 4> value,
                                                 PatternRewriter& rewriter) {
  auto conv_op = *this;
  SmallVector<Value, 4> operands;
  operands.push_back(value[0]);
  operands.push_back(value[1]);
  if (conv_op.getBias()) operands.push_back(conv_op.getBias());
  if (this->get_addToOutput()) { operands.push_back(value[2]); }
  NamedAttrList attributes = conv_op->getAttrs();
  attributes.set(conv_op.getDataFormatAttrName(), rewriter.getStringAttr("channels_last"));
  auto res = rewriter
                 .create<oneflow::Conv2DOp>(conv_op.getLoc(), getNHWCResultTypes(conv_op), operands,
                                            attributes)
                 ->getResults();
  llvm::SmallVector<Value, 4> results;
  results.push_back(res[0]);
  return results;
}

bool BiasAddOp::IsNCHW() { return this->getAxisAttr().getValue().getSExtValue() == 1; }

llvm::DenseSet<Value> BiasAddOp::OperandsToTranspose() { return {this->getA()}; }

llvm::DenseSet<Value> BiasAddOp::ResultsToTranspose() { return {this->getOut()}; }

llvm::SmallVector<Value, 4> BiasAddOp::NchwToNhwc(llvm::SmallVector<Value, 4> value,
                                                  PatternRewriter& rewriter) {
  auto bias_add_op = *this;
  SmallVector<Value, 4> operands;
  operands.push_back(value[0]);
  operands.push_back(bias_add_op.getB());
  NamedAttrList attributes = bias_add_op->getAttrs();
  attributes.set(bias_add_op.getAxisAttrName(), rewriter.getSI32IntegerAttr(3));
  auto res = rewriter
                 .create<oneflow::BiasAddOp>(bias_add_op.getLoc(), getNHWCResultTypes(bias_add_op),
                                             operands, attributes)
                 ->getResults();
  llvm::SmallVector<Value, 4> results;
  results.push_back(res[0]);
  return results;
}

bool BroadcastAddOp::IsNCHW() { return false; }

llvm::DenseSet<Value> BroadcastAddOp::OperandsToTranspose() { return {this->getX(), this->getY()}; }

llvm::DenseSet<Value> BroadcastAddOp::ResultsToTranspose() { return {this->getZ()}; }

llvm::SmallVector<Value, 4> BroadcastAddOp::NchwToNhwc(llvm::SmallVector<Value, 4> values,
                                                       PatternRewriter& rewriter) {
  auto broadcast_op = *this;
  NamedAttrList attributes = broadcast_op->getAttrs();
  auto res = rewriter
                 .create<oneflow::BroadcastAddOp>(
                     broadcast_op.getLoc(), getNHWCResultTypes(broadcast_op), values, attributes)
                 .getZ();
  llvm::SmallVector<Value, 4> results;
  results.push_back(res);
  return results;
}

bool NormalizationOp::IsNCHW() { return this->getAxisAttr().getValue().getSExtValue() == 1; }

bool NormalizationInferenceOp::IsNCHW() {
  return this->getAxisAttr().getValue().getSExtValue() == 1;
}

llvm::DenseSet<Value> NormalizationOp::OperandsToTranspose() { return {this->getX()}; }

llvm::DenseSet<Value> NormalizationInferenceOp::OperandsToTranspose() { return {this->getX()}; }

llvm::DenseSet<Value> NormalizationOp::ResultsToTranspose() { return {this->getY()}; }

llvm::DenseSet<Value> NormalizationInferenceOp::ResultsToTranspose() { return {this->getY()}; }

llvm::SmallVector<Value, 4> NormalizationOp::NchwToNhwc(llvm::SmallVector<Value, 4> value,
                                                        PatternRewriter& rewriter) {
  auto normalization_op = *this;
  SmallVector<Value, 4> operands;
  operands.push_back(value[0]);
  if (normalization_op.getMovingMean()) operands.push_back(normalization_op.getMovingMean());
  if (normalization_op.getMovingVariance())
    operands.push_back(normalization_op.getMovingVariance());
  operands.push_back(normalization_op.getGamma());
  operands.push_back(normalization_op.getBeta());
  if (normalization_op.get_addToOutput()) operands.push_back(normalization_op.get_addToOutput());
  NamedAttrList attributes = normalization_op->getAttrs();
  attributes.set(normalization_op.getAxisAttrName(), rewriter.getSI32IntegerAttr(3));
  auto res =
      rewriter
          .create<oneflow::NormalizationOp>(
              normalization_op.getLoc(), getNHWCResultTypes(normalization_op), operands, attributes)
          ->getResults();
  llvm::SmallVector<Value, 4> results;
  results.push_back(res[0]);
  return results;
}

llvm::SmallVector<Value, 4> NormalizationInferenceOp::NchwToNhwc(llvm::SmallVector<Value, 4> value,
                                                                 PatternRewriter& rewriter) {
  auto normalization_op = *this;
  SmallVector<Value, 4> operands;
  operands.push_back(value[0]);
  if (normalization_op.getMovingMean()) operands.push_back(normalization_op.getMovingMean());
  if (normalization_op.getMovingVariance())
    operands.push_back(normalization_op.getMovingVariance());
  operands.push_back(normalization_op.getGamma());
  operands.push_back(normalization_op.getBeta());
  if (normalization_op.get_addToOutput()) operands.push_back(normalization_op.get_addToOutput());
  NamedAttrList attributes = normalization_op->getAttrs();
  attributes.set(normalization_op.getAxisAttrName(), rewriter.getSI32IntegerAttr(3));
  auto res =
      rewriter
          .create<oneflow::NormalizationInferenceOp>(
              normalization_op.getLoc(), getNHWCResultTypes(normalization_op), operands, attributes)
          ->getResults();
  llvm::SmallVector<Value, 4> results;
  results.push_back(res[0]);
  return results;
}

bool MaxPool2DOp::IsNCHW() { return this->getDataFormat().str() == "channels_first"; }

llvm::DenseSet<Value> MaxPool2DOp::OperandsToTranspose() { return {this->getX()}; }

llvm::DenseSet<Value> MaxPool2DOp::ResultsToTranspose() {
  return {this->getY(), this->getIndice()};
}

llvm::SmallVector<Value, 4> MaxPool2DOp::NchwToNhwc(llvm::SmallVector<Value, 4> value,
                                                    PatternRewriter& rewriter) {
  auto max_pool_2d_op = *this;
  SmallVector<Value, 4> operands;
  operands.push_back(value[0]);
  NamedAttrList attributes = max_pool_2d_op->getAttrs();
  attributes.set(max_pool_2d_op.getDataFormatAttrName(), rewriter.getStringAttr("channels_last"));
  auto res =
      rewriter
          .create<oneflow::MaxPool2DOp>(max_pool_2d_op.getLoc(), getNHWCResultTypes(max_pool_2d_op),
                                        operands, attributes)
          ->getResults();
  llvm::SmallVector<Value, 4> results;
  results.push_back(res[0]);
  results.push_back(res[1]);
  return results;
}

bool ReluOp::IsNCHW() { return false; }

llvm::DenseSet<Value> ReluOp::OperandsToTranspose() { return {this->getX()}; }

llvm::DenseSet<Value> ReluOp::ResultsToTranspose() { return {this->getY()}; }

llvm::SmallVector<Value, 4> ReluOp::NchwToNhwc(llvm::SmallVector<Value, 4> value,
                                               PatternRewriter& rewriter) {
  auto relu_op = *this;
  SmallVector<Value, 4> operands{value[0]};
  auto res = rewriter
                 .create<oneflow::ReluOp>(relu_op.getLoc(), getNHWCResultTypes(relu_op), operands,
                                          relu_op->getAttrs())
                 ->getResults();
  return {res[0]};
}

bool ScalarDivOp::IsNCHW() { return false; }

llvm::DenseSet<Value> ScalarDivOp::OperandsToTranspose() { return {this->getIn()}; }

llvm::DenseSet<Value> ScalarDivOp::ResultsToTranspose() { return {this->getOut()}; }

llvm::SmallVector<Value, 4> ScalarDivOp::NchwToNhwc(llvm::SmallVector<Value, 4> value,
                                                    PatternRewriter& rewriter) {
  auto elementwise_op = *this;
  SmallVector<Value, 4> operands{value[0]};
  auto res =
      rewriter
          .create<oneflow::ScalarDivOp>(elementwise_op.getLoc(), getNHWCResultTypes(elementwise_op),
                                        operands, elementwise_op->getAttrs())
          ->getResults();
  return {res[0]};
}

bool SiluOp::IsNCHW() { return false; }

llvm::DenseSet<Value> SiluOp::OperandsToTranspose() { return {this->getIn()}; }

llvm::DenseSet<Value> SiluOp::ResultsToTranspose() { return {this->getOut()}; }

llvm::SmallVector<Value, 4> SiluOp::NchwToNhwc(llvm::SmallVector<Value, 4> value,
                                               PatternRewriter& rewriter) {
  auto elementwise_op = *this;
  SmallVector<Value, 4> operands{value[0]};
  auto res =
      rewriter
          .create<oneflow::SiluOp>(elementwise_op.getLoc(), getNHWCResultTypes(elementwise_op),
                                   operands, elementwise_op->getAttrs())
          ->getResults();
  return {res[0]};
}

bool CastOp::IsNCHW() { return false; }

llvm::DenseSet<Value> CastOp::OperandsToTranspose() { return {this->getIn()}; }

llvm::DenseSet<Value> CastOp::ResultsToTranspose() { return {this->getOut()}; }

llvm::SmallVector<Value, 4> CastOp::NchwToNhwc(llvm::SmallVector<Value, 4> value,
                                               PatternRewriter& rewriter) {
  auto elementwise_op = *this;
  SmallVector<Value, 4> operands{value[0]};
  auto res =
      rewriter
          .create<oneflow::CastOp>(elementwise_op.getLoc(), getNHWCResultTypes(elementwise_op),
                                   operands, elementwise_op->getAttrs())
          ->getResults();
  return {res[0]};
}

bool Add2Op::IsNCHW() { return false; }

llvm::DenseSet<Value> Add2Op::OperandsToTranspose() { return {this->getIn0(), this->getIn1()}; }

llvm::DenseSet<Value> Add2Op::ResultsToTranspose() { return {this->getOut()}; }

llvm::SmallVector<Value, 4> Add2Op::NchwToNhwc(llvm::SmallVector<Value, 4> value,
                                               PatternRewriter& rewriter) {
  auto add2_op = *this;
  SmallVector<Value, 4> operands{value[0], value[1]};
  auto res = rewriter
                 .create<oneflow::Add2Op>(add2_op.getLoc(), getNHWCResultTypes(add2_op), operands,
                                          add2_op->getAttrs())
                 ->getResults();
  return {res[0]};
}

bool ConcatOp::IsNCHW() { return this->getAxisAttr().getValue().getSExtValue() == 1; }

llvm::DenseSet<Value> ConcatOp::OperandsToTranspose() {
  llvm::DenseSet<Value> operands;
  for (auto operand : this->getIn()) { operands.insert(operand); }
  return operands;
}

llvm::DenseSet<Value> ConcatOp::ResultsToTranspose() { return {this->getOut()}; }

llvm::SmallVector<Value, 4> ConcatOp::NchwToNhwc(llvm::SmallVector<Value, 4> values,
                                                 PatternRewriter& rewriter) {
  auto elementwise_op = *this;
  NamedAttrList attributes = elementwise_op->getAttrs();
  attributes.set(elementwise_op.getAxisAttrName(),
                 IntegerAttr::get(rewriter.getIntegerType(64, /*isSigned=*/true),
                                  APInt(64, 3, /*isSigned=*/true)));
  auto out = rewriter
                 .create<oneflow::ConcatOp>(elementwise_op.getLoc(),
                                            getNHWCResultTypes(elementwise_op), values, attributes)
                 .getOut();
  return {out};
}

bool GroupNormOp::IsNCHW() { return this->getDataFormat().str() == "channels_first"; }

llvm::DenseSet<Value> GroupNormOp::OperandsToTranspose() { return {this->getX()}; }

llvm::DenseSet<Value> GroupNormOp::ResultsToTranspose() { return {this->getY()}; }

llvm::SmallVector<Value, 4> GroupNormOp::NchwToNhwc(llvm::SmallVector<Value, 4> value,
                                                    PatternRewriter& rewriter) {
  auto group_norm_op = *this;
  SmallVector<Value, 4> operands;
  operands.push_back(value[0]);
  if (this->getAffine()) {
    operands.push_back(this->getBeta());
    operands.push_back(this->getGamma());
  }
  NamedAttrList attributes = group_norm_op->getAttrs();
  attributes.set(group_norm_op.getDataFormatAttrName(), rewriter.getStringAttr("channels_last"));
  auto res =
      rewriter
          .create<oneflow::GroupNormOp>(group_norm_op.getLoc(), getNHWCResultTypes(group_norm_op),
                                        operands, attributes)
          ->getResults();
  llvm::SmallVector<Value, 4> results;
  results.push_back(res[0]);
  results.push_back(res[1]);
  results.push_back(res[2]);
  return results;
}

}  // namespace oneflow

}  // namespace mlir

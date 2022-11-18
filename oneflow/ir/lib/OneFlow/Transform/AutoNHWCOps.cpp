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

bool Conv2DOp::IsNCHW() { return this->data_format().str() == "channels_first"; }

llvm::DenseSet<Value> Conv2DOp::OperandsToTranspose() {
  if (this->_add_to_output()) {
    return {this->in(), this->weight(), this->_add_to_output()};
  } else {
    return {this->in(), this->weight()};
  }
}

llvm::DenseSet<Value> Conv2DOp::ResultsToTranspose() { return {this->out()}; }

llvm::SmallVector<Value, 4> Conv2DOp::NchwToNhwc(llvm::SmallVector<Value, 4> value,
                                                 PatternRewriter& rewriter) {
  auto conv_op = *this;
  SmallVector<Value, 4> operands;
  operands.push_back(value[0]);
  operands.push_back(value[1]);
  if (conv_op.bias()) operands.push_back(conv_op.bias());
  if (this->_add_to_output()) { operands.push_back(value[2]); }
  NamedAttrList attributes = conv_op->getAttrs();
  attributes.set(conv_op.data_formatAttrName(), rewriter.getStringAttr("channels_last"));
  auto res = rewriter
                 .create<oneflow::Conv2DOp>(conv_op.getLoc(), getNHWCResultTypes(conv_op), operands,
                                            attributes)
                 ->getResults();
  llvm::SmallVector<Value, 4> results;
  results.push_back(res[0]);
  return results;
}

bool BiasAddOp::IsNCHW() { return this->axisAttr().getValue().getSExtValue() == 1; }

llvm::DenseSet<Value> BiasAddOp::OperandsToTranspose() { return {this->a()}; }

llvm::DenseSet<Value> BiasAddOp::ResultsToTranspose() { return {this->out()}; }

llvm::SmallVector<Value, 4> BiasAddOp::NchwToNhwc(llvm::SmallVector<Value, 4> value,
                                                  PatternRewriter& rewriter) {
  auto bias_add_op = *this;
  SmallVector<Value, 4> operands;
  operands.push_back(value[0]);
  operands.push_back(bias_add_op.b());
  NamedAttrList attributes = bias_add_op->getAttrs();
  attributes.set(bias_add_op.axisAttrName(), rewriter.getSI32IntegerAttr(3));
  auto res = rewriter
                 .create<oneflow::BiasAddOp>(bias_add_op.getLoc(), getNHWCResultTypes(bias_add_op),
                                             operands, attributes)
                 ->getResults();
  llvm::SmallVector<Value, 4> results;
  results.push_back(res[0]);
  return results;
}

bool BroadcastAddOp::IsNCHW() { return false; }

llvm::DenseSet<Value> BroadcastAddOp::OperandsToTranspose() { return {this->x(), this->y()}; }

llvm::DenseSet<Value> BroadcastAddOp::ResultsToTranspose() { return {this->z()}; }

llvm::SmallVector<Value, 4> BroadcastAddOp::NchwToNhwc(llvm::SmallVector<Value, 4> values,
                                                       PatternRewriter& rewriter) {
  auto broadcast_op = *this;
  NamedAttrList attributes = broadcast_op->getAttrs();
  auto res = rewriter
                 .create<oneflow::BroadcastAddOp>(
                     broadcast_op.getLoc(), getNHWCResultTypes(broadcast_op), values, attributes)
                 .z();
  llvm::SmallVector<Value, 4> results;
  results.push_back(res);
  return results;
}

bool NormalizationOp::IsNCHW() { return this->axisAttr().getValue().getSExtValue() == 1; }

llvm::DenseSet<Value> NormalizationOp::OperandsToTranspose() { return {this->x()}; }

llvm::DenseSet<Value> NormalizationOp::ResultsToTranspose() { return {this->y()}; }

llvm::SmallVector<Value, 4> NormalizationOp::NchwToNhwc(llvm::SmallVector<Value, 4> value,
                                                        PatternRewriter& rewriter) {
  auto normalization_op = *this;
  SmallVector<Value, 4> operands;
  operands.push_back(value[0]);
  if (normalization_op.moving_mean()) operands.push_back(normalization_op.moving_mean());
  if (normalization_op.moving_variance()) operands.push_back(normalization_op.moving_variance());
  operands.push_back(normalization_op.gamma());
  operands.push_back(normalization_op.beta());
  if (normalization_op._add_to_output()) operands.push_back(normalization_op._add_to_output());
  NamedAttrList attributes = normalization_op->getAttrs();
  attributes.set(normalization_op.axisAttrName(), rewriter.getSI32IntegerAttr(3));
  auto res =
      rewriter
          .create<oneflow::NormalizationOp>(
              normalization_op.getLoc(), getNHWCResultTypes(normalization_op), operands, attributes)
          ->getResults();
  llvm::SmallVector<Value, 4> results;
  results.push_back(res[0]);
  return results;
}

bool MaxPool2DOp::IsNCHW() { return this->data_format().str() == "channels_first"; }

llvm::DenseSet<Value> MaxPool2DOp::OperandsToTranspose() { return {this->x()}; }

llvm::DenseSet<Value> MaxPool2DOp::ResultsToTranspose() { return {this->y(), this->indice()}; }

llvm::SmallVector<Value, 4> MaxPool2DOp::NchwToNhwc(llvm::SmallVector<Value, 4> value,
                                                    PatternRewriter& rewriter) {
  auto max_pool_2d_op = *this;
  SmallVector<Value, 4> operands;
  operands.push_back(value[0]);
  NamedAttrList attributes = max_pool_2d_op->getAttrs();
  attributes.set(max_pool_2d_op.data_formatAttrName(), rewriter.getStringAttr("channels_last"));
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

llvm::DenseSet<Value> ReluOp::OperandsToTranspose() { return {this->x()}; }

llvm::DenseSet<Value> ReluOp::ResultsToTranspose() { return {this->y()}; }

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

llvm::DenseSet<Value> ScalarDivOp::OperandsToTranspose() { return {this->in()}; }

llvm::DenseSet<Value> ScalarDivOp::ResultsToTranspose() { return {this->out()}; }

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

llvm::DenseSet<Value> SiluOp::OperandsToTranspose() { return {this->in()}; }

llvm::DenseSet<Value> SiluOp::ResultsToTranspose() { return {this->out()}; }

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

llvm::DenseSet<Value> CastOp::OperandsToTranspose() { return {this->in()}; }

llvm::DenseSet<Value> CastOp::ResultsToTranspose() { return {this->out()}; }

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

llvm::DenseSet<Value> Add2Op::OperandsToTranspose() { return {this->in0(), this->in1()}; }

llvm::DenseSet<Value> Add2Op::ResultsToTranspose() { return {this->out()}; }

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

bool ConcatOp::IsNCHW() { return this->axisAttr().getValue().getSExtValue() == 1; }

llvm::DenseSet<Value> ConcatOp::OperandsToTranspose() {
  llvm::DenseSet<Value> operands;
  for (auto operand : this->in()) { operands.insert(operand); }
  return operands;
}

llvm::DenseSet<Value> ConcatOp::ResultsToTranspose() { return {this->out()}; }

llvm::SmallVector<Value, 4> ConcatOp::NchwToNhwc(llvm::SmallVector<Value, 4> values,
                                                 PatternRewriter& rewriter) {
  auto elementwise_op = *this;
  NamedAttrList attributes = elementwise_op->getAttrs();
  attributes.set(elementwise_op.axisAttrName(),
                 IntegerAttr::get(rewriter.getIntegerType(64, /*isSigned=*/true),
                                  APInt(64, 3, /*isSigned=*/true)));
  auto out = rewriter
                 .create<oneflow::ConcatOp>(elementwise_op.getLoc(),
                                            getNHWCResultTypes(elementwise_op), values, attributes)
                 .out();
  return {out};
}

bool GroupNormOp::IsNCHW() { return this->data_format().str() == "channels_first"; }

llvm::DenseSet<Value> GroupNormOp::OperandsToTranspose() { return {this->x()}; }

llvm::DenseSet<Value> GroupNormOp::ResultsToTranspose() { return {this->y()}; }

llvm::SmallVector<Value, 4> GroupNormOp::NchwToNhwc(llvm::SmallVector<Value, 4> value,
                                                    PatternRewriter& rewriter) {
  auto group_norm_op = *this;
  SmallVector<Value, 4> operands;
  operands.push_back(value[0]);
  if (this->affine()) {
    operands.push_back(this->beta());
    operands.push_back(this->gamma());
  }
  NamedAttrList attributes = group_norm_op->getAttrs();
  attributes.set(group_norm_op.data_formatAttrName(), rewriter.getStringAttr("channels_last"));
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

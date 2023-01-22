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
#include "OneFlow/Transform/BufferizableOpInterfaceImpl.h"
#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/OneFlowOps.h"
#include "OneFlow/Passes.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::oneflow;
using namespace mlir::bufferization;

struct ReluOpInterface : public BufferizableOpInterface::ExternalModel<ReluOpInterface, ReluOp> {
  bool bufferizesToMemoryRead(Operation*, OpOperand&, const AnalysisState&) const { return true; }
  bool bufferizesToMemoryWrite(Operation*, OpOperand&, const AnalysisState&) const { return false; }
  SmallVector<OpResult> getAliasingOpResult(Operation* op, OpOperand&, const AnalysisState&) const {
    return {};
  }
  LogicalResult bufferize(Operation* op, RewriterBase& rewriter,
                          const BufferizationOptions& options) const {
    if (op->getOperands().front().getType().template isa<MemRefType>()) { return success(); }

    ReluOp raw_op = cast<ReluOp>(op);
    FailureOr<Value> maybe_buffer = getBuffer(rewriter, raw_op.x(), options);
    if (failed(maybe_buffer)) { return failure(); }
    auto new_op = rewriter.create<ReluOp>(op->getLoc(), maybe_buffer->getType(), *maybe_buffer,
                                          op->getAttrs());
    bufferization::replaceOpWithBufferizedValues(rewriter, op, new_op->getResults());
    return success();
  }
};

struct TanhOpInterface : public BufferizableOpInterface::ExternalModel<TanhOpInterface, TanhOp> {
  bool bufferizesToMemoryRead(Operation*, OpOperand&, const AnalysisState&) const { return true; }
  bool bufferizesToMemoryWrite(Operation*, OpOperand&, const AnalysisState&) const { return false; }
  SmallVector<OpResult> getAliasingOpResult(Operation* op, OpOperand&, const AnalysisState&) const {
    return {};
  }
  LogicalResult bufferize(Operation* op, RewriterBase& rewriter,
                          const BufferizationOptions& options) const {
    if (op->getOperands().front().getType().template isa<MemRefType>()) { return success(); }

    TanhOp raw_op = cast<TanhOp>(op);
    FailureOr<Value> maybe_buffer = getBuffer(rewriter, raw_op.x(), options);
    if (failed(maybe_buffer)) { return failure(); }
    auto new_op = rewriter.create<TanhOp>(op->getLoc(), maybe_buffer->getType(), *maybe_buffer,
                                          op->getAttrs());
    bufferization::replaceOpWithBufferizedValues(rewriter, op, new_op->getResults());
    return success();
  }
};

void mlir::oneflow::registerBufferizableOpInterfaceExternalModels(DialectRegistry& registry) {
  registry.addExtension(+[](MLIRContext* ctx, oneflow::OneFlowDialect* dialect) {
    ReluOp::attachInterface<ReluOpInterface>(*ctx);
    TanhOp::attachInterface<TanhOpInterface>(*ctx);
  });
}

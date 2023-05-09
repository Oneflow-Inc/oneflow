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
#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/Passes.h"
#include "OneFlow/OneFlowDevice.h"
#include "OneFlow/Transform/PostTransformChecks.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace oneflow {

namespace {
class OneFlowPostTransformChecksPass
    : public OneFlowPostTransformChecksBase<OneFlowPostTransformChecksPass> {
 public:
  void runOnOperation() override final {
    func::FuncOp func = getOperation();
    MLIRContext* ctx = &getContext();
    ConversionTarget target(*ctx);
    target.addIllegalDialect<OneFlowDialect>();
    target.addIllegalDialect<tosa::TosaDialect>();
    target.addIllegalOp<arith::ConstantOp>();
    target.addLegalOp<arith::ConstantIndexOp>();

    auto device = device::FuncOp2DeviceBuilder(func);
    if (device.isGPU()) {
      target.addIllegalDialect<arith::ArithDialect>();
      target.addIllegalDialect<math::MathDialect>();
      target.addIllegalDialect<scf::SCFDialect>();
    } else {
      target.addLegalDialect<arith::ArithDialect>();
      target.addLegalDialect<math::MathDialect>();
      target.addLegalDialect<scf::SCFDialect>();
    }

    RewritePatternSet patterns(ctx);
    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      func->emitError("find illegal ops in current func.func body");
      signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<Pass> createOneFlowPostTransformChecksPass() {
  return std::make_unique<OneFlowPostTransformChecksPass>();
}

}  // namespace oneflow
}  // namespace mlir

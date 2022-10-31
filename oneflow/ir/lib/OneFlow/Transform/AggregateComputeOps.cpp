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
#include "OneFlow/OKL/OKLDialect.h"
#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/OneFlowOps.h"
#include "OneFlow/Passes.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <iostream>
#include <string>

using namespace mlir;

namespace mlir {
namespace oneflow {

struct AggregateComputeOpsPattern : public mlir::OpRewritePattern<OutputOp> {
  explicit AggregateComputeOpsPattern(mlir::MLIRContext* context)
      : OpRewritePattern<OutputOp>(context, /*benefit=*/0) {}

  mlir::LogicalResult matchAndRewrite(OutputOp op, mlir::PatternRewriter& rewriter) const override {
    if (op->getNumResults() != 1) { return failure(); }
    if (llvm::isa<oneflow::ReturnOp>(op->getNextNode())) { return failure(); }
    // oneflow.output only have a single result
    for (auto user : op->getResult(0).getUsers()) {
      if (!llvm::isa<oneflow::ReturnOp>(user)) { return failure(); }
      rewriter.setInsertionPoint(user);
    }

    auto new_val = rewriter.clone(*op)->getResults();
    rewriter.replaceOp(op, new_val);
    return success();
  };
};

namespace {

class AggregateComputeOpsPass : public AggregateComputeOpsPassBase<AggregateComputeOpsPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<oneflow::OneFlowDialect>();
  }

  void runOnOperation() override {
    Operation* op = getOperation();
    RewritePatternSet patterns(op->getContext());
    patterns.add<AggregateComputeOpsPattern>(patterns.getContext());
    (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
  }
};

}  // namespace

std::unique_ptr<Pass> createAggregateComputeOpsPass() {
  return std::make_unique<AggregateComputeOpsPass>();
}

}  // namespace oneflow
}  // namespace mlir

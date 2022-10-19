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
#include "OneFlow/OKL/OKLOps.h"
#include "OneFlow/OKL/OKLTypes.h"
#include "OneFlow/OKL/passes.h"
#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include <algorithm>
#include <string>
#include <vector>
#include <glog/logging.h>

namespace mlir {

namespace okl {

struct OnlyKeepComputeOpPattern : public mlir::OpRewritePattern<func::FuncOp> {
  explicit OnlyKeepComputeOpPattern(mlir::MLIRContext* context)
      : mlir::OpRewritePattern<func::FuncOp>(context, 0) {}
  mlir::LogicalResult matchAndRewrite(func::FuncOp op,
                                      mlir::PatternRewriter& rewriter) const override {
    if (op.getSymName() != "okl_compute") {
      rewriter.eraseOp(op);
      return success();
    }
    return success();
  }
};

namespace {
struct OnlyKeepComputeOpsPass : public OnlyKeepComputeOpsPassBase<OnlyKeepComputeOpsPass> {
  void runOnOperation() override;

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<okl::OKLDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<arith::ArithmeticDialect>();
  }
};
}  // namespace

std::unique_ptr<Pass> createOnlyKeepComputeOpsPass() {
  return std::make_unique<OnlyKeepComputeOpsPass>();
}

void OnlyKeepComputeOpsPass::runOnOperation() {
  Operation* op = getOperation();
  RewritePatternSet patterns(op->getContext());
  patterns.add<OnlyKeepComputeOpPattern>(patterns.getContext());
  (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
}

}  // namespace okl

}  // namespace mlir

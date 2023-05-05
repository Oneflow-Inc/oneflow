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
#include "OneFlow/OneFlowPDLLPatterns.h"
#include "OneFlow/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <glog/logging.h>
#include <functional>

namespace mlir {
namespace oneflow {

namespace {

struct MgpuToOneFlowStreamPattern final : public OpRewritePattern<LLVM::CallOp> {
 public:
  explicit MgpuToOneFlowStreamPattern(mlir::MLIRContext* context)
      : OpRewritePattern<LLVM::CallOp>(context, /*benefit=*/0) {}
  mlir::LogicalResult matchAndRewrite(LLVM::CallOp op,
                                      mlir::PatternRewriter& rewriter) const override {
    auto ptr_type = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto func = op->getParentOfType<LLVM::LLVMFuncOp>();
    auto callee = op.getCallee();
    if (!func || !callee) return failure();
    Value stream = func.getArguments().back();
    if (stream.getType() != ptr_type) {
      LOG(ERROR) << "failed to find stream in llvm.func block arguments";
      return failure();
    }

    DenseMap<StringRef,
             std::pair<std::function<bool(LLVM::CallOp&, Value&)>,
                       std::function<void(mlir::PatternRewriter&, LLVM::CallOp&, Value&)>>>
        oneflow_abi = {
            {"mgpuStreamCreate",
             {[](LLVM::CallOp& op, Value& stream) { return true; },
              [](mlir::PatternRewriter& rewriter, LLVM::CallOp& op, Value& stream) {
                rewriter.replaceOp(op, {stream});
              }}},
            {"mgpuLaunchKernel",
             {[](LLVM::CallOp& op, Value& stream) {
                unsigned idx = op->getNumOperands();
                return op.getOperand(idx - 3) != stream;
              },
              [](mlir::PatternRewriter& rewriter, LLVM::CallOp& op, Value& stream) {
                unsigned idx = op->getNumOperands();
                auto target = op.getOperand(idx - 3).getDefiningOp();
                rewriter.replaceOp(target, {stream});
              }}},
            // this sync operation is created by gpu-to-llvm-pass from gpu.launch_func op.
            {"mgpuStreamSynchronize",
             {[](LLVM::CallOp& op, Value& stream) { return true; },
              [](mlir::PatternRewriter& rewriter, LLVM::CallOp& op, Value& stream) {
                rewriter.eraseOp(op);
              }}},
            {"mgpuStreamDestroy",
             {[](LLVM::CallOp& op, Value& stream) { return true; },
              [](mlir::PatternRewriter& rewriter, LLVM::CallOp& op, Value& stream) {
                rewriter.eraseOp(op);
              }}},
        };
    auto out = oneflow_abi.find(callee.value().str());
    if (out != oneflow_abi.end() && out->getSecond().first(op, stream)) {
      out->getSecond().second(rewriter, op, stream);
    }
    return success();
  }
};

struct AppendOneFlowStreamPattern final : public OpRewritePattern<func::FuncOp> {
 public:
  explicit AppendOneFlowStreamPattern(mlir::MLIRContext* context)
      : OpRewritePattern<func::FuncOp>(context, /*benefit=*/0) {}
  mlir::LogicalResult matchAndRewrite(func::FuncOp op,
                                      mlir::PatternRewriter& rewriter) const override {
    auto ptr_type = LLVM::LLVMPointerType::get(rewriter.getContext());
    if (llvm::dyn_cast<LLVM::LLVMPointerType>(op.getFunctionType().getInputs().back()))
      return success();

    llvm::SmallVector<Type> new_operand_type;
    for (auto type : op.getFunctionType().getInputs()) { new_operand_type.push_back(type); }
    new_operand_type.push_back(ptr_type);
    auto function_type =
        rewriter.getFunctionType(new_operand_type, op.getFunctionType().getResults());

    auto func = rewriter.create<func::FuncOp>(op.getLoc(), op.getName(), function_type);
    for (auto pair : op->getDialectAttrs()) { func->setAttr(pair.getName(), pair.getValue()); }
    op.getBody().addArgument(ptr_type, func->getLoc());
    IRMapping bvm;
    op.getRegion().cloneInto(&func.getRegion(), bvm);
    rewriter.eraseOp(op);
    return success();
  }
};

class AppendOneFlowStreamPass : public AppendOneFlowStreamPassBase<AppendOneFlowStreamPass> {
  void runOnOperation() override {
    Operation* op = getOperation();
    auto ctx = op->getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<AppendOneFlowStreamPattern>(ctx);
    (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
  }
};

class MgpuToOneFlowStreamPass : public MgpuToOneFlowStreamPassBase<MgpuToOneFlowStreamPass> {
  void runOnOperation() override {
    Operation* op = getOperation();
    auto ctx = op->getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<MgpuToOneFlowStreamPattern>(ctx);
    (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
  }
};

}  // namespace

std::unique_ptr<Pass> createAppendOneFlowStreamPass() {
  return std::make_unique<AppendOneFlowStreamPass>();
}

std::unique_ptr<Pass> createMgpuToOneFlowStreamPass() {
  return std::make_unique<MgpuToOneFlowStreamPass>();
}

}  // namespace oneflow
}  // namespace mlir
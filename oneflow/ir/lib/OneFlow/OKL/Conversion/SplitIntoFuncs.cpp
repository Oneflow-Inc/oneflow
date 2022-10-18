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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
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

struct SplitIntoFuncsPattern : public mlir::OpRewritePattern<func::FuncOp> {
  static mlir::LogicalResult SplitIntoFuncs(func::FuncOp func, mlir::PatternRewriter& rewriter) {
    SmallVector<StringLiteral> new_op_list{BuildKernelOp::getOperationName(),
                                           BuildRegContextOp::getOperationName(),
                                           BuildRunContextOp::getOperationName()};
    SmallVector<StringLiteral> del_op_list{DestroyRegContextOp::getOperationName(),
                                           DestroyRunContextOp::getOperationName()};
    SmallVector<SmallVector<Operation*>> new_ops(new_op_list.size()), del_ops(del_op_list.size());
    SmallVector<Operation*> compute_ops;
    auto& ops = func.getBody().front();

    auto is_in_list = [](const SmallVector<StringLiteral>& list, Operation& op,
                         SmallVector<SmallVector<Operation*>>& ops) {
      for (int i = 0; i < list.size(); ++i) {
        if (op.getName().getStringRef() == list[i]) {
          ops[i].push_back(&op);
          return true;
        }
      }
      return false;
    };
    for (auto& op : ops) {
      if (is_in_list(new_op_list, op, new_ops) || is_in_list(del_op_list, op, del_ops)) {
        continue;
      }
      compute_ops.push_back(&op);
    }
    auto new_ops_type = rewriter.getFunctionType(
        TypeRange{LauncherContextType::get(rewriter.getContext())}, TypeRange{});
    auto launcher_to_void_func = rewriter.getFunctionType(
        TypeRange{LauncherContextType::get(rewriter.getContext())}, TypeRange{});
    rewriter.create<func::FuncOp>(func.getLoc(), new_ops_func, new_ops_type);
    rewriter.create<func::FuncOp>(func.getLoc(), del_ops_func, launcher_to_void_func);
    rewriter.create<func::FuncOp>(func.getLoc(), compute_ops_func, launcher_to_void_func);

    // mlir::VectorType::get({})
    return success();
  }

  explicit SplitIntoFuncsPattern(mlir::MLIRContext* context)
      : mlir::OpRewritePattern<func::FuncOp>(context, 0) {}
  mlir::LogicalResult matchAndRewrite(func::FuncOp op,
                                      mlir::PatternRewriter& rewriter) const override {
    SmallVector<StringLiteral> legal_func_list{new_ops_func, del_ops_func, compute_ops_func};
    if (std::find(legal_func_list.begin(), legal_func_list.end(), op.getSymName())
        != legal_func_list.end()) {
      return success();
    }

    mlir::ModuleOp module_op;
    if (!(module_op = op->getParentOfType<mlir::ModuleOp>())) {
      op->emitError("Failed on -split-into-funcs pass because there are some funcs is not a child "
                    "of the module");
      exit(1);
    }

    if (!(op.getFunctionType().getNumInputs() == 1 && op.getFunctionType().getNumResults() == 0
          && op.getFunctionType().getInput(0).dyn_cast<LauncherContextType>())) {
      op->emitError("Failed on -split-into-funcs pass because the illegal func type is not "
                    "(!okl.launcher_ctx) -> ()");
      exit(1);
    }

    // the legal functions and illegal func can not exist at the same time, because the legalization
    // of illegal func will generate new legal functions.
    for (auto name : legal_func_list) {
      if (module_op.lookupSymbol(name)) {
        op->emitError("Failed on -split-into-funcs pass because illegal funcs and legal funcs "
                      "exist at the same time");
        exit(1);
      }
    }

    return SplitIntoFuncs(op, rewriter);
  }

  static const StringLiteral new_ops_func, del_ops_func, compute_ops_func;
};

const StringLiteral SplitIntoFuncsPattern::new_ops_func = "okl_get_resources",
                    SplitIntoFuncsPattern::del_ops_func = "okl_recycle",
                    SplitIntoFuncsPattern::compute_ops_func = "okl_compute";

namespace {
struct SplitIntoFuncsPass : public SplitIntoFuncsPassBase<SplitIntoFuncsPass> {
  void runOnOperation() override;

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<okl::OKLDialect>();
  }
};
}  // namespace

std::unique_ptr<Pass> createSplitIntoFuncsPass() { return std::make_unique<SplitIntoFuncsPass>(); }

void SplitIntoFuncsPass::runOnOperation() {
  Operation* op = getOperation();
  RewritePatternSet patterns(op->getContext());
  patterns.add<SplitIntoFuncsPattern>(patterns.getContext());
  (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
}

}  // namespace okl

}  // namespace mlir

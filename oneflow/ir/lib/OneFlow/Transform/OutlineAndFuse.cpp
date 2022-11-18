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
#include "OneFlow/OneFlowOps.h"
#include "OneFlow/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <iostream>
#include <string>

using namespace mlir;

namespace mlir {
namespace oneflow {

namespace {

class OutlineJitFunctionPass : public OutlineJitFunctionPassBase<OutlineJitFunctionPass> {
  void runOnOperation() override {
    Operation* op = getOperation();
    RewritePatternSet patterns(op->getContext());
    populateFuserPasses(patterns);
    (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
  }
};

class ConvertOFKLCalleeToLLVMPass
    : public ConvertOFKLCalleeToLLVMPassBase<ConvertOFKLCalleeToLLVMPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    Operation* op = getOperation();
    RewritePatternSet patterns(op->getContext());
    populateConvertOFKLCalleeToLLVMPasses(patterns);
    (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
  }
};

class KernelLaunchFunctionPass : public KernelLaunchFunctionPassBase<KernelLaunchFunctionPass> {
  void runOnOperation() override {
    Operation* op = getOperation();
    RewritePatternSet patterns(op->getContext());
    populateKernelWrapperPasses(patterns);
    (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
  }
};

class FuseIntoExistingOpPass : public FuseIntoExistingOpPassBase<FuseIntoExistingOpPass> {
  void runOnOperation() override {
    Operation* op = getOperation();
    RewritePatternSet patterns(op->getContext());
    populateFuserForExistingOp(patterns);
    (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
  }
};

struct GroupMatMulPattern : public mlir::OpInterfaceRewritePattern<MatMulCompatible> {
  explicit GroupMatMulPattern(mlir::MLIRContext* context)
      : OpInterfaceRewritePattern<MatMulCompatible>(context, /*benefit=*/1) {}
  mlir::LogicalResult matchAndRewrite(MatMulCompatible op,
                                      mlir::PatternRewriter& rewriter) const override {
    if (!op.isLinear()) { return failure(); }
    BiasAddCompatible bias_add;
    for (auto u : op.out().getUsers()) {
      if (auto b = dyn_cast<BiasAddCompatible>(u)) {
        bias_add = b;
        break;
      }
    }
    if (bias_add) {
      if (!bias_add.isLastDim()) { return failure(); }
    }
    llvm::SmallVector<MatMulCompatible, 4> all_matmuls{};
    llvm::SmallVector<BiasAddCompatible, 4> all_bias_adds{};
    for (auto u : op.a().getUsers()) {
      if (auto another_matmul = dyn_cast<MatMulCompatible>(u)) {
        if (!another_matmul.isLinear()) { continue; }
        bool has_another_bias_add = false;
        for (auto u : another_matmul.out().getUsers()) {
          if (auto another_bias_add = dyn_cast<BiasAddCompatible>(u)) {
            if (!another_bias_add.isLastDim()) { continue; }
            all_bias_adds.push_back(another_bias_add);
            has_another_bias_add = true;
            break;
          }
        }
        if (!!bias_add == has_another_bias_add) { all_matmuls.push_back(another_matmul); }
      }
    }
    // all_matmuls has only self, means no other matmul can be grouped
    if (all_matmuls.size() == 1) { return failure(); }
    llvm::SmallVector<Value, 4> operands{};
    for (auto matmul : all_matmuls) { operands.push_back(matmul.a()); }
    for (auto matmul : all_matmuls) { operands.push_back(matmul.b()); }
    for (auto bias_adds : all_bias_adds) { operands.push_back(bias_adds.b()); }
    llvm::SmallVector<Type, 4> results{};
    for (auto matmul : all_matmuls) { results.push_back(matmul.out().getType()); }
    NamedAttrList attributes{};
    attributes.set(OpTrait::IsOpConfCompatible<void>::getDeviceTagAttr(),
                   OpTrait::IsOpConfCompatible<void>::getDeviceTag(op));
    attributes.set(OpTrait::IsOpConfCompatible<void>::getDeviceNameAttr(),
                   OpTrait::IsOpConfCompatible<void>::getDeviceName(op));
    if (auto hierarchy = OpTrait::IsOpConfCompatible<void>::getHierarchy(op)) {
      attributes.set(OpTrait::IsOpConfCompatible<void>::getHierarchyAttr(), hierarchy);
    }
    if (auto scope_symbol_id = OpTrait::IsOpConfCompatible<void>::getScopeSymbolID(op)) {
      attributes.set(OpTrait::IsOpConfCompatible<void>::getScopeSymbolIDAttr(), scope_symbol_id);
    }
    attributes.set("operand_segment_sizes",
                   rewriter.getI32VectorAttr({static_cast<int>(all_matmuls.size()),
                                              static_cast<int>(all_matmuls.size()),
                                              static_cast<int>(all_bias_adds.size())}));
    attributes.set(OpTrait::IsOpConfCompatible<void>::getOpNameAttr(),
                   rewriter.getStringAttr(
                       "grouped_matmul_" + OpTrait::IsOpConfCompatible<void>::getOpName(op).str()));
    auto grouped_matmul =
        rewriter.create<GroupedMatmulBiasOp>(op->getLoc(), results, operands, attributes);
    if (all_bias_adds.empty()) {
      for (const auto& matmul : llvm::enumerate(all_matmuls)) {
        matmul.value().out().replaceAllUsesWith(grouped_matmul.ys()[matmul.index()]);
      }
    } else {
      CHECK(all_bias_adds.size() == all_matmuls.size());
      for (const auto& bias_add : llvm::enumerate(all_bias_adds)) {
        bias_add.value().out().replaceAllUsesWith(grouped_matmul.ys()[bias_add.index()]);
      }
    }
    return success();
  }
};

class GroupMatMulPass : public GroupMatMulBase<GroupMatMulPass> {
  void runOnOperation() override {
    Operation* op = getOperation();
    RewritePatternSet patterns(op->getContext());
    patterns.add<GroupMatMulPattern>(op->getContext());
    (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
  }
};

struct GroupNormActivationPattern : public OpRewritePattern<GroupNormOp> {
  explicit GroupNormActivationPattern(MLIRContext* context)
      : OpRewritePattern<GroupNormOp>(context, /*benefit=*/1) {}
  LogicalResult matchAndRewrite(oneflow::GroupNormOp op, PatternRewriter& rewriter) const override {
    if (op.activation() == "none") {
      llvm::SmallVector<Operation*, 4> act_ops{};
      for (auto& u : op.y().getUses()) {
        if (auto act_op = dyn_cast<oneflow::SiluOp>(u.getOwner())) { act_ops.push_back(act_op); }
      }
      NamedAttrList attributes(op->getAttrs());
      attributes.set(OpTrait::IsOpConfCompatible<void>::getOpNameAttr(),
                     rewriter.getStringAttr(OpTrait::IsOpConfCompatible<void>::getOpName(op).str()
                                            + "_with_activation"));
      attributes.set("activation", rewriter.getStringAttr("silu"));
      auto gn_with_act = rewriter.create<GroupNormOp>(op->getLoc(), op->getResultTypes(),
                                                      op.getOperands(), attributes);
      for (auto act : act_ops) {
        if (auto op = dyn_cast<oneflow::SiluOp>(act)) {
          op.out().replaceAllUsesWith(gn_with_act.y());
        }
      }
      return success();
    }
    return failure();
  }
};

class FuseForwardOpsPass : public FuseForwardOpsBase<FuseForwardOpsPass> {
  void runOnOperation() override {
    Operation* op = getOperation();
    RewritePatternSet patterns(op->getContext());
    patterns.add<GroupNormActivationPattern>(op->getContext());
    (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
  }
};

}  // namespace

std::unique_ptr<Pass> createOutlineJitFunctionPass() {
  return std::make_unique<OutlineJitFunctionPass>();
}

std::unique_ptr<Pass> createKernelLaunchFunctionPass() {
  return std::make_unique<KernelLaunchFunctionPass>();
}

std::unique_ptr<mlir::Pass> createConvertOFKLCalleeToLLVMPass() {
  return std::make_unique<ConvertOFKLCalleeToLLVMPass>();
}

std::unique_ptr<Pass> createFuseIntoExistingOpPass() {
  return std::make_unique<FuseIntoExistingOpPass>();
}

std::unique_ptr<Pass> createGroupMatMul() { return std::make_unique<GroupMatMulPass>(); }

std::unique_ptr<Pass> createFuseForwardOps() { return std::make_unique<FuseForwardOpsPass>(); }

}  // namespace oneflow
}  // namespace mlir

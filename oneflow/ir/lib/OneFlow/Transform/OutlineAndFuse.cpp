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
#include "OneFlow/Transform/OutlineAndFuse.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "OneFlow/OKL/OKLDialect.h"
#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/OneFlowOps.h"
#include "OneFlow/Passes.h"
#include "OneFlow/OneFlowPDLLPatterns.h"
#include "OneFlow/OneFlowPatternUtils.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <iostream>
#include <string>

namespace mlir {
namespace oneflow {

namespace {

class WrapOpsToKernelLaunchPass : public WrapOpsToKernelLaunchPassBase<WrapOpsToKernelLaunchPass> {
 public:
  WrapOpsToKernelLaunchPass() = default;
  WrapOpsToKernelLaunchPass(const WrapOpsToKernelLaunchPass& other)
      : WrapOpsToKernelLaunchPassBase(other) {}

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<oneflow::OneFlowDialect>();
  }

  void runOnOperation() override {
    Operation* op = getOperation();
    RewritePatternSet patterns(op->getContext());
    populateWrapOpsToKernelLaunchPatterns(patterns, wrap_ops_mode_.c_str());
    (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
  }

 private:
  Option<std::string> wrap_ops_mode_{*this, "mode",
                                     llvm::cl::desc("the mode of this pass to wrap ops"),
                                     llvm::cl::init(wrap_mode::SIMPLE)};
};

class FuseIntoExistingOpPass : public FuseIntoExistingOpPassBase<FuseIntoExistingOpPass> {
  void runOnOperation() override {
    Operation* op = getOperation();
    RewritePatternSet patterns(op->getContext());
    populateFuserForExistingOp(patterns);
    (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
  }
};

namespace {

BiasAddCompatible getBiasAddCompatibleOp(MatMulCompatible op) {
  BiasAddCompatible bias_add;
  auto self_bias_op = dyn_cast<BiasAddCompatible>(op.getOperation());
  if (self_bias_op) /* matmul itself is also bias add op */ {
    bias_add = self_bias_op;
  } else /* there is bias add op */ {
    for (auto u : op.matMulGetY().getUsers()) {
      if (auto b = dyn_cast<BiasAddCompatible>(u)) {
        bias_add = b;
        break;
      }
    }
  }
  if (bias_add && bias_add.isLastDim()) {
    return bias_add;
  } else {
    return BiasAddCompatible{};
  }
}

}  // namespace
struct GroupMatMulPattern : public mlir::OpInterfaceRewritePattern<MatMulCompatible> {
  explicit GroupMatMulPattern(mlir::MLIRContext* context)
      : OpInterfaceRewritePattern<MatMulCompatible>(context, /*benefit=*/1) {}
  mlir::LogicalResult matchAndRewrite(MatMulCompatible op,
                                      mlir::PatternRewriter& rewriter) const override {
    if (!op.isLinear()) { return failure(); }
    auto bias_add = getBiasAddCompatibleOp(op);
    llvm::SmallVector<MatMulCompatible, 4> all_matmuls{};
    llvm::SmallVector<BiasAddCompatible, 4> all_bias_adds{};
    for (auto xUser : op.matMulGetX().getUsers()) {
      if (auto matmul = dyn_cast<MatMulCompatible>(xUser)) {
        if (!matmul.isLinear()) { continue; }
        auto each_bias_add = getBiasAddCompatibleOp(matmul);
        if (each_bias_add) { all_bias_adds.push_back(each_bias_add); }
        if (!!bias_add == !!each_bias_add) { all_matmuls.push_back(matmul); }
      }
    }
    // all_matmuls has only self, means no other matmul can be grouped
    if (all_matmuls.size() == 1) { return failure(); }
    llvm::SmallVector<Value, 4> operands{};
    for (auto matmul : all_matmuls) { operands.push_back(matmul.matMulGetX()); }
    for (auto matmul : all_matmuls) { operands.push_back(matmul.matMulGetW()); }
    for (auto bias_adds : all_bias_adds) { operands.push_back(bias_adds.biasAddGetBias()); }
    llvm::SmallVector<Type, 4> results{};
    for (auto matmul : all_matmuls) { results.push_back(matmul.matMulGetY().getType()); }
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
    attributes.set(OpTrait::AttrSizedOperandSegments<void>::getOperandSegmentSizeAttr(),
                   rewriter.getDenseI32ArrayAttr({static_cast<int>(all_matmuls.size()),
                                                  static_cast<int>(all_matmuls.size()),
                                                  static_cast<int>(all_bias_adds.size())}));
    attributes.set(OpTrait::IsOpConfCompatible<void>::getOpNameAttr(),
                   rewriter.getStringAttr(
                       "grouped_matmul_" + OpTrait::IsOpConfCompatible<void>::getOpName(op).str()));
    auto grouped_matmul =
        rewriter.create<GroupedMatmulBiasOp>(op->getLoc(), results, operands, attributes);
    if (all_bias_adds.empty()) {
      for (const auto& matmul : llvm::enumerate(all_matmuls)) {
        matmul.value().matMulGetY().replaceAllUsesWith(grouped_matmul.getYs()[matmul.index()]);
      }
    } else {
      CHECK(all_bias_adds.size() == all_matmuls.size());
      for (const auto& bias_add : llvm::enumerate(all_bias_adds)) {
        bias_add.value().biasAddGetOut().replaceAllUsesWith(
            grouped_matmul.getYs()[bias_add.index()]);
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

namespace {

bool MatmulQuantOpHasInputScale(MatmulQuantOp op) {
  return op.getODSOperands(3).empty() ? false : true;
}

bool MatmulQuantOpHasScale(MatmulQuantOp op) { return op.getODSOperands(6).empty() ? false : true; }

bool MatmulQuantOpHasBias(MatmulQuantOp op) { return op.getODSOperands(7).empty() ? false : true; }

bool MatmulQuantOpHasAddToOutput(MatmulQuantOp op) { return !(op.getODSOperands(8).empty()); }

}  // namespace

struct GroupMatMulQuantPattern : public mlir::OpRewritePattern<MatmulQuantOp> {
  explicit GroupMatMulQuantPattern(mlir::MLIRContext* context)
      : OpRewritePattern<MatmulQuantOp>(context, /*benefit=*/1) {}
  mlir::LogicalResult matchAndRewrite(MatmulQuantOp op,
                                      mlir::PatternRewriter& rewriter) const override {
    llvm::SmallVector<MatmulQuantOp, 4> all_matmuls{};
    bool has_in_scale = MatmulQuantOpHasInputScale(op);
    bool has_scale = MatmulQuantOpHasScale(op);
    bool has_bias = MatmulQuantOpHasBias(op);
    bool has_add_to_output = MatmulQuantOpHasAddToOutput(op);
    for (auto xUser : op.getA().getUsers()) {
      if (auto matmul_quant = dyn_cast<MatmulQuantOp>(xUser)) {
        if (has_in_scale != MatmulQuantOpHasInputScale(matmul_quant)
            || has_scale != MatmulQuantOpHasScale(matmul_quant)
            || has_bias != MatmulQuantOpHasBias(matmul_quant)
            || has_add_to_output != MatmulQuantOpHasBias(matmul_quant)) {
          continue;
        }
        all_matmuls.push_back(matmul_quant);
      }
    }
    // all_matmuls has only self, means no other matmul can be grouped
    if (all_matmuls.size() == 1) { return failure(); }
    int a_size = 0;
    int b_size = 0;
    int in_zero_size = 0;
    int in_scale_size = 0;
    int weaght_scale_size = 0;
    int weaght_acc_size = 0;
    int scale_size = 0;
    int bias_size = 0;
    int add_to_out_put_size = 0;

    llvm::SmallVector<Value, 4> operands{};
    for (auto matmul : all_matmuls) { operands.push_back(matmul.getA()); }
    a_size = all_matmuls.size();
    for (auto matmul : all_matmuls) { operands.push_back(matmul.getB()); }
    b_size = all_matmuls.size();
    if (has_in_scale) {
      for (auto matmul : all_matmuls) { operands.push_back(matmul.getInZeroPoint()); }
      for (auto matmul : all_matmuls) { operands.push_back(matmul.getInScale()); }
      for (auto matmul : all_matmuls) { operands.push_back(matmul.getWeightScale()); }
      for (auto matmul : all_matmuls) { operands.push_back(matmul.getWeightAcc()); }
      in_zero_size = all_matmuls.size();
      in_scale_size = all_matmuls.size();
      weaght_scale_size = all_matmuls.size();
      weaght_acc_size = all_matmuls.size();
    }
    if (has_scale) {
      for (auto matmul : all_matmuls) { operands.push_back(matmul.getScale()); }
      scale_size = all_matmuls.size();
    }
    if (has_bias) {
      for (auto matmul : all_matmuls) { operands.push_back(matmul.getBias()); }
      bias_size = all_matmuls.size();
    }
    if (has_add_to_output) {
      for (auto matmul : all_matmuls) { operands.push_back(matmul.get_addToOutput()); }
      add_to_out_put_size = all_matmuls.size();
    }
    llvm::SmallVector<Type, 4> results{};
    for (auto matmul : all_matmuls) { results.push_back(matmul.getOut().getType()); }
    NamedAttrList attributes{};
    attributes.set(OpTrait::IsOpConfCompatible<void>::getDeviceTagAttr(),
                   OpTrait::IsOpConfCompatible<void>::getDeviceTag(op));
    attributes.set(OpTrait::IsOpConfCompatible<void>::getDeviceNameAttr(),
                   OpTrait::IsOpConfCompatible<void>::getDeviceName(op));
    attributes.set("transpose_a", op.getTransposeAAttr());
    attributes.set("transpose_b", op.getTransposeBAttr());
    attributes.set("alpha", op.getAlphaAttr());
    attributes.set("out_dtype", op.getOutDtypeAttr());
    attributes.set("tuning_cache", op.getTuningCacheAttr());
    if (auto hierarchy = OpTrait::IsOpConfCompatible<void>::getHierarchy(op)) {
      attributes.set(OpTrait::IsOpConfCompatible<void>::getHierarchyAttr(), hierarchy);
    }
    if (auto scope_symbol_id = OpTrait::IsOpConfCompatible<void>::getScopeSymbolID(op)) {
      attributes.set(OpTrait::IsOpConfCompatible<void>::getScopeSymbolIDAttr(), scope_symbol_id);
    }
    attributes.set(OpTrait::AttrSizedOperandSegments<void>::getOperandSegmentSizeAttr(),
                   rewriter.getDenseI32ArrayAttr({a_size, b_size, in_zero_size, in_scale_size,
                                                  weaght_scale_size, weaght_acc_size, scale_size,
                                                  bias_size, add_to_out_put_size}));
    attributes.set(
        OpTrait::IsOpConfCompatible<void>::getOpNameAttr(),
        rewriter.getStringAttr("grouped_matmul_quant_"
                               + OpTrait::IsOpConfCompatible<void>::getOpName(op).str()));
    auto grouped_matmul_quant_op =
        rewriter.create<GroupedMatmulQuantOp>(op->getLoc(), results, operands, attributes);
    for (const auto& matmul : llvm::enumerate(all_matmuls)) {
      matmul.value().getOut().replaceAllUsesWith(
          grouped_matmul_quant_op.getOutputs()[matmul.index()]);
    }
    return success();
  }
};

class GroupMatMulQuantPass : public GroupMatMulQuantBase<GroupMatMulQuantPass> {
  void runOnOperation() override {
    Operation* op = getOperation();
    RewritePatternSet patterns(op->getContext());
    patterns.add<GroupMatMulQuantPattern>(op->getContext());
    (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
  }
};

struct GroupNormActivationPattern : public OpRewritePattern<GroupNormOp> {
  explicit GroupNormActivationPattern(MLIRContext* context)
      : OpRewritePattern<GroupNormOp>(context, /*benefit=*/1) {}
  LogicalResult matchAndRewrite(oneflow::GroupNormOp op, PatternRewriter& rewriter) const override {
    if (op.getActivation() == "none") {
      llvm::SmallVector<Operation*, 4> act_ops{};
      for (auto& u : op.getY().getUses()) {
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
          op.getOut().replaceAllUsesWith(gn_with_act.getY());
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

class FuseOpsWithBackwardImplPass
    : public FuseOpsWithBackwardImplBase<FuseOpsWithBackwardImplPass> {
  void runOnOperation() override {
    Operation* op = getOperation();
    RewritePatternSet patterns(op->getContext());
    populateFuseOpsWithBackwardImplPattern(patterns);
    rewrites::populateRewrites(patterns);
    (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
  }
};

class FuseNormalizationOpsPass : public FuseNormalizationOpsBase<FuseNormalizationOpsPass> {
  void runOnOperation() override {
    Operation* op = getOperation();
    RewritePatternSet patterns(op->getContext());
    populateNormalizationOpPatterns(patterns);
    rewrites::populateRewrites(patterns);
    (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
  }
};

}  // namespace

std::unique_ptr<Pass> createWrapOpsToKernelLaunchPass() {
  return std::make_unique<WrapOpsToKernelLaunchPass>();
}

std::unique_ptr<Pass> createFuseIntoExistingOpPass() {
  return std::make_unique<FuseIntoExistingOpPass>();
}

std::unique_ptr<Pass> createGroupMatMul() { return std::make_unique<GroupMatMulPass>(); }

std::unique_ptr<Pass> createGroupMatMulQuant() { return std::make_unique<GroupMatMulQuantPass>(); }

std::unique_ptr<Pass> createFuseForwardOps() { return std::make_unique<FuseForwardOpsPass>(); }
std::unique_ptr<Pass> createFuseOpsWithBackwardImpl() {
  return std::make_unique<FuseOpsWithBackwardImplPass>();
}

std::unique_ptr<Pass> createFuseNormalizationOps() {
  return std::make_unique<FuseNormalizationOpsPass>();
}

}  // namespace oneflow
}  // namespace mlir

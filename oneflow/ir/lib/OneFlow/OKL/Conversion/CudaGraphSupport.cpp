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

#include "oneflow/core/framework/op_kernel.h"
#include "OneFlow/OKL/OKLDialect.h"
#include "OneFlow/OKL/OKLOps.h"
#include "OneFlow/OKL/OKLTypes.h"
#include "OneFlow/OKL/passes.h"
#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/Passes.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/ir/oneflow-extension/include/OneFlow/kernel_launch/RegContext.h"

namespace mlir {
namespace okl {
struct TagCudaGraphSupportPattern final : public mlir::OpRewritePattern<func::FuncOp> {
  static mlir::Operation* FindOneFlowOp(mlir::Operation* op) {
    mlir::Operation* reg_op = nullptr;
    for (auto& op_it : op->getRegion(0).front().getOperations()) {
      if (op_it.getDialect()->getNamespace() != "oneflow") { continue; }
      reg_op = &op_it;
      break;
    }
    return reg_op;
  }
  static LogicalResult CheckChild(func::FuncOp func) {
    using namespace ::oneflow::user_op;
    for (auto& op : func->getRegion(0).front()) {
      if (auto reg_ctx_op = llvm::dyn_cast_or_null<mlir::okl::BuildRegContextOp>(&op)) {
        // iter reg context op
        const auto reg_op = FindOneFlowOp(&op);
        if (!reg_op) {
          func->emitError("Failed to find reg_op in okl.build_reg_context_op");
          return failure();
        }
        // generate kernel from oneflow.{compute op}
        ::oneflow::okl::RegContext reg_ctx(reg_op);

        auto* kernel = const_cast<OpKernel*>(reg_ctx.GetKernel());

        // check whether cuda graph support is base class
        if (const auto* cuda_graph_support = dynamic_cast<CudaGraphSupport*>(kernel)) {
          // TODO: more check
          continue;
        }
        return failure();
      }
    }
    return success();
  }

 public:
  explicit TagCudaGraphSupportPattern(mlir::MLIRContext* context)
      : OpRewritePattern<func::FuncOp>(context, /*benefit=*/0) {}
  mlir::LogicalResult matchAndRewrite(func::FuncOp op,
                                      mlir::PatternRewriter& rewriter) const override {
    const auto tag_name = "cuda_support";
    // check whether this op is okl init context function  op
    if (op.getSymName().str() != "okl_init_context") { return failure(); }
    // check whether this op has been taged before
    if (op->getAttr(tag_name).dyn_cast_or_null<BoolAttr>() != nullptr) { return success(); }
    // check whether its childern is all cuda graph supported
    const auto outcome = succeeded(CheckChild(op));

    // set cuda graph support tag on init_context and compute function ops
    op->setAttr(tag_name, rewriter.getBoolAttr(outcome));
    if (auto module_op = op->getParentOfType<ModuleOp>()) {
      if (auto compute_func = module_op.lookupSymbol("okl_compute")) {
        compute_func->setAttr(tag_name, rewriter.getBoolAttr(outcome));
        return success();
      }
    }
    // tagging is not completed successfully
    op->emitError("Tag cuda graph pass: tagging is not completed successfully");
    return failure();
  }
};

namespace {
struct TagCudaGraphSupportPass : public TagCudaGraphSupportPassBase<TagCudaGraphSupportPass> {
  void runOnOperation() override;

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<okl::OKLDialect>();
  }
};
}  // namespace

std::unique_ptr<Pass> createTagCudaGraphSupportPass() {
  return std::make_unique<TagCudaGraphSupportPass>();
}

void TagCudaGraphSupportPass::runOnOperation() {
  MLIRContext* context = &getContext();
  RewritePatternSet patterns(context);

  patterns.add<TagCudaGraphSupportPattern>(context);

  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

}  // namespace okl
}  // namespace mlir

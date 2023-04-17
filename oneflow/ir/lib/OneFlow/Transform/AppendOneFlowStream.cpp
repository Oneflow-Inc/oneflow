#include "OneFlow/OneFlowPDLLPatterns.h"
#include "OneFlow/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <glog/logging.h>

namespace mlir {
namespace oneflow {

namespace {

struct AppendOneFlowStreamPattern final : public OpRewritePattern<func::FuncOp> {
 public:
  explicit AppendOneFlowStreamPattern(mlir::MLIRContext* context)
      : OpRewritePattern<func::FuncOp>(context, /*benefit=*/0) {}
  mlir::LogicalResult matchAndRewrite(func::FuncOp op,
                                      mlir::PatternRewriter& rewriter) const override {
    auto ptr_type = LLVM::LLVMPointerType::get(IntegerType::get(rewriter.getContext(), 8));
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

}  // namespace

std::unique_ptr<Pass> createAppendOneFlowStreamPass() {
  return std::make_unique<AppendOneFlowStreamPass>();
}

}  // namespace oneflow
}  // namespace mlir
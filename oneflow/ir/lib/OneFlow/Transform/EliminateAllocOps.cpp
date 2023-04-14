#include "OneFlow/OneFlowPDLLPatterns.h"
#include "OneFlow/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace oneflow {

namespace {
class EliminateAllocOpsPass : public EliminateAllocOpsPassBase<EliminateAllocOpsPass> {
  void runOnOperation() override {
    Operation* op = getOperation();
    RewritePatternSet patterns(op->getContext());
    mlir::oneflow::populateAllocEliminationPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
  }
};

}  // namespace

std::unique_ptr<Pass> createEliminateAllocOpsPass() {
  return std::make_unique<EliminateAllocOpsPass>();
}

}  // namespace oneflow
}  // namespace mlir

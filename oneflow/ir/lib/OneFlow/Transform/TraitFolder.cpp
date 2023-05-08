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
class TestOneFlowTraitFolderPass : public TestOneFlowTraitFolderPassBase<TestOneFlowTraitFolderPass> {
  void runOnOperation() override {
    if (failed(applyPatternsAndFoldGreedily(getOperation(), RewritePatternSet(&getContext())))) {
      exit(1);
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createTestOneFlowTraitFolderPass() {
  return std::make_unique<TestOneFlowTraitFolderPass>();
}

}  // namespace oneflow
}  // namespace mlir

#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
namespace mlir {
namespace oneflow {

namespace {

// 1. collect ops to outline
// 2. create func.func jit ops to call
// 3. replace the usages with jit ops' results

// entries: non-oneflow ops which have operands are from oneflow ops
// exits: result consumed by oneflow ops

class OutlineJitFunctionPass : public OutlineJitFunctionPassBase<OutlineJitFunctionPass> {
  void runOnOperation() override {
    FunctionOpInterface func = getOperation();
    for (auto& op : func.getBody().front()) {
      if (!llvm::dyn_cast<OneFlowDialect>(op.getDialect())) { op.dump(); }
    }
  };
};

}  // namespace

std::unique_ptr<Pass> createOutlineJitFunctionPass() {
  return std::make_unique<OutlineJitFunctionPass>();
}

}  // namespace oneflow
}  // namespace mlir

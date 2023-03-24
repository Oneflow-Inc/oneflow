#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
namespace mlir {
namespace oneflow {

namespace {

// general lowering path:
// 1. outline linalg ops to a func.func and an oneflow.jit op
// 2. bufferize the func.func and update oneflow.jit op's tmp buffer size

// 1. collect ops to outline
// 2. create func.func jit ops to call
// 3. replace the usages with jit ops' results

// entries: non-oneflow ops which have operands are from oneflow ops
// exits: result consumed by oneflow ops

// NOTE: we assume all arg values are produced by an oneflow op and won't be an argument
void cloneOpsToNewBody(OpBuilder& builder, Operation* op, Block& body,
                       llvm::DenseSet<Operation*>& visitedEntryOps,
                       llvm::DenseSet<Value>& argValues, llvm::DenseSet<Value> returnValues,
                       BlockAndValueMapping& mapping) {
  for (auto operand : op->getOperands()) {
    if (!mapping.lookup(operand)) {
      if (auto defOp = operand.getDefiningOp()) {
        if (llvm::dyn_cast<OneFlowDialect>(defOp->getDialect())) {
          visitedEntryOps.insert(op);
          argValues.insert(operand);
          mapping.map(operand, body.addArgument(operand.getType(), operand.getLoc()));
        } else {
          cloneOpsToNewBody(builder, defOp, body, visitedEntryOps, argValues, returnValues,
                            mapping);
        }
      }
    }
  }
  ImplicitLocOpBuilder nb(op->getLoc(), builder);
  nb.clone(*op, mapping);
  for (auto& use : op->getUses()) {
    if (llvm::dyn_cast<OneFlowDialect>(use.getOwner()->getDialect())) {
      returnValues.insert(use.get());
    } else {
      cloneOpsToNewBody(builder, use.getOwner(), body, visitedEntryOps, argValues, returnValues,
                        mapping);
    }
  }
}

class OutlineJitFunctionPass : public OutlineJitFunctionPassBase<OutlineJitFunctionPass> {
  void runOnOperation() override {
    llvm::DenseSet<Operation*> entryOps, visitedEntryOps;
    llvm::DenseSet<Value> argValues, returnValues;
    FunctionOpInterface func = getOperation();
    auto& operations = func.getBody().front().getOperations();
    // collect non-oneflow operations
    for (auto& op : operations) {
      if (llvm::dyn_cast<OneFlowDialect>(op.getDialect())) {
        for (auto result : op.getResults()) {
          for (auto user : result.getUsers()) {
            if (!llvm::dyn_cast<OneFlowDialect>(user->getDialect())) { entryOps.insert(user); }
          }
        }
      }
    }
    OpBuilder builder{&getContext()};
    for (auto entryOp : entryOps) {
      OpBuilder::InsertionGuard guard(builder);
      Block body{};
      builder.setInsertionPointToStart(&body);
      BlockAndValueMapping mapping;
      if (visitedEntryOps.contains(entryOp)) { continue; }
      cloneOpsToNewBody(builder, entryOp, body, visitedEntryOps, argValues, returnValues, mapping);
    }
    llvm::errs() << entryOps.size() << "\n";
  };
};

}  // namespace

std::unique_ptr<Pass> createOutlineJitFunctionPass() {
  return std::make_unique<OutlineJitFunctionPass>();
}

}  // namespace oneflow
}  // namespace mlir

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
void cloneOpsToNewBody(OpBuilder& builder, Operation* op, Block* body,
                       llvm::DenseSet<Operation*>& visitedEntryOps,
                       llvm::DenseSet<Value>& argValues, llvm::DenseSet<Value> returnValues,
                       BlockAndValueMapping& mapping) {
  for (auto operand : op->getOperands()) {
    if (!mapping.lookup(operand)) {
      if (auto defOp = operand.getDefiningOp()) {
        if (llvm::dyn_cast<OneFlowDialect>(defOp->getDialect())) {
          visitedEntryOps.insert(op);
          argValues.insert(operand);
          mapping.map(operand, body->addArgument(operand.getType(), operand.getLoc()));
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
    FunctionOpInterface job = getOperation();
    auto& operations = job.getBody().front().getOperations();

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
      llvm::DenseSet<Value> argValues, returnValues;
      OpBuilder::InsertionGuard guard(builder);
      auto block = new Block();
      builder.setInsertionPointToStart(block);
      BlockAndValueMapping mapping;
      if (visitedEntryOps.contains(entryOp)) { continue; }
      cloneOpsToNewBody(builder, entryOp, block, visitedEntryOps, argValues, returnValues, mapping);

      SmallVector<::mlir::Value, 4> mapped_results;
      SmallVector<Type, 4> argument_types, result_types;

      for (auto ret : returnValues) {
        mapped_results.push_back(mapping.lookup(ret));
        result_types.push_back(ret.getType());
      }
      builder.setInsertionPointToEnd(block);
      builder.create<func::ReturnOp>(entryOp->getLoc(), mapped_results);

      for (auto argument : block->getArguments()) { argument_types.push_back(argument.getType()); }
      auto func_type = builder.getFunctionType(argument_types, result_types);
      auto function = builder.create<func::FuncOp>(entryOp->getLoc(), "func_name", func_type);
      function.getBody().push_back(block);
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createOutlineJitFunctionPass() {
  return std::make_unique<OutlineJitFunctionPass>();
}

}  // namespace oneflow
}  // namespace mlir

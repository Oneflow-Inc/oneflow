#include <queue>
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

class Outliner {
 private:
  OpBuilder& builder;
  Block* body;
  llvm::DenseSet<Operation*>& visitedOps;
  std::deque<Operation*> worklist{};
  void cloneOpsToNewBody(Operation* op) {
    if (visitedOps.contains(op)) { return; }
    for (auto operand : op->getOperands()) {
      if (!mapping.lookup(operand)) {
        if (auto defOp = operand.getDefiningOp()) {
          if (visitedOps.contains(defOp)) { continue; }
          if (llvm::dyn_cast<OneFlowDialect>(defOp->getDialect())) {
            argValues.insert(operand);
            mapping.map(operand, body->addArgument(operand.getType(), operand.getLoc()));
          } else {
            cloneOpsToNewBody(defOp);
          }
        }
      }
    }
    ImplicitLocOpBuilder nb(op->getLoc(), builder);
    nb.clone(*op, mapping);
    visitedOps.insert(op);
    llvm::errs() << "[clone] ";
    op->dump();

    for (auto& use : op->getUses()) {
      auto owner = use.getOwner();
      if (llvm::dyn_cast<OneFlowDialect>(owner->getDialect())) {
        returnValues.insert(use.get());
      } else {
        if (visitedOps.contains(owner)) { continue; }
        // worklist.push_front(owner);
        cloneOpsToNewBody(owner);
      }
    }

    // while (!worklist.empty()) {
    //   auto op = worklist.front();
    //   worklist.pop_front();
    //   cloneOpsToNewBody(op);
    // }
  }

 public:
  Outliner(OpBuilder& builder, Block* body, Operation* op, llvm::DenseSet<Operation*>& visitedOps)
      : builder{builder}, body{body}, visitedOps{visitedOps} {
    cloneOpsToNewBody(op);
  }

  BlockAndValueMapping mapping;
  llvm::DenseSet<Value> argValues{}, returnValues{};
};

class OutlineJitFunctionPass : public OutlineJitFunctionPassBase<OutlineJitFunctionPass> {
  void runOnOperation() override {
    llvm::DenseSet<Operation*> entryOps, visitedOps;
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
      if (visitedOps.contains(entryOp)) { continue; }
      OpBuilder::InsertionGuard guard(builder);
      auto block = new Block();
      builder.setInsertionPointToStart(block);
      auto outliner = Outliner(builder, block, entryOp, visitedOps);

      SmallVector<::mlir::Value, 4> mappedResults;
      SmallVector<Type, 4> argumentTypes, resultTypes;

      for (auto ret : outliner.returnValues) {
        mappedResults.push_back(outliner.mapping.lookup(ret));
        resultTypes.push_back(ret.getType());
      }
      builder.setInsertionPointToEnd(block);
      builder.create<func::ReturnOp>(entryOp->getLoc(), mappedResults);

      for (auto argument : block->getArguments()) { argumentTypes.push_back(argument.getType()); }
      auto funcType = builder.getFunctionType(argumentTypes, resultTypes);
      funcType.dump();
      auto function = builder.create<func::FuncOp>(entryOp->getLoc(), "TODO-func_name", funcType);
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

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

llvm::SmallVector<llvm::DenseSet<Operation*>, 4> mergeEntries(
    const llvm::SmallVector<llvm::DenseSet<Operation*>, 4>& groups, size_t iter = 100) {
  llvm::SmallVector<llvm::DenseSet<Operation*>, 4> results{};
  for (auto g : groups) {
    bool hasIntersection = false;
    for (auto& r : results) {
      for (auto op : g) {
        if (r.count(op)) {
          hasIntersection = true;
          break;
        }
      }
      if (hasIntersection) {
        (void)results.emplace_back();
        std::set_union(g.begin(), g.end(), r.begin(), r.end(), results.back().begin());
        break;
      }
    }
    if (!hasIntersection) { results.push_back(g); }
  }
  if (iter > 0) {
    return mergeEntries(results, iter - 1);
  } else {
    return results;
  }
}

void findEntriesOfRoot(Operation* root, llvm::DenseSet<Operation*>& entryOps,
                       llvm::DenseSet<Operation*>& results) {
  if (entryOps.count(root)) { results.insert(root); }
  for (auto operand : root->getOperands()) {
    if (auto defOp = operand.getDefiningOp()) { findEntriesOfRoot(defOp, entryOps, results); }
  }
}

class OutlineJitFunctionPass : public OutlineJitFunctionPassBase<OutlineJitFunctionPass> {
  void runOnOperation() override {
    llvm::DenseSet<Operation*> entryOps{};
    llvm::DenseSet<Operation*> exitOps{};
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
        for (auto operand : op.getOperands()) {
          if (auto defOp = operand.getDefiningOp()) {
            if (!llvm::dyn_cast<OneFlowDialect>(defOp->getDialect())) { exitOps.insert(defOp); }
          }
        }
      }
    }
    llvm::SmallVector<llvm::DenseSet<Operation*>, 4> entryGroups{};
    for (auto exitOp : exitOps) {
      llvm::DenseSet<Operation*> group;
      findEntriesOfRoot(exitOp, entryOps, group);
      entryGroups.push_back(group);
    }
    entryGroups = mergeEntries(entryGroups);
    llvm::errs() << entryOps.size() << "\n";
    llvm::errs() << exitOps.size() << "\n";
    llvm::errs() << entryGroups.size() << "\n";
  };
};

}  // namespace

std::unique_ptr<Pass> createOutlineJitFunctionPass() {
  return std::make_unique<OutlineJitFunctionPass>();
}

}  // namespace oneflow
}  // namespace mlir

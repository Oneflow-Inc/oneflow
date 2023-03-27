#include <queue>
#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/OneFlowOps.h"
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

NamedAttrList GetJitOpAttributes(Builder& rewriter, StringRef op_name, int32_t input_size,
                                 int32_t output_size, Operation* op) {
  NamedAttrList attributes;
  attributes.set(OpTrait::IsOpConfCompatible<void>::getDeviceTagAttr(),
                 OpTrait::IsOpConfCompatible<void>::getDeviceTag(op));
  attributes.set(OpTrait::IsOpConfCompatible<void>::getDeviceNameAttr(),
                 OpTrait::IsOpConfCompatible<void>::getDeviceName(op));
  if (auto hierarchy = OpTrait::IsOpConfCompatible<void>::getHierarchy(op)) {
    attributes.set(OpTrait::IsOpConfCompatible<void>::getHierarchyAttr(), hierarchy);
  }
  attributes.set(OpTrait::IsOpConfCompatible<void>::getOpNameAttr(),
                 rewriter.getStringAttr(op_name));
  if (auto scope_symbol_id = OpTrait::IsOpConfCompatible<void>::getScopeSymbolID(op)) {
    attributes.set(OpTrait::IsOpConfCompatible<void>::getScopeSymbolIDAttr(), scope_symbol_id);
  }
  return attributes;
}

bool isOneFlowOp(Operation* op) { return llvm::dyn_cast<OneFlowDialect>(op->getDialect()); }
class Outliner {
 private:
  OpBuilder& builder;
  Block* body;
  llvm::DenseSet<Operation*>& visitedOps;
  std::queue<Operation*> worklist{};
  void cloneOpsToNewBody(Operation* op, bool defer = false) {
    if (visitedOps.contains(op)) { return; }
    for (auto operand : op->getOperands()) {
      if (!mapping.lookup(operand)) {
        if (auto defOp = operand.getDefiningOp()) {
          if (isOneFlowOp(defOp)) {
            entries.insert(operand);
            mapping.map(operand, body->addArgument(operand.getType(), operand.getLoc()));
          } else {
            cloneOpsToNewBody(defOp, true);
          }
        }
      }
    }
    ImplicitLocOpBuilder nb(op->getLoc(), builder);
    nb.clone(*op, mapping);
    visitedOps.insert(op);

    for (auto& use : op->getUses()) {
      auto owner = use.getOwner();
      if (isOneFlowOp(owner)) {
        exits.insert(use.get());
      } else {
        if (defer) {
          worklist.push(owner);
        } else {
          cloneOpsToNewBody(owner);
        }
      }
    }
    if (!defer) {
      while (!worklist.empty()) {
        auto op = worklist.front();
        worklist.pop();
        cloneOpsToNewBody(op);
      }
    }
  }

 public:
  Outliner(OpBuilder& builder, Block* body, Operation* op, llvm::DenseSet<Operation*>& visitedOps)
      : builder{builder}, body{body}, visitedOps{visitedOps} {
    cloneOpsToNewBody(op);
  }

  BlockAndValueMapping mapping{};
  llvm::DenseSet<Value> entries{}, exits{};
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
            if (!isOneFlowOp(user)) { entryOps.insert(user); }
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

      SmallVector<::mlir::Value, 4> entries, exits, mappedExits;
      SmallVector<Type, 4> argumentTypes, resultTypes;

      for (auto exit : outliner.exits) {
        exits.push_back(exit);
        mappedExits.push_back(outliner.mapping.lookup(exit));
        resultTypes.push_back(exit.getType());
      }
      builder.setInsertionPointToEnd(block);
      builder.create<func::ReturnOp>(entryOp->getLoc(), mappedExits);

      for (auto argument : outliner.entries) {
        entries.push_back(argument);
        argumentTypes.push_back(argument.getType());
      }
      auto funcType = builder.getFunctionType(argumentTypes, resultTypes);
      if (auto mod = job->getParentOfType<ModuleOp>()) {
        std::string name = "TODO-func_name";

        builder.setInsertionPointToStart(&mod.getRegion().front());
        auto function = builder.create<func::FuncOp>(entryOp->getLoc(), name, funcType);
        function.getBody().push_front(block);

        auto lastOp = exits.end()->getDefiningOp();
        if (!lastOp) {
          job->emitError() << "fail to outline, nowhere to replace";
          signalPassFailure();
        }
        builder.setInsertionPointAfter(lastOp);
        NamedAttrList attributes =
            GetJitOpAttributes(builder, name, argumentTypes.size(), resultTypes.size(),
                               entryOp->getOperand(0).getDefiningOp());
        // auto created = builder.create<MlirJitOp>(entryOp->getLoc(), function, attributes,
        // operands);
        for (const auto& old : llvm::enumerate(exits)) {
          // old.value().replaceAllUsesWith(created->getResult(old.index()));
        }
      } else {
        job->emitError() << "fail to outline";
        signalPassFailure();
      }
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createOutlineJitFunctionPass() {
  return std::make_unique<OutlineJitFunctionPass>();
}

}  // namespace oneflow
}  // namespace mlir
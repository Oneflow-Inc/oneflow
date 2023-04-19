#include "OneFlow/Passes.h"
#include "OneFlow/Transform/OneFlowMemPool.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <glog/logging.h>

namespace mlir {
namespace oneflow {
namespace {

struct InsertOneFlowMemPoolPattern final : public OpRewritePattern<func::FuncOp> {
  Type GetMemPoolElemType(MLIRContext* ctx) const { return IntegerType::get(ctx, 8); }

  // GetAllocOpSize(funop) -> <is_legal, size_of_mem_pool>
  std::pair<bool, unsigned> GetAllocOpSize(func::FuncOp func) const {
    int num = 0, size = 0;
    auto& ops = func->getBlock()->getOperations();
    for (auto& op : ops) {
      if (auto alloc = llvm::dyn_cast_or_null<memref::AllocOp>(op)) {
        if (num > 0) return {false, size};
        num++;
        auto type = alloc->getResult(0).getType().dyn_cast_or_null<MemRefType>();
        if (type.getRank() != 1 && type.getElementType() != GetMemPoolElemType(func->getContext()))
          return {false, size};
        size = type.getDimSize(0);
      }
    }
    return {true, size};
  }

 public:
  explicit InsertOneFlowMemPoolPattern(mlir::MLIRContext* context)
      : OpRewritePattern<func::FuncOp>(context, /*benefit=*/0) {}
  mlir::LogicalResult matchAndRewrite(func::FuncOp op,
                                      mlir::PatternRewriter& rewriter) const override {
    if (op->getAttr(codegen::mempool::MEMPOOL_ATTR_NAME)) return success();

    auto [is_legal, size] = GetAllocOpSize(op);
    if (is_legal) {
      LOG(FATAL) << "you should run -fold-memref-alloc before insert-ofmem-pool pass";
      return failure();
    }

    llvm::SmallVector<Type> new_operand_type;
    Type mempool_type = MemRefType::get({size}, GetMemPoolElemType(op->getContext()));
    new_operand_type.push_back(mempool_type);
    for (auto type : op.getFunctionType().getInputs()) { new_operand_type.push_back(type); }
    auto function_type =
        rewriter.getFunctionType(new_operand_type, op.getFunctionType().getResults());

    auto func = rewriter.create<func::FuncOp>(op.getLoc(), op.getName(), function_type);
    for (auto pair : op->getDialectAttrs()) { func->setAttr(pair.getName(), pair.getValue()); }
    op.getBody().insertArgument(unsigned(0), mempool_type, op->getLoc());
    IRMapping bvm;
    op.getRegion().cloneInto(&func.getRegion(), bvm);
    rewriter.eraseOp(op);
    func->setAttr(codegen::mempool::MEMPOOL_ATTR_NAME, rewriter.getUI32IntegerAttr(size));
    return success();
  }
};

class InsertOneFlowMemPoolPass : public InsertOneFlowMemPoolPassBase<InsertOneFlowMemPoolPass> {
  void runOnOperation() override {
    Operation* op = getOperation();
    auto ctx = op->getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<InsertOneFlowMemPoolPattern>(ctx);
    (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
  }
};

}  // namespace

std::unique_ptr<Pass> createInsertOneFlowMemPoolPass() {
  return std::make_unique<InsertOneFlowMemPoolPass>();
}

}  // namespace oneflow
}  // namespace mlir
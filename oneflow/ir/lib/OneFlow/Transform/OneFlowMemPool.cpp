/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "OneFlow/Passes.h"
#include "OneFlow/Transform/OneFlowMemPool.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "oneflow/core/common/hash_container.h"
#include "oneflow/core/job/intra_job_mem_sharing_util.h"
#include <glog/logging.h>
#include <algorithm>
#include <climits>
#include <tuple>
#include <vector>

namespace mlir {
namespace oneflow {
namespace {

Type getMemPoolElemType(MLIRContext* ctx) { return IntegerType::get(ctx, 8); }

const int align_size_ = ::oneflow::kBlobBodyAlignSize;

struct AllocOpInfo {
  Operation* val_ = nullptr;
  int32_t start_lifetime_ = 0;
  int32_t end_lifetime_ = 0;
  size_t size_ = 0;
};

template<class T>
std::vector<AllocOpInfo> getAllocInfoList(T op) {
  std::vector<memref::AllocOp> list;
  // collect all memref.alloc ops and gpu.launch_func ops
  op->walk([&](memref::AllocOp alloc) { list.push_back(alloc); });

  std::vector<AllocOpInfo> ret;
  for (auto alloc : list) {
    // compute size
    MemRefType type = alloc->getResult(0).getType().dyn_cast<MemRefType>();
    size_t size = type.getElementTypeBitWidth() / 8;
    for (int64_t i : type.getShape()) { size *= i; }
    size = (size / align_size_ + ((size % align_size_) != 0)) * align_size_;
    // compute lifetime
    // TODO: support lifetime analysis
    int32_t start_lifetime = 0;
    int32_t end_lifetime = INT_MAX;
    ret.push_back({alloc, start_lifetime, end_lifetime, size});
  }
  return ret;
}

void replaceAllocwithSubview(func::FuncOp func, OpBuilder& builder,
                             const ::oneflow::MemBlockResultInfo<Operation*>& ret) {
  // create the uni memref.alloc op
  builder.setInsertionPointToStart(&func.getBody().front());
  auto output_type = MemRefType::get({static_cast<long>(ret.mem_block_size)},
                                     getMemPoolElemType(func->getContext()));
  Value mempool = builder.create<memref::AllocOp>(func->getLoc(), output_type);
  // replace alloc with subview
  for (auto [op, offset] : ret.regst_desc2offset) {
    MemRefType type = op->getResult(0).getType().cast<MemRefType>();
    Value byte_shift = builder.create<arith::ConstantIndexOp>(op->getLoc(), offset);
    Value new_op =
        builder.create<memref::ViewOp>(op->getLoc(), type, mempool, byte_shift, ValueRange{});
    op->replaceAllUsesWith(ValueRange{new_op});
    op->erase();
  }
}

bool isMemPool(Operation* op) {
  auto alloc = dyn_cast<memref::AllocOp>(op);
  if (!alloc) return false;
  MemRefType type = alloc->getOpResult(0).getType().cast<MemRefType>();
  if (!type) return false;
  return type.getRank() == 1 && type.getElementType() == getMemPoolElemType(op->getContext());
}

struct InsertOneFlowMemPoolPattern final : public OpRewritePattern<func::FuncOp> {
  // GetAllocOpSize(funop) -> <is_legal, size_of_mem_pool>
  std::pair<bool, memref::AllocOp> getAllocOp(func::FuncOp func) const {
    memref::AllocOp ret;
    auto& ops = func.getBody().front();
    for (auto& op : ops) {
      if (auto alloc = llvm::dyn_cast_or_null<memref::AllocOp>(op)) {
        if (ret) return {false, ret};
        ret = alloc;
      }
    }
    return {true, ret};
  }

  MemRefType getNullMemType(mlir::PatternRewriter& rewriter) const {
    return MemRefType::get({1}, getMemPoolElemType(rewriter.getContext()));
  }

 public:
  explicit InsertOneFlowMemPoolPattern(mlir::MLIRContext* context)
      : OpRewritePattern<func::FuncOp>(context, /*benefit=*/0) {}
  mlir::LogicalResult matchAndRewrite(func::FuncOp op,
                                      mlir::PatternRewriter& rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    if (module && module->getAttr(codegen::mempool::MEMPOOL_ATTR_NAME)) return success();

    auto [is_legal, alloc_op] = getAllocOp(op);
    if (!is_legal) {
      LOG(FATAL) << "you should run -fold-memref-alloc before insert-ofmem-pool pass";
      return failure();
    }

    auto type = alloc_op ? alloc_op->getResult(0).getType().dyn_cast_or_null<MemRefType>()
                         : getNullMemType(rewriter);
    if (type.getRank() != 1 || type.getElementType() != getMemPoolElemType(op->getContext())) {
      LOG(FATAL) << "the alloc op fail to matching memref<?xi8>";
      return failure();
    }
    llvm::SmallVector<Type> new_operand_types;
    new_operand_types.push_back(type);
    for (auto type : op.getFunctionType().getInputs()) { new_operand_types.push_back(type); }
    auto function_type =
        rewriter.getFunctionType(new_operand_types, op.getFunctionType().getResults());

    auto func = rewriter.create<func::FuncOp>(op.getLoc(), op.getName(), function_type);
    for (auto pair : op->getDialectAttrs()) { func->setAttr(pair.getName(), pair.getValue()); }
    op.getBody().insertArgument(unsigned(0), type, op->getLoc());
    if (alloc_op) rewriter.replaceOp(alloc_op, {op.getArgument(0)});
    IRMapping bvm;
    op.getRegion().cloneInto(&func.getRegion(), bvm);
    rewriter.eraseOp(op);
    module->setAttr(codegen::mempool::MEMPOOL_ATTR_NAME,
                    rewriter.getI64IntegerAttr(type.getDimSize(0)));
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

class FoldAllocToSubviewPass : public FoldAllocToSubviewPassBase<FoldAllocToSubviewPass> {
  void runOnOperation() override {
    func::FuncOp op = getOperation();
    applyFoldAlloc(op);
  }
};

}  // namespace

void applyFoldAlloc(func::FuncOp op) {
  std::vector<AllocOpInfo> list;
  // TODO-1: support cpu memory fold
  // TODO-2: support multiple gpu.launch
  op->walk([&](gpu::LaunchOp launchOp) { list = getAllocInfoList(launchOp); });
  op->walk([&](scf::ForallOp launchOp) { list = getAllocInfoList(launchOp); });

  {
    std::vector<AllocOpInfo> body_list;
    body_list = getAllocInfoList(op);
    list.insert(list.end(), body_list.begin(), body_list.end());
  }

  auto ctx = op->getContext();
  OpBuilder builder(ctx);

  // Note: no malloc op should be folded.
  if (!list.size()) { return; }
  // Note: the single malloc op with out type of memref<?xi8> means it has been folded.
  if (list.size() == 1 && oneflow::isMemPool(list.front().val_)) { return; }

  ::oneflow::HashMap<Operation*, std::pair<int32_t, int32_t>> val2lifetime;
  ::oneflow::HashMap<Operation*, size_t> val2size;
  for (const auto& info : list) {
    val2lifetime[info.val_] = {info.start_lifetime_, info.end_lifetime_};
    val2size[info.val_] = info.size_;
  }
  ::oneflow::MemBlockResultInfo<Operation*> ret;

  ::oneflow::MemReusedMemSizeFirstAlgo(false, val2lifetime, val2size, &ret);
  oneflow::replaceAllocwithSubview(op, builder, ret);
}

std::unique_ptr<Pass> createInsertOneFlowMemPoolPass() {
  return std::make_unique<InsertOneFlowMemPoolPass>();
}

std::unique_ptr<Pass> createFoldAllocToSubviewPass() {
  return std::make_unique<FoldAllocToSubviewPass>();
}

}  // namespace oneflow
}  // namespace mlir
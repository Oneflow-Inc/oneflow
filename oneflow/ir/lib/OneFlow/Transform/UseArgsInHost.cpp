
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

namespace {

class UseArgsInHostPass : public UseArgsInHostPassBase<UseArgsInHostPass> {
  void runOnOperation() override;
};

Operation* getFuncElemOp(Operation* op) {
  while (op->getParentOp() && !dyn_cast_or_null<func::FuncOp>(op->getParentOp()))
    op = op->getParentOp();
  return op;
}

LogicalResult setInsertionPoint(Operation* op, OpBuilder& builder) {
  op = getFuncElemOp(op);
  if (!op->getParentOp()) return failure();
  builder.setInsertionPoint(op);
  return success();
}

void UseArgsInHostPass::runOnOperation() {
  func::FuncOp op = getOperation();
  auto ctx = &getContext();
  OpBuilder builder(ctx);

  for (auto arg : op.getArguments()) {
    auto arg_type = arg.getType().dyn_cast_or_null<MemRefType>();
    if (!arg_type)
      LOG(FATAL) << "Fail to work on use-args-in-host pass, you should bufferization first.";
    Value currentVal = arg;

    // Note: sort the user of current argument in order.
    SmallVector<Operation*> list;
    for (auto& elemOp : op.getBody().front()) {
      for (auto* use : arg.getUsers()) {
        if (getFuncElemOp(use) == &elemOp) list.push_back(use);
      }
    };

    auto asyncType = gpu::AsyncTokenType::get(ctx);
    for (auto* owner : list) {
      if (currentVal == arg && !owner->getDialect()->getNamespace().equals("gpu")
          && succeeded(setInsertionPoint(owner, builder))) {
        currentVal = builder.create<memref::AllocOp>(arg.getLoc(), arg_type);
        auto token =
            builder.create<gpu::WaitOp>(arg.getLoc(), asyncType, ValueRange{})->getResult(0);
        token =
            builder
                .create<gpu::MemcpyOp>(arg.getLoc(), asyncType, ValueRange{token}, currentVal, arg)
                ->getResult(0);
        builder.create<gpu::WaitOp>(arg.getLoc(), asyncType, ValueRange{});
        owner->replaceUsesOfWith(arg, currentVal);
      }

      if (currentVal != arg && owner->getDialect()->getNamespace().equals("gpu")
          && succeeded(setInsertionPoint(owner, builder))) {
        auto token =
            builder.create<gpu::WaitOp>(arg.getLoc(), asyncType, ValueRange{})->getResult(0);
        token =
            builder
                .create<gpu::MemcpyOp>(arg.getLoc(), asyncType, ValueRange{token}, arg, currentVal)
                ->getResult(0);
        builder.create<gpu::WaitOp>(arg.getLoc(), asyncType, ValueRange{});
        currentVal = arg;
      }
    }
  }
}

}  // namespace

namespace oneflow {

std::unique_ptr<Pass> createUseArgsInHostPass() { return std::make_unique<UseArgsInHostPass>(); }

}  // namespace oneflow
}  // namespace mlir
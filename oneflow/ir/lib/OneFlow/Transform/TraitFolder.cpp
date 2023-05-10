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
class TestOneFlowTraitFolderPass
    : public TestOneFlowTraitFolderPassBase<TestOneFlowTraitFolderPass> {
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

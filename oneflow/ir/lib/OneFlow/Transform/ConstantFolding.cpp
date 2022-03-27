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
#include <iostream>
#include <string>
#include "OneFlow/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

class ConstantFoldingPass : public ConstantFoldingPassBase<ConstantFoldingPass> {
  void runOnOperation() override {
    Operation* op = getOperation();
    RewritePatternSet patterns(op->getContext());
    oneflow::populateConstantFolding(patterns);
    (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
  }
};

class PostConstantFoldingPass : public PostConstantFoldingPassBase<PostConstantFoldingPass> {
  void runOnOperation() override {
    Operation* op = getOperation();
    RewritePatternSet patterns(op->getContext());
    oneflow::populatePostConstantFolding(patterns);
    (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
  }
};

}  // namespace

namespace mlir {

namespace oneflow {

std::unique_ptr<Pass> createConstantFoldingPass() {
  return std::make_unique<ConstantFoldingPass>();
}

std::unique_ptr<Pass> createPostConstantFoldingPass() {
  return std::make_unique<PostConstantFoldingPass>();
}

}  // namespace oneflow

}  // namespace mlir

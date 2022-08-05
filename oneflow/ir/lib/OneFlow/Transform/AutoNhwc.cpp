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

class AutoNhwcPass : public AutoNhwcPassBase<AutoNhwcPass> {
  void runOnOperation() override {
    Operation* op = getOperation();
    RewritePatternSet patterns(op->getContext());
    oneflow::populateAutoNhwcPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
  }
};

}  // namespace

namespace mlir {

namespace oneflow {

std::unique_ptr<Pass> createAutoNhwcPass() { return std::make_unique<AutoNhwcPass>(); }

}  // namespace oneflow

}  // namespace mlir

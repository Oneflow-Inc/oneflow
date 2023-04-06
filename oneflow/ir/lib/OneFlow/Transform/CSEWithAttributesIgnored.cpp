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
#include "OneFlow/OneFlowOps.h"
#include "OneFlow/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir {

namespace oneflow {

namespace {

static const auto MAGIC_OP_NAME = "ONEFLOW_ERASE_MAGIC";
static const auto MAGIC_SCOPE_SYMBOL_ID = 77777;

struct EraseAttributes : public mlir::OpInterfaceRewritePattern<UserOpCompatible> {
  explicit EraseAttributes(mlir::MLIRContext* context, std::shared_ptr<CSEState> state)
      : OpInterfaceRewritePattern<UserOpCompatible>(context, /*benefit=*/1), state_{state} {}
  mlir::LogicalResult matchAndRewrite(UserOpCompatible op,
                                      mlir::PatternRewriter& rewriter) const override {
    if (op->getAttrOfType<StringAttr>(OpTrait::IsOpConfCompatible<void>::getOpNameAttr())
            .getValue()
            .str()
        != MAGIC_OP_NAME) {
      if (state_) {
        state_->opNames[op] =
            op->getAttrOfType<StringAttr>(OpTrait::IsOpConfCompatible<void>::getOpNameAttr());
        state_->scopeSymbolIDs[op] = op->getAttrOfType<IntegerAttr>(
            OpTrait::IsOpConfCompatible<void>::getScopeSymbolIDAttr());
      }
      op->setAttr(OpTrait::IsOpConfCompatible<void>::getOpNameAttr(),
                  rewriter.getStringAttr(MAGIC_OP_NAME));
      op->setAttr(OpTrait::IsOpConfCompatible<void>::getScopeSymbolIDAttr(),
                  rewriter.getI64IntegerAttr(MAGIC_SCOPE_SYMBOL_ID));
      return success();
    } else {
      return failure();
    }
  }

 private:
  std::shared_ptr<CSEState> state_;
};

struct PutAttributes : public mlir::OpInterfaceRewritePattern<UserOpCompatible> {
  explicit PutAttributes(mlir::MLIRContext* context, std::shared_ptr<CSEState> state)
      : OpInterfaceRewritePattern<UserOpCompatible>(context, /*benefit=*/1), state_{state} {}
  mlir::LogicalResult matchAndRewrite(UserOpCompatible op,
                                      mlir::PatternRewriter& rewriter) const override {
    if (op->getAttrOfType<StringAttr>(OpTrait::IsOpConfCompatible<void>::getOpNameAttr())
            .getValue()
            .str()
        == MAGIC_OP_NAME) {
      if (state_) {
        op->setAttr(OpTrait::IsOpConfCompatible<void>::getOpNameAttr(), state_->opNames[op]);
        op->setAttr(OpTrait::IsOpConfCompatible<void>::getScopeSymbolIDAttr(),
                    state_->scopeSymbolIDs[op]);
      }
      return success();
    } else {
      return failure();
    }
  }

 private:
  std::shared_ptr<CSEState> state_;
};

class CSEWithAttributesIgnored : public CSEWithAttributesIgnoredBase<CSEWithAttributesIgnored> {
 public:
  explicit CSEWithAttributesIgnored() {}
  explicit CSEWithAttributesIgnored(std::shared_ptr<CSEState> state) : state_(state) {}
  void runOnOperation() override {
    Operation* op = getOperation();
    RewritePatternSet patterns(op->getContext());
    patterns.add<EraseAttributes>(op->getContext(), state_);
    (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
  }

 private:
  std::shared_ptr<CSEState> state_;
};

class CSEPutAttributes : public CSEPutAttributesBase<CSEPutAttributes> {
 public:
  explicit CSEPutAttributes() {}
  explicit CSEPutAttributes(std::shared_ptr<CSEState> state) { state_ = state; }

  void runOnOperation() override {
    Operation* op = getOperation();
    RewritePatternSet patterns(op->getContext());
    patterns.add<PutAttributes>(op->getContext(), state_);
    (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
  }

 private:
  std::shared_ptr<CSEState> state_;
};

}  // namespace

std::unique_ptr<Pass> createCSEWithAttributesIgnored() {
  return std::make_unique<CSEWithAttributesIgnored>();
}

std::unique_ptr<Pass> createCSEPutAttributes() { return std::make_unique<CSEPutAttributes>(); }

std::pair<std::unique_ptr<Pass>, std::unique_ptr<Pass>> createCSEPasses(
    std::shared_ptr<CSEState> state) {
  return std::make_pair(std::make_unique<CSEWithAttributesIgnored>(state),
                        std::make_unique<CSEPutAttributes>(state));
}

void registerCSEPasses(std::shared_ptr<CSEState> state) {
  ::mlir::registerPass([state]() -> std::unique_ptr<::mlir::Pass> {
    return std::make_unique<CSEWithAttributesIgnored>(state);
  });
  ::mlir::registerPass([state]() -> std::unique_ptr<::mlir::Pass> {
    return std::make_unique<CSEPutAttributes>(state);
  });
}

}  // namespace oneflow

}  // namespace mlir

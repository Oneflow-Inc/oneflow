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
#include "OneFlow/Transform/InferPlacement.h"

#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/OneFlowOps.h"
#include "OneFlow/OneFlowOpTraits.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"

namespace mlir {
namespace oneflow {
namespace lite {

static bool CanScheduleOnTarget(Operation* op, StringRef target) {
  if (!op->hasTrait<OpTrait::IsOpConfCompatible>()) { return false; }
  if (llvm::dyn_cast<oneflow::InputOp>(op) || llvm::dyn_cast<oneflow::OutputOp>(op)) {
    return false;
  }
  // TODO()
  return true;
}

struct InferPlacementPass : public PassWrapper<InferPlacementPass, OperationPass<ModuleOp>> {
  StringRef target_;
  explicit InferPlacementPass(StringRef target) : target_(target) {}

  void runOnOperation() override;
};

void InferPlacementPass::runOnOperation() {
  getOperation().walk([&](Operation* op) {
    if (!op->hasTrait<OpTrait::IsOpConfCompatible>()) { return; }
    auto target = [&]() -> StringRef {
      if (CanScheduleOnTarget(op, target_)) { return target_; }
      return StringRef("host");
    }();

    OpBuilder builder(&getContext());
    op->setAttr(OpTrait::IsOpConfCompatible<void>::getDeviceTagAttr(),
                builder.getStringAttr(target));
  });
}

std::unique_ptr<mlir::Pass> createLiteInferPlacementPass(StringRef target) {
  return std::unique_ptr<mlir::Pass>(new InferPlacementPass(target));
}

}  // namespace lite
}  // namespace oneflow
}  // namespace mlir

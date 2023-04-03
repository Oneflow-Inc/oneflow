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
#include "OneFlow/Transform/InsertTransferOp.h"

#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/OneFlowOps.h"
#include "OneFlow/OneFlowOpTraits.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"

namespace mlir {
namespace oneflow {
namespace lite {

struct InsertTransferOpPass : public PassWrapper<InsertTransferOpPass, OperationPass<ModuleOp>> {
  void runOnOperation() override;

  StringAttr InferTargetDevice(StringAttr from, StringAttr to) const;
};

StringAttr InsertTransferOpPass::InferTargetDevice(StringAttr from, StringAttr to) const {
  auto IsHostDevice = [](StringAttr device) {
    return device == "host" || device == "cpu" || device == "x86" || device == "arm";
  };
  return IsHostDevice(from) ? to : from;
}

void InsertTransferOpPass::runOnOperation() {
  auto opNameAttrkey = OpTrait::IsOpConfCompatible<void>::getOpNameAttr();
  auto deviceTagAttrKey = OpTrait::IsOpConfCompatible<void>::getDeviceTagAttr();
  auto deviceNameAttrKey = OpTrait::IsOpConfCompatible<void>::getDeviceNameAttr();

  OpBuilder builder(&getContext());

  getOperation().walk([&](Operation* op) {
    if (!op->hasTrait<OpTrait::IsOpConfCompatible>()) { return; }
    auto device = op->getAttrOfType<StringAttr>(deviceTagAttrKey);

    for (Value result : op->getResults()) {
      llvm::DenseMap<StringAttr, SmallVector<OpOperand*, 4>> operandsToReplace;
      for (auto& use : result.getUses()) {
        if (!use.getOwner()->hasTrait<OpTrait::IsOpConfCompatible>()) { continue; }
        auto use_device = use.getOwner()->getAttrOfType<StringAttr>(deviceTagAttrKey);
        if (use_device != device) { operandsToReplace[use_device].push_back(&use); }
      }
      for (const auto& it : operandsToReplace) {
        NamedAttrList attrs;
        attrs.set(opNameAttrkey, builder.getStringAttr("copy"));
        attrs.set(deviceTagAttrKey, InferTargetDevice(device, it.first));
        attrs.set(deviceNameAttrKey,
                  builder.getArrayAttr(llvm::to_vector<8>(llvm::map_range(
                      ArrayRef<StringRef>({"@0:0"}),
                      [&](StringRef v) -> Attribute { return builder.getStringAttr(v); }))));
        attrs.set(builder.getStringAttr("device_type"), it.first);

        builder.setInsertionPointAfter(op);
        SmallVector<mlir::Value, 4> operands{result};
        auto copy_op =
            builder.create<oneflow::CopyOp>(op->getLoc(), op->getResultTypes(), operands, attrs);

        for (OpOperand* operand : it.second) {
          operand->getOwner()->setOperand(operand->getOperandNumber(), copy_op.out());
        }
      }
    }
  });
}

std::unique_ptr<mlir::Pass> createLiteInsertTransferOpPass() {
  return std::unique_ptr<mlir::Pass>(new InsertTransferOpPass());
}

}  // namespace lite
}  // namespace oneflow
}  // namespace mlir

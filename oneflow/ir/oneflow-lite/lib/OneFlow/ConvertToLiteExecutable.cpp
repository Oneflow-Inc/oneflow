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
#include "OneFlow/ConvertToLiteExecutable.h"

#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/OneFlowOps.h"
#include "OneFlow/OneFlowOpTraits.h"
#include "OneFlow/Passes.h"
#include "OneFlow/OneFlowUtils.h"
#include "OneFlow/Transform/FoldVariable.h"
#include "OneFlow/Transform/InferPlacement.h"
#include "OneFlow/Transform/InsertTransferOp.h"
#include "OneFlow/Transform/MemoryPlanning.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-qual"
#include "schemas/executable_generated.h"
#pragma GCC diagnostic pop

namespace mlir {
namespace oneflow {

namespace lite {

static oneflow_lite_OpDef_ref_t createLiteOpDef(
    FlatbufferBuilder& builder, Operation* op,
    const llvm::DenseMap<StringAttr, int>& deviceOrdering) {
  oneflow_lite_OpDef_start(builder);

  return oneflow_lite_OpDef_end(builder);
}

LogicalResult ConvertToLiteExecutable(MLIRContext* context, ModuleOp module, ConvertOptions options,
                                      llvm::SmallVector<uint8_t, 32>* executable) {
  mlir::PassManager pm(context);
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createLiteFoldVariablePass());
  pm.addPass(createLiteInferPlacementPass(options.target));
  pm.addPass(createLiteInsertTransferOpPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createLiteMemoryPlanningPass());
  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Failed to run oneflow lite compilation passes.\n";
    return failure();
  }

  Operation* root = nullptr;
  module.getOperation()->walk([&](oneflow::Job job) -> WalkResult {
    root = job.getOperation();
    return WalkResult::interrupt();
  });
  if (!root) {
    llvm::errs() << "Job not found in module: " << *module;
    return failure();
  }
  llvm::errs() << *module << "\n";
  auto funcName = root->getAttrOfType<StringAttr>("sym_name");

  const llvm::DenseMap<StringAttr, size_t>& deviceSegmentSize = getDeviceSegmentSize();
  const llvm::DenseMap<Value, size_t>& valueSegmentOffset = getValueSegmentOffset();

  FlatbufferBuilder builder;
  oneflow_lite_ExecutableDef_start_as_root(builder);
  oneflow_lite_ExecutableDef_version_add(builder, 0);
  oneflow_lite_ExecutableDef_name_add(builder, builder.createString(funcName.getValue()));

  llvm::DenseMap<StringAttr, int> deviceOrdering;
  llvm::SmallVector<StringRef, 4> devices;
  for (const auto& it : deviceSegmentSize) {
    deviceOrdering[it.first] = deviceOrdering.size();
    devices.push_back(it.first.getValue());
  }
  oneflow_lite_ExecutableDef_devices_add(builder, builder.createStringVec(devices));

  llvm::SmallVector<oneflow_lite_OpDef_ref_t, 4> ops;
  root->walk([&](Operation* op) {
    if (!op->hasTrait<OpTrait::IsOpConfCompatible>() || llvm::dyn_cast<InputOp>(op)
        || llvm::dyn_cast<OutputOp>(op)) {
      return;
    }
    ops.push_back(createLiteOpDef(builder, op, deviceOrdering));
  });
  oneflow_lite_ExecutableDef_ops_add(builder, builder.createOffsetVecDestructive(ops));

  oneflow_lite_ExecutableDef_end_as_root(builder);

  size_t packedSize = flatcc_builder_get_buffer_size(builder);
  executable->resize(packedSize);
  flatcc_builder_copy_buffer(builder, executable->data(), packedSize);
  return success();
}

}  // namespace lite

}  // namespace oneflow
}  // namespace mlir

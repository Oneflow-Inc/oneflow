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

namespace mlir {
namespace oneflow {

namespace lite {

LogicalResult ConvertToLiteExecutable(MLIRContext* context, ModuleOp module, ConvertOptions options,
                                      LiteExecutable* executable) {
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

  Operation* job_op = nullptr;
  auto find_first_job = [&](oneflow::Job job) -> WalkResult {
    job_op = job.getOperation();
    return WalkResult::interrupt();
  };
  module.getOperation()->walk(find_first_job);
  if (!job_op) {
    llvm::errs() << "Job not found in module: " << *module;
    return failure();
  }
  llvm::errs() << *module << "\n";
  return success();
}

}  // namespace lite

}  // namespace oneflow
}  // namespace mlir

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
#include "OneFlow/Transform/LoweringLaunchJob.h"

#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/OneFlowOps.h"
#include "OneFlow/OneFlowOpTraits.h"
#include "OneFlow/OneFlowLiteUtils.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"

#ifdef LITE_USE_ASCEND_NPU
#include "OneFlow/Transform/Lowering/LoweringAscend.h"
#endif  // LITE_USE_ASCEND_NPU

namespace mlir {
namespace oneflow {
namespace lite {

struct LoweringLaunchJobPass : public PassWrapper<LoweringLaunchJobPass, OperationPass<ModuleOp>> {
  StringRef checkpointDir;

  explicit LoweringLaunchJobPass(StringRef checkpointDir) : checkpointDir(checkpointDir) {}

  void runOnOperation() override;

  LogicalResult loweringLaunchJob(OpBuilder& builder, Operation* callee, StringRef backend,
                                  llvm::SmallVector<uint8_t, 4>* loweringData);
};

LogicalResult LoweringLaunchJobPass::loweringLaunchJob(
    OpBuilder& builder, Operation* callee, StringRef backend,
    llvm::SmallVector<uint8_t, 4>* loweringData) {
  if (backend == "ascend") {
#ifdef LITE_USE_ASCEND_NPU
    return loweringAscend(builder, callee, checkpointDir, loweringData);
#else
    llvm::errs() << "please recompile with LITE_USE_ASCEND_NPU=ON\n";
    return failure();
#endif  // LITE_USE_ASCEND_NPU
  } else {
    llvm::errs() << "lowering for backend " << backend << " is not supported yet\n";
    return failure();
  }
  return success();
}

void LoweringLaunchJobPass::runOnOperation() {
  SmallVector<Operation*, 4> launchOps;
  Operation* entryJobOp = getEntryJobOp(getOperation());
  entryJobOp->walk([&](Operation* op) {
    if (dyn_cast<oneflow::MlirJitOp>(op)) { launchOps.push_back(op); }
  });

  SymbolTable symbolTable(getOperation());
  OpBuilder builder(&getContext());

  // TODO(): register backend converters
  for (Operation* op : launchOps) {
    auto launchOp = dyn_cast<oneflow::MlirJitOp>(op);
    Operation* callee = symbolTable.lookup(launchOp.callee());
    if (!callee) {
      llvm::errs() << "can not find a callee named " << launchOp.callee() << "\n";
      return signalPassFailure();
    }
    llvm::SmallVector<uint8_t, 4> loweringData;
    if (failed(loweringLaunchJob(builder, callee, launchOp.device_tag(), &loweringData))) {
      llvm::errs() << "failed to lowerring job " << launchOp.callee() << "\n";
    }
    op->setAttr("mlir_assembly",
                builder.getStringAttr(StringRef(reinterpret_cast<const char*>(loweringData.data()),
                                                loweringData.size())));
  }
}

std::unique_ptr<mlir::Pass> createLiteLoweringLaunchJobPass(StringRef checkpointDir) {
  return std::unique_ptr<mlir::Pass>(new LoweringLaunchJobPass(checkpointDir));
}

}  // namespace lite
}  // namespace oneflow
}  // namespace mlir

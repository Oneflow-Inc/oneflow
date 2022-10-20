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
#include "OneFlow/OKL/Conversion/Conversion.h"
#include "OneFlow/OKL/Conversion/FetchFromLauncher.h"
#include "OneFlow/OKL/Conversion/OKLToLLVM.h"
#include "OneFlow/OKL/Conversion/OnlyKeepComputeOps.h"
#include "OneFlow/OKL/Conversion/SplitIntoFuncs.h"
#include "OneFlow/Passes.h"
#include "OneFlow/Transform/OutlineAndFuse.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Pass/PassManager.h"
#include "oneflow/ir/include/OneFlow/OneFlowUtils.h"
namespace mlir {
namespace okl {

LogicalResult LowerWrapOpsToOKL(ModuleOp module) {
  PassManager pm(module->getContext());
  pm.addPass(oneflow::createExtractKernelLaunchTensorPass());  // extract-kernel-launch-tensor
  pm.addPass(oneflow::createTrimReturnAsVoidPass());           // trim-return-as-void
  pm.addPass(oneflow::createLowerToOKLPass());                 // lower-to-okl
  pm.addPass(createSplitIntoFuncsPass());                      // split-into-funcs
  pm.addPass(createFetchFromLauncherPass());                   // fetch-from-launcher
  oneflow::CheckEnableIRPrinting(pm);
  return pm.run(module);
}

LogicalResult LowerOKLComputeToLLVM(ModuleOp module) {
  PassManager pm(module->getContext());
  pm.addPass(createOnlyKeepComputeOpsPass());        // only-keep-compute-ops
  pm.addPass(createLowerOKLToLLVMFuncPass());        // lower-okl-to-llvm-func
  pm.addPass(createLowerOKLToLLVMCallPass());        // lower-okl-to-llvm-call
  pm.addPass(createReconcileUnrealizedCastsPass());  // reconcile-unrealized-casts
  oneflow::CheckEnableIRPrinting(pm);
  return pm.run(module);
}

}  // namespace okl
}  // namespace mlir

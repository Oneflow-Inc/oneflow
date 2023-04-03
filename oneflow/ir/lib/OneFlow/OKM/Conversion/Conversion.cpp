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
#include "OneFlow/OKM/passes.h"
#include "OneFlow/OneFlowUtils.h"

namespace mlir {
namespace okm {

LogicalResult LowerWrapOpsToOKL(ModuleOp module) {
  PassManager pm(module->getContext());
  pm.addPass(createExtractOKMTensorPass());
  pm.addPass(createWrapOKMKernelPass());
  pm.addPass(createOptOKMMemrefPass());
  pm.addPass(createConvertOKMToOKLPass());
  pm.addPass(okl::createTagCudaGraphSupportPass());
  oneflow::CheckEnableIRPrinting(pm);
  return pm.run(module);
}

}  // namespace okm
}  // namespace mlir

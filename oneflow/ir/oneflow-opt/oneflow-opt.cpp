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
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "OneFlow/SBP/SBPDialect.h"
#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/OneFlowOps.h"
#include "OneFlow/Passes.h"

namespace mlir {
struct TestOneFlowTraitFolder
    : public PassWrapper<TestOneFlowTraitFolder, OperationPass<func::FuncOp>> {
  void runOnOperation() override {
    if (failed(applyPatternsAndFoldGreedily(getOperation(), RewritePatternSet(&getContext())))) {
      exit(1);
    }
  }
  StringRef getArgument() const final { return "test-oneflow-trait-folder"; }
};
void registerTestOneFlowTraitsPass() { PassRegistration<TestOneFlowTraitFolder>(); }

}  // namespace mlir

int32_t main(int32_t argc, char** argv) {
  mlir::registerAllPasses();
  mlir::registerTestOneFlowTraitsPass();
  mlir::registerLowerOneFlowToTosaPassPass();
  mlir::registerGpuMapParallelLoopsPassPass();
  mlir::registerBufferHostRegisterPassPass();
  mlir::registerGpuCopyArgPassPass();
#ifdef WITH_MLIR_CUDA_CODEGEN
  mlir::oneflow::registerGpuSerializeToCubinPass();
#endif  // WITH_MLIR_CUDA_CODEGEN
  mlir::registerOutlineJitFunctionPassPass();
  mlir::DialectRegistry registry;
  registry.insert<mlir::sbp::SBPDialect>();
  registry.insert<mlir::oneflow::OneFlowDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::tosa::TosaDialect>();
  registry.insert<mlir::linalg::LinalgDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  registry.insert<mlir::gpu::GPUDialect>();
  registry.insert<mlir::AffineDialect>();
  registry.insert<mlir::bufferization::BufferizationDialect>();
  return failed(mlir::MlirOptMain(argc, argv, "OneFlow optimizer driver\n", registry));
}

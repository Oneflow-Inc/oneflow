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
#include "mlir/Support/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "OneFlow/OneFlowDialect.h"

namespace mlir {
struct TestOneFlowTraitFolder : public PassWrapper<TestOneFlowTraitFolder, FunctionPass> {
  void runOnFunction() override {
    applyPatternsAndFoldGreedily(getFunction(), OwningRewritePatternList());
  }
};
void registerTestOneFlowTraitsPass() {
  PassRegistration<TestOneFlowTraitFolder>("test-oneflow-trait-folder", "Run trait folding");
}
}  // namespace mlir

int main(int argc, char** argv) {
  mlir::registerAllPasses();
  mlir::registerTestOneFlowTraitsPass();
  mlir::DialectRegistry registry;
  registry.insert<mlir::oneflow::OneFlowDialect>();
  registry.insert<mlir::StandardOpsDialect>();
  return failed(mlir::MlirOptMain(argc, argv, "OneFlow optimizer driver\n", registry));
}

//===- oneflow-translate.cpp ---------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Translate a list of OneFlow user ops to MLIR.
// In principle, the list of user ops should be a graph
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Module.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Translation.h"

#include "OneFlow/OneFlowDialect.h"

using namespace mlir;
namespace {
OwningModuleRef translateOneFlowJobToModule(llvm::StringRef str, MLIRContext *context) {
  OwningModuleRef module(
      ModuleOp::create(FileLineColLoc::get("", /*line=*/0, /*column=*/0, context)));
  return module;
}
}  // namespace
namespace mlir {
void registerFromOneFlowJobTranslation() {
  TranslateToMLIRRegistration fromOneFlowJob("import-oneflow-job",
                                             [](llvm::StringRef str, MLIRContext *context) {
                                               return ::translateOneFlowJobToModule(str, context);
                                             });
}
}  // namespace mlir

int main(int argc, char **argv) {
  mlir::registerAllTranslations();

  registerFromOneFlowJobTranslation();

  return failed(mlir::mlirTranslateMain(argc, argv, "MLIR Translation Testing Tool"));
}

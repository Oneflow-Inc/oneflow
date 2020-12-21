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

#include <google/protobuf/text_format.h>
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include <iostream>

namespace mlir {

namespace {
using PbMessage = google::protobuf::Message;

OwningModuleRef translateOneFlowJobToModule(llvm::StringRef str, MLIRContext *context) {
  std::string cpp_str = str.str();
  ::oneflow::Job job;
  google::protobuf::TextFormat::ParseFromString(cpp_str, &job);
  OwningModuleRef module(
      ModuleOp::create(FileLineColLoc::get("", /*line=*/0, /*column=*/0, context)));
  for (size_t i = 0; i < job.net().op_size(); i++) {
    ::oneflow::OperatorConf op = job.net().op(i);
    if (op.has_user_conf()) {
      std::cout << "loading user op: " << op.name() << "\n";
      std::cout << op.DebugString() << "\n";
    }
  }
  return module;
}
}  // namespace

void registerFromOneFlowJobTranslation() {
  TranslateToMLIRRegistration fromOneFlowJob("import-oneflow-job",
                                             [](llvm::StringRef str, MLIRContext *context) {
                                               return translateOneFlowJobToModule(str, context);
                                             });
}

}  // namespace mlir

int main(int argc, char **argv) {
  mlir::registerAllTranslations();

  mlir::registerFromOneFlowJobTranslation();

  return failed(mlir::mlirTranslateMain(argc, argv, "MLIR Translation Testing Tool"));
}

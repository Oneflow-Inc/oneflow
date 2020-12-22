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

#include "OneFlow/OneFlowOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Module.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Translation.h"
#include "mlir/IR/Builders.h"

#include "OneFlow/OneFlowDialect.h"

#include <google/protobuf/text_format.h>
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include <iostream>
#include <new>

namespace mlir {

namespace {

using PbMessage = google::protobuf::Message;

class Importer {
 public:
  Importer(MLIRContext *context, ModuleOp module)
      : b(context),
        context(context),
        module(module),
        unknownLoc(FileLineColLoc::get("imported-protobuf", 0, 0, context)) {}
  LogicalResult processUserOp(const ::oneflow::OperatorConf &op);
  LogicalResult processJob(::oneflow::Job *job);

 private:
  /// The current builder, pointing at where the next Instruction should be
  /// generated.
  OpBuilder b;
  /// The current context.
  MLIRContext *context;
  /// The current module being created.
  ModuleOp module;
  /// Cached FileLineColLoc::get("imported-protobuf", 0, 0).
  Location unknownLoc;
};

LogicalResult Importer::processUserOp(const ::oneflow::OperatorConf &op) {
  if (op.has_user_conf() == false) { return failure(); }
  const std::string &type_name = op.user_conf().op_type_name();
  if (type_name == "relu") {
    return success();
  } else if (type_name == "constant") {
    if (op.user_conf().attr().at("is_floating_value").at_bool()) {
      b.create<oneflow::ConstantOp>(unknownLoc,
                                    op.user_conf().attr().at("floating_value").at_double());
      std::cout << "processed user op: " << op.name() << "\n\n";
    } else {
      // b.create<ConstantOp>(unknownLoc, op.user_conf().attr().at("integer_value").at_int64());
    }
    return success();
  } else {
    return failure();
  }
}

LogicalResult Importer::processJob(::oneflow::Job *job) {
  for (size_t i = 0; i < job->net().op_size(); i++) {
    ::oneflow::OperatorConf op = job->net().op(i);
    if (op.has_user_conf()) {
      std::cout << "processing user op: " << op.name() << "\n";
      std::cout << op.DebugString() << "\n";
      if (failed(processUserOp(op))) { return failure(); }
    }
  }
  return success();
}

OwningModuleRef translateOneFlowJobToModule(llvm::StringRef str, MLIRContext *context) {
  std::string cpp_str = str.str();
  ::oneflow::Job job;
  google::protobuf::TextFormat::ParseFromString(cpp_str, &job);
  context->loadDialect<oneflow::OneFlowDialect>();
  OwningModuleRef module(
      ModuleOp::create(FileLineColLoc::get("", /*line=*/0, /*column=*/0, context)));
  Importer imp(context, module.get());
  if (failed(imp.processJob(&job))) { return {}; }
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

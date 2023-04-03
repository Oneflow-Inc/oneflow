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
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"

#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Transforms/Passes.h"

#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/ConvertToLiteExecutable.h"

namespace mlir {
namespace oneflow {
namespace lite {

LogicalResult Compile(int argc, char** argv) {
  llvm::InitLLVM y(argc, argv);
  static llvm::cl::OptionCategory mainOptions("OneFlowLite Compile Main Options");

  llvm::cl::opt<std::string> inputFiledir(llvm::cl::Positional,
                                          llvm::cl::desc("<Input saved model directory>"),
                                          llvm::cl::Required, llvm::cl::cat(mainOptions));

  llvm::cl::opt<std::string> outputFilename("o", llvm::cl::desc("Output filename"),
                                            llvm::cl::value_desc("filename"), llvm::cl::init("-"),
                                            llvm::cl::cat(mainOptions));

  llvm::cl::list<std::string> targets("targets",
                                      llvm::cl::desc("Target backends for executable compilation"),
                                      llvm::cl::ZeroOrMore, llvm::cl::cat(mainOptions));

  llvm::cl::ParseCommandLineOptions(argc, argv, "OneFlowLite compile\n");

  llvm::SmallString<128> inputFilename = StringRef(inputFiledir + "/model.mlir");
  llvm::sys::path::native(inputFilename);

  mlir::MLIRContext context;
  context.getOrLoadDialect<oneflow::OneFlowDialect>();
  context.loadDialect<mlir::func::FuncDialect>();

  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(inputFilename, &context);

  ConvertOptions options;
  options.checkpointDir = inputFiledir;
  if (targets.empty()) {
    options.target = "host";
  } else {
    if (targets.size() > 1) {
      llvm::errs() << "Support only one target currently.\n";
      return failure();
    }
    options.target = targets[0];
  }
  llvm::errs() << "Enable compilation for target: " << options.target << "\n";

  llvm::SmallVector<uint8_t, 32> executable;
  if (failed(ConvertToLiteExecutable(&context, module.get(), options, &executable))) {
    return failure();
  }
  std::string errorMessage;
  auto output = mlir::openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }
  output->os().write(reinterpret_cast<char*>(executable.data()), executable.size());
  output->keep();
  return success();
}

}  // namespace lite
}  // namespace oneflow
}  // namespace mlir

int main(int argc, char** argv) {
  if (mlir::failed(mlir::oneflow::lite::Compile(argc, argv))) { return 1; }
  return 0;
}

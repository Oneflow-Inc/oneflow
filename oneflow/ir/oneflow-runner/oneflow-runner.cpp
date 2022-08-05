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
//===- mlir-cpu-runner.cpp - MLIR CPU Execution Driver---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry point to a command line utility that executes an MLIR file on the
// CPU by  translating MLIR to LLVM IR before JIT-compiling and executing the
// latter.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/ExecutionEngine/JitRunner.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"

#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"
#include "OneFlow/OneFlowDialect.h"

int main(int argc, char** argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  // llvm::InitializeNativeTargetAsmParser(); // link fails

  mlir::DialectRegistry registry;
  mlir::registerAllToLLVMIRTranslations(registry);
  registry.insert<mlir::oneflow::OneFlowDialect>();
  return mlir::JitRunnerMain(argc, argv, registry);
}

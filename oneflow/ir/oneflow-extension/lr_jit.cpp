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
#include "PyAst/Ast.h"
#include "PyAst/AstMlirGen.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/Transforms/RequestCWrappers.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/InitAllDialects.h"
#include "mlir/IR/Builders.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/MemRefUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/StringRef.h"

#include <glog/logging.h>
#include <numeric>
#include <any>
#include <functional>
#include <memory>

using llvm::ArrayRef;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

static struct LLVMInitializer {
  LLVMInitializer() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
  }
} initializer;

static mlir::LogicalResult lowerToLLVMDialect(mlir::ModuleOp module) {
  mlir::PassManager pm(module.getContext());

  pm.addNestedPass<mlir::func::FuncOp>(mlir::LLVM::createRequestCWrappersPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createMemRefToLLVMPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.addPass(mlir::cf::createConvertControlFlowToLLVMPass());
  pm.addPass(mlir::createConvertMathToLLVMPass());
  pm.addPass(mlir::arith::createArithmeticExpandOpsPass());
  pm.addPass(mlir::arith::createConvertArithmeticToLLVMPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  return pm.run(module);
}

// generate a simple mlir module for test
static mlir::OwningOpRef<mlir::ModuleOp> GenModuleForTest(mlir::MLIRContext& context) {
  std::string moduleStr = R"mlir(
  func.func @get_lr(%arg0 : f32, %arg1 : i32) -> f32 attributes { llvm.emit_c_interface } {
    return %arg0 : f32
  }
  )mlir";
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(moduleStr, &context);
  return module;
}

// generate a module op from a function def python ast
static mlir::OwningOpRef<mlir::ModuleOp> GenModule(mlir::MLIRContext& context,
                                                   pyast::FunctionDef& ast) {
  using namespace pyast;

  MLIRGenImpl mlir_gen(context);
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir_gen.GenModule(&ast);
  // module->dump();
  return module;
}

// generate store of lr jit registry from a function def python ast
static LRJITRegistry_Store_ GenFunc(pyast::FunctionDef& ast, bool is_dump) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::MLIRContext context(registry);
  context.loadDialect<mlir::memref::MemRefDialect>();
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::arith::ArithmeticDialect>();
  context.loadDialect<mlir::math::MathDialect>();
  context.loadDialect<mlir::scf::SCFDialect>();
  context.loadDialect<mlir::cf::ControlFlowDialect>();
  context.loadDialect<mlir::AffineDialect>();

  auto module = GenModule(context, ast);
  if (is_dump) { module->dump(); }
  // auto module = genModuleForTest(context);
  CHECK(!!module) << "failed to parse module";
  CHECK(succeeded(lowerToLLVMDialect(*module))) << "failed to lower to llvm dialect";
  auto jit_or_err = mlir::ExecutionEngine::create(*module);
  CHECK(jit_or_err) << "failed to create JIT exe engine, "
                    << llvm::toString(jit_or_err.takeError());

  std::shared_ptr<mlir::ExecutionEngine> engine = cantFail(std::move(jit_or_err));

  std::weak_ptr<mlir::ExecutionEngine> engine_ = engine;

  auto func = [engine_](double base_lr, double step) {
    float res = 0;
    if (!engine_.expired()) {
      auto engine = engine_.lock();
      auto&& out = mlir::ExecutionEngine::result(res);
      auto base_lr_jit = static_cast<float>(base_lr);
      auto step_jit = static_cast<float>(step);
      auto err = engine->invoke("get_lr", base_lr_jit, step_jit, out);
    }
    return res;
  };
  return {engine, func};
}

void LRJITRegistry::Register(const std::string& function_id, pyast::FunctionDef& ast,
                             bool is_dump) {
  auto jit = GenFunc(ast, is_dump);
  functionId2engine_[function_id] = jit;
}

std::function<double(double, double)> LRJITRegistry::LookUp(const std::string& function_id) {
  auto iter = functionId2engine_.find(function_id);
  if (iter != functionId2engine_.end()) { return iter->second.second; }
  llvm::errs() << "function '" << function_id << "' not be registered before lookup.";
  return nullptr;
};

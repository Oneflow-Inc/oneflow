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
#ifndef ONEFLOW_IR_ONEFLOW_EXTENSION_INCLUDE_PYAST_AST_MLIR_GEN_H_
#define ONEFLOW_IR_ONEFLOW_EXTENSION_INCLUDE_PYAST_AST_MLIR_GEN_H_

#include "OneFlow/OneFlowLRJITRegistry.h"
#include "PyAst/Ast.h"

#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Builders.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/MemRefUtils.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include <glog/logging.h>
#include <numeric>
#include <any>
#include <functional>
#include <memory>

class BuilderWithSymbolTable {
 protected:
  mlir::OpBuilder builder_;
  mlir::ModuleOp theModule_;
  std::map<std::string, mlir::Value> symbolTable_;
  mlir::Block* symbolTableForDeclareBlock_{};

  explicit BuilderWithSymbolTable(mlir::MLIRContext& context) : builder_(&context) {}
  virtual ~BuilderWithSymbolTable() = default;

  mlir::LogicalResult Declare(const std::string& var, mlir::Value value);
  mlir::Value LoopUp(const std::string& var);
  mlir::Location Loc(const std::string& file_name = "unknown", int line = 0, int col = 0);
  void Dump();
};

class MLIRGenImpl : public BuilderWithSymbolTable {
 public:
  explicit MLIRGenImpl(mlir::MLIRContext& context) : BuilderWithSymbolTable(context) {}

  mlir::ModuleOp GenModule(pyast::FunctionDef* func);

  mlir::Value MlirGen(pyast::Compare* expr);
  mlir::Value MlirGen(pyast::BinOp* expr);
  mlir::Value MlirGen(pyast::Call* expr);
  mlir::Value MlirGen(pyast::Constant* expr);
  mlir::Value MlirGen(pyast::Name* expr);

  mlir::Value MlirGen(pyast::expr* expr);

  void MlirGen(pyast::If* stmt);
  void MlirGen(pyast::Assign* stmt);
  void MlirGen(pyast::Return* stmt);

  void MlirGen(pyast::stmt* stmt);
};

#endif  // ONEFLOW_IR_ONEFLOW_EXTENSION_INCLUDE_PYAST_AST_MLIR_GEN_H_

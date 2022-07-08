#ifndef ONEFLOW_IR_ONEFLOW_EXTENSION_INCLUDE_PYAST_AST_MLIR_GEN_H_
#define ONEFLOW_IR_ONEFLOW_EXTENSION_INCLUDE_PYAST_AST_MLIR_GEN_H_

#include <glog/logging.h>
#include <any>
#include <functional>
#include <memory>
#include "llvm/ADT/StringRef.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
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
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/TypeSwitch.h"
#include <numeric>

#include "mlir/Transforms/Passes.h"
#include "oneflow/ir/oneflow-extension/include/OneFlow/OneFlowLRJITRegistry.h"
#include "oneflow/ir/oneflow-extension/include/PyAst/Ast.h"

class BuilderWithSymbolTable {
 protected:
  mlir::OpBuilder builder;
  mlir::ModuleOp theModule;
  std::map<std::string, mlir::Value> symbolTable;
  mlir::Block* symbolTableForDeclareBlock;

  BuilderWithSymbolTable(mlir::MLIRContext& context) : builder(&context) {}
  virtual ~BuilderWithSymbolTable() = default;

  mlir::LogicalResult declare(const std::string& var, mlir::Value value);
  mlir::Value lookup(const std::string& var);
  mlir::Location loc(const std::string& file_name = "unknown", int line = 0, int col = 0);
  void dump();
};


class MLIRGenImpl : public BuilderWithSymbolTable {
 public:
  explicit MLIRGenImpl(mlir::MLIRContext& context) : BuilderWithSymbolTable(context) {}

  mlir::ModuleOp genModule(pyast::FunctionDef* func);

  mlir::Value mlirGen(pyast::Compare* expr);
  mlir::Value mlirGen(pyast::BinOp* expr);
  mlir::Value mlirGen(pyast::Call* expr);
  mlir::Value mlirGen(pyast::Constant* expr);
  mlir::Value mlirGen(pyast::Name* expr);

  mlir::Value mlirGen(pyast::expr* expr);

  void mlirGen(pyast::If* stmt);
  void mlirGen(pyast::Assign* stmt);
  void mlirGen(pyast::Return* stmt);

  void mlirGen(pyast::stmt* stmt);
};

#endif // ONEFLOW_IR_ONEFLOW_EXTENSION_INCLUDE_PYAST_AST_MLIR_GEN_H_

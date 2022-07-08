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
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/TypeSwitch.h"
#include <numeric>

#include "mlir/Transforms/Passes.h"
#include "oneflow/ir/oneflow-extension/include/PyAst/Ast.h"
#include "oneflow/ir/oneflow-extension/include/PyAst/AstMlirGen.h"

using llvm::ArrayRef;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

static mlir::OwningOpRef<mlir::ModuleOp> genModuleForTest(mlir::MLIRContext& context) {
  using namespace pyast;
  auto func = FunctionDef::FunctionDef_(
      "get_lr",
      arguments::arguments_({
          arg::arg_("base_lr"),
          arg::arg_("step"),
      }),
      {
          If::If_(Compare::Compare_(Name::Name_("step"), {Compare::kLt}, {Constant::Constant_(5)}),
                  {
                      Assign::Assign_({Name::Name_("base_lr")},
                                      (BinOp::BinOp_(Name::Name_("base_lr"), BinOp::kAdd,
                                                     Constant::Constant_(1.0 / 3)))),
                  },
                  {}),
          Return::Return_(Name::Name_("base_lr")),
      });

  MLIRGenImpl mlir_gen(context);
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir_gen.genModule(func.get());
  module->dump();
  return module;
}

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

static mlir::OwningOpRef<mlir::ModuleOp> genModuleForBuild(mlir::MLIRContext& context) {
  std::string moduleStr = R"mlir(
  func.func @get_lr(%arg0 : f32, %arg1 : i32) -> f32 attributes { llvm.emit_c_interface } {
    return %arg0 : f32
  }
  )mlir";
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(moduleStr, &context);
  return module;
}

static mlir::OwningOpRef<mlir::ModuleOp> genModule(mlir::MLIRContext& context,
                                                   pyast::FunctionDef& ast) {
  using namespace pyast;

  MLIRGenImpl mlir_gen(context);
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir_gen.genModule(&ast);
  // module->dump();
  return module;
}
static LRJITRegistry_Store_ genFunc(pyast::FunctionDef& ast, bool is_dump) {
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

  auto module = genModule(context, ast);
  if (is_dump) { module->dump(); }
  // auto module = genModuleForTest(context);
  CHECK(!!module) << "failed to parse module";
  CHECK(succeeded(lowerToLLVMDialect(*module))) << "failed to lower to llvm dialect";
  auto jit_or_err = mlir::ExecutionEngine::create(*module);
  CHECK(jit_or_err) << "failed to create JIT exe engine, "
                    << llvm::toString(jit_or_err.takeError());

  std::shared_ptr<mlir::ExecutionEngine> engine = cantFail(move(jit_or_err));

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
  auto jit = genFunc(ast, is_dump);
  function_id2engine_[function_id] = jit;
}

std::function<double(double, double)> LRJITRegistry::LookUp(const std::string& function_id) {
  auto iter = function_id2engine_.find(function_id);
  if (iter != function_id2engine_.end()) { return iter->second.second; }
  llvm::errs() << "function '" << function_id << "' not be registered before lookup.";
  return nullptr;
};

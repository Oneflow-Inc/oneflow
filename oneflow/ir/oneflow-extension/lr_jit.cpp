#include <glog/logging.h>
#include <memory>
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/IR/Attributes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/MemRefUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include "oneflow/ir/oneflow-extension/include/OneFlow/OneFlowAstJIT.h"

using namespace std;
using namespace mlir;

static struct LLVMInitializer {
  LLVMInitializer() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
  }
} initializer;

static LogicalResult lowerToLLVMDialect(ModuleOp module) {
  PassManager pm(module.getContext());
  pm.addPass(createMemRefToLLVMPass());
  pm.addNestedPass<func::FuncOp>(arith::createConvertArithmeticToLLVMPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
  return pm.run(module);
}

class JIT_Engine final {
  unique_ptr<ExecutionEngine> _engine;

 public:
  explicit JIT_Engine(unique_ptr<ExecutionEngine> _engine) : _engine(move(_engine)){};
  explicit JIT_Engine(const PyASTNode& ast);
  double Invoke(double base_lr, int64_t step);
};

JIT_Engine::JIT_Engine(const PyASTNode& ast) {
  std::string moduleStr = R"mlir(
  func.func @get_lr(%arg0 : f32, %arg1 : i32) -> f32 attributes { llvm.emit_c_interface } {
    return %arg0 : f32
  }
  )mlir";
  DialectRegistry registry;
  registerAllDialects(registry);
  registerLLVMDialectTranslation(registry);
  MLIRContext context(registry);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(moduleStr, &context);
  CHECK(!!module) << "failed to parse module";
  CHECK(succeeded(lowerToLLVMDialect(*module))) << "failed to lower to llvm dialect";
  auto jit_or_err = ExecutionEngine::create(*module);
  CHECK(jit_or_err) << "failed to create JIT exe engine, "
                    << llvm::toString(jit_or_err.takeError());

  _engine = cantFail(move(jit_or_err));
}

double JIT_Engine::Invoke(double base_lr, int64_t step) {
  float res = 0;
  auto&& out = ExecutionEngine::result(res);
  auto base_lr_jit = static_cast<float>(base_lr);
  auto step_jit = static_cast<int>(step);
  auto err = _engine->invoke("get_lr", base_lr_jit, step_jit, out);
  return res;
}

void LR_JIT::Register(const string& function_id, const PyASTNode& ast) {
  auto jit = make_shared<JIT_Engine>(ast);
  function_id2engine_[function_id] = jit;
}

std::shared_ptr<JIT_Engine> LR_JIT::LookUp(const std::string& function_id) {
  if (function_id2engine_.count(function_id)) { return function_id2engine_[function_id]; }
  return nullptr;
};

double LR_JIT::Invoke(shared_ptr<JIT_Engine> engine, double base_lr, int64_t step) {
  if (engine == nullptr) llvm::errs() << "engine is null";
  return engine->Invoke(base_lr, step);
};

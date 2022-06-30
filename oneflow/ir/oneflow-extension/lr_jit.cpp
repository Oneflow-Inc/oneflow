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
#include "py_ast.h"

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
  explicit JIT_Engine(PyASTNodeWrapper& ast);
  double Invoke(double base_lr, int64_t step);
};

static OwningOpRef<ModuleOp> genModuleForTest() {
  // FunctionDef(name='get_lr', args=arguments(posonlyargs=[], args=[arg(arg='self',
  // annotation=None, type_comment=None), arg(arg='base_lr', annotation=None, type_comment=None),
  // arg(arg='step', annotation=None, type_comment=None)], vararg=None, kwonlyargs=[],
  // kw_defaults=[], kwarg=None, defaults=[]), body=[Assign(targets=[Name(id='decay_batch',
  // ctx=Store())], value=Constant(value=5, kind=None), type_comment=None),
  // Assign(targets=[Name(id='cur_batch', ctx=Store())], value=Name(id='step', ctx=Load()),
  // type_comment=None), If(test=Constant(value=False, kind=None),
  // body=[If(test=Compare(left=Name(id='cur_batch', ctx=Load()), ops=[Eq()],
  // comparators=[Constant(value=0, kind=None)]), body=[Assign(targets=[Name(id='cur_batch',
  // ctx=Store())], value=Constant(value=1, kind=None), type_comment=None)], orelse=[]),
  // Assign(targets=[Name(id='decay_batch', ctx=Store())], value=BinOp(left=Name(id='decay_batch',
  // ctx=Load()), op=Mult(), right=Call(args=[BinOp(left=Name(id='cur_batch', ctx=Load()), op=Div(),
  // right=Name(id='decay_batch', ctx=Load()))], keywords=[])), type_comment=None)],
  // orelse=[Assign(targets=[Name(id='cur_batch', ctx=Store())], value=Call(func=Name(id='min',
  // ctx=Load()), args=[Name(id='cur_batch', ctx=Load()), Name(id='decay_batch', ctx=Load())],
  // keywords=[]), type_comment=None)]), Assign(targets=[Name(id='factor', ctx=Store())],
  // value=BinOp(left=BinOp(left=Constant(value=1, kind=None), op=Sub(),
  // right=BinOp(left=Name(id='cur_batch', ctx=Load()), op=Div(), right=Name(id='decay_batch',
  // ctx=Load()))), op=Pow(), right=Constant(value=1.0, kind=None)), type_comment=None),
  // Return(value=BinOp(left=BinOp(left=BinOp(left=Name(id='base_lr', ctx=Load()), op=Sub(),
  // right=Constant(value=0.0001, kind=None)), op=Mult(), right=Name(id='factor', ctx=Load())),
  // op=Add(), right=Constant(value=0.0001, kind=None)))], decorator_list=[], returns=None,
  // type_comment=None)
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
  return module;
}

static OwningOpRef<ModuleOp> genModuleForBuild() {
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
  return module;
}

JIT_Engine::JIT_Engine(PyASTNodeWrapper& ast) {
  auto module = genModuleForBuild();
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

void LR_JIT::Register(const string& function_id, PyASTNodeWrapper& ast) {
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

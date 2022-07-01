#include <glog/logging.h>
#include <any>
#include <memory>
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/IR/Attributes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/IR/Builders.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
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
#include <numeric>

#include "oneflow/ir/oneflow-extension/include/OneFlow/OneFlowAstJIT.h"
#include "py_ast.h"

using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::makeArrayRef;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

using namespace pyast;
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

namespace {
using namespace pyast;

class EvalVisitor : public BaseVisitor {
  mlir::OpBuilder builder;
  mlir::ModuleOp theModule;
  llvm::ScopedHashTable<StringRef, mlir::Value> symbolTable;
  mlir::Location unknown = builder.getUnknownLoc();

  mlir::LogicalResult declare(llvm::StringRef var, mlir::Value value) {
    if (symbolTable.count(var)) return mlir::failure();
    symbolTable.insert(var, value);
    return mlir::success();
  }

 public:
  explicit EvalVisitor(mlir::MLIRContext& context) : builder(&context) {}

  mlir::ModuleOp getModule(stmt_t func) {
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
    if (failed(mlir::verify(theModule))) {
      theModule.emitError("module verification error");
      return nullptr;
    }
    theModule->dump();

    return theModule;
  }

  std::any visitFunctionDef(FunctionDef_t node) {
    // crate function
    ScopedHashTableScope<llvm::StringRef, mlir::Value> varScope(symbolTable);
    builder.setInsertionPointToEnd(theModule.getBody());
    llvm::SmallVector<mlir::Type> arg_types(node->args->args.size(),
                                            mlir::Float64Type::getF64(builder.getContext()));
    auto func_type = builder.getFunctionType(arg_types, llvm::None);
    auto function = builder.create<mlir::func::FuncOp>(unknown, node->name, func_type);

    // register args
    mlir::Block& entry_block = function.front();
    for (const auto nameValue : llvm::zip(node->args->args, entry_block.getArguments())) {
      if (failed(declare(std::get<0>(nameValue)->arg, std::get<1>(nameValue)))) return nullptr;
    }

    // build body
    builder.setInsertionPointToStart(&entry_block);
    for (auto stmt : node->body) { visit(stmt); }
  }
  std::any visitReturn(Return_t node) {}
  std::any visitAssign(Assign_t node) {}
  std::any visitIf(If_t node) {}
  std::any visitRaise(Raise_t node) {}
  std::any visitAssert(Assert_t node) {}
  std::any visitExpr(Expr_t node) {}
  std::any visitBoolOp(BoolOp_t node) {}
  std::any visitBinOp(BinOp_t node) {}
  std::any visitLambda(Lambda_t node) {}
  std::any visitCompare(Compare_t node) {}
  std::any visitCall(Call_t node) {}
  std::any visitNum(Num_t node) {}
  std::any visitConstant(Constant_t node) {}
  std::any visitAttribute(Attribute_t node) {}
  std::any visitName(Name_t node) {}
  std::any visitBoolop(boolop_t value) {}
  std::any visitOperator(operator_t value) {}
  std::any visitCmpop(cmpop_t value) {}
  std::any visitArguments(arguments_t node) {}
  std::any visitArg(arg_t node) {}
};

}  // namespace

static OwningOpRef<ModuleOp> genModuleForTest() {
  auto func = FunctionDef("get_lr",
                          arguments({
                              arg("self"),
                              arg("base_lr"),
                              arg("step"),
                          }),
                          {
                              Assign(
                                  {
                                      Name("step_stage"),
                                  },
                                  Call(pyast::Attribute(Name("math"), "floor"),
                                       {BinOp(Name("step"), operator_t::kDiv, Constant(5))})),
                              Assign(
                                  {
                                      Name("factor"),
                                  },
                                  BinOp(Constant(0.1), operator_t::kPow, Name("step_stage"))),
                              Return(BinOp(Name("base_lr"), operator_t::kMult, Name("factor"))),
                          });

  DialectRegistry registry;
  registerAllDialects(registry);
  registerLLVMDialectTranslation(registry);
  MLIRContext context(registry);
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::arith::ArithmeticDialect>();
  EvalVisitor eval_visitor(context);
  OwningOpRef<ModuleOp> module = eval_visitor.getModule(func);
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
  auto module = genModuleForTest();
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

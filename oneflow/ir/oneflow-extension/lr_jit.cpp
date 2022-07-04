#include <glog/logging.h>
#include <any>
#include <functional>
#include <memory>
#include "llvm/ADT/StringRef.h"
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
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
#include "llvm/ADT/TypeSwitch.h"
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

class JIT_Engine final {
  unique_ptr<ExecutionEngine> _engine;

 public:
  explicit JIT_Engine(unique_ptr<ExecutionEngine> _engine) : _engine(move(_engine)){};
  explicit JIT_Engine(PyASTNodeWrapper& ast);
  double Invoke(double base_lr, int64_t step);
};

namespace {
using namespace pyast;

class MLIRGenImpl {
  OpBuilder builder;
  ModuleOp theModule;
  map<string, Value> symbolTable;

  LogicalResult declare(const string& var, Value value) {
    if (symbolTable.count(var)) return failure();
    symbolTable[var] = value;
    return success();
  }

  Value lookup(const string& var) {
    if (symbolTable.count(var) == 1) { return symbolTable[var]; }
    theModule->emitError("error: unknown variable '" + var + "'");
    return nullptr;
  }

  Location loc() { return FileLineColLoc::get(builder.getStringAttr("unknown"), 0, 0); }

 public:
  explicit MLIRGenImpl(MLIRContext& context) : builder(&context) {}

  void dump() { theModule->dump(); }

  void mlirGen(stmt_& stmt) {
    llvm::TypeSwitch<stmt_*>(&stmt)
        .Case<Return_>([&](auto* node) { mlirGen(*dynamic_cast<Return_*>(node)); })
        .Case<Assign_>([&](auto* node) { mlirGen(*dynamic_cast<Assign_*>(node)); })
        // .Case<If_>([&](auto* node) { mlirGen(cast<If_*>(node)); })
        // .Case<Raise_>([&](auto* node) { mlirGen(cast<Raise_*>(node)); })
        // .Case<Assert_>([&](auto* node) { mlirGen(cast<Assert_*>(node)); })
        // .Case<Expr_>([&](auto* node) { mlirGen(cast<Expr_*>(node)); })
        .Default([&](auto* node) { theModule->emitError("StmtKind not support yet"); });
  }

  Value mlirGen(expr_& stmt) {
    Value res;
    llvm::TypeSwitch<expr_*>(&stmt)
        .Case<BinOp_>([&](auto* node) { res = mlirGen(*dynamic_cast<BinOp_*>(node)); })
        //     .Case<Lambda_>([&](auto* node) { mlirGen(cast<Lambda_*>(node)); })
        //     .Case<Compare_>([&](auto* node) { mlirGen(cast<Compare_*>(node)); })
        .Case<Call_>([&](auto* node) { res = mlirGen(*dynamic_cast<Call_*>(node)); })
        //     .Case<Num_>([&](auto* node) { mlirGen(cast<Num_*>(node)); })
        .Case<Constant_>([&](auto* node) { res = mlirGen(*dynamic_cast<Constant_*>(node)); })
        //     .Case<Attribute_>([&](auto* node) { mlirGen(cast<Attribute_*>(node)); })
        .Case<Name_>([&](auto* node) { res = mlirGen(*dynamic_cast<Name_*>(node)); })
        .Default([&](auto* node) { theModule->emitError("ExprKind not support yet"); });
    return res;
  }

  Value mlirGen(BinOp_& expr) {
    auto lhs = mlirGen(*expr.get_left().get());
    auto rhs = mlirGen(*expr.get_right().get());
    Value res;
    switch (expr.get_op()) {
      case BinOp_::kDiv: res = builder.create<arith::DivFOp>(loc(), lhs, rhs); break;
      case BinOp_::kMult: res = builder.create<arith::MulFOp>(loc(), lhs, rhs); break;
      case BinOp_::kPow: res = builder.create<math::PowFOp>(loc(), lhs, rhs); break;
      default: break;
    }
    return res;
  }

  Value mlirGen(Call_& expr) {
    if (expr.get_func()->get_kind() != expr_::kAttribute) {
      theModule->emitError("only support call func is attribute node");
    }
    auto func = *dynamic_cast<Attribute_*>(expr.get_func().get());
    if (func.get_value()->get_kind() != expr_::kName
        || dynamic_cast<Name_*>(func.get_value().get())->get_id() != "math") {
      theModule->emitError("only support call func is python math lib");
    }
    if (expr.get_args().size() != 1) {
      theModule->emitError("only support call func with one param");
    }
    auto value = mlirGen(*expr.get_args()[0].get());
    auto attr = func.get_attr();
    Value res;
    if (attr == "floor") {
      res = builder.create<math::FloorOp>(loc(), value);
      return res;
    } else {
      theModule->emitError(attr + " not support yet");
    }
    return res;
  }

  Value mlirGen(Constant_& expr) {
    float value = expr.get_value();
    auto constant = builder.create<arith::ConstantOp>(loc(), builder.getF32FloatAttr(value));
    return constant;
  }

  Value mlirGen(Name_& expr) { return lookup(expr.get_id()); }

  void mlirGen(Assign_& stmt) {
    auto value = mlirGen(*stmt.get_value().get());
    for (const auto& target : stmt.get_targets()) {
      if (target->get_kind() != expr_::kName) {
        theModule->emitError("only support assign to name node");
      }
      auto name = dynamic_cast<Name_*>(target.get())->get_id();
      declare(name, value);
    }
  }
  void mlirGen(Return_& stmt) {
    auto value = mlirGen(*stmt.get_value().get());

    builder.create<func::ReturnOp>(loc(), ValueRange({value}));
  }

  ModuleOp mlirGen(FunctionDef_& func) {
    theModule = ModuleOp::create(loc());

    if (failed(verify(theModule))) {
      theModule.emitError("module verification error");
      return nullptr;
    }

    builder.setInsertionPointToEnd(theModule.getBody());

    auto args = func.get_args()->get_args();
    llvm::SmallVector<Type> arg_types(args.size(), Float32Type::getF32(builder.getContext()));
    llvm::SmallVector<Type> res_types(1, Float32Type::getF32(builder.getContext()));

    auto func_type = builder.getFunctionType(arg_types, res_types);

    auto function = func::FuncOp::create(loc(), func.get_name(), func_type);
    auto* entry_block = function.addEntryBlock();
    theModule.push_back(function);
    builder.setInsertionPointToStart(entry_block);

    for (const auto nameValue : llvm::zip(args, entry_block->getArguments())) {
      if (failed(declare(get<0>(nameValue)->get_arg(), get<1>(nameValue)))) { return nullptr; }
    }

    builder.setInsertionPointToStart(entry_block);
    for (const auto& stmt : func.get_body()) { mlirGen(*stmt.get()); }

    return theModule;
  }
};

}  // namespace

static OwningOpRef<ModuleOp> genModuleForTest(MLIRContext& context) {
  auto func =
      FunctionDef("get_lr",
                  arguments({
                      arg("base_lr"),
                      arg("step"),
                  }),
                  {
                      Assign(
                          {
                              Name("step_stage"),
                          },
                          Call(pyast::Attribute(Name("math"), "floor"),
                               {BinOp(Name("step"), BinOp_::operator_t::kDiv, Constant(5))})),
                      Assign(
                          {
                              Name("factor"),
                          },
                          BinOp(Constant(0.1), BinOp_::operator_t::kPow, Name("step_stage"))),
                      Return(BinOp(Name("base_lr"), BinOp_::operator_t::kMult, Name("factor"))),
                  });

  MLIRGenImpl mlir_gen(context);
  OwningOpRef<ModuleOp> module = mlir_gen.mlirGen(*dynamic_cast<FunctionDef_*>(func.get()));
  mlir_gen.dump();
  return module;
}

static struct LLVMInitializer {
  LLVMInitializer() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
  }
} initializer;

static LogicalResult lowerToLLVMDialect(ModuleOp module) {
  PassManager pm(module.getContext());
  pm.addPass(createMemRefToLLVMPass());
  pm.addNestedPass<func::FuncOp>(createConvertMathToLLVMPass());
  pm.addNestedPass<func::FuncOp>(arith::createConvertArithmeticToLLVMPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
  return pm.run(module);
}
static OwningOpRef<ModuleOp> genModuleForBuild(MLIRContext& context) {
  string moduleStr = R"mlir(
  func.func @get_lr(%arg0 : f32, %arg1 : i32) -> f32 attributes { llvm.emit_c_interface } {
    return %arg0 : f32
  }
  )mlir";
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(moduleStr, &context);
  return module;
}

JIT_Engine::JIT_Engine(PyASTNodeWrapper& ast) {
  DialectRegistry registry;
  registerAllDialects(registry);
  registerLLVMDialectTranslation(registry);
  MLIRContext context(registry);
  context.loadDialect<func::FuncDialect>();
  context.loadDialect<arith::ArithmeticDialect>();
  context.loadDialect<math::MathDialect>();

  auto module = genModuleForTest(context);
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

shared_ptr<JIT_Engine> LR_JIT::LookUp(const string& function_id) {
  if (function_id2engine_.count(function_id)) { return function_id2engine_[function_id]; }
  return nullptr;
};

double LR_JIT::Invoke(shared_ptr<JIT_Engine> engine, double base_lr, int64_t step) {
  if (engine == nullptr) llvm::errs() << "engine is null";
  return engine->Invoke(base_lr, step);
};

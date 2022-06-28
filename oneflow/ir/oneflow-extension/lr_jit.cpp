#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "oneflow/ir/oneflow-extension/include/OneFlow/OneFlowPyIr.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/MemRefUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

static std::function<double(double, int64_t)> AstToJIT(const ast& ast,
                                                       const std::string& function_id) {
  std::string moduleStr = R"mlir(
  func.func @foo(%arg0 : i32) -> i32 attributes { llvm.emit_c_interface } {
    %res = arith.addi %arg0, %arg0 : i32
    return %res : i32
  }
  )mlir";
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::MLIRContext context(registry);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(moduleStr, &context);
  auto jit_or_error = mlir::ExecutionEngine::create(*module);
  CHECK(!!jit_or_error) << "failed to create JIT exe engine, "
                        << llvm::toString(jit_or_error.takeError());
  auto jit = std::move(jit_or_error->get());
  auto lr_func = [jit](double base_lr, int64_t step) {
    double res = 0;
    auto&& r_res = mlir::ExecutionEngine::Result<double>(res);
    auto err = jit->invoke("get_lr", base_lr, step, r_res);
    CHECK(!!err) << "failed to run JIT exe engine";
    return res;
  };
  return lr_func;
}

void LR_JIT::Register(const std::string& function_id,
                      std::function<double(double, int64_t)> lr_func) {
  function_id2lr_func_[function_id] = std::move(lr_func);
};

bool LR_JIT::Invoke(const std::string& function_id, double& lr, double base_lr, int64_t step) {
  if (function_id2lr_func_.count(function_id)) {
    lr = function_id2lr_func_[function_id](base_lr, step);
    return true;
  }
  return false;
};

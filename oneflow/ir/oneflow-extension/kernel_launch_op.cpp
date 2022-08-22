#include <string>
#include "llvm/Support/TargetSelect.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/Transforms/RequestCWrappers.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"
#include "oneflow/core/common/singleton.h"
#include "oneflow/core/framework/op_kernel.h"
#include "oneflow/core/framework/user_op_kernel_registry.h"

extern "C" {

void oneflow_launch_kernel(void* ctx_, std::string& name) {
  auto ctx = (oneflow::user_op::KernelComputeContext*)ctx_;
  auto kernel = oneflow::Singleton<oneflow::user_op::KernelLaunchRegistry>::Get()->LookUp(name)();
  kernel->Compute(ctx);
}

}  // extern "C"

static struct LLVMInitializer {
  LLVMInitializer() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
  }
} initializer;

static mlir::LogicalResult lowerToLLVMDialect(mlir::ModuleOp module) {
  mlir::PassManager pm(module.getContext());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::LLVM::createRequestCWrappersPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::createCSEPass());
  return pm.run(module);
}

mlir::ModuleOp gen(std::string& func_name) {
  llvm::StringRef callee = "oneflow_launch_kernel";
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::MLIRContext context(registry);
  context.loadDialect<mlir::func::FuncDialect>();

  auto builder = mlir::OpBuilder(&context);
  auto loc = mlir::UnknownLoc::get(&context);
  auto module = mlir::ModuleOp::create(loc);
  builder.setInsertionPointToEnd(module.getBody());

  const auto& proto_name = func_name;
  auto proto_type = builder.getFunctionType({}, {});
  auto func = mlir::func::FuncOp::create(loc, proto_name, proto_type);

  module.push_back(func);
  auto* entry_block = func.addEntryBlock();
  builder.setInsertionPointToStart(entry_block);

  mlir::TypeRange res;
  builder.create<mlir::func::CallOp>(loc, callee, res);
  return module;
}

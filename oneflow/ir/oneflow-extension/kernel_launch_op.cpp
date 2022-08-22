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

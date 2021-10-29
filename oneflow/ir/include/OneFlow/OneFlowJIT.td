#ifndef ONEFLOW_JIT_PASSES
#define ONEFLOW_JIT_PASSES

include "mlir/Pass/PassBase.td"

def ReturnAllLeaveResultPass : FunctionPass<"return-all-leave-result"> {
  let summary = "";
  let constructor = "mlir::oneflow::createReturnAllLeaveResultPass()";
}

def CreateComputeCtxPass : FunctionPass<"create-compute-ctx-pass"> {
  let summary = "";
  let constructor = "mlir::oneflow::createCreateComputeCtxPass()";
}

#endif // ONEFLOW_JIT_PASSES

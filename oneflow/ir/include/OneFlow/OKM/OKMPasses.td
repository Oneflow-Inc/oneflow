#ifndef ONEFLOW_OKM_PASSES
#define ONEFLOW_OKM_PASSES

include "OneFlow/OKM/OKMBase.td"

def ExtractOKMTensorPass : Pass<"extract-okm-tensor", "ModuleOp"> {
  let summary = "extract okm tensors from args and rets";
  let constructor = "mlir::okm::createExtractOKMTensorPass()";
}

def WrapOKMKernelPass : Pass<"wrap-okm-kernel", "ModuleOp"> {
  let summary = "wrap kernel in okm";
  let constructor = "mlir::okm::createWrapOKMKernelPass()";
}

def OptOKMMemrefPass : Pass<"opt-okm-memref", "ModuleOp"> {
  let summary = "optimize okm memref";
  let constructor = "mlir::okm::createOptOKMMemrefPass()";
}

def BuildOKLFromOKMPass : Pass<"build-okl-from-okm", "ModuleOp"> {
  let summary = "convert okm to okl";
  let constructor = "mlir::okm::createBuildOKLFromOKMPass()";
}
#endif // ONEFLOW_OKM_PASSES


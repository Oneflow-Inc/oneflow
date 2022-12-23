#ifndef ONEFLOW_PASSES
#define ONEFLOW_PASSES

include "OneFlow/OneFlowOps.td"

def LowerOneFlowToTosaPass : Pass<"lower-oneflow-to-tosa", "ModuleOp"> {
  let summary = "lower oneflow dialect to tosa dialect";
  let constructor = "mlir::oneflow::createLowerOneFlowToTosaPass()";
  let dependentDialects = ["tosa::TosaDialect", "memref::MemRefDialect", "mlir::func::FuncDialect"];
  let options = [
    Option<"variableAsConstant", "variable-as-constant", "int", "0",
           "convert variable op as const op of tosa">,
  ];
}

def BufferHostRegisterPass : Pass<"buffer-host-register", "func::FuncOp"> {
  let summary = "";
  let constructor = "mlir::oneflow::createBufferHostRegisterPass()";
  let dependentDialects = ["gpu::GPUDialect"];
}

def GpuCopyArgPass : Pass<"gpu-copy-arg", "func::FuncOp"> {
  let summary = "";
  let constructor = "mlir::oneflow::createGpuCopyArgPass()";
  let dependentDialects = ["memref::MemRefDialect", "gpu::GPUDialect"];
}

def OutlineJitFunctionPass : Pass<"outline-jit-function", "ModuleOp"> {
  let summary = "move ops could be jitted to jit function";
  let constructor = "mlir::oneflow::createOutlineJitFunctionPass()";
}

def AggregateComputeOpsPass : Pass<"aggregate-compute-ops", "ModuleOp"> {
  let summary = "aggregate compute ops together";
  let constructor = "mlir::oneflow::createAggregateComputeOpsPass()";
}

def AggregateCudaGraphSupportOpsPass : Pass<"aggregate-cuda-graph-support-ops", "ModuleOp"> {
  let summary = "aggregate cuda graph support ops together for wrapping them into a single okl op";
  let constructor = "mlir::oneflow::createAggregateCudaGraphSupportOpsPass()";
}

def WrapOpsToKernelLaunchPass : Pass<"wrap-ops-to-kernel-launch", "ModuleOp"> {
  let summary = "wrap user ops with a single kernel launch op in OneFlow Job";
  let constructor = "mlir::oneflow::createWrapOpsToKernelLaunchPass()";
}

def ExtractKernelLaunchTensorPass : Pass<"extract-kernel-launch-tensor", "ModuleOp"> {
  let summary = "extract kernel launch tensor from !okl.launcher_ctx";
  let constructor = "mlir::oneflow::createExtractKernelLaunchTensorPass()";
}

def TrimReturnAsVoidPass : Pass<"trim-return-to-void", "ModuleOp"> {
  let summary = "trim return types to void type";
  let constructor = "mlir::oneflow::createTrimReturnAsVoidPass()";
}

def LowerToOKLPass : Pass<"lower-to-okl", "ModuleOp"> {
  let summary = "lower oneflow dialect ops to okl dialect";
  let constructor = "mlir::oneflow::createLowerToOKLPass()";
}

def FuseIntoExistingOpPass : Pass<"fuse-into-existing-op", "ModuleOp"> {
  let summary = "";
  let constructor = "mlir::oneflow::createFuseIntoExistingOpPass()";
  let dependentDialects = ["pdl_interp::PDLInterpDialect", "pdl::PDLDialect"];
}

def AutoNhwcPass : Pass<"auto-nhwc", "ModuleOp"> {
  let summary = "";
  let constructor = "mlir::oneflow::createAutoNhwcPass()";
}

def PreConvertInferenceOpPass : Pass<"pre-convert-inference-op", "ModuleOp"> {
  let summary = "Convert variable op to variable ir op for constant folding.";
  let constructor = "mlir::oneflow::createPreConvertInferenceOpPass()";
}

def ConvertInferenceOpPass : Pass<"convert-inference-op", "ModuleOp"> {
  let summary = "Convert ops to their inference version and rewrite them with a more performant equivalent in inference workflow.";
  let constructor = "mlir::oneflow::createConvertInferenceOpPass()";
}

def PostConvertInferenceOpPass : Pass<"post-convert-inference-op", "ModuleOp"> {
  let summary = "Convert variable ir op to variable op after contant folding.";
  let constructor = "mlir::oneflow::createPostConvertInferenceOpPass()";
}


def ConvertToSignlessForTosaPass : Pass<"convert-to-signless-for-tosa", "ModuleOp"> {
  let summary = "convert func type to unsigned before lowering to tosa";
  let description = [{
    In oneflow, int typed tensor is explicit signed. Convert them before lowering to TOSA.
  }];
  let constructor = "mlir::oneflow::createConvertToSignlessForTosaPass()";
  let dependentDialects = ["func::FuncDialect"];
}

def CSEWithAttributesIgnored : Pass<"cse-with-attributes-ignored", "ModuleOp"> {
  let summary = "ignore oneflow attributes to have cse work";
  let description = [{
    cse and ignore oneflow attributes like op name, symbol id, etc.
  }];
  let constructor = "mlir::oneflow::createCSEWithAttributesIgnored()";
  let dependentDialects = [];
}

def CSEPutAttributes : Pass<"cse-put-attributes", "ModuleOp"> {
  let summary = "cse and ignore oneflow attributes";
  let description = [{
    put back oneflow attributes like op name, symbol id, etc.
  }];
  let constructor = "mlir::oneflow::createCSEPutAttributes()";
  let dependentDialects = [];
}

def GroupMatMul : Pass<"group-matmul", "ModuleOp"> {
  let summary = "group matmul together";
  let description = [{
    group matmul ops together and use cudnn batched matmul
  }];
  let constructor = "mlir::oneflow::createGroupMatMul()";
  let dependentDialects = [];
}

def FuseForwardOps : Pass<"fuse-forward-only-ops", "ModuleOp"> {
  let summary = "fuse forward ops";
  let description = [{
    fuse forward ops. Usually they are actions after an op.
  }];
  let constructor = "mlir::oneflow::createFuseForwardOps()";
  let dependentDialects = [];
}

def FuseNormalizationOps : Pass<"fuse-normalization-ops", "ModuleOp"> {
  let summary = "fuse forward ops";
  let description = [{
    fuse forward ops. Usually they are actions after an op.
  }];
  let constructor = "mlir::oneflow::createFuseNormalizationOps()";
  let dependentDialects = ["pdl_interp::PDLInterpDialect", "pdl::PDLDialect"];
}

#endif // ONEFLOW_PASSES

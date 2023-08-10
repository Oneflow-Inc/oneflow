#ifndef ONEFLOW_PASSES
#define ONEFLOW_PASSES

include "OneFlow/OneFlowOps.td"

#ifdef WITH_MLIR_CUDA_CODEGEN
def NVVMToCubinPass : InterfacePass<"nvvm-to-cubin", "SymbolOpInterface"> {
  let summary = "convert nvvm ir to cubin";
  let constructor = "mlir::oneflow::createNVVMToCubinPass()";
  let options = [
    Option<"triple", "triple", "StringRef", "\"nvptx64-nvidia-cuda\"", "Target triple">,
    Option<"chip", "chip", "StringRef", "mlir::oneflow::getArchVersion()", "Target architecture">,
    Option<"features", "features", "StringRef", "\"+ptx60\"", "Target features">,
  ];
}
#endif // WITH_MLIR_CUDA_CODEGEN

def TestOneFlowTraitFolderPass : Pass<"test-oneflow-trait-folder", "func::FuncOp"> {
  let constructor = "mlir::oneflow::createTestOneFlowTraitFolderPass()";
}

def LowerOneFlowToTosaPass : Pass<"lower-oneflow-to-tosa", "ModuleOp"> {
  let summary = "lower oneflow dialect to tosa dialect";
  let constructor = "mlir::oneflow::createLowerOneFlowToTosaPass()";
  let dependentDialects = ["tosa::TosaDialect", "memref::MemRefDialect", "mlir::func::FuncDialect"];
  let options = [
    Option<"variableAsConstant", "variable-as-constant", "int", "0",
           "convert variable op as const op of tosa">,
    Option<"fullyConvert", "full", "bool",
           /*default=*/"true", "Fully convert operations and make OneFlow dialect illegal target">,
    Option<"lowerJob", "lower-job", "bool",
           /*default=*/"true", "Convert oneflow.job to func.func">,
  ];
}

def LowerOneFlowToLinalgPass : Pass<"lower-oneflow-to-linalg", "ModuleOp"> {
  let summary = "lower oneflow dialect to Linalg dialect";
  let constructor = "mlir::oneflow::createLowerOneFlowToLinalgPass()";
  let dependentDialects = ["linalg::LinalgDialect", "memref::MemRefDialect", "mlir::func::FuncDialect"];
}

def FuncToOneFlowJobPass : Pass<"func-to-ofjob", "ModuleOp"> {
  let summary = "convert func Ops to oneflow Ops";
  let constructor = "mlir::oneflow::createFuncToOneFlowJobPass()";
  let dependentDialects = ["mlir::func::FuncDialect"];
}

def OneFlowJobToFuncPass : Pass<"ofjob-to-func", "ModuleOp"> {
  let summary = "convert oneflow Ops to func Ops";
  let constructor = "mlir::oneflow::createOneFlowJobToFuncPass()";
  let dependentDialects = ["mlir::func::FuncDialect"];
}

def CastOneFlowOpsToSignlessPass : Pass<"cast-ofops-to-signless", "ModuleOp"> {
  let summary = "cast oneflow ops to singless";
  let constructor = "mlir::oneflow::createCastOneFlowOpsToSignlessPass()";
  let dependentDialects = ["mlir::func::FuncDialect", "mlir::BuiltinDialect"];
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

def OutlineJitFunctionPass : InterfacePass<"outline-jit-function", "FunctionOpInterface"> {
  let summary = "move ops could be jitted to jit function";
  let constructor = "mlir::oneflow::createOutlineJitFunctionPass()";
  let dependentDialects = ["pdl_interp::PDLInterpDialect", "pdl::PDLDialect", "LLVM::LLVMDialect"];
  let options = [
    Option<"compileToLLVM", "compile-to-llvm", "bool",
           /*default=*/"true", "Convert to llvm dialect in this pass">,
  ];
}

def AggregateComputeOpsPass : Pass<"aggregate-compute-ops", "ModuleOp"> {
  let summary = "aggregate compute ops together";
  let constructor = "mlir::oneflow::createAggregateComputeOpsPass()";
}

def WrapOpsToKernelLaunchPass : Pass<"wrap-ops-to-kernel-launch", "ModuleOp"> {
  let summary = "wrap user ops with a single kernel launch op in OneFlow Job";
  let constructor = "mlir::oneflow::createWrapOpsToKernelLaunchPass()";
}

def EliminateAllocOpsPass : Pass<"eliminate-alloc-ops", "ModuleOp"> {
  let summary = "eliminate memref.alloc and memref.copy which target is a block argument";
  let constructor = "mlir::oneflow::createEliminateAllocOpsPass()";
  let dependentDialects = ["pdl_interp::PDLInterpDialect", "pdl::PDLDialect"];
}

def AppendOneFlowStreamPass : Pass<"append-ofstream", "ModuleOp"> {
  let summary = "append oneflow stream to gpu function arguments";
  let constructor = "mlir::oneflow::createAppendOneFlowStreamPass()";
}

def MgpuToOneFlowStreamPass : Pass<"mgpu-to-ofstream", "ModuleOp"> {
  let summary = "convert mlir abi about mgpu to oneflow stream, this pass should be invoked after append-ofstream pass";
  let constructor = "mlir::oneflow::createMgpuToOneFlowStreamPass()";
}

def FuseIntoExistingOpPass : Pass<"fuse-into-existing-op", "ModuleOp"> {
  let summary = "";
  let constructor = "mlir::oneflow::createFuseIntoExistingOpPass()";
  let dependentDialects = ["pdl_interp::PDLInterpDialect", "pdl::PDLDialect"];
}

def InsertOneFlowMemPoolPass : Pass<"insert-ofmempool", "ModuleOp"> {
  let summary = "insert oneflow tmp buffer as a memory pool in mlir codegen";
  let constructor = "mlir::oneflow::createInsertOneFlowMemPoolPass()";
}

def FoldAllocToSubviewPass : Pass<"fold-alloc-to-subview", "func::FuncOp"> {
  let summary = "fold dispersed memref.alloc ops with memory optimize algo to a single memref.alloc op and memref.subview ops";
  let constructor = "mlir::oneflow::createFoldAllocToSubviewPass()";
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
  let dependentDialects = ["pdl_interp::PDLInterpDialect", "pdl::PDLDialect"];
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

def FuseOpsWithBackwardImpl : Pass<"fuse-ops-with-backward-impl", "ModuleOp"> {
  let summary = "fuse ops with backward impl";
  let description = [{
    fuse ops with backward impl.
  }];
  let constructor = "mlir::oneflow::createFuseOpsWithBackwardImpl()";
  let dependentDialects = ["pdl_interp::PDLInterpDialect", "pdl::PDLDialect"];
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

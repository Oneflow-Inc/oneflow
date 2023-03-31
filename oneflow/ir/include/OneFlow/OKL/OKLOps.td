/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_IR_INCLUDE_OKL_OKLOPS
#define ONEFLOW_IR_INCLUDE_OKL_OKLOPS

include "OneFlow/OKL/OKLDialect.td"
include "OneFlow/OKL/OKLBase.td"
include "OneFlow/OKL/OKLTypes.td"
include "mlir/Pass/PassBase.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"


def GetTensorFromArgOp : OKL_Op<"get_tensor_from_arg"> {
  let summary = "get tensor as arguments from operands of context";
  let description = [{
    get tensor value from launcher context as arguments.
  }];

  let arguments = (ins
    LauncherContextType:$launcher_ctx,
    I32Attr:$index
  );
  let results = (outs AnyTensor);
}

def GetTensorFromRetOp : OKL_Op<"get_tensor_from_ret"> {
  let summary = "get tensor as arguments from results of context";
  let description = [{
    get tensor value from launcher context as arguments.
  }];

  let arguments = (ins
    LauncherContextType:$launcher_ctx,
    I32Attr:$index
  );
  let results = (outs AnyTensor);
}

def GetTensorAsRetOp : OKL_Op<"get_tensor_as_ret"> {
  let summary = "get tensor as outcomes from results of context";
  let description = [{
    get tensor value from launcher context as outcomes.
  }];

  let arguments = (ins
    LauncherContextType:$launcher_ctx,
    AnyTensor:$tensor,
    I32Attr:$index
  );
  let results = (outs AnyTensor);
}

def PoolToTensorOp : OKL_Op<"pool_to_tensor"> {
  let arguments = (ins
    LauncherContextType:$launcher_ctx,
    I64Attr:$offset
  );
  let results = (outs AnyTensor);
}

def PoolToBufferOp : OKL_Op<"pool_to_buffer"> {
  let arguments = (ins
    LauncherContextType:$launcher_ctx,
    I64Attr:$offset
  );
  let results = (outs AnyTensor);
}

def TensorToPoolOp : OKL_Op<"tensor_to_pool"> {
  let arguments = (ins
    LauncherContextType:$launcher_ctx,
    AnyTensor:$tensor,
    I64Attr:$offset
  );
  let results = (outs AnyTensor);
}

def WrapperKernelOp : OKL_Op<"wrapper_kernel"> {
  let summary = "build reg context operation";
  let description = [{
    this context is generated from module op and used on kernel/run_ctx build phase.
    each wrapped op has their own reg_ctx with their own attrs.
  }];

  let arguments = (ins
    I32Attr:$index
  );

  let regions = (region AnyRegion:$body);
}

def ReturnOp : OKL_Op<"return", [HasParent<"WrapperKernelOp">, Terminator]> {
  let summary = "return operation";
  let description = [{
    return oneflow ops in reg context
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<(ins),
    [{ build($_builder, $_state, llvm::None); }]>];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def LowerLauncherToLLVMPtrPass : Pass<"lower-launcher-to-llvm-ptr", "ModuleOp"> {
  let summary = "convert okl dialect func to llvm dialect";
  let constructor = "mlir::okl::createLowerLauncherToLLVMPtrPass()";
}

def LowerOKLToLLVMCallPass : Pass<"lower-okl-to-llvm-call", "ModuleOp"> {
  let summary = "convert okl dialect ops to llvm dialect llvm.call";
  let constructor = "mlir::okl::createLowerOKLToLLVMCallPass()";
}

def TagCudaGraphSupportPass : Pass<"tag-cuda-graph-support", "ModuleOp"> {
  let summary = "tag cuda graph support according to its wrapped ops";
  let constructor = "mlir::okl::createTagCudaGraphSupportPass()";
}

#endif // ONEFLOW_IR_INCLUDE_OKL_OKLOPS

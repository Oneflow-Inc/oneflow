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
#ifndef ONEFLOW_IR_INCLUDE_OKL_OKLOPS_H_
#define ONEFLOW_IR_INCLUDE_OKL_OKLOPS_H_

include "OneFlow/OKL/OKLDialect.td"
include "OneFlow/OKL/OKLBase.td"
include "OneFlow/OKL/OKLTypes.td"
include "mlir/Pass/PassBase.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"

def OKL_ArgumentTensor : I32EnumAttrCase<"TT_Argument", 0>;
def OKL_BufferTensor: I32EnumAttrCase<"TT_Buffer", 1>;
def OKL_ReturnTensor: I32EnumAttrCase<"TT_Return", 2>;

def OKL_TensorAttr: I32EnumAttr<"TensorType", "OKL Tensor Type enum",
  [
    OKL_ArgumentTensor,
    OKL_BufferTensor,
    OKL_ReturnTensor
  ]
> {
  let cppNamespace = "::mlir::okl";
  let stringToSymbolFnName = "ConvertToEnum";
  let symbolToStringFnName = "ConvertToString";
}

def GetTensorFromArgOp : OKL_Op<"get_tensor_from_arg"> {
  let summary = "get tensor from context as arguments";
  let description = [{
    get tensor value from launcher context as arguments.
  }];

  let arguments = (ins
    LauncherContextType:$launcher_ctx,
    OKL_TensorAttr:$tensor_type
  );
  let results = (outs AnyTensor);
}

def GetTensorAsRetOp : OKL_Op<"get_tensor_as_ret"> {
  let summary = "get tensor from context as outcomes";
  let description = [{
    get tensor value from launcher context as outcomes.
  }];

  let arguments = (ins
    LauncherContextType:$launcher_ctx,
    AnyTensor:$tensor,
    OKL_TensorAttr:$tensor_type
  );
  let results = (outs AnyTensor);
}


def BuildRegContextOp : OKL_Op<"build_reg_ctx"> {
  let summary = "build reg context operation";
  let description = [{
    this context is generated from module op and used on kernel/run_ctx build phase.
    each wrapped op has their own reg_ctx with their own attrs.
  }];

  let arguments = (ins
    StrAttr:$mlir_assembly
  );
  let results = (outs RegContextType:$out);
}

def DestroyRegContextOp : OKL_Op<"destroy_reg_ctx"> {
  let summary = "destroy reg context operation";
  let description = [{
    delete the pointer source.
  }];

  let arguments = (ins RegContextType:$ptr);
}

def BuildRunContextOp : OKL_Op<"build_run_ctx"> {
  let summary = "build run context operation";
  let description = [{
    this context is generated from reg_ctx and compute_ctx.
    compute_ctx is passed by jit engine from llvm function operand.
    each wrapped op has their own run_ctx with their own reg_ctx.
  }];

  let arguments = (ins
    RegContextType:$reg_ctx,
    LauncherContextType:$launcher_ctx
  );
  let results = (outs RunContextType:$out);
}

def DestroyRunContextOp : OKL_Op<"destroy_run_ctx"> {
  let summary = "destroy run context operation";
  let description = [{
    delete the pointer source.
  }];

  let arguments = (ins RunContextType:$ptr);
}

def BuildKernelOp : OKL_Op<"build_op_kernel"> {
  let summary = "build kernel operation";
  let description = [{
    kernel is generated from reg_ctx and op_name.
  }];

  let arguments = (ins
    RegContextType:$reg_ctx,
    StrAttr:$op_type_name
  );
  let results = (outs KernelType:$out);
}

def LaunchOp : OKL_Op<"launch"> {
  let summary = "launch operation";
  let description = [{
    launch use kernel and run_ctx to run the real kernel compute func.
  }];

  let arguments = (ins
    RegContextType:$reg_ctx,
    RunContextType:$run_ctx,
    KernelType:$kernel
  );
}

def LowerOKLToLLVMPass : Pass<"lower-okl-to-llvm", "ModuleOp"> {
  let summary = "convert okl dialect to llvm dialect";
  let constructor = "mlir::okl::createLowerOKLToLLVMPass()";
}

#endif // ONEFLOW_IR_INCLUDE_OKL_OKLOPS_H_

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
include "mlir/Pass/PassBase.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

def RegContextOp : OKL_Op<"build_reg_ctx"> {
  let summary = "build reg context operation";
  let description = [{
    this context is generated from module op and used on kernel/run_ctx build phase.
    each wrapped op has their own reg_ctx with their own attrs.
  }];

  let arguments = (ins
    StrAttr:$mlir_assembly
  );
  let results = (outs LLVM_AnyPointer:$out);
}

def DestroyRegContextOp : OKL_Op<"destroy_reg_ctx"> {
  let summary = "destroy reg context operation";
  let description = [{
    delete the pointer source.
  }];

  let arguments = (ins LLVM_AnyPointer:$ptr);
}

def RunContextOp : OKL_Op<"build_run_ctx"> {
  let summary = "build run context operation";
  let description = [{
    this context is generated from reg_ctx and compute_ctx.
    compute_ctx is passed by jit engine from llvm function operand.
    each wrapped op has their own run_ctx with their own reg_ctx.
  }];

  let arguments = (ins
    LLVM_AnyPointer:$reg_ctx,
    LLVM_AnyPointer:$compute_ctx
  );
  let results = (outs LLVM_AnyPointer:$out);
}

def DestroyRunContextOp : OKL_Op<"destroy_run_ctx"> {
  let summary = "destroy run context operation";
  let description = [{
    delete the pointer source.
  }];

  let arguments = (ins LLVM_AnyPointer:$ptr);
}

def KernelOp : OKL_Op<"build_op_kernel"> {
  let summary = "build kernel operation";
  let description = [{
    kernel is generated from reg_ctx and op_name.
  }];

  let arguments = (ins
    LLVM_AnyPointer:$reg_ctx,
    StrAttr:$op_type_name
  );
  let results = (outs LLVM_AnyPointer:$out);
}

def LaunchOp : OKL_Op<"launch"> {
  let summary = "launch operation";
  let description = [{
    launch use kernel and run_ctx to run the real kernel compute func.
  }];

  let arguments = (ins
    LLVM_AnyPointer:$reg_ctx,
    LLVM_AnyPointer:$run_ctx,
    LLVM_AnyPointer:$kernel
  );
}

def LowerOKLToLLVMPass : Pass<"lower-okl-to-llvm", "ModuleOp"> {
  let summary = "convert okl dialect to llvm dialect";
  let constructor = "mlir::okl::createLowerOKLToLLVMPass()";
}

#endif // ONEFLOW_IR_INCLUDE_OKL_OKLOPS_H_

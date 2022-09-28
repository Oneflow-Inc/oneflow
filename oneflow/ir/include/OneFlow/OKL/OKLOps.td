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

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Pass/PassBase.td"

include "mlir/IR/OpBase.td"

def BuildContextOp : OKL_Op<"reg_ctx"> {
  let summary = "build reg context operation";
  let description = [{
    this context is generated from module op and used on kernel/run_ctx build phase.
    each wrapped op has their own reg_ctx with their own attrs.
  }];
}

def BuildContextOp : OKL_Op<"run_ctx"> {
  let summary = "build run context operation";
  let description = [{
    this context is generated from reg_ctx and compute_ctx.
    compute_ctx is passed by jit engine from llvm function operand.
    each wrapped op has their own run_ctx with their own reg_ctx.
  }];
}

def BuildKernelOp : OKL_Op<"kernel"> {
  let summary = "build kernel operation";
  let description = [{
    kernel is generated from reg_ctx and op_name.
  }];
}

def LaunchOp : OKL_Op<"launch"> {
  let summary = "launch operation";
  let description = [{
    launch use kernel and run_ctx to run the real kernel compute func.
  }];
}

#endif // ONEFLOW_IR_INCLUDE_OKL_OKLOPS_H_

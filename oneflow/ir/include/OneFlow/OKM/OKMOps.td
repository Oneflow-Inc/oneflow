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
#ifndef ONEFLOW_IR_INCLUDE_ONEFLOW_OKM_OKMOPS
#define ONEFLOW_IR_INCLUDE_ONEFLOW_OKM_OKMOPS

include "OneFlow/OKM/OKMAttributes.td"
include "OneFlow/OKM/OKMPasses.td"
include "mlir/IR/OpBase.td"

def ArgToTensorOp : OKM_Op<"arg_to_tensor"> {
  let arguments = (ins
    I32Attr:$index
  );
  let results = (outs AnyTensor);
}

def ArgToMemrefOp : OKM_Op<"arg_to_memref"> {
  let arguments = (ins
    I32Attr:$index
  );
  let results = (outs AnyMemRef);
}

def RetToMemrefOp : OKM_Op<"ret_to_memref"> {
  let arguments = (ins
    I32Attr:$index
  );
  let results = (outs AnyMemRef);
}

def AllocMemrefOp : OKM_Op<"alloc_memref"> {
  let results = (outs AnyMemRef);
}

def PlanMemrefOp : OKM_Op<"plan_memref"> {
  let results = (outs AnyMemRef);
}

def TensorToRetOp : OKM_Op<"tensor_to_ret"> {
  let arguments = (ins
    AnyTensor:$tensor,
    I32Attr:$index
  );
  let results = (outs AnyTensor);
}

def MemrefToRetOp : OKM_Op<"memref_to_ret"> {
  let arguments = (ins
    AnyMemRef:$tensor,
    I32Attr:$index
  );
  let results = (outs AnyMemRef);
}

def WrapperOp : OKM_Op<"wrapper_kernel"> {
  let arguments = (ins
    Variadic<AnyType>:$operands
  );
  let results = (outs Variadic<AnyType>);
  let regions = (region AnyRegion:$body);
}


def ReturnOp : OKM_Op<"return", [HasParent<"WrapperOp">, Terminator]> {
  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<(ins),
    [{ build($_builder, $_state, llvm::None); }]>];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

#endif // ONEFLOW_IR_INCLUDE_ONEFLOW_OKM_OKMOPS

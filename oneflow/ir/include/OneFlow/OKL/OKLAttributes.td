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

#ifndef ONEFLOW_IR_INCLUDE_OKL_OKLATTRIBUTES
#define ONEFLOW_IR_INCLUDE_OKL_OKLATTRIBUTES

include "OneFlow/OKL/OKLDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def OKL_TensorFromArgsAttr : OKL_Attr<"TensorFromArgs", "arg"> {
  let summary = "Tensor From Args";
  let description = [{
    represent this tensor come from args:index
  }];
  let parameters = (ins "int":$index);
  let assemblyFormat = "`<` $index `>`";
}

def OKL_TensorFromRetsAttr : OKL_Attr<"TensorFromRets", "ret"> {
  let summary = "Tensor From Rets";
  let description = [{
    represent this tensor come from rets:index
  }];
  let parameters = (ins "int":$index);
  let assemblyFormat = "`<` $index `>`";
}

def OKL_TensorSignatureAttr : OKL_Attr<"TensorSignature", "tensor_signature"> {
  let summary = "Tensor signature of OKL Op";
  let description = [{
    To represent a signature, with a arrow in beween, pass two listes corepondent to the data input and data output tensors. For example:
    ```
    #okl.tensor_signature<[#okl.arg<0>] -> [#okl.ret<0>]>
    ```
  }];
  let parameters = (ins "ArrayAttr":$inputs, "ArrayAttr":$outputs);
  let assemblyFormat = "`<` $inputs ` ` `->` ` ` $outputs `>`";
}

#endif // ONEFLOW_IR_INCLUDE_OKL_OKLATTRIBUTES

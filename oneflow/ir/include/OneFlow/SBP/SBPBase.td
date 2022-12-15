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
#ifndef ONEFLOW_IR_INCLUDE_SBP_SBPBASE
#define ONEFLOW_IR_INCLUDE_SBP_SBPBASE

include "OneFlow/SBP/SBPDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"


class SBP_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<SBP_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def SBP_SplitAttr : SBP_Attr<"Split", "S"> {
  let summary = "Signature S";
  let description = [{
    signature split, representing a sharded tensor at the `axis`
  }];
  let parameters = (ins "int":$axis);
  let assemblyFormat = "`<` $axis `>`";
}

def SBP_BroadcastAttr : SBP_Attr<"Broadcast", "B"> {
  let summary = "Signature B";
  let description = [{
    signature broadcast, representing a tensor to be duplicated
  }];
}

def SBP_PartialSumAttr : SBP_Attr<"PartialSum", "P"> {
  let summary = "Signature P";
  let description = [{
    signature partial sum, representing a shareded tensor will be reduced lazily
  }];
}

def SBP_AnyAttr : SBP_Attr<"Any", "Any"> {
  let summary = "Signature Any";
  let description = [{
    signature any, representing one of sbp tensor;
  }];
}

def SBP_ParallelSignatureAttr : SBP_Attr<"ParallelSignature", "parallel"> {
  let summary = "Parallel signature of OneFlow Op, aka. SBP";
  let description = [{
    To represent a signature, with a arrow in beween, pass two listes corepondent to the data input and data output tensors. For example:
    ```
    #sbp.parallel<[#sbp.S<0>] -> [#sbp.S<0>]>
    ```
    One level nested list is used to represent a 2D parallelism signature. For example:
    ```
    #sbp.parallel<[[#sbp.S<0>, #sbp.P]] -> [#sbp.S<0>]>
    ```
  }];
  let parameters = (ins "ArrayAttr":$inputs, "ArrayAttr":$outputs);
  let assemblyFormat = "`<` custom<SBP>($inputs)` ` `->` ` ` custom<SBP>($outputs) `>`";
}

#endif // ONEFLOW_IR_INCLUDE_SBP_SBPBASE

/*
Copyright 2020 The sbp Authors. All rights reserved.

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
#ifndef ONEFLOW_IR_INCLUDE_SBP_SBPBASE_H_
#define ONEFLOW_IR_INCLUDE_SBP_SBPBASE_H_

include "OneFlow/SBP/SBPDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"


class SBP_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<SBP_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def SBP_SplitAttr : SBP_Attr<"Split", "s"> {
  let summary = "Signature S";
  let description = [{
    signature split, representing a sharded tensor at the `axis`
  }];
  let parameters = (ins "int":$axis);
  let assemblyFormat = "`<` $axis `>`";
}

def SBP_BroadcastAttr : SBP_Attr<"Broadcast", "b"> {
  let summary = "Signature B";
  let description = [{
    signature broadcast, representing a tensor to be duplicated
  }];
}

def SBP_PartialSumAttr : SBP_Attr<"PartialSum", "p"> {
  let summary = "Signature P";
  let description = [{
    signature partial sum, representing a shareded tensor will be reduced lazily
  }];
}

def SBP_2DAttr : SBP_Attr<"_2D", "_2d"> {
  let summary = "2D sbp";
  let description = [{
    2d sbp container
  }];
  let parameters = (ins "::mlir::Attribute":$axis0, "::mlir::Attribute":$axis1);
  let assemblyFormat = "`(` $axis0 `,` $axis1 `)`";
}

def SBP_ParallelSignatureAttr : SBP_Attr<"ParallelSignature", "parallel_signature"> {
  let summary = "Parallel signature of SBP Op, aka. SBP";
  let description = [{
    To represent a signature, with a arrow in beween, pass two listes corepondent to the data input and data output tensors. For example:
    ```
    #sbp.parallel_signature<[#sbp.s<0>] -> [#sbp.s<0>]>
    ```
    One level nested list is used to represent a 2D parallelism signature. For example:
    ```
    #sbp.parallel_signature<[[#sbp.s<0>, #sbp.p]] -> [#sbp.s<0>]>
    ```
  }];
  let parameters = (ins "::mlir::ArrayAttr":$inputs, "::mlir::ArrayAttr":$outputs);
  let assemblyFormat = "`<` $inputs `->` $outputs `>`";
}

#endif // ONEFLOW_IR_INCLUDE_SBP_SBPBASE_H_

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
#ifndef ONEFLOW_IR_INCLUDE_OKL_OKLBASE_H_
#define ONEFLOW_IR_INCLUDE_OKL_OKLBASE_H_

include "OneFlow/OKL/OKLDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

class OKL_Op<string mnemonic, list<Trait> traits = []> :
    Op<OKL_Dialect, mnemonic, traits>;

class OKL_Type<string name, list<Trait> traits = []>
    : TypeDef<OKL_Dialect, name, traits> {
}
#endif // ONEFLOW_IR_INCLUDE_OKL_OKLBASE_H_

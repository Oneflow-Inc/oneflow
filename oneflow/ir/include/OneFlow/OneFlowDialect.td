//===- OneFlowDialect.td - OneFlow dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ONEFLOW_DIALECT
#define ONEFLOW_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// OneFlow dialect definition.
//===----------------------------------------------------------------------===//

def OneFlow_Dialect : Dialect {
    let name = "oneflow";
    let summary = "OneFlow MLIR dialect.";
    let description = [{
        This dialect is the IR of OneFlow.
    }];
    let cppNamespace = "::mlir::oneflow";
}

//===----------------------------------------------------------------------===//
// Base oneflow operation definition.
//===----------------------------------------------------------------------===//

class OneFlow_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<OneFlow_Dialect, mnemonic, traits>;

#endif // ONEFLOW_DIALECT

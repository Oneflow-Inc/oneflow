//===- OneFlowOps.td - OneFlow dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ONEFLOW_OPS
#define ONEFLOW_OPS

include "OneFlowDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def OneFlow_FooOp : OneFlow_Op<"foo", [NoSideEffect,
                                             SameOperandsAndResultType]> {
    let summary = "Illustrates how to define an operation.";

    let arguments = (ins I32:$input);
    let results = (outs I32:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

def ConstantOp : OneFlow_Op<"constant", [NoSideEffect]> {
  let summary = "constant";
  let arguments = (ins F64ElementsAttr:$value);

  let results = (outs F64Tensor);

  let parser = [{ return ::parseConstantOp(parser, result); }];

  let builders = [
    // Build a constant with a given constant tensor value.
    OpBuilderDAG<(ins "DenseElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>,

    OpBuilderDAG<(ins "double":$value)>
  ];

  let verifier = [{ return ::verify(*this); }];
}

#endif // ONEFLOW_OPS

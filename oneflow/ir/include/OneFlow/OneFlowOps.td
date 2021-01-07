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

class OneFlow_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<OneFlow_Dialect, mnemonic, traits>;

class OneFlow_UserOp<string mnemonic, list<OpTrait> traits = []> :
        Op<OneFlow_Dialect, mnemonic, traits>;

def ConstantOp : OneFlow_Op<"constant", [NoSideEffect]> {
  // Provide a summary and description for this operation. This can be used to
  // auto-generate documentation of the operations within our dialect.
  let summary = "constant";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = oneflow.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>
                        : tensor<2x3xf64>
    ```
  }];

  // The constant operation takes an attribute as the only input.
  let arguments = (ins F64Attr:$double_value);

  // The constant operation returns a single value of TensorType.
  let results = (outs F32Tensor);

  // Specify a parser and printer method.
  let parser = [{ return ::parseConstantOp(parser, result); }];

  // Add custom build methods for the constant operation. These method populates
  // the `state` that MLIR uses to create operations, i.e. these are used when
  // using `builder.create<ConstantOp>(...)`.
    let builders = [
    OpBuilderDAG<(ins "double":$value)>
  ];

  // Invoke a static verify method to verify this constant operation.
  let verifier = [{ return ::verify(*this); }];
}

def OneFlow_ReluOp : OneFlow_Op<"relu", [Idempotent, NoSideEffect, SameOperandsAndResultType]> {
    let summary = "ReLU op";

    let arguments = (ins F32Tensor:$input);
    let results = (outs F32Tensor:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

def OneFlow_UserOp : OneFlow_Op<"user", []> {
    let summary = "Generic op for all kinds of user op";

    let arguments = (ins
      Variadic<AnyType>:$operands,
      StrAttr:$op_type_name,
      DictionaryAttr:$attr
    );
    let results = (outs Variadic<AnyType>:$output);
}

#endif // ONEFLOW_OPS

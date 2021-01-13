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

// TODO: Define SBP attribute type
// TODO: OneFlow_Op should has a placement attribute

// Replace i64 arr with shape arr in op
class OneFlow_ShapeAttr: TypedArrayAttrBase<I64Attr, "">;

class OneFlow_Blob<list<Type> allowedTypes> :
  AnyTypeOf<[TensorOf<allowedTypes>, NoneType], "">;

// TODO: add control in operand, which should be optional
// TODO: add control out result, could be optional
class OneFlow_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<OneFlow_Dialect, mnemonic, traits> {
  dag sys_attrs = (ins
    StrAttr:$op_name,
    StrAttr:$device,
    // TODO: change placement to dict and parse the literal fmt like "0:0-0"
    StrArrayAttr:$placement
  );
  dag attrs = (ins);
  dag data_in = (ins);
  dag ctrl_in = (ins);
  let arguments = !con(
      data_in,
      ctrl_in,
      sys_attrs,
      attrs
  );
}

def OneFlow_ConstantOp : OneFlow_Op<"constant", [NoSideEffect]> {
  let summary = "constant";
  let attrs = (ins F64Attr:$double_value);
  let results = (outs F32Tensor);
}

def OneFlow_ReluOp : OneFlow_Op<"relu", [Idempotent, NoSideEffect, SameOperandsAndResultType]> {
    let summary = "ReLU op";

    let arguments = (
      ins AnyType:$input,
      StrAttr:$op_name
    );
    let results = (outs AnyType:$res);
    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

def OneFlow_UserOp : OneFlow_Op<"user", []> {
    let data_in = (ins Variadic<AnyType>:$operands);
    let attrs = (ins
        StrAttr:$op_type_name,
        DictionaryAttr:$attr
    );
    let results = (outs Variadic<AnyType>:$outputs);
}

def OneFlow_PlaceholderOp : OneFlow_Op<"NoOp", [NoSideEffect]> {
  let summary = "Placeholder for ops with control edge or foreign input, etc";
  let data_in = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$outputs);
}

#endif // ONEFLOW_OPS

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

// TODO: make placement hashable and comparable
class OneFlow_Placement : TypedArrayAttrBase<StrAttr, "">;

def OneFlow_PlacementAttr : OneFlow_Placement;

// TODO: add control in operand, which should be optional
// TODO: add control out result, could be optional
class OneFlow_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<OneFlow_Dialect, mnemonic, traits> {
  dag mandatory_attrs = (ins
    OneFlow_PlacementAttr:$placement
  );
  dag attrs = (ins);
  dag data_in = (ins);
  dag ctrl_in = (ins);
  let arguments = !con(
      mandatory_attrs,
      attrs,
      data_in,
      ctrl_in
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
    let summary = "Generic op for all kinds of user op";

    let arguments = (ins
      Variadic<AnyType>:$operands,
      StrAttr:$op_name,
      StrAttr:$op_type_name,
      DictionaryAttr:$attr
    );
    let results = (outs Variadic<AnyType>:$outputs);
}

def OneFlow_NoOp : OneFlow_Op<"NoOp", [NoSideEffect]> {
  let summary = "Placeholder for ops with control edge or foreign input, etc";

  let arguments = (ins);

  let results = (outs);
}

#endif // ONEFLOW_OPS

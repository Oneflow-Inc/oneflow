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

class OneFlow_Blob<list<Type> allowedTypes> :
  AnyTypeOf<[TensorOf<allowedTypes>, NoneType], "">;

// TODO: Use traits AttrSizedOperandSegments and AttrSizedResultSegments to split ctrl and data
// I32ElementsAttr:$result_segment_sizes,
// I32ElementsAttr:$operand_segment_sizes

class OneFlow_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<OneFlow_Dialect, mnemonic, traits> {
  dag sys_attrs = (ins
    StrAttr:$op_name,
    StrAttr:$device,
    // TODO: change placement to dict and parse the literal fmt like "0:0-0"
    StrArrayAttr:$placement
  );
  dag attrs = (ins);
  dag input = (ins);
  dag ctrl_input = (ins);
  dag ctrl_output = (outs);
  let arguments = !con(
      input,
      ctrl_input,
      sys_attrs,
      attrs
  );
}

def OneFlow_ConstantOp : OneFlow_Op<"constant", [NoSideEffect]> {
  let summary = "";
  let attrs = (ins
    OptionalAttr<F64Attr>:$floating_value,
    OptionalAttr<I64Attr>:$integer_value
  );
  let results = (outs F32Tensor);
}

def OneFlow_ShardingOp : OneFlow_Op<"sharding", [NoSideEffect, SameOperandsAndResultType]> {
    let summary = "Sharding op for SBP";

    let arguments = (
      ins AnyType:$input,
      StrArrayAttr:$input_signature,
      StrArrayAttr:$output_signature,
      I64ArrayAttr:$shape
    );
    let results = (outs AnyType:$res);
    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

def OneFlow_ReluOp : OneFlow_Op<"relu", [Idempotent, NoSideEffect, SameOperandsAndResultType]> {
    let summary = "";

    let arguments = (
      ins AnyType:$input,
      StrAttr:$op_name
    );
    let results = (outs AnyType:$res);
    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

def OneFlow_SystemOp : OneFlow_Op<"system", []> {
  let summary = "";
  let input = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$outputs);
}

#endif // ONEFLOW_OPS

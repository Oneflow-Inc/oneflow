#ifndef ONEFLOW_OPS
#define ONEFLOW_OPS

include "OneFlowDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class OneFlow_BaseOp<string mnemonic, list<OpTrait> traits = []> :
        Op<OneFlow_Dialect, mnemonic, traits> {
  dag sys_attrs = (ins
    StrAttr:$op_name,
    OptionalAttr<BoolAttr>:$trainable,
    OptionalAttr<StrAttr>:$device,
    // TODO: change placement to dict and parse the literal fmt like "0:0-0"
    StrArrayAttr:$placement,
    OptionalAttr<I64Attr>:$scope_symbol_id
  );
  dag attrs = (ins);
  dag trait_attrs = (ins);
  dag input = (ins Variadic<AnyType>:$operands);
  dag output = (outs Variadic<AnyType>:$results);
  dag ctrl_input = (ins);
  dag ctrl_output = (outs);
  let arguments = !con(
      input,
      ctrl_input,
      sys_attrs,
      trait_attrs,
      attrs
  );
  let results = !con(
    output,
    ctrl_output
  );
}

class OneFlow_Op<string mnemonic, list<OpTrait> traits = []> :
        OneFlow_BaseOp<mnemonic, !listconcat(traits, [AttrSizedOperandSegments, AttrSizedResultSegments])> {
  let ctrl_input = (ins Variadic<AnyType>:$ctrl_inputs);
  let ctrl_output = (outs Optional<AnyType>:$ctrl_output);
  let trait_attrs = (ins
    I32ElementsAttr:$operand_segment_sizes,
    I32ElementsAttr:$result_segment_sizes
  );
}

def OneFlow_ConstantOp : OneFlow_BaseOp<"constant", [NoSideEffect]> {
  let summary = "";
  let attrs = (ins
    OptionalAttr<F64Attr>:$floating_value,
    OptionalAttr<I64Attr>:$integer_value
  );
  let output = (outs F32Tensor);
}

def OneFlow_ShardingOp : OneFlow_BaseOp<"sharding", [NoSideEffect, SameOperandsAndResultType]> {
    let summary = "Sharding op for SBP";
    let attrs = (ins
      StrArrayAttr:$input_signature,
      StrArrayAttr:$output_signature,
      I64ArrayAttr:$shape
    );
    let output = (outs AnyType:$res);
}

def OneFlow_UserOp : OneFlow_BaseOp<"user", []> {
    let summary = "";
    let attrs = (ins
      StrArrayAttr:$input_lbn_segment_keys,
      I32ArrayAttr:$input_lbn_segment_sizes,
      StrArrayAttr:$output_lbn_segment_keys,
      I32ArrayAttr:$output_lbn_segment_sizes,
      StrArrayAttr:$output_lbns
    );
}

def OneFlow_ReluOp : OneFlow_BaseOp<"relu", [Idempotent, NoSideEffect, SameOperandsAndResultType]> {
    let summary = "";
    let input = (ins AnyType:$x);
    let output = (outs AnyType:$res);
}

def OneFlow_SystemOp : OneFlow_Op<"system", []> {
  let summary = "";
  let attrs = (ins
    StrArrayAttr:$input_bns,
    StrArrayAttr:$output_lbns
  );
  let input = (ins Variadic<AnyType>:$operands);
  let output = (outs Variadic<AnyType>:$outputs);
}

#endif // ONEFLOW_OPS

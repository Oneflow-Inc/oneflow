#ifndef ONEFLOW_OPS
#define ONEFLOW_OPS

include "OneFlowDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// TODO: Use traits AttrSizedOperandSegments and AttrSizedResultSegments to split ctrl and data
// I32ElementsAttr:$result_segment_sizes,
// I32ElementsAttr:$operand_segment_sizes

class OneFlow_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<OneFlow_Dialect, mnemonic, !listconcat(traits, [AttrSizedOperandSegments])> {
  dag sys_attrs = (ins
    StrAttr:$op_name,
    OptionalAttr<BoolAttr>:$trainable,
    OptionalAttr<StrAttr>:$device,
    // TODO: change placement to dict and parse the literal fmt like "0:0-0"
    StrArrayAttr:$placement,
    OptionalAttr<I64Attr>:$scope_symbol_id
  );
  dag attrs = (ins);
  dag trait_attrs = (ins I32ElementsAttr:$operand_segment_sizes);
  dag input = (ins Variadic<AnyType>:$operands);
  dag ctrl_input = (ins Variadic<AnyType>:$ctrls);
  dag ctrl_output = (outs);
  let arguments = !con(
      input,
      ctrl_input,
      sys_attrs,
      trait_attrs,
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
    let attrs = (ins
      StrArrayAttr:$input_signature,
      StrArrayAttr:$output_signature,
      I64ArrayAttr:$shape
    );
    let results = (outs AnyType:$res);
}

// TODO: MLIR Idempotent requires op to have only one operand,
// so we need to implement a special Idempotent trait aware of control edges in oneflow
def OneFlow_ReluOp : OneFlow_Op<"relu", [NoSideEffect, SameOperandsAndResultType]> {
    let summary = "";
    let results = (outs AnyType:$res);
}

def OneFlow_SystemOp : OneFlow_Op<"system", []> {
  let summary = "";
  let attrs = (ins
    StrArrayAttr:$input_bns,
    StrArrayAttr:$output_lbns
  );
  let input = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$outputs);
}

#endif // ONEFLOW_OPS

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
    StrArrayAttr:$placement, // TODO: change placement to dict and parse the literal fmt like "0:0-0"
    OptionalAttr<I64Attr>:$scope_symbol_id
  );
  dag attrs = (ins);
  dag trait_attrs = (ins);
  dag user_op_lbn_attrs = (ins);
  dag input = (ins Variadic<AnyType>:$data_input);
  dag output = (outs Variadic<AnyType>:$data_output);
  dag ctrl_input = (ins);
  dag ctrl_output = (outs);
  let arguments = !con(
      input,
      ctrl_input,
      sys_attrs,
      trait_attrs,
      user_op_lbn_attrs,
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

class OneFlow_UserBaseOp<string mnemonic, list<OpTrait> traits = []> :
        OneFlow_BaseOp<mnemonic, traits> {
    let summary = "";
    let user_op_lbn_attrs = (ins
      StrArrayAttr:$input_lbn_segment_keys,
      I32ArrayAttr:$input_lbn_segment_sizes,
      StrArrayAttr:$output_lbn_segment_keys,
      I32ArrayAttr:$output_lbn_segment_sizes,
      StrArrayAttr:$output_lbns
    );
}

class OneFlow_UserBaseWithCtrlOp<string mnemonic, list<OpTrait> traits = []> :
        OneFlow_UserBaseOp<mnemonic, !listconcat(traits, [AttrSizedOperandSegments, AttrSizedResultSegments])> {
    let summary = "";
    let ctrl_input = (ins Variadic<AnyType>:$ctrl_inputs);
    let ctrl_output = (outs Optional<AnyType>:$ctrl_output);
    let trait_attrs = (ins
      I32ElementsAttr:$operand_segment_sizes,
      I32ElementsAttr:$result_segment_sizes
    );
}

def OneFlow_UserOp : OneFlow_UserBaseOp<"user", [AttrSizedOperandSegments, AttrSizedResultSegments]> {
    let summary = "";
    let ctrl_input = (ins Variadic<AnyType>:$ctrl_inputs);
    let ctrl_output = (outs Optional<AnyType>:$ctrl_output);
    let trait_attrs = (ins
      I32ElementsAttr:$operand_segment_sizes,
      I32ElementsAttr:$result_segment_sizes
    );
    let hasCanonicalizer = 1;
}

// TODO: support canonicalization
def OneFlow_ConstantOp : OneFlow_UserBaseOp<"constant", [AttrSizedOperandSegments, NoSideEffect]> {
  let summary = "";
  let ctrl_input = (ins Variadic<AnyType>:$ctrl_inputs);
  let attrs = (ins
    OptionalAttr<F64Attr>:$floating_value,
    OptionalAttr<SI64Attr>:$integer_value
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

def OneFlow_ReluOp : OneFlow_BaseOp<"relu", [Idempotent, NoSideEffect, SameOperandsAndResultType]> {
    let summary = "";
    let input = (ins AnyType:$x);
    let output = (outs AnyType:$y);
}

def OneFlow_SystemOp : OneFlow_Op<"system", []> {
  let summary = "";
  let attrs = (ins
    StrArrayAttr:$input_bns,
    StrArrayAttr:$output_lbns
  );
}

#endif // ONEFLOW_OPS

#ifndef ONEFLOW_OPS
#define ONEFLOW_OPS

include "OneFlowDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class OneFlow_BaseOp<string mnemonic, list<OpTrait> traits = []> :
        Op<OneFlow_Dialect, mnemonic, traits> {
  dag sys_attrs = (ins
    StrAttr:$op_name,
    OptionalAttr<BoolAttr>:$trainable,
    StrAttr:$device_tag,
    StrArrayAttr:$device_name, // TODO: change device_name to dict and parse the literal fmt like "0:0-0"
    OptionalAttr<I64Attr>:$scope_symbol_id,
    OptionalAttr<I64ArrayAttr>:$hierarchy
  );
  dag attrs = (ins);
  dag trait_attrs = (ins);
  dag user_op_attrs = (ins);
  dag input = (ins Variadic<AnyType>:$data_input);
  dag output = (outs Variadic<AnyType>:$data_output);
  dag ctrl_input = (ins);
  dag ctrl_output = (outs);
  let arguments = !con(
      input,
      ctrl_input,
      sys_attrs,
      trait_attrs,
      user_op_attrs,
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
    let user_op_attrs = (ins
      StrAttr:$op_type_name,
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

// TODO: support ConstantLike trait
def OneFlow_ConstantOp : OneFlow_UserBaseOp<"constant", [AttrSizedOperandSegments, NoSideEffect]> {
  let summary = "";
  let ctrl_input = (ins Variadic<AnyType>:$ctrl_inputs);
  let attrs = (ins
    OptionalAttr<F64Attr>:$floating_value,
    OptionalAttr<SI64Attr>:$integer_value
  );
  let output = (outs AnyTypeOf<[F32Tensor, F64Tensor, I32Tensor, I64Tensor]>);
}

def OneFlow_SystemOp : OneFlow_Op<"system", []> {
  let summary = "";
  let attrs = (ins
    StrArrayAttr:$input_bns,
    StrArrayAttr:$output_lbns
  );
  let hasCanonicalizer = 1;
}

def OneFlow_CastOp : OneFlow_BaseOp<"cast", []> {}

def OneFlow_ScalarMulByTensorOp : OneFlow_BaseOp<"scalar_mul_by_tensor", []> {}

class OneFlow_UnaryBaseOp<string mnemonic, list<OpTrait> traits = []> :
        OneFlow_BaseOp<mnemonic, !listconcat(traits, [SameOperandsAndResultType, NoSideEffect])> {
    let summary = "";
    let input = (ins AnyType:$x);
    let output = (outs AnyType:$y);
}


// Idempotent ops

def OneFlow_Idempotent : NativeOpTrait<"IsIdempotentOfIdenticalPlacement">;

class OneFlow_IdempotentBaseOp<string mnemonic, list<OpTrait> traits = []> :
        OneFlow_UnaryBaseOp<mnemonic, !listconcat(traits, [OneFlow_Idempotent])> {
}

def OneFlow_AbsOp : OneFlow_IdempotentBaseOp<"abs", []> {}

def OneFlow_CeilOp : OneFlow_IdempotentBaseOp<"ceil", []> {}

def OneFlow_floorOp : OneFlow_IdempotentBaseOp<"floor", []> {}

// TODO: pattern match constant_like to create ones_like when constant_like is port to user op
def OneFlow_OnesLikeOp : OneFlow_IdempotentBaseOp<"ones_like", []> {}

def OneFlow_ReluOp : OneFlow_IdempotentBaseOp<"relu", []> {}

// TODO: pattern match hardtanh to create Relu6 op
def OneFlow_Relu6Op : OneFlow_IdempotentBaseOp<"relu6", []> {}

def OneFlow_RintOp : OneFlow_IdempotentBaseOp<"rint", []> {}

def OneFlow_RoundOp : OneFlow_IdempotentBaseOp<"round", []> {}

def OneFlow_SignOp : OneFlow_IdempotentBaseOp<"sign", []> {}

// TODO: pattern match constant_like to create zeros_like when constant_like is port to user op
def OneFlow_ZerosLikeOp : OneFlow_IdempotentBaseOp<"zeros_like", []> {}

// Involution ops

def OneFlow_Involution : NativeOpTrait<"IsInvolutionOfIdenticalPlacement">;

// TODO: add invert, logical_not when they are implemented in OneFlow

class OneFlow_InvolutionBaseOp<string mnemonic, list<OpTrait> traits = []> :
        OneFlow_UnaryBaseOp<mnemonic, !listconcat(traits, [OneFlow_Involution])> {
}

def OneFlow_NegativeOp : OneFlow_InvolutionBaseOp<"negative", []> {}

def OneFlow_ReciprocalOp : OneFlow_InvolutionBaseOp<"reciprocal", []> {}

#endif // ONEFLOW_OPS

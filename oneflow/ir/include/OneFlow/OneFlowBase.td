#ifndef ONEFLOW_IR_INCLUDE_ONEFLOW_ONEFLOWBASE_H_
#define ONEFLOW_IR_INCLUDE_ONEFLOW_ONEFLOWBASE_H_

include "OneFlow/OneFlowDialect.td"
include "OneFlow/OneFlowInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "/mlir/Interfaces/InferTypeOpInterface.td"

def OneFlow_InvalidElement: TypeDef<OneFlow_Dialect, "InvalidElement"> {
  let mnemonic = "invalid_element";
}
def OneFlow_CharElement: TypeDef<OneFlow_Dialect, "CharElement"> {
  let mnemonic = "char_element";
}
def OneFlow_TensorBufferElement: TypeDef<OneFlow_Dialect, "TensorBufferElement"> {
  let mnemonic = "tensor_buffer_element";
}
def OneFlow_OFRecordElement: TypeDef<OneFlow_Dialect, "OFRecordElement"> {
  let mnemonic = "of_record_element";
}

def OneFlow_OFRecordTensor : TensorOf<[OneFlow_OFRecordElement]>;
def OneFlow_TensorBufferTensor : TensorOf<[OneFlow_TensorBufferElement]>;

def OneFlow_Tensor : TensorOf<[AnyType]>;

def SI32ArrayAttr : TypedArrayAttrBase<SI32Attr, "signed 32-bit integer array attribute"> {}

def SI64ArrayAttr : TypedArrayAttrBase<SI64Attr, "signed 64-bit integer array attribute"> {}

def ShapeAttr : TypedArrayAttrBase<SI64Attr, ""> {}

def DTArrayAttr : TypedArrayAttrBase<OneFlow_DataType, ""> {}

def ShapeArrayAttr : TypedArrayAttrBase<ShapeAttr, ""> {}

def ComplexDoubleAttr : TypedArrayAttrBase<F64Attr, ""> {}

def OneFlow_IsOpConfCompatible : NativeOpTrait<"IsOpConfCompatible">;
def OneFlow_IsImportCompatible : NativeOpTrait<"IsImportCompatible">;
def OneFlow_AlternativeOp : NativeOpTrait<"IsAlternative">;
def OneFlow_TensorSource : NativeOpTrait<"TensorSource">;
def OneFlow_OnlyExistsInIR : NativeOpTrait<"OnlyExistsInIR">;
def OneFlow_ElementwiseOp : NativeOpTrait<"IsElementwise">;

class OneFlow_IROp<string mnemonic, list<Trait> traits = []> :
        Op<OneFlow_Dialect, mnemonic, !listconcat(traits, [OneFlow_OnlyExistsInIR])> {}

class OneFlow_BaseOp<string mnemonic, list<Trait> traits = []> :
        Op<OneFlow_Dialect, mnemonic, !listconcat(traits, [OneFlow_IsOpConfCompatible])> {
  dag op_conf_attrs = (ins
    StrAttr:$op_name,
    StrAttr:$device_tag,
    StrArrayAttr:$device_name, // TODO: change device_name to dict and parse the literal fmt like "0:0-0"
    OptionalAttr<I64Attr>:$scope_symbol_id,
    OptionalAttr<I64ArrayAttr>:$hierarchy
  );
  dag attrs = (ins);
  dag trait_attrs = (ins);
  dag user_op_attrs = (ins);
  dag input = (ins
    Optional<OneFlow_Tensor>:$UserSourceOpTickInput
  );
  dag output = (outs);
  dag ctrl_input = (ins);
  dag ctrl_output = (outs);
  let arguments = !con(
      input,
      ctrl_input,
      op_conf_attrs,
      trait_attrs,
      user_op_attrs,
      attrs
  );
  let results = !con(
    output,
    ctrl_output
  );
  int same_output_regst_num = -1;

  bit has_check_fn = 0;
  bit has_logical_tensor_desc_infer_fn = 0;
  bit has_physical_tensor_desc_infer_fn = 0;
  bit has_get_sbp_fn = 0;
  bit has_sbp_signature_infer_fn = 0;
  bit has_data_type_infer_fn = 0;
  bit has_device_and_stream_infer_fn = 0;
  bit has_input_arg_modify_fn = 0;
  bit has_output_arg_modify_fn = 0;
  bit has_output_blob_time_shape_infer_fn = 0;
  bit has_nd_sbp_infer_fn = 0;
  bit has_compute_complexity_fn = 0;
  bit has_get_nd_sbp_fn = 0;
  bit has_enumerate_nd_sbp_signatures_fn = 0;
  bit has_dump_nd_sbp_signature_for_op_conf_fn = 0;
}

class OneFlow_Op<string mnemonic, list<Trait> traits = []> :
        OneFlow_BaseOp<mnemonic, !listconcat(traits, [AttrSizedOperandSegments, AttrSizedResultSegments, DeclareOpInterfaceMethods<ControlEdgeCompatibleInterface>])> {
  let ctrl_input = (ins Variadic<AnyType>:$ctrl_inputs);
  let ctrl_output = (outs Optional<AnyType>:$ctrl_output);
  let trait_attrs = (ins
    I32ElementsAttr:$operand_segment_sizes,
    I32ElementsAttr:$result_segment_sizes
  );
}

class OneFlow_UserBaseOp<string mnemonic, list<Trait> traits = [OneFlow_AlternativeOp]> :
        OneFlow_BaseOp<mnemonic, traits> {
    let summary = "";
    let user_op_attrs = (ins
      StrAttr:$op_type_name,
      // NOTE: vector types must have positive constant sizes, so we can't use I32ElementsAttr
      I32ArrayAttr:$input_sizes,
      I32ArrayAttr:$output_sizes
    );
}

// Why don't we merge ctrl in/out and data in/out into operand_segment/result_segment_sizes?
// 1. We only need to erase operand_segment/result_segment_sizes when we are creating a concrete user op
// 2. Isolating data and ctrl make debug easier and produced IR more human-readable
class OneFlow_UserBaseWithCtrlOp<string mnemonic, list<Trait> traits = []> :
        OneFlow_UserBaseOp<mnemonic, !listconcat(traits, [AttrSizedOperandSegments, AttrSizedResultSegments, DeclareOpInterfaceMethods<ControlEdgeCompatibleInterface>])> {
    let summary = "";
    let ctrl_input = (ins Variadic<AnyType>:$ctrl_inputs);
    let ctrl_output = (outs Optional<AnyType>:$ctrl_output);
    let trait_attrs = (ins
      I32ElementsAttr:$operand_segment_sizes,
      I32ElementsAttr:$result_segment_sizes
    );
}


class OneFlow_ConvolutionBaseOp<string mnemonic, list<Trait> traits = []> :
        OneFlow_BaseOp<mnemonic, !listconcat(traits, [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>])> {
    let summary = "OneFlow convolution operation";
    let description = [{
    "The convolution operator consumes an input tensor and a filter, and"
    "computes the output."
    }];
    let input = (ins
      OneFlow_Tensor:$in,
      OneFlow_Tensor:$weight,
      Optional<OneFlow_Tensor>:$bias,
      Optional<OneFlow_Tensor>:$_add_to_output
    );
    let output = (outs OneFlow_Tensor:$out);
    let attrs = (ins
      DefaultValuedAttr<SI32Attr, "0">:$filters,
      SI32ArrayAttr:$padding_before,
      StrAttr:$data_format,
      SI32ArrayAttr:$kernel_size,
      SI32ArrayAttr:$strides,
      SI32ArrayAttr:$dilation_rate,
      DefaultValuedAttr<SI32Attr, "1">:$groups,
      DefaultValuedAttr<StrAttr, "\"\"">:$tuning_cache
    );
    let trait_attrs = (ins
      I32ElementsAttr:$operand_segment_sizes
    );
    let has_check_fn = 1;
    let has_logical_tensor_desc_infer_fn = 1;
    let has_physical_tensor_desc_infer_fn = 1;
    let has_get_sbp_fn = 1;
    let has_data_type_infer_fn = 1;
    let has_compute_complexity_fn = 1;
}

class OneFlow_TFPoolBaseOp<string mnemonic, list<Trait> traits = []> :
        OneFlow_BaseOp<mnemonic, !listconcat(traits, [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>])> {
    let summary = "OneFlow pooling operation, align with TensorFlow";
    let input = (ins AnyType:$x);
    let output = (outs AnyType:$y);
    let attrs = (ins
    StrAttr:$padding,
    SI32ArrayAttr:$padding_before,
    SI32ArrayAttr:$padding_after,
    StrAttr:$data_format,
    SI32ArrayAttr:$pool_size,
    SI32ArrayAttr:$strides,
    BoolAttr:$ceil_mode
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_compute_complexity_fn = 1;
}

class OneFlow_TFPoolGradBaseOp<string mnemonic, list<Trait> traits = []> :
        OneFlow_BaseOp<mnemonic, !listconcat(traits, [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>])> {
  let summary = "OneFlow pooling grad operation, align with TensorFlow";
  let input = (ins
    AnyType:$x,
    AnyType:$y,
    AnyType:$dy
  );
  let output = (outs AnyType:$dx);
  let attrs = (ins
    StrAttr:$padding,
    SI32ArrayAttr:$padding_before,
    SI32ArrayAttr:$padding_after,
    StrAttr:$data_format,
    SI32ArrayAttr:$pool_size,
    SI32ArrayAttr:$strides,
    BoolAttr:$ceil_mode
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_compute_complexity_fn = 1;
}


class OneFlow_MaxPoolBaseOp<string mnemonic, list<Trait> traits = []> :
        OneFlow_BaseOp<mnemonic, !listconcat(traits, [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>])> {
  let summary = "OneFlow Max Pooling operation";
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y,
    AnyType:$indice
  );
  let attrs = (ins
    SI32ArrayAttr:$padding,
    StrAttr:$data_format,
    SI32ArrayAttr:$kernel_size,
    SI32ArrayAttr:$stride,
    SI32ArrayAttr:$dilation,
    DefaultValuedAttr<BoolAttr, "false">:$return_indices,
    DefaultValuedAttr<BoolAttr, "false">:$ceil_mode
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_compute_complexity_fn = 1;
}

class OneFlow_MaxUnpoolBaseOp<string mnemonic, list<Trait> traits = []> :
        OneFlow_BaseOp<mnemonic, !listconcat(traits, [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>])> {
  let summary = "OneFlow Max Unpooling operation";
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$indices
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let attrs = (ins
    SI64ArrayAttr:$kernel_size,
    SI64ArrayAttr:$stride,
    SI64ArrayAttr:$padding,
    DefaultValuedAttr<BoolAttr, "false">:$has_output_size,
    ShapeAttr:$output_size
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

class OneFlow_AvgPoolBaseOp<string mnemonic, list<Trait> traits = []> :
        OneFlow_BaseOp<mnemonic, !listconcat(traits, [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>])> {
  let summary = "OneFlow Average Pooling operation";
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    SI32ArrayAttr:$padding,
    StrAttr:$data_format,
    SI32ArrayAttr:$kernel_size,
    SI32ArrayAttr:$stride,
    DefaultValuedAttr<BoolAttr, "false">:$ceil_mode,
    DefaultValuedAttr<BoolAttr, "false">:$count_include_pad,
    DefaultValuedAttr<SI32Attr, "0">:$divisor_override
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_compute_complexity_fn = 1;
}

class OneFlow_MaxPoolGradBaseOp<string mnemonic, list<Trait> traits = []> :
        OneFlow_BaseOp<mnemonic, !listconcat(traits, [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>])> {
  let summary = "OneFlow Max Pooling Grad operation";
  let input = (ins
    AnyType:$x,
    AnyType:$indice,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    SI32ArrayAttr:$padding,
    StrAttr:$data_format,
    SI32ArrayAttr:$kernel_size,
    SI32ArrayAttr:$stride,
    SI32ArrayAttr:$dilation,
    DefaultValuedAttr<BoolAttr, "false">:$return_indices,
    DefaultValuedAttr<BoolAttr, "false">:$ceil_mode
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_compute_complexity_fn = 1;
}

class OneFlow_AvgPoolGradBaseOp<string mnemonic, list<Trait> traits = []> :
        OneFlow_BaseOp<mnemonic, !listconcat(traits, [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>])> {
  let summary = "OneFlow Average Pooling Grad operation";
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    SI32ArrayAttr:$padding,
    StrAttr:$data_format,
    SI32ArrayAttr:$kernel_size,
    SI32ArrayAttr:$stride,
    DefaultValuedAttr<BoolAttr, "false">:$ceil_mode,
    DefaultValuedAttr<BoolAttr, "false">:$count_include_pad,
    DefaultValuedAttr<SI32Attr, "0">:$divisor_override
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_compute_complexity_fn = 1;
}

class OneFlow_MaxUnpoolGradBaseOp<string mnemonic, list<Trait> traits = []> :
        OneFlow_BaseOp<mnemonic, !listconcat(traits, [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>])> {
  let summary = "OneFlow Max Unpooling Grad operation";
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$indices,
    OneFlow_Tensor:$dy
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

class OneFlow_AdaptivePoolBaseOp<string mnemonic, list<Trait> traits = []> :
       OneFlow_BaseOp<mnemonic, !listconcat(traits, [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>])> {
  let summary = "OneFlow adaptive pool operation";
  let input = (ins
    AnyType:$x
  );
  let output = (outs AnyType:$y);
  let attrs = (ins
    SI64ArrayAttr:$output_size
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

class OneFlow_AdaptivePoolGradBaseOp<string mnemonic, list<Trait> traits = []> :
       OneFlow_BaseOp<mnemonic, !listconcat(traits, [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>])> {
  let summary = "OneFlow adaptive pool operation";
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs AnyType:$dx);
  let attrs = (ins
    SI64ArrayAttr:$output_size
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

class OneFlow_UnaryBaseOp<string mnemonic, list<Trait> traits = []> :
        OneFlow_BaseOp<mnemonic, !listconcat(traits, [SameOperandsAndResultType, NoSideEffect])> {
  let summary = "";
  let input = (ins AnyType:$x);
  let output = (outs AnyType:$y);
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

class OneFlow_AdaptiveMaxPoolBaseOp<string mnemonic, list<Trait> traits = []> :
       OneFlow_BaseOp<mnemonic, !listconcat(traits, [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>])> {
  let summary = "OneFlow adaptive max pool operation";
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y,
    AnyType:$index
  );
  let attrs = (ins
    SI64ArrayAttr:$output_size
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

class OneFlow_AdaptiveMaxPoolGradBaseOp<string mnemonic, list<Trait> traits = []> :
       OneFlow_BaseOp<mnemonic, !listconcat(traits, [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>])> {
  let summary = "OneFlow adaptive max pool grad operation";
  let input = (ins
    AnyType:$dy,
    AnyType:$x,
    AnyType:$index
  );
  let output = (outs AnyType:$dx);
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_Idempotent : NativeOpTrait<"IsIdempotentOfIdenticalPlacement">;

class OneFlow_IdempotentBaseOp<string mnemonic, list<Trait> traits = []> :
        OneFlow_UnaryBaseOp<mnemonic, !listconcat(traits, [OneFlow_Idempotent])> {}

def OneFlow_Involution : NativeOpTrait<"IsInvolutionOfIdenticalPlacement">;

class OneFlow_InvolutionBaseOp<string mnemonic, list<Trait> traits = []> :
        OneFlow_UnaryBaseOp<mnemonic, !listconcat(traits, [OneFlow_Involution])> {}

#define GET_ONEFLOW_BASE_OP_DEFINITIONS
include "OneFlow/OneFlowUserOps.td"

#endif  // ONEFLOW_IR_INCLUDE_ONEFLOW_ONEFLOWBASE_H_

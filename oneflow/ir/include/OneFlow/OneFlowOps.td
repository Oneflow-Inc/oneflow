#ifndef ONEFLOW_OPS
#define ONEFLOW_OPS

include "OneFlowDialect.td"
include "OneFlowEnums.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Pass/PassBase.td"
include "mlir/Interfaces/CallInterfaces.td"

def SI32ArrayAttr : TypedArrayAttrBase<SI32Attr,
                                      "signed 32-bit integer array attribute"> {
  let constBuilderCall = "$_builder.getArrayAttr(llvm::to_vector<8>(llvm::map_range(values, [this](int32_t v) -> Attribute { return builder_.getSI32IntegerAttr($0); })))";
}

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

// Why don't we merge ctrl in/out and data in/out into operand_segment/result_segment_sizes?
// 1. We only need to erase operand_segment/result_segment_sizes when we are creating a concrete user op
// 2. Isolating data and ctrl make debug easier and produced IR more human-readable
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

def OneFlow_CastOp : OneFlow_BaseOp<"cast", []> {
  let input = (ins AnyType:$x);
  let output = (outs AnyType:$y);
}

def OneFlow_ScalarMulByTensorOp : OneFlow_BaseOp<"scalar_mul_by_tensor", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$scalar
  );
  let output = (outs AnyType:$y);
}


class OneFlow_ScalarMathOp<string mnemonic, list<OpTrait> traits = []> :
        OneFlow_BaseOp<mnemonic, !listconcat(traits, [SameOperandsAndResultType, NoSideEffect])> {
    let summary = "OneFlow Scalar Math Ops";
    let input = (ins AnyType:$x);
    let output = (outs AnyType:$y);
    let attrs = (ins
    BoolAttr:$has_int_operand,
    BoolAttr:$has_float_operand,
    SI64Attr:$int_operand,
    F64Attr:$float_operand
  );
}

def OneFlow_ScalarAddOp : OneFlow_ScalarMathOp<"scalar_add", []> {}

def OneFlow_ScalarFloorDivOp : OneFlow_ScalarMathOp<"scalar_floordiv", []> {}

def OneFlow_ScalarFmodOp : OneFlow_ScalarMathOp<"scalar_fmod", []> {}

def OneFlow_ScalarMulOp : OneFlow_ScalarMathOp<"scalar_mul", []> {}

def OneFlow_ScalarPowOp : OneFlow_ScalarMathOp<"scalar_pow", []> {}

class OneFlow_ReduceBaseOp<string mnemonic, list<OpTrait> traits = []> :
        OneFlow_BaseOp<mnemonic, !listconcat(traits, [NoSideEffect])> {
    let summary = "OneFlow Reduce Mean Ops";
    let input = (ins AnyType:$input_tensor);
    let output = (outs AnyType:$output_tensor);
    let attrs = (ins
    SI32ArrayAttr:$axis,
    BoolAttr:$keepdims
  );
}

def OneFlow_ReduceMinOp : OneFlow_ReduceBaseOp<"reduce_min", []> {}

def OneFlow_ReduceMaxOp : OneFlow_ReduceBaseOp<"reduce_max", []> {}

def OneFlow_ReduceSumOp : OneFlow_ReduceBaseOp<"reduce_sum", []> {}

def OneFlow_ReduceProdOp : OneFlow_ReduceBaseOp<"reduce_prod", []> {}

def OneFlow_Reshape : OneFlow_BaseOp<"reshape", []> {
  let summary = "OneFlow Reshape operation";
  let description = [{
  "Reshape the input tensor similar to numpy.reshape."
  "First input is the data tensor, second input is a shape tensor which specifies the output shape. It outputs the reshaped tensor."
  "At most one dimension of the new shape can be -1. In this case, the value is"
  "inferred from the size of the tensor and the remaining dimensions. A dimension"
  "could also be 0, in which case the actual dimension value is unchanged (i.e. taken"
  "from the input tensor)."
  }];
  let input = (ins
    AnyType:$in
  );
  let output = (outs AnyType:$out);
}

def OneFlow_MatmulOp : OneFlow_BaseOp<"matmul", []> {
  let input = (ins
    AnyType:$a,
    AnyType:$b
  );
  let output = (outs AnyType:$out);
  let attrs = (ins
    BoolAttr:$transpose_a,
    BoolAttr:$transpose_b,
    OptionalAttr<BoolAttr>:$_add_to_output,
    F64Attr:$alpha
  );
}

class OneFlow_ConvolutionBaseOp<string mnemonic, list<OpTrait> traits = []> :
        OneFlow_BaseOp<mnemonic, !listconcat(traits, [NoSideEffect])> {
    let summary = "OneFlow convolution operation";
    let description = [{
    "The convolution operator consumes an input tensor and a filter, and"
    "computes the output."
    }];
    let input = (ins
      AnyType:$in,
      AnyType:$weight
    );
    let output = (outs AnyType:$out);
    let attrs = (ins
      SI32Attr:$filters,
      SI32ArrayAttr:$padding_before,
      StrAttr:$data_format,
      SI32ArrayAttr:$kernel_size,
      SI32ArrayAttr:$strides,
      SI32ArrayAttr:$dilation_rate,
      DefaultValuedAttr<SI32Attr, "1">:$group
    );
}

def OneFlow_Convolution1dOp : OneFlow_ConvolutionBaseOp<"conv1d", []> {}

def OneFlow_Convolution2dOp : OneFlow_ConvolutionBaseOp<"conv2d", []> {}

def OneFlow_Convolution3dOp : OneFlow_ConvolutionBaseOp<"conv3d", []> {}

def OneFlow_ConvFilterGradOp : OneFlow_BaseOp<"conv_filter_grad", []> {
  let summary = "OneFlow Conv Filter Grad operation";
  let input = (ins
    AnyType:$dy,
    AnyType:$x
  );
  let output = (outs AnyType:$filter_diff);
  let attrs = (ins
    SI32Attr:$num_spatial_dims,
    SI32ArrayAttr:$padding_before,
    StrAttr:$data_format,
    SI32ArrayAttr:$kernel_size,
    SI32ArrayAttr:$strides,
    SI32ArrayAttr:$dilation_rate,
    DefaultValuedAttr<SI32Attr, "1">:$group
  );
}

def OneFlow_ConvDataGradOp : OneFlow_BaseOp<"conv_data_grad", []> {
  let summary = "OneFlow Conv Data Grad operation";
  let input = (ins
    AnyType:$dy,
    AnyType:$filter,
    AnyType:$x_like
  );
  let output = (outs AnyType:$dx);
  let attrs = (ins
    SI32Attr:$num_spatial_dims,
    SI32ArrayAttr:$padding_before,
    StrAttr:$data_format,
    SI32ArrayAttr:$kernel_size,
    SI32ArrayAttr:$strides,
    SI32ArrayAttr:$dilation_rate,
    DefaultValuedAttr<SI32Attr, "1">:$group
  );
}

def OneFlow_BiasAddOp : OneFlow_BaseOp<"bias_add", []> {
  let summary = "OneFlow Bias Add operation";
  let input = (ins
    AnyType:$a,
    AnyType:$b
  );
  let output = (outs AnyType:$out);
  let attrs = (ins
    SI32Attr:$axis
  );
}

def OneFlow_FusedBiasAddGeluOp : OneFlow_BaseOp<"fused_bias_add_gelu", []>  {
  let summary = "";
  let input = (ins
    AnyType:$a,
    AnyType:$b
  );
  let output = (outs AnyType:$out);
  let attrs = (ins
    SI32Attr:$axis
  );
  let hasCanonicalizer = 1;
}

def OneFlow_DropoutOp : OneFlow_BaseOp<"dropout", []>  {
  let summary = "";
  let input = (ins
    AnyType:$in,
    AnyType:$mask
  );
  let output = (outs AnyType:$out);
  let attrs = (ins
    F32Attr:$scale
  );
}

def OneFlow_TrilOp : OneFlow_BaseOp<"tril", []>  {
  let summary = "";
  let input = (ins
    AnyType:$in
  );
  let output = (outs AnyType:$out);
  let attrs = (ins
    SI64Attr:$diagonal,
    F64Attr:$floating_fill_value,
    SI64Attr:$integer_fill_value,
    BoolAttr:$is_floating_fill_value
  );
}

def OneFlow_FusedScaleTrilOp : OneFlow_BaseOp<"fused_scale_tril", []>  {
  let summary = "";
  let input = (ins
    AnyType:$in
  );
  let output = (outs AnyType:$out);
  let attrs = (ins
    SI64Attr:$diagonal,
    F64Attr:$floating_fill_value,
    SI64Attr:$integer_fill_value,
    BoolAttr:$is_floating_fill_value,
    F64Attr:$floating_scale_value,
    SI64Attr:$integer_scale_value,
    BoolAttr:$is_floating_scale_value
  );
  let hasCanonicalizer = 1;
}

def OneFlow_FusedBiasAddMaskScaleOp : OneFlow_BaseOp<"fused_bias_add_mask_scale", []>  {
  let summary = "";
  let input = (ins
    AnyType:$a,
    AnyType:$b,
    AnyType:$mask
  );
  let output = (outs AnyType:$out);
  let attrs = (ins
    SI32Attr:$axis,
    F32Attr:$scale
  );
  let hasCanonicalizer = 1;
}

def OneFlow_RandomMaskLikeOp : OneFlow_BaseOp<"random_mask_like", []>  {
  let summary = "";
  let input = (ins
    AnyType:$like
  );
  let output = (outs AnyType:$out);
  let attrs = (ins
    F32Attr:$rate,
    SI64Attr:$seed
  );
}

class OneFlow_LazyPoolBaseOp<string mnemonic, list<OpTrait> traits = []> :
        OneFlow_BaseOp<mnemonic, !listconcat(traits, [NoSideEffect])> {
    let summary = "OneFlow Lazy Pooling operation";
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
}

class OneFlow_LazyPoolGradBaseOp<string mnemonic, list<OpTrait> traits = []> :
        OneFlow_BaseOp<mnemonic, !listconcat(traits, [NoSideEffect])> {
    let summary = "OneFlow Lazy Pooling Grad operation";
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
}


class OneFlow_EagerPoolBaseOp<string mnemonic, list<OpTrait> traits = []> :
        OneFlow_BaseOp<mnemonic, !listconcat(traits, [NoSideEffect])> {
    let summary = "OneFlow Eager Pooling operation";
    let input = (ins AnyType:$x);
    let output = (outs AnyType:$y);
    let attrs = (ins
    SI32ArrayAttr:$padding,
    StrAttr:$data_format,
    SI32ArrayAttr:$kernel_size,
    SI32ArrayAttr:$stride,
    BoolAttr:$ceil_mode,
    BoolAttr:$count_include_pad,
    SI64Attr:$divisor_override
  );
}

class OneFlow_EagerPoolGradBaseOp<string mnemonic, list<OpTrait> traits = []> :
        OneFlow_BaseOp<mnemonic, !listconcat(traits, [NoSideEffect])> {
    let summary = "OneFlow Eager Pooling Grad operation";
    let input = (ins
      AnyType:$x,
      AnyType:$y,
      AnyType:$dy
    );
    let output = (outs AnyType:$dx);
    let attrs = (ins
    SI32ArrayAttr:$padding,
    StrAttr:$data_format,
    SI32ArrayAttr:$kernel_size,
    SI32ArrayAttr:$stride,
    BoolAttr:$ceil_mode,
    BoolAttr:$count_include_pad,
    SI64Attr:$divisor_override
  );
}

def OneFlow_LazyMaxPool1dOp : OneFlow_LazyPoolBaseOp<"max_pool_1d", []> {}

def OneFlow_LazyMaxPoolGrad1dOp : OneFlow_LazyPoolGradBaseOp<"max_pool_1d_grad", []> {}

def OneFlow_LazyMaxPool2dOp : OneFlow_LazyPoolBaseOp<"max_pool_2d", []> {}

def OneFlow_LazyMaxPoolGrad2dOp : OneFlow_LazyPoolGradBaseOp<"max_pool_2d_grad", []> {}

def OneFlow_LazyMaxPool3dOp : OneFlow_LazyPoolBaseOp<"max_pool_3d", []> {}

def OneFlow_LazyMaxPoolGrad3dOp : OneFlow_LazyPoolGradBaseOp<"max_pool_3d_grad", []> {}

def OneFlow_LazyAvgPool1dOp : OneFlow_LazyPoolBaseOp<"avg_pool_1d", []> {}

def OneFlow_LazyAvgPoolGrad1dOp : OneFlow_LazyPoolGradBaseOp<"avg_pool_1d_grad", []> {}

def OneFlow_LazyAvgPool2dOp : OneFlow_LazyPoolBaseOp<"avg_pool_2d", []> {}

def OneFlow_LazyAvgPoolGrad2dOp : OneFlow_LazyPoolGradBaseOp<"avg_pool_2d_grad", []> {}

def OneFlow_LazyAvgPool3dOp : OneFlow_LazyPoolBaseOp<"avg_pool_3d", []> {}

def OneFlow_LazyAvgPoolGrad3dOp : OneFlow_LazyPoolGradBaseOp<"avg_pool_3d_grad", []> {}

def OneFlow_EagerAvgPool1dOp : OneFlow_EagerPoolBaseOp<"avgpool_1d", []> {}

def OneFlow_EagerAvgPoolGrad1dOp : OneFlow_EagerPoolGradBaseOp<"avgpool_1d_grad", []> {}

def OneFlow_EagerAvgPool2dOp : OneFlow_EagerPoolBaseOp<"avgpool_2d", []> {}

def OneFlow_EagerAvgPoolGrad2dOp : OneFlow_EagerPoolGradBaseOp<"avgpool_2d_grad", []> {}

def OneFlow_EagerAvgPool3dOp : OneFlow_EagerPoolBaseOp<"avgpool_3d", []> {}

def OneFlow_EagerAvgPoolGrad3dOp : OneFlow_EagerPoolGradBaseOp<"avgpool_3d_grad", []> {}


def OneFlow_GatherOp : OneFlow_BaseOp<"gather", []> {
  let input = (ins
    AnyType:$in,
    AnyType:$indices
  );
  let output = (outs AnyType:$out);
  let attrs = (ins
    SI64Attr:$axis
  );
}

def OneFlow_GeluGradOp : OneFlow_BaseOp<"gelu_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs AnyType:$dx);
}

class OneFlow_UnaryBaseOp<string mnemonic, list<OpTrait> traits = []> :
        OneFlow_BaseOp<mnemonic, !listconcat(traits, [SameOperandsAndResultType, NoSideEffect])> {
    let summary = "";
    let input = (ins AnyType:$x);
    let output = (outs AnyType:$y);
}



def OneFlow_GeluOp : OneFlow_UnaryBaseOp<"gelu", []> {}

def OneFlow_RsqrtOp : OneFlow_UnaryBaseOp<"rsqrt", []> {}

def OneFlow_TanhOp : OneFlow_UnaryBaseOp<"tanh", []> {}


def OneFlow_AssignOp : OneFlow_BaseOp<"assign", []> {
  let summary = "OneFlow assign operation";
  let input = (ins
    AnyType:$ref,
    AnyType:$value
  );
}

def OneFlow_AddNOp : OneFlow_BaseOp<"add_n", []> {
  let summary = "OneFlow add_n operation";
  let input = (ins Variadic<AnyType>:$in);
  let output = (outs AnyType:$out);
}


// optimizer ops

def OneFlow_SgdUpdateOp : OneFlow_BaseOp<"sgd_update", [AttrSizedOperandSegments]> {
  let summary = "OneFlow sgd_update operation";
  let input = (ins
    AnyType:$model,
    AnyType:$model_diff,
    Optional<AnyType>:$learning_rate,
    Optional<AnyType>:$scale_by_tensor,
    Optional<AnyType>:$skip_if
  );
  let attrs = (ins
    F32Attr:$learning_rate_val,
    F64Attr:$scale,
    F32Attr:$l1,
    F32Attr:$l2,
    F32Attr:$weight_decay
  );
  let trait_attrs = (ins
    I32ElementsAttr:$operand_segment_sizes
  );
}



// data preprocessing ops

def OneFlow_OFRecordReaderOp : OneFlow_BaseOp<"OFRecordReader", []> {
  let summary = "OneFlow OFRecordReader operation";
  let output = (outs AnyType:$out);
  let attrs = (ins
    StrAttr:$data_dir,
    SI32Attr:$data_part_num,
    SI32Attr:$batch_size,
    StrAttr:$part_name_prefix,
    SI32Attr:$part_name_suffix_length,
    BoolAttr:$random_shuffle,
    SI64Attr:$seed,
    SI32Attr:$shuffle_buffer_size,
    BoolAttr:$shuffle_after_epoch,
    StrArrayAttr:$nd_sbp
  );
}

def OneFlow_OFRecordRawDecoderOp : OneFlow_BaseOp<"ofrecord_raw_decoder", []> {
  let summary = "OneFlow ofrecord_raw_decoder operation";
  let input = (ins AnyType:$in);
  let output = (outs AnyType:$out);
  let attrs = (ins
    StrAttr:$name,
    AnyI64ElementsAttr:$shape,
    StrAttr:$data_type,
    BoolAttr:$dim1_varying_length,
    BoolAttr:$truncate
  );
}

def OneFlow_OFRecordImageDecoderOp : OneFlow_BaseOp<"ofrecord_image_decoder", []> {
  let summary = "OneFlow ofrecord_image_decoder operation";
  let input = (ins AnyType:$in);
  let output = (outs AnyType:$out);
  let attrs = (ins
    StrAttr:$name,
    StrAttr:$color_space
  );
}

def OneFlowImageResizeToFixedOp : OneFlow_BaseOp<"image_resize_to_fixed", []> {
  let summary = "OneFlow image_resize_to_fixed operation";
  let input = (ins AnyType:$in);
  let output = (outs
    AnyType:$out,
    AnyType:$scale
  );
  let attrs = (ins
    SI64Attr:$target_width,
    SI64Attr:$target_height,
    SI64Attr:$channels,
    StrAttr:$data_type,
    StrAttr:$interpolation_type
  );
}

def OneFlowCropMirrorNormalizeFromUINT8Op : OneFlow_BaseOp<"crop_mirror_normalize_from_uint8", []> {
  let summary = "OneFlow crop_mirror_normalize_from_uint8 operation";
  let input = (ins AnyType:$in);
  let output = (outs AnyType:$out);
  let attrs = (ins
    StrAttr:$color_space,
    StrAttr:$output_layout,
    F32ArrayAttr:$mean,
    F32ArrayAttr:$std,
    SI64Attr:$crop_h,
    SI64Attr:$crop_w,
    F32Attr:$crop_pos_x,
    F32Attr:$crop_pos_y,
    StrAttr:$output_dtype
  );
}


// Loss ops

def OneFlowSparseSoftmaxCrossEntropyOp : OneFlow_BaseOp<"sparse_softmax_cross_entropy", []> {
  let input = (ins
    AnyType:$prediction,
    AnyType:$label
  );
  let output = (outs
    AnyType:$prob,
    AnyType:$out
  );
  let attrs = (ins
    SI64Attr:$depth
  );
}

def OneFlowSparseSoftmaxCrossEntropyGradOp : OneFlow_BaseOp<"sparse_softmax_cross_entropy_grad", []> {
  let input = (ins
    AnyType:$label,
    AnyType:$dy,
    AnyType:$prob
  );
  let output = (outs
    AnyType:$prediction_diff
  );
  let attrs = (ins
    SI64Attr:$depth
  );
}

// Idempotent ops

def OneFlow_Idempotent : NativeOpTrait<"IsIdempotentOfIdenticalPlacement">;

class OneFlow_IdempotentBaseOp<string mnemonic, list<OpTrait> traits = []> :
        OneFlow_UnaryBaseOp<mnemonic, !listconcat(traits, [OneFlow_Idempotent])> {
}

def OneFlow_AbsOp : OneFlow_IdempotentBaseOp<"abs", []> {}

def OneFlow_CeilOp : OneFlow_IdempotentBaseOp<"ceil", []> {}

def OneFlow_FloorOp : OneFlow_IdempotentBaseOp<"floor", []> {}

// TODO: pattern match constant_like to create ones_like when constant_like is port to user op
def OneFlow_OnesLikeOp : OneFlow_IdempotentBaseOp<"ones_like", []> {}

def OneFlow_ReluOp : OneFlow_IdempotentBaseOp<"relu", []> {}

def OneFlowReLUGradOp : OneFlow_BaseOp<"relu_grad", []> {
  let input = (ins
    AnyType:$y,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

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

// JIT ops

def OneFlow_MlirJitOp : OneFlow_UserBaseOp<"mlir_jit", [
      CallOpInterface
    ]
  > {
  let attrs = (ins
    FlatSymbolRefAttr:$callee,
    StrAttr:$mlir_assembly
  );
  let builders = [
    OpBuilder<(ins
      "FuncOp":$callee,
      "NamedAttrList":$attributes,
      CArg<"ValueRange", "{}">:$data_input
    ), [{
      $_state.addOperands(data_input);
      $_state.addAttribute("callee",$_builder.getSymbolRefAttr(callee));
      $_state.addTypes(callee.getType().getResults());
      $_state.addAttributes(attributes);
    }]>,
    OpBuilder<(ins
      "SymbolRefAttr":$callee,
      "TypeRange":$results,
      "NamedAttrList":$attributes,
      CArg<"ValueRange", "{}">:$data_input
    ), [{
      $_state.addOperands(data_input);
      $_state.addAttribute("callee", callee);
      $_state.addTypes(results);
      $_state.addAttributes(attributes);
    }]>
  ];
    let extraClassDeclaration = [{
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }
  }];
  let assemblyFormat = [{
    $callee `(` $data_input `)` attr-dict `:` functional-type($data_input, results)
  }];
}

#endif // ONEFLOW_OPS

#ifndef ONEFLOW_PATTERNS
#define ONEFLOW_PATTERNS

def IsNotNestedInJit: Constraint<CPred<"(!$0.getDefiningOp()->getParentOfType<FuncOp>()->hasAttr(\"llvm.emit_c_interface\"))">, "">;
def OutlineMulCast : NativeCodeCall<"OutlineMulCast($_builder, $0, $1)">;
// TODO: remove attr binding if possible
def MulCastPattern : Pat<
  (
    OneFlow_ScalarMulByTensorOp : $mul_op
    (
      OneFlow_CastOp : $cast_op
        $cast_x,
        $cast_op_name,
        $cast_trainable,
        $cast_device_tag,
        $cast_device_name,
        $cast_scope_symbol_id,
        $cast_hierarchy
    ),
    $scalar,
    $mul_op_name,
    $mul_trainable,
    $mul_device_tag,
    $mul_device_name,
    $mul_scope_symbol_id,
    $mul_hierarchy
  ),
  (OutlineMulCast $mul_op, $cast_op),
  [
    (IsNotNestedInJit $mul_op)
  ]
>;

// TODO: FusedSelfAttentionPattern

def IsGPU: Constraint<CPred<"$0.getValue().equals(\"gpu\")">, "is GPU device">;
def FusedScaleTrilPattern : Pat<
  (
    OneFlow_TrilOp
    (
      OneFlow_ScalarMulOp
        $x,
        $scale_op_name,
        $scale_trainable,
        $scale_device_tag,
        $scale_device_name,
        $scale_scope_symbol_id,
        $scale_hierarchy,
        $has_int_operand,
        $has_float_operand,
        $int_operand,
        $float_operand
    ),
    $tril_op_name,
    $tril_trainable,
    $tril_device_tag,
    $tril_device_name,
    $tril_scope_symbol_id,
    $tril_hierarchy,
    $diagonal,
    $floating_fill_value,
    $integer_fill_value,
    $is_floating_fill_value
  ),
  (OneFlow_FusedScaleTrilOp $x,
    $tril_op_name,
    $tril_trainable,
    $tril_device_tag,
    $tril_device_name,
    $tril_scope_symbol_id,
    $tril_hierarchy,
    $diagonal,
    $floating_fill_value,
    $integer_fill_value,
    $is_floating_fill_value,
    $float_operand,
    $int_operand,
    $has_float_operand
  ),
  [
    (IsGPU $tril_device_tag),
    (IsGPU $scale_device_tag)
  ]
>;

def FusedScaleTrilPattern2 : Pat<
  (
    OneFlow_ScalarMulOp
    (
      OneFlow_TrilOp
      $x,
      $tril_op_name,
      $tril_trainable,
      $tril_device_tag,
      $tril_device_name,
      $tril_scope_symbol_id,
      $tril_hierarchy,
      $diagonal,
      $floating_fill_value,
      $integer_fill_value,
      $is_floating_fill_value
    ),
    $scale_op_name,
    $scale_trainable,
    $scale_device_tag,
    $scale_device_name,
    $scale_scope_symbol_id,
    $scale_hierarchy,
    $has_int_operand,
    $has_float_operand,
    $int_operand,
    $float_operand
  ),
  (OneFlow_FusedScaleTrilOp $x,
    $scale_op_name,
    $scale_trainable,
    $scale_device_tag,
    $scale_device_name,
    $scale_scope_symbol_id,
    $scale_hierarchy,
    $diagonal,
    $floating_fill_value,
    $integer_fill_value,
    $is_floating_fill_value,
    $float_operand,
    $int_operand,
    $has_float_operand
  ),
  [
    (IsGPU $tril_device_tag),
    (IsGPU $scale_device_tag)
  ]
>;

def FusedBiasAddDropoutPattern : Pat<
  (
    OneFlow_DropoutOp
    (
      OneFlow_BiasAddOp
        $a,
        $b,
        $bias_add_op_name,
        $bias_add_trainable,
        $bias_add_device_tag,
        $bias_add_device_name,
        $bias_add_scope_symbol_id,
        $bias_add_hierarchy,
        $axis
    ),
    $mask,
    $dropout_op_name,
    $dropout_trainable,
    $dropout_device_tag,
    $dropout_device_name,
    $dropout_scope_symbol_id,
    $dropout_hierarchy,
    $scale
  ),
  (OneFlow_FusedBiasAddMaskScaleOp $a, $b, $mask,
    $dropout_op_name,
    $dropout_trainable,
    $dropout_device_tag,
    $dropout_device_name,
    $dropout_scope_symbol_id,
    $dropout_hierarchy,
    $axis,
    $scale
  ),
  []
>;

def FusedBiasAddGeluPattern : Pat<
  (
    OneFlow_GeluOp : $gelu_op
    (
      OneFlow_BiasAddOp
        $a,
        $b,
        $bias_add_op_name,
        $bias_add_trainable,
        $bias_add_device_tag,
        $bias_add_device_name,
        $bias_add_scope_symbol_id,
        $bias_add_hierarchy,
        $axis
    ),
    $gelu_op_name,
    $gelu_trainable,
    $gelu_device_tag,
    $gelu_device_name,
    $gelu_scope_symbol_id,
    $gelu_hierarchy
  ),
  (OneFlow_FusedBiasAddGeluOp $a, $b,
    $gelu_op_name,
    $gelu_trainable,
    $gelu_device_tag,
    $gelu_device_name,
    $gelu_scope_symbol_id,
    $gelu_hierarchy,
    $axis
  ),
  []
>;

#endif // ONEFLOW_PATTERNS

#ifndef ONEFLOW_PASSES
#define ONEFLOW_PASSES

def LowerOneFlowToTosaPass : Pass<"lower-oneflow-to-tosa", "ModuleOp"> {
  let summary = "";
  let constructor = "mlir::oneflow::createLowerOneFlowToTosaPass()";
  let dependentDialects = ["tosa::TosaDialect", "memref::MemRefDialect", "StandardOpsDialect"];
}

def MapSCFToGPUPass : Pass<"gpu-greedy-parallel-loop-mapping", "FuncOp"> {
  let summary = "Greedily maps all parallel loops to gpu hardware ids";
  let constructor = "mlir::oneflow::createMapSCFToGPUPass()";
  let dependentDialects = ["scf::SCFDialect"];
}

def BufferHostRegisterPass : FunctionPass<"buffer-host-register"> {
  let summary = "";
  let constructor = "mlir::oneflow::createBufferHostRegisterPass()";
  let dependentDialects = ["gpu::GPUDialect"];
}

def OutlineJitFunctionPass : Pass<"outline-jit-function", "ModuleOp"> {
  let summary = "move ops could be jitted to jit function";
  let constructor = "mlir::oneflow::createOutlineJitFunctionPass()";
}

def FuseIntoExistingOpPass : Pass<"fuse-into-existing-op", "FuncOp"> {
  let summary = "";
  let constructor = "mlir::oneflow::createFuseIntoExistingOpPass()";
}

#endif // ONEFLOW_PASSES

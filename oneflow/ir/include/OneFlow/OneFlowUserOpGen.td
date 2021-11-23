def OneFlow_ZeroLikeOp : OneFlow_BaseOp<"zero_like", []> {
  let input = (ins
    AnyType:$like
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_WhereScalarYOp : OneFlow_BaseOp<"where_scalar_y", []> {
  let input = (ins
    AnyType:$condition,
    AnyType:$x
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    BoolAttr:$has_int_operand,
    BoolAttr:$has_float_operand,
    SI64Attr:$int_operand,
    F64Attr:$float_operand
  );
}

def OneFlow_WhereScalarXOp : OneFlow_BaseOp<"where_scalar_x", []> {
  let input = (ins
    AnyType:$condition,
    AnyType:$y
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    BoolAttr:$has_int_operand,
    BoolAttr:$has_float_operand,
    SI64Attr:$int_operand,
    F64Attr:$float_operand
  );
}

def OneFlow_UpsampleTrilinear_3dGradOp : OneFlow_BaseOp<"upsample_trilinear_3d_grad", []> {
  let input = (ins
    AnyType:$dy,
    AnyType:$x
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    F32Attr:$depth_scale,
    F32Attr:$height_scale,
    F32Attr:$width_scale,
    BoolAttr:$align_corners,
    StrAttr:$data_format
  );
}

def OneFlow_UpsampleNearest_2dGradOp : OneFlow_BaseOp<"upsample_nearest_2d_grad", []> {
  let input = (ins
    AnyType:$dy,
    AnyType:$x
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    F32Attr:$height_scale,
    F32Attr:$width_scale,
    StrAttr:$data_format
  );
}

def OneFlow_UpsampleNearest_1dGradOp : OneFlow_BaseOp<"upsample_nearest_1d_grad", []> {
  let input = (ins
    AnyType:$dy,
    AnyType:$x
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    F32Attr:$scale_factor,
    StrAttr:$data_format
  );
}

def OneFlow_UpsampleTrilinear_3dOp : OneFlow_BaseOp<"upsample_trilinear_3d", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    F32Attr:$depth_scale,
    F32Attr:$height_scale,
    F32Attr:$width_scale,
    BoolAttr:$align_corners,
    StrAttr:$data_format
  );
}

def OneFlow_UpsampleNearest_2dOp : OneFlow_BaseOp<"upsample_nearest_2d", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    F32Attr:$height_scale,
    F32Attr:$width_scale,
    StrAttr:$data_format
  );
}

def OneFlow_UpsampleNearest_1dOp : OneFlow_BaseOp<"upsample_nearest_1d", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    F32Attr:$scale_factor,
    StrAttr:$data_format
  );
}

def OneFlow_UnsortedSegmentSumLikeOp : OneFlow_BaseOp<"unsorted_segment_sum_like", []> {
  let input = (ins
    AnyType:$data,
    AnyType:$segment_ids,
    AnyType:$like
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI64Attr:$axis
  );
}

def OneFlow_UnpackOp : OneFlow_BaseOp<"unpack", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI32Attr:$unpack_num
  );
}

def OneFlow_UnfoldTensorGradOp : OneFlow_BaseOp<"unfold_tensor_grad", []> {
  let input = (ins
    AnyType:$dy,
    AnyType:$x
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    SI32Attr:$dimension,
    SI32Attr:$size,
    SI32Attr:$step
  );
}

def OneFlow_FoldOp : OneFlow_BaseOp<"fold", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    SI32ArrayAttr:$output_size,
    SI32ArrayAttr:$kernel_size,
    SI32ArrayAttr:$strides,
    SI32ArrayAttr:$padding,
    SI32ArrayAttr:$dilation_rate
  );
}

def OneFlow_UnfoldOp : OneFlow_BaseOp<"unfold", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    StrAttr:$data_format,
    SI32ArrayAttr:$kernel_size,
    SI32ArrayAttr:$padding,
    SI32ArrayAttr:$strides,
    SI32ArrayAttr:$dilation_rate
  );
}

def OneFlow_ReduceMaxGlobalStageGradOp : OneFlow_BaseOp<"reduce_max_global_stage_grad", []> {
  let input = (ins
    AnyType:$out_diff,
    AnyType:$mask,
    AnyType:$device_count
  );
  let output = (outs
    AnyType:$in_diff
  );
  let attrs = (ins
    SI32ArrayAttr:$axis,
    BoolAttr:$keepdims
  );
}

def OneFlow_ReduceMinGlobalStageGradOp : OneFlow_BaseOp<"reduce_min_global_stage_grad", []> {
  let input = (ins
    AnyType:$out_diff,
    AnyType:$mask,
    AnyType:$device_count
  );
  let output = (outs
    AnyType:$in_diff
  );
  let attrs = (ins
    SI32ArrayAttr:$axis,
    BoolAttr:$keepdims
  );
}

def OneFlow_ReduceMaxGlobalStageOp : OneFlow_BaseOp<"reduce_max_global_stage", []> {
  let input = (ins
    AnyType:$in,
    AnyType:$device_count
  );
  let output = (outs
    AnyType:$out,
    AnyType:$mask
  );
  let attrs = (ins
    SI32ArrayAttr:$axis,
    BoolAttr:$keepdims
  );
}

def OneFlow_ReduceMaxDeviceStageOp : OneFlow_BaseOp<"reduce_max_device_stage", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out,
    AnyType:$mask,
    AnyType:$count
  );
  let attrs = (ins
    SI32ArrayAttr:$axis
  );
}

def OneFlow_TopKOp : OneFlow_BaseOp<"top_k", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI32Attr:$k,
    BoolAttr:$sorted
  );
}

def OneFlow_TfPreluGradOp : OneFlow_BaseOp<"tf_prelu_grad", []> {
  let input = (ins
    AnyType:$dy,
    AnyType:$x,
    AnyType:$alpha
  );
  let output = (outs
    AnyType:$dx,
    AnyType:$alpha_diff
  );
}

def OneFlow_CpuOnlyReluTestOp : OneFlow_BaseOp<"cpu_only_relu_test", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_TestUserOpAttrAutoTypeOp : OneFlow_BaseOp<"test_user_op_attr_auto_type", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI32Attr:$int1,
    SI32Attr:$int2
  );
}

def OneFlow_TestListDataTypeAndListShapeAndListStringAttrOp : OneFlow_BaseOp<"TestListDataTypeAndListShapeAndListStringAttr", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    ShapeArrayAttr:$out_shapes,
    DTArrayAttr:$out_types,
    StrArrayAttr:$string_list
  );
}

def OneFlow_TestDynamicSourceOp : OneFlow_BaseOp<"TestDynamicSource", []> {
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_TestMultiInputGradOp : OneFlow_BaseOp<"TestMultiInputGrad", []> {
  let input = (ins
    AnyType:$x1,
    AnyType:$x2,
    AnyType:$y_diff
  );
  let output = (outs
    AnyType:$x1_diff,
    AnyType:$x2_diff
  );
}

def OneFlow_TestMultiInputOp : OneFlow_BaseOp<"TestMultiInput", []> {
  let input = (ins
    AnyType:$x1,
    AnyType:$x2
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_TestSourceMultiGpuFixedOutNumOp : OneFlow_BaseOp<"TestSourceMultiGpuFixedOutNum", []> {
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI64Attr:$out_num
  );
}

def OneFlow_CcreluOp : OneFlow_BaseOp<"ccrelu", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_TensorBufferToListOfTensorsOp : OneFlow_BaseOp<"tensor_buffer_to_list_of_tensors", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    AnyI64ElementsAttr:$out_shape,
    StrAttr:$out_dtype,
    BoolAttr:$dynamic_out
  );
}

def OneFlow_TensorBufferToTensorOp : OneFlow_BaseOp<"tensor_buffer_to_tensor", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    AnyI64ElementsAttr:$instance_shape,
    StrAttr:$dtype
  );
}

def OneFlow_SummaryWriteImageOp : OneFlow_BaseOp<"summary_write_image", []> {
  let input = (ins
    AnyType:$in,
    AnyType:$step,
    AnyType:$tag
  );
}

def OneFlow_SspVariableProxyOp : OneFlow_BaseOp<"ssp_variable_proxy", []> {
  let input = (ins
    AnyType:$var
  );
  let output = (outs
    AnyType:$ref,
    AnyType:$value
  );
  let attrs = (ins
    SI64Attr:$buffer_size
  );
}

def OneFlow_SparseSoftmaxCrossEntropyMsOp : OneFlow_BaseOp<"sparse_softmax_cross_entropy_ms", []> {
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

def OneFlow_SparseSoftmaxCrossEntropyOp : OneFlow_BaseOp<"sparse_softmax_cross_entropy", []> {
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

def OneFlow_SparseCrossEntropyMsGradOp : OneFlow_BaseOp<"sparse_cross_entropy_ms_grad", []> {
  let input = (ins
    AnyType:$prediction,
    AnyType:$label,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$prediction_diff
  );
  let attrs = (ins
    SI64Attr:$depth
  );
}

def OneFlow_SparseCrossEntropyMsOp : OneFlow_BaseOp<"sparse_cross_entropy_ms", []> {
  let input = (ins
    AnyType:$prediction,
    AnyType:$label
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI64Attr:$depth
  );
}

def OneFlow_SparseCrossEntropyOp : OneFlow_BaseOp<"sparse_cross_entropy", []> {
  let input = (ins
    AnyType:$prediction,
    AnyType:$label
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI64Attr:$depth
  );
}

def OneFlow_SoftsignGradOp : OneFlow_BaseOp<"softsign_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_SoftsignOp : OneFlow_BaseOp<"softsign", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_SoftmaxGradOp : OneFlow_BaseOp<"softmax_grad", []> {
  let input = (ins
    AnyType:$y,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_UpsampleOp : OneFlow_BaseOp<"upsample", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    F32Attr:$height_scale,
    F32Attr:$width_scale,
    BoolAttr:$align_corners,
    StrAttr:$data_format,
    StrAttr:$interpolation
  );
}

def OneFlow_SoftmaxCrossEntropyGradOp : OneFlow_BaseOp<"softmax_cross_entropy_grad", []> {
  let input = (ins
    AnyType:$dy,
    AnyType:$label,
    AnyType:$prob
  );
  let output = (outs
    AnyType:$prediction_diff
  );
}

def OneFlow_SmoothL1LossGradOp : OneFlow_BaseOp<"smooth_l1_loss_grad", []> {
  let input = (ins
    AnyType:$input,
    AnyType:$target,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    StrAttr:$reduction,
    F32Attr:$beta
  );
}

def OneFlow_SliceGradOp : OneFlow_BaseOp<"slice_grad", []> {
  let input = (ins
    AnyType:$dy,
    AnyType:$like
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    SI64ArrayAttr:$start,
    SI64ArrayAttr:$stop,
    SI64ArrayAttr:$step
  );
}

def OneFlow_SummaryWriteHistogramOp : OneFlow_BaseOp<"summary_write_histogram", []> {
  let input = (ins
    AnyType:$in,
    AnyType:$step,
    AnyType:$tag
  );
}

def OneFlow_SliceOp : OneFlow_BaseOp<"slice", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    SI64ArrayAttr:$start,
    SI64ArrayAttr:$stop,
    SI64ArrayAttr:$step
  );
}

def OneFlow_SiluGradOp : OneFlow_BaseOp<"silu_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_SigmoidGradOp : OneFlow_BaseOp<"sigmoid_grad", []> {
  let input = (ins
    AnyType:$y,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_TestRandomSourceOp : OneFlow_BaseOp<"TestRandomSource", []> {
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI64Attr:$seed
  );
}

def OneFlow_SigmoidOp : OneFlow_BaseOp<"sigmoid", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_SigmoidCrossEntropyOp : OneFlow_BaseOp<"sigmoid_cross_entropy", []> {
  let input = (ins
    AnyType:$prediction,
    AnyType:$label
  );
  let output = (outs
    AnyType:$loss
  );
}

def OneFlow_WhereOp : OneFlow_BaseOp<"where", []> {
  let input = (ins
    AnyType:$condition,
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_SeluGradOp : OneFlow_BaseOp<"selu_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_SeluOp : OneFlow_BaseOp<"selu", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_SiluOp : OneFlow_BaseOp<"silu", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_ScalarPowGradOp : OneFlow_BaseOp<"scalar_pow_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    BoolAttr:$has_int_operand,
    BoolAttr:$has_float_operand,
    SI64Attr:$int_operand,
    F64Attr:$float_operand
  );
}

def OneFlow_ReduceMinDeviceStageGradOp : OneFlow_BaseOp<"reduce_min_device_stage_grad", []> {
  let input = (ins
    AnyType:$out_diff,
    AnyType:$mask,
    AnyType:$count
  );
  let output = (outs
    AnyType:$in_diff
  );
  let attrs = (ins
    SI32ArrayAttr:$axis
  );
}

def OneFlow_ScalarMulOp : OneFlow_BaseOp<"scalar_mul", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    BoolAttr:$has_int_operand,
    BoolAttr:$has_float_operand,
    SI64Attr:$int_operand,
    F64Attr:$float_operand
  );
}

def OneFlow_TestReshapeOp : OneFlow_BaseOp<"TestReshape", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    AnyI64ElementsAttr:$shape
  );
}

def OneFlow_ScalarLogicalLessEqualOp : OneFlow_BaseOp<"scalar_logical_less_equal", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    BoolAttr:$has_int_operand,
    BoolAttr:$has_float_operand,
    SI64Attr:$int_operand,
    F64Attr:$float_operand
  );
}

def OneFlow_ReshapeOp : OneFlow_BaseOp<"reshape", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    AnyI64ElementsAttr:$shape
  );
}

def OneFlow_RepeatOp : OneFlow_BaseOp<"repeat", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI32Attr:$repeat_num
  );
}

def OneFlow_TfPreluOp : OneFlow_BaseOp<"tf_prelu", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$alpha
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_ReluOp : OneFlow_IdempotentBaseOp<"relu", []> {}

def OneFlow_ReduceMaxOp : OneFlow_BaseOp<"reduce_max", []> {
  let input = (ins
    AnyType:$input_tensor
  );
  let output = (outs
    AnyType:$output_tensor
  );
  let attrs = (ins
    SI32ArrayAttr:$axis,
    BoolAttr:$keepdims
  );
}

def OneFlow_UpsampleLinear_1dGradOp : OneFlow_BaseOp<"upsample_linear_1d_grad", []> {
  let input = (ins
    AnyType:$dy,
    AnyType:$x
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    F32Attr:$scale_factor,
    BoolAttr:$align_corners,
    StrAttr:$data_format
  );
}

def OneFlow_ReduceSumOp : OneFlow_BaseOp<"reduce_sum", []> {
  let input = (ins
    AnyType:$input_tensor
  );
  let output = (outs
    AnyType:$output_tensor
  );
  let attrs = (ins
    SI32ArrayAttr:$axis,
    BoolAttr:$keepdims
  );
}

def OneFlow_ReduceProdOp : OneFlow_BaseOp<"reduce_prod", []> {
  let input = (ins
    AnyType:$input_tensor
  );
  let output = (outs
    AnyType:$output_tensor
  );
  let attrs = (ins
    SI32ArrayAttr:$axis,
    BoolAttr:$keepdims
  );
}

def OneFlow_ReduceAllOp : OneFlow_BaseOp<"reduce_all", []> {
  let input = (ins
    AnyType:$input_tensor
  );
  let output = (outs
    AnyType:$output_tensor
  );
  let attrs = (ins
    SI32ArrayAttr:$axis,
    BoolAttr:$keepdims
  );
}

def OneFlow_QuantizationOp : OneFlow_BaseOp<"quantization", []> {
  let input = (ins
    AnyType:$in,
    AnyType:$scale,
    AnyType:$zero_point
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    StrAttr:$quantization_formula,
    SI32Attr:$quantization_bit,
    StrAttr:$quantization_scheme
  );
}

def OneFlow_Maxpool_3dGradOp : OneFlow_BaseOp<"maxpool_3d_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$y,
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
    BoolAttr:$return_indices,
    BoolAttr:$ceil_mode
  );
}

def OneFlow_Maxpool_2dGradOp : OneFlow_BaseOp<"maxpool_2d_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$y,
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
    BoolAttr:$return_indices,
    BoolAttr:$ceil_mode
  );
}

def OneFlow_Maxpool_2dOp : OneFlow_BaseOp<"maxpool_2d", []> {
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
    BoolAttr:$return_indices,
    BoolAttr:$ceil_mode
  );
}

def OneFlow_Maxpool_1dOp : OneFlow_BaseOp<"maxpool_1d", []> {
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
    BoolAttr:$return_indices,
    BoolAttr:$ceil_mode
  );
}

def OneFlow_MaxPool_3dGradOp : OneFlow_BaseOp<"max_pool_3d_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$y,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
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

def OneFlow_UnsortedSegmentSumOp : OneFlow_BaseOp<"unsorted_segment_sum", []> {
  let input = (ins
    AnyType:$data,
    AnyType:$segment_ids
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI64Attr:$axis,
    SI64Attr:$num_segments
  );
}

def OneFlow_MaxPool_3dOp : OneFlow_BaseOp<"max_pool_3d", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
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

def OneFlow_MaxPool_2dGradOp : OneFlow_BaseOp<"max_pool_2d_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$y,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
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

def OneFlow_MaxPool_1dOp : OneFlow_BaseOp<"max_pool_1d", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
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

def OneFlow_AvgPool_3dGradOp : OneFlow_BaseOp<"avg_pool_3d_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$y,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
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

def OneFlow_AvgPool_3dOp : OneFlow_BaseOp<"avg_pool_3d", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
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

def OneFlow_AvgPool_2dGradOp : OneFlow_BaseOp<"avg_pool_2d_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$y,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
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

def OneFlow_AvgPool_1dOp : OneFlow_BaseOp<"avg_pool_1d", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
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

def OneFlow_DistributedPartialFcSampleDisableBoxingOp : OneFlow_BaseOp<"distributed_partial_fc_sample_disable_boxing", []> {
  let input = (ins
    AnyType:$sampled_weight_diff,
    AnyType:$sampled_label
  );
  let output = (outs
    AnyType:$boxing_disabled_sampled_weight_diff,
    AnyType:$boxing_disabled_sampled_label
  );
}

def OneFlow_PreluGradOp : OneFlow_BaseOp<"prelu_grad", []> {
  let input = (ins
    AnyType:$dy,
    AnyType:$x,
    AnyType:$alpha
  );
  let output = (outs
    AnyType:$dx,
    AnyType:$alpha_diff
  );
}

def OneFlow_DistributedPartialFcSampleOp : OneFlow_BaseOp<"distributed_partial_fc_sample", []> {
  let input = (ins
    AnyType:$weight,
    AnyType:$label
  );
  let output = (outs
    AnyType:$mapped_label,
    AnyType:$sampled_label,
    AnyType:$sampled_weight
  );
  let attrs = (ins
    SI64Attr:$num_sample,
    SI64Attr:$seed
  );
}

def OneFlow_CcreluGradOp : OneFlow_BaseOp<"ccrelu_grad", []> {
  let input = (ins
    AnyType:$y,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_ParallelCastOp : OneFlow_BaseOp<"parallel_cast", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    StrAttr:$sbp_parallel,
    StrAttr:$grad_sbp_parallel
  );
}

def OneFlow_ConstantPad3dGradOp : OneFlow_BaseOp<"constant_pad3d_grad", []> {
  let input = (ins
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    SI64ArrayAttr:$padding,
    F64Attr:$floating_value,
    SI64Attr:$integral_value
  );
}

def OneFlow_ConstantPad2dGradOp : OneFlow_BaseOp<"constant_pad2d_grad", []> {
  let input = (ins
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    SI64ArrayAttr:$padding,
    F64Attr:$floating_value,
    SI64Attr:$integral_value
  );
}

def OneFlow_ConstantPad2dOp : OneFlow_BaseOp<"constant_pad2d", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    SI64ArrayAttr:$padding,
    F64Attr:$floating_value,
    SI64Attr:$integral_value
  );
}

def OneFlow_TrilOp : OneFlow_BaseOp<"tril", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI64Attr:$diagonal,
    F64Attr:$floating_fill_value,
    SI64Attr:$integer_fill_value,
    BoolAttr:$is_floating_fill_value
  );
}

def OneFlow_ConstantPad1dGradOp : OneFlow_BaseOp<"constant_pad1d_grad", []> {
  let input = (ins
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    SI64ArrayAttr:$padding,
    F64Attr:$floating_value,
    SI64Attr:$integral_value
  );
}

def OneFlow_ReplicationPad2dOp : OneFlow_BaseOp<"replication_pad2d", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    SI64ArrayAttr:$padding
  );
}

def OneFlow_ReluGradOp : OneFlow_IdempotentBaseOp<"relu_grad", []> {}

def OneFlow_ReflectionPad2dGradOp : OneFlow_BaseOp<"reflection_pad2d_grad", []> {
  let input = (ins
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    SI64ArrayAttr:$padding
  );
}

def OneFlow_ReduceMinOp : OneFlow_BaseOp<"reduce_min", []> {
  let input = (ins
    AnyType:$input_tensor
  );
  let output = (outs
    AnyType:$output_tensor
  );
  let attrs = (ins
    SI32ArrayAttr:$axis,
    BoolAttr:$keepdims
  );
}

def OneFlow_ReflectionPad2dOp : OneFlow_BaseOp<"reflection_pad2d", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    SI64ArrayAttr:$padding
  );
}

def OneFlow_PadGradOp : OneFlow_BaseOp<"pad_grad", []> {
  let input = (ins
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    SI64ArrayAttr:$padding_before,
    SI64ArrayAttr:$padding_after,
    F64Attr:$floating_constant_value,
    SI64Attr:$integral_constant_value
  );
}

def OneFlow_PadOp : OneFlow_BaseOp<"pad", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    SI64ArrayAttr:$padding_before,
    SI64ArrayAttr:$padding_after,
    F64Attr:$floating_constant_value,
    SI64Attr:$integral_constant_value
  );
}

def OneFlow_RecvOp : OneFlow_BaseOp<"recv", []> {
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI64Attr:$src_process_id,
    StrAttr:$dtype,
    AnyI64ElementsAttr:$shape,
    StrAttr:$device_type,
    SI64Attr:$device_id
  );
}

def OneFlow_SendOp : OneFlow_BaseOp<"send", []> {
  let input = (ins
    AnyType:$in
  );
  let attrs = (ins
    SI64Attr:$dst_process_id
  );
}

def OneFlow_OfrecordImageClassificationReaderOp : OneFlow_BaseOp<"ofrecord_image_classification_reader", []> {
  let output = (outs
    AnyType:$image,
    AnyType:$label
  );
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
    StrAttr:$color_space,
    StrAttr:$image_feature_name,
    StrAttr:$label_feature_name,
    SI32Attr:$decode_buffer_size_per_thread,
    SI32Attr:$num_decode_threads_per_machine
  );
}

def OneFlow_TestDataTypeAttrOp : OneFlow_BaseOp<"TestDataTypeAttr", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    StrAttr:$output_type
  );
}

def OneFlow_SoftmaxOp : OneFlow_BaseOp<"softmax", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_ScalarLogicalNotEqualOp : OneFlow_BaseOp<"scalar_logical_not_equal", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    BoolAttr:$has_int_operand,
    BoolAttr:$has_float_operand,
    SI64Attr:$int_operand,
    F64Attr:$float_operand
  );
}

def OneFlow_OfrecordImageDecoderRandomCropOp : OneFlow_BaseOp<"ofrecord_image_decoder_random_crop", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    StrAttr:$name,
    StrAttr:$color_space,
    SI32Attr:$num_attempts,
    SI64Attr:$seed,
    BoolAttr:$has_seed,
    F32ArrayAttr:$random_area,
    F32ArrayAttr:$random_aspect_ratio
  );
}

def OneFlow_OfrecordImageDecoderOp : OneFlow_BaseOp<"ofrecord_image_decoder", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    StrAttr:$name,
    StrAttr:$color_space
  );
}

def OneFlow_OfrecordRawDecoderOp : OneFlow_BaseOp<"ofrecord_raw_decoder", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    StrAttr:$name,
    AnyI64ElementsAttr:$shape,
    StrAttr:$data_type,
    BoolAttr:$dim1_varying_length,
    BoolAttr:$truncate
  );
}

def OneFlow_NvtxStartOp : OneFlow_BaseOp<"nvtx_start", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    StrAttr:$mark_prefix
  );
}

def OneFlow_NormalizationAddReluGradOp : OneFlow_BaseOp<"normalization_add_relu_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy,
    AnyType:$mean,
    AnyType:$inv_variance,
    AnyType:$gamma,
    AnyType:$beta,
    AnyType:$reserve_space,
    AnyType:$y
  );
  let output = (outs
    AnyType:$gamma_diff,
    AnyType:$beta_diff,
    AnyType:$dx,
    AnyType:$addend_diff
  );
  let attrs = (ins
    SI32Attr:$axis,
    F32Attr:$epsilon
  );
}

def OneFlow_SparseSoftmaxCrossEntropyGradOp : OneFlow_BaseOp<"sparse_softmax_cross_entropy_grad", []> {
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

def OneFlow_NormalizationGradOp : OneFlow_BaseOp<"normalization_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy,
    AnyType:$mean,
    AnyType:$inv_variance,
    AnyType:$gamma
  );
  let output = (outs
    AnyType:$gamma_diff,
    AnyType:$beta_diff,
    AnyType:$dx
  );
  let attrs = (ins
    SI32Attr:$axis,
    F32Attr:$epsilon
  );
}

def OneFlow_OnesLikeOp : OneFlow_IdempotentBaseOp<"ones_like", []> {}

def OneFlow_CudnnFusedNormalizationAddReluOp : OneFlow_BaseOp<"cudnn_fused_normalization_add_relu", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$addend,
    AnyType:$moving_mean,
    AnyType:$moving_variance,
    AnyType:$gamma,
    AnyType:$beta
  );
  let output = (outs
    AnyType:$y,
    AnyType:$reserve_space,
    AnyType:$mean,
    AnyType:$inv_variance
  );
  let attrs = (ins
    SI32Attr:$axis,
    F32Attr:$epsilon,
    F32Attr:$momentum
  );
}

def OneFlow_UpsampleLinear_1dOp : OneFlow_BaseOp<"upsample_linear_1d", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    F32Attr:$scale_factor,
    BoolAttr:$align_corners,
    StrAttr:$data_format
  );
}

def OneFlow_NormalizationAddReluOp : OneFlow_BaseOp<"normalization_add_relu", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$addend,
    AnyType:$moving_mean,
    AnyType:$moving_variance,
    AnyType:$gamma,
    AnyType:$beta
  );
  let output = (outs
    AnyType:$y,
    AnyType:$reserve_space,
    AnyType:$mean,
    AnyType:$inv_variance
  );
  let attrs = (ins
    SI32Attr:$axis,
    F32Attr:$epsilon,
    BoolAttr:$training,
    F32Attr:$momentum
  );
}

def OneFlow_ConstantPad3dOp : OneFlow_BaseOp<"constant_pad3d", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    SI64ArrayAttr:$padding,
    F64Attr:$floating_value,
    SI64Attr:$integral_value
  );
}

def OneFlow_NmsOp : OneFlow_BaseOp<"nms", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    F32Attr:$iou_threshold,
    SI32Attr:$keep_n
  );
}

def OneFlow_NllOp : OneFlow_BaseOp<"nll", []> {
  let input = (ins
    AnyType:$input,
    AnyType:$target,
    AnyType:$weight
  );
  let output = (outs
    AnyType:$out,
    AnyType:$total_weight
  );
  let attrs = (ins
    SI64Attr:$ignore_index,
    StrAttr:$reduction
  );
}

def OneFlow_TensorScatterNdUpdateOp : OneFlow_BaseOp<"tensor_scatter_nd_update", []> {
  let input = (ins
    AnyType:$params,
    AnyType:$updates,
    AnyType:$indices
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_GatherNdOp : OneFlow_BaseOp<"gather_nd", []> {
  let input = (ins
    AnyType:$params,
    AnyType:$indices
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow__ncclLogicalS2sOp : OneFlow_BaseOp<"_nccl_logical_s2s", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI64Attr:$in_split_axis,
    SI64Attr:$out_split_axis
  );
}

def OneFlow__ncclLogicalAllGatherOp : OneFlow_BaseOp<"_nccl_logical_all_gather", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_ScalarFmodOp : OneFlow_BaseOp<"scalar_fmod", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    BoolAttr:$has_int_operand,
    BoolAttr:$has_float_operand,
    SI64Attr:$int_operand,
    F64Attr:$float_operand
  );
}

def OneFlow__ncclLogicalReduceScatterOp : OneFlow_BaseOp<"_nccl_logical_reduce_scatter", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow__ncclLogical_2DSameDim0All2allOp : OneFlow_BaseOp<"_nccl_logical_2D_same_dim0_all2all", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI64Attr:$in_dim1_split_axis,
    SI64Attr:$out_dim1_split_axis
  );
}

def OneFlow__ncclLogical_2DSameDim0AllGatherNoncontinuousOp : OneFlow_BaseOp<"_nccl_logical_2D_same_dim0_all_gather_noncontinuous", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI64Attr:$in_dim1_split_axis
  );
}

def OneFlow__ncclLogical_2DSameDim0AllGatherOp : OneFlow_BaseOp<"_nccl_logical_2D_same_dim0_all_gather", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow__ncclLogical_2DSameDim0AllReduceOp : OneFlow_BaseOp<"_nccl_logical_2D_same_dim0_all_reduce", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_GenTensorBufferOp : OneFlow_BaseOp<"gen_tensor_buffer", []> {
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    AnyI64ElementsAttr:$shape,
    ShapeArrayAttr:$shape_list,
    F32ArrayAttr:$value_list,
    StrAttr:$data_type,
    BoolAttr:$dynamic_out
  );
}

def OneFlow_NarrowOp : OneFlow_BaseOp<"narrow", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI64Attr:$dim,
    SI64Attr:$start,
    SI64Attr:$length
  );
}

def OneFlow_MultiplyOp : OneFlow_BaseOp<"multiply", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_UpsampleBicubic_2dGradOp : OneFlow_BaseOp<"upsample_bicubic_2d_grad", []> {
  let input = (ins
    AnyType:$dy,
    AnyType:$x
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    F32Attr:$height_scale,
    F32Attr:$width_scale,
    BoolAttr:$align_corners,
    StrAttr:$data_format
  );
}

def OneFlow_MovingAverageMinMaxObserverOp : OneFlow_BaseOp<"moving_average_min_max_observer", []> {
  let input = (ins
    AnyType:$in,
    AnyType:$current_train_step,
    AnyType:$moving_max,
    AnyType:$moving_min
  );
  let output = (outs
    AnyType:$scale,
    AnyType:$zero_point
  );
  let attrs = (ins
    BoolAttr:$training,
    StrAttr:$quantization_formula,
    SI64Attr:$stop_update_after_iters,
    SI32Attr:$quantization_bit,
    StrAttr:$quantization_scheme,
    F32Attr:$momentum
  );
}

def OneFlow_RmspropUpdateOp : OneFlow_BaseOp<"rmsprop_update", []> {
  let input = (ins
    AnyType:$model,
    AnyType:$model_diff,
    AnyType:$learning_rate,
    AnyType:$scale_by_tensor,
    AnyType:$skip_if,
    AnyType:$mean_square,
    AnyType:$mean_gradient
  );
  let attrs = (ins
    F32Attr:$learning_rate_val,
    F64Attr:$scale,
    F32Attr:$l1,
    F32Attr:$l2,
    BoolAttr:$centered,
    F32Attr:$epsilon,
    F32Attr:$decay_rate,
    F32Attr:$weight_decay
  );
}

def OneFlow_AdamBiasCorrectionFactorOp : OneFlow_BaseOp<"adam_bias_correction_factor", []> {
  let input = (ins
    AnyType:$train_step
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    F32Attr:$beta
  );
}

def OneFlow_SparseCrossEntropyGradOp : OneFlow_BaseOp<"sparse_cross_entropy_grad", []> {
  let input = (ins
    AnyType:$prediction,
    AnyType:$label,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$prediction_diff
  );
  let attrs = (ins
    SI64Attr:$depth
  );
}

def OneFlow_AdamUpdateOp : OneFlow_BaseOp<"adam_update", []> {
  let input = (ins
    AnyType:$model,
    AnyType:$model_diff,
    AnyType:$learning_rate,
    AnyType:$scale_by_tensor,
    AnyType:$skip_if,
    AnyType:$bias_correction1,
    AnyType:$bias_correction2,
    AnyType:$m,
    AnyType:$v,
    AnyType:$max_v
  );
  let attrs = (ins
    F32Attr:$learning_rate_val,
    F32Attr:$bias_correction1_val,
    F32Attr:$bias_correction2_val,
    F64Attr:$scale,
    F32Attr:$l1,
    F32Attr:$l2,
    F32Attr:$beta1,
    F32Attr:$beta2,
    F32Attr:$epsilon,
    F32Attr:$weight_decay,
    BoolAttr:$amsgrad,
    BoolAttr:$do_bias_correction
  );
}

def OneFlow_IndexedSlicesSgdUpdateOp : OneFlow_BaseOp<"indexed_slices_sgd_update", []> {
  let input = (ins
    AnyType:$model,
    AnyType:$model_diff_indices,
    AnyType:$model_diff_values,
    AnyType:$learning_rate
  );
  let attrs = (ins
    F32Attr:$weight_decay
  );
}

def OneFlow_SgdUpdateOp : OneFlow_BaseOp<"sgd_update", []> {
  let input = (ins
    AnyType:$model,
    AnyType:$model_diff,
    AnyType:$learning_rate,
    AnyType:$scale_by_tensor,
    AnyType:$skip_if
  );
  let attrs = (ins
    F32Attr:$learning_rate_val,
    F64Attr:$scale,
    F32Attr:$l1,
    F32Attr:$l2,
    F32Attr:$weight_decay
  );
}

def OneFlow_MishGradOp : OneFlow_BaseOp<"mish_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_UpsampleNearest_3dOp : OneFlow_BaseOp<"upsample_nearest_3d", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    F32Attr:$depth_scale,
    F32Attr:$height_scale,
    F32Attr:$width_scale,
    StrAttr:$data_format
  );
}

def OneFlow_ScalarMulByTensorOp : OneFlow_BaseOp<"scalar_mul_by_tensor", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$scalar
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_MinMaxObserverOp : OneFlow_BaseOp<"min_max_observer", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$scale,
    AnyType:$zero_point
  );
  let attrs = (ins
    StrAttr:$quantization_formula,
    SI32Attr:$quantization_bit,
    StrAttr:$quantization_scheme,
    BoolAttr:$per_layer_quantization
  );
}

def OneFlow_ScalarLogicalXorOp : OneFlow_BaseOp<"scalar_logical_xor", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    BoolAttr:$has_int_operand,
    BoolAttr:$has_float_operand,
    SI64Attr:$int_operand,
    F64Attr:$float_operand
  );
}

def OneFlow_BroadcastMatmulGradBOp : OneFlow_BaseOp<"broadcast_matmul_grad_b", []> {
  let input = (ins
    AnyType:$a,
    AnyType:$b,
    AnyType:$_add_to_output
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    F64Attr:$alpha
  );
}

def OneFlow_BroadcastMatmulOp : OneFlow_BaseOp<"broadcast_matmul", []> {
  let input = (ins
    AnyType:$a,
    AnyType:$b,
    AnyType:$_add_to_output
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    BoolAttr:$transpose_a,
    BoolAttr:$transpose_b,
    F64Attr:$alpha
  );
}

def OneFlow_MatmulOp : OneFlow_BaseOp<"matmul", []> {
  let input = (ins
    AnyType:$a,
    AnyType:$b,
    AnyType:$_add_to_output
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    BoolAttr:$transpose_a,
    BoolAttr:$transpose_b,
    F64Attr:$alpha
  );
}

def OneFlow__ncclLogicalAllGatherNoncontinuousOp : OneFlow_BaseOp<"_nccl_logical_all_gather_noncontinuous", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI64Attr:$in_split_axis
  );
}

def OneFlow_TanhOp : OneFlow_BaseOp<"tanh", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_TanGradOp : OneFlow_BaseOp<"tan_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_TanOp : OneFlow_BaseOp<"tan", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_SquareGradOp : OneFlow_BaseOp<"square_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_SoftplusGradOp : OneFlow_BaseOp<"softplus_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_SinhGradOp : OneFlow_BaseOp<"sinh_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_SinhOp : OneFlow_BaseOp<"sinh", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_SortOp : OneFlow_BaseOp<"sort", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    StrAttr:$direction
  );
}

def OneFlow_SinGradOp : OneFlow_BaseOp<"sin_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_GenerateRandomBatchPermutationIndicesOp : OneFlow_BaseOp<"generate_random_batch_permutation_indices", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    SI64Attr:$seed
  );
}

def OneFlow_PowYGradOp : OneFlow_BaseOp<"pow_y_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$y,
    AnyType:$dz
  );
  let output = (outs
    AnyType:$dy
  );
}

def OneFlow_GeluOp : OneFlow_BaseOp<"gelu", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_HierarchicalParallelCastOp : OneFlow_BaseOp<"hierarchical_parallel_cast", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    StrArrayAttr:$nd_sbp,
    StrAttr:$grad_mode,
    StrArrayAttr:$grad_nd_sbp
  );
}

def OneFlow_GatherOp : OneFlow_BaseOp<"gather", []> {
  let input = (ins
    AnyType:$in,
    AnyType:$indices
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI64Attr:$axis
  );
}

def OneFlow_LarsUpdateOp : OneFlow_BaseOp<"lars_update", []> {
  let input = (ins
    AnyType:$model,
    AnyType:$model_diff,
    AnyType:$learning_rate,
    AnyType:$momentum,
    AnyType:$scale_by_tensor,
    AnyType:$skip_if
  );
  let attrs = (ins
    F64Attr:$scale,
    F32Attr:$l1,
    F32Attr:$l2,
    F32Attr:$momentum_beta,
    F32Attr:$epsilon,
    F32Attr:$lars_coefficient,
    F32Attr:$weight_decay
  );
}

def OneFlow_ImageResizeKeepAspectRatioOp : OneFlow_BaseOp<"image_resize_keep_aspect_ratio", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out,
    AnyType:$size,
    AnyType:$scale
  );
  let attrs = (ins
    SI32Attr:$target_size,
    SI32Attr:$min_size,
    SI32Attr:$max_size,
    BoolAttr:$resize_longer,
    StrAttr:$interpolation_type
  );
}

def OneFlow_FusedSelfAttentionQueryMulKeyAndValueGradOp : OneFlow_BaseOp<"fused_self_attention_query_mul_key_and_value_grad", []> {
  let input = (ins
    AnyType:$query_mul_key_grad,
    AnyType:$value_grad,
    AnyType:$hidden_states
  );
  let output = (outs
    AnyType:$hidden_states_grad
  );
  let attrs = (ins
    F32Attr:$alpha
  );
}

def OneFlow_L1L2RegularizeGradientOp : OneFlow_BaseOp<"l1_l2_regularize_gradient", []> {
  let input = (ins
    AnyType:$model,
    AnyType:$model_diff
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    F32Attr:$l1,
    F32Attr:$l2
  );
}

def OneFlow_FusedTrilScaleSoftmaxMaskScaleOp : OneFlow_BaseOp<"fused_tril_scale_softmax_mask_scale", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$mask
  );
  let output = (outs
    AnyType:$y,
    AnyType:$softmax_y
  );
  let attrs = (ins
    SI64Attr:$diagonal,
    F32Attr:$tril_fill_value,
    F32Attr:$tril_scale_value,
    F32Attr:$mask_scale_value
  );
}

def OneFlow__ncclLogicalAllReduceOp : OneFlow_BaseOp<"_nccl_logical_all_reduce", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_FusedBiasAddGeluGradOp : OneFlow_BaseOp<"fused_bias_add_gelu_grad", []> {
  let input = (ins
    AnyType:$a,
    AnyType:$b,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    SI32Attr:$axis
  );
}

def OneFlow_SqrtOp : OneFlow_BaseOp<"sqrt", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_HardswishGradOp : OneFlow_BaseOp<"hardswish_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_EagerNcclReduceScatterOp : OneFlow_BaseOp<"eager_nccl_reduce_scatter", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    StrAttr:$parallel_conf,
    StrAttr:$op_type
  );
}

def OneFlow_CropMirrorNormalizeFromTensorbufferOp : OneFlow_BaseOp<"crop_mirror_normalize_from_tensorbuffer", []> {
  let input = (ins
    AnyType:$in,
    AnyType:$mirror
  );
  let output = (outs
    AnyType:$out
  );
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

def OneFlow_ReplicationPad2dGradOp : OneFlow_BaseOp<"replication_pad2d_grad", []> {
  let input = (ins
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    SI64ArrayAttr:$padding
  );
}

def OneFlow_LambUpdateOp : OneFlow_BaseOp<"lamb_update", []> {
  let input = (ins
    AnyType:$m,
    AnyType:$v,
    AnyType:$beta1_t,
    AnyType:$beta2_t,
    AnyType:$model,
    AnyType:$model_diff,
    AnyType:$learning_rate,
    AnyType:$scale_by_tensor,
    AnyType:$skip_if
  );
  let attrs = (ins
    F32Attr:$beta1,
    F32Attr:$beta2,
    F32Attr:$epsilon,
    F64Attr:$scale,
    F32Attr:$l1,
    F32Attr:$l2,
    F32Attr:$weight_decay
  );
}

def OneFlow_TanhGradOp : OneFlow_BaseOp<"tanh_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_CoshGradOp : OneFlow_BaseOp<"cosh_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_FlipGradOp : OneFlow_BaseOp<"flip_grad", []> {
  let input = (ins
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    SI32ArrayAttr:$dims
  );
}

def OneFlow_DiagOp : OneFlow_BaseOp<"diag", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI32Attr:$diagonal
  );
}

def OneFlow_ReduceAnyOp : OneFlow_BaseOp<"reduce_any", []> {
  let input = (ins
    AnyType:$input_tensor
  );
  let output = (outs
    AnyType:$output_tensor
  );
  let attrs = (ins
    SI32ArrayAttr:$axis,
    BoolAttr:$keepdims
  );
}

def OneFlow_BroadcastDivGradOp : OneFlow_BaseOp<"broadcast_div_grad", []> {
  let input = (ins
    AnyType:$y,
    AnyType:$z,
    AnyType:$dz
  );
  let output = (outs
    AnyType:$dy
  );
}

def OneFlow_NllGradOp : OneFlow_BaseOp<"nll_grad", []> {
  let input = (ins
    AnyType:$input,
    AnyType:$target,
    AnyType:$total_weight,
    AnyType:$weight,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    SI64Attr:$ignore_index,
    StrAttr:$reduction
  );
}

def OneFlow_NegativeOp : OneFlow_InvolutionBaseOp<"negative", []> {}

def OneFlow_ScatterNdOp : OneFlow_BaseOp<"scatter_nd", []> {
  let input = (ins
    AnyType:$indices,
    AnyType:$updates
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    AnyI64ElementsAttr:$shape
  );
}

def OneFlow_FakeQuantizationOp : OneFlow_BaseOp<"fake_quantization", []> {
  let input = (ins
    AnyType:$in,
    AnyType:$scale,
    AnyType:$zero_point
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    StrAttr:$quantization_formula,
    SI32Attr:$quantization_bit,
    StrAttr:$quantization_scheme
  );
}

def OneFlow_EyeOp : OneFlow_BaseOp<"eye", []> {
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI64Attr:$rows,
    SI64Attr:$cols,
    StrAttr:$dtype,
    StrArrayAttr:$nd_sbp
  );
}

def OneFlow_ReduceMinGlobalStageOp : OneFlow_BaseOp<"reduce_min_global_stage", []> {
  let input = (ins
    AnyType:$in,
    AnyType:$device_count
  );
  let output = (outs
    AnyType:$out,
    AnyType:$mask
  );
  let attrs = (ins
    SI32ArrayAttr:$axis,
    BoolAttr:$keepdims
  );
}

def OneFlow_ExpandGradOp : OneFlow_BaseOp<"expand_grad", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI32ArrayAttr:$logical_out_shape,
    SI32ArrayAttr:$logical_expand_shape
  );
}

def OneFlow_AdagradUpdateOp : OneFlow_BaseOp<"adagrad_update", []> {
  let input = (ins
    AnyType:$model,
    AnyType:$model_diff,
    AnyType:$learning_rate,
    AnyType:$scale_by_tensor,
    AnyType:$skip_if,
    AnyType:$train_step,
    AnyType:$sum
  );
  let attrs = (ins
    SI32Attr:$train_step_val,
    F32Attr:$learning_rate_val,
    F64Attr:$scale,
    F32Attr:$l1,
    F32Attr:$l2,
    F32Attr:$lr_decay,
    F32Attr:$weight_decay,
    F32Attr:$epsilon
  );
}

def OneFlow_Avgpool_3dGradOp : OneFlow_BaseOp<"avgpool_3d_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$y,
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
    BoolAttr:$ceil_mode,
    BoolAttr:$count_include_pad,
    SI64Attr:$divisor_override
  );
}

def OneFlow_XlogyOp : OneFlow_BaseOp<"xlogy", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
}

def OneFlow_LogicalSliceAssignOp : OneFlow_BaseOp<"logical_slice_assign", []> {
  let input = (ins
    AnyType:$ref,
    AnyType:$value
  );
  let attrs = (ins
    SI64ArrayAttr:$start,
    SI64ArrayAttr:$stop,
    SI64ArrayAttr:$step
  );
}

def OneFlow_SamePaddingOp : OneFlow_BaseOp<"same_padding", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    StrAttr:$padding,
    StrAttr:$data_format,
    SI32ArrayAttr:$kernel_size,
    SI32ArrayAttr:$strides,
    SI32ArrayAttr:$dilation_rate
  );
}

def OneFlow_RandpermOp : OneFlow_BaseOp<"randperm", []> {
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI32Attr:$n,
    StrAttr:$nd_sbp
  );
}

def OneFlow_ExpandDimsOp : OneFlow_BaseOp<"expand_dims", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI32Attr:$axis
  );
}

def OneFlow_EmptyOp : OneFlow_BaseOp<"empty", []> {
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    StrAttr:$dtype,
    AnyI64ElementsAttr:$shape,
    StrArrayAttr:$nd_sbp
  );
}

def OneFlow_EluGradOp : OneFlow_BaseOp<"elu_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    F64Attr:$alpha
  );
}

def OneFlow_CosGradOp : OneFlow_BaseOp<"cos_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_CastToStaticShapeOp : OneFlow_BaseOp<"cast_to_static_shape", []> {
  let input = (ins
    AnyType:$input
  );
  let output = (outs
    AnyType:$output
  );
}

def OneFlow_MomentumUpdateOp : OneFlow_BaseOp<"momentum_update", []> {
  let input = (ins
    AnyType:$model,
    AnyType:$model_diff,
    AnyType:$momentum,
    AnyType:$learning_rate,
    AnyType:$scale_by_tensor,
    AnyType:$skip_if
  );
  let attrs = (ins
    F32Attr:$learning_rate_val,
    F64Attr:$scale,
    F32Attr:$l1,
    F32Attr:$l2,
    F32Attr:$beta,
    F32Attr:$weight_decay
  );
}

def OneFlow_AmpWhiteIdentityOp : OneFlow_BaseOp<"amp_white_identity", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_SoftplusOp : OneFlow_BaseOp<"softplus", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_EluOp : OneFlow_BaseOp<"elu", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    F64Attr:$alpha
  );
}

def OneFlow_ElementwiseMinimumOp : OneFlow_BaseOp<"elementwise_minimum", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
}

def OneFlow_UniqueWithCountsOp : OneFlow_BaseOp<"unique_with_counts", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y,
    AnyType:$idx,
    AnyType:$count,
    AnyType:$num_unique
  );
  let attrs = (ins
    StrAttr:$out_idx
  );
}

def OneFlow_ElementwiseMinimumBackwardOp : OneFlow_BaseOp<"elementwise_minimum_backward", []> {
  let input = (ins
    AnyType:$dz,
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$dx,
    AnyType:$dy
  );
}

def OneFlow_GridSampleOp : OneFlow_BaseOp<"grid_sample", []> {
  let input = (ins
    AnyType:$input,
    AnyType:$grid
  );
  let output = (outs
    AnyType:$output
  );
  let attrs = (ins
    StrAttr:$interpolation_mode,
    StrAttr:$padding_mode,
    BoolAttr:$align_corners
  );
}

def OneFlow_ElementwiseMaximumOp : OneFlow_BaseOp<"elementwise_maximum", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
}

def OneFlow_EagerSToBOp : OneFlow_BaseOp<"eager_s_to_b", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI64Attr:$in_split_axis,
    StrAttr:$in_parallel_conf,
    StrAttr:$out_parallel_conf,
    AnyI64ElementsAttr:$shape
  );
}

def OneFlow_ElementwiseMaximumBackwardOp : OneFlow_BaseOp<"elementwise_maximum_backward", []> {
  let input = (ins
    AnyType:$dz,
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$dx,
    AnyType:$dy
  );
}

def OneFlow_EagerPToBOp : OneFlow_BaseOp<"eager_p_to_b", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    StrAttr:$in_parallel_conf,
    StrAttr:$out_parallel_conf,
    AnyI64ElementsAttr:$shape
  );
}

def OneFlow_KlDivLossGradOp : OneFlow_BaseOp<"kl_div_loss_grad", []> {
  let input = (ins
    AnyType:$input,
    AnyType:$target,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    StrAttr:$reduction,
    BoolAttr:$log_target
  );
}

def OneFlow_EagerPToSOp : OneFlow_BaseOp<"eager_p_to_s", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI64Attr:$out_split_axis,
    StrAttr:$in_parallel_conf,
    StrAttr:$out_parallel_conf,
    AnyI64ElementsAttr:$shape
  );
}

def OneFlow_ImageResizeToFixedOp : OneFlow_BaseOp<"image_resize_to_fixed", []> {
  let input = (ins
    AnyType:$in
  );
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

def OneFlow_BroadcastAddOp : OneFlow_BaseOp<"broadcast_add", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
}

def OneFlow_UpsampleNearest_3dGradOp : OneFlow_BaseOp<"upsample_nearest_3d_grad", []> {
  let input = (ins
    AnyType:$dy,
    AnyType:$x
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    F32Attr:$depth_scale,
    F32Attr:$height_scale,
    F32Attr:$width_scale,
    StrAttr:$data_format
  );
}

def OneFlow_SplitLikeOp : OneFlow_BaseOp<"split_like", []> {
  let input = (ins
    AnyType:$in,
    AnyType:$like
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI64Attr:$axis
  );
}

def OneFlow_ImageFlipOp : OneFlow_BaseOp<"image_flip", []> {
  let input = (ins
    AnyType:$in,
    AnyType:$flip_code
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_FlipOp : OneFlow_BaseOp<"flip", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    SI32ArrayAttr:$dims
  );
}

def OneFlow_SmoothL1LossOp : OneFlow_BaseOp<"smooth_l1_loss", []> {
  let input = (ins
    AnyType:$input,
    AnyType:$target
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    StrAttr:$reduction,
    F32Attr:$beta
  );
}

def OneFlow_BinaryCrossEntropyWithLogitsOp : OneFlow_BaseOp<"binary_cross_entropy_with_logits", []> {
  let input = (ins
    AnyType:$input,
    AnyType:$target,
    AnyType:$weight,
    AnyType:$pos_weight
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    BoolAttr:$has_pos_weight,
    StrAttr:$reduction
  );
}

def OneFlow_ScalarLogicalAndOp : OneFlow_BaseOp<"scalar_logical_and", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    BoolAttr:$has_int_operand,
    BoolAttr:$has_float_operand,
    SI64Attr:$int_operand,
    F64Attr:$float_operand
  );
}

def OneFlow_ConstantOp : OneFlow_BaseOp<"constant", []> {
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    F64Attr:$floating_value,
    SI64Attr:$integer_value,
    BoolAttr:$is_floating_value,
    StrAttr:$dtype,
    AnyI64ElementsAttr:$shape,
    StrArrayAttr:$nd_sbp
  );
}

def OneFlow_DynamicLossScaleScheduleOp : OneFlow_BaseOp<"dynamic_loss_scale_schedule", []> {
  let input = (ins
    AnyType:$count_not_finite,
    AnyType:$loss_scale,
    AnyType:$good_step_counter
  );
  let attrs = (ins
    SI64Attr:$increment_period,
    F32Attr:$multiplier
  );
}

def OneFlow_DimScatterMulOp : OneFlow_BaseOp<"dim_scatter_mul", []> {
  let input = (ins
    AnyType:$input,
    AnyType:$index,
    AnyType:$src
  );
  let output = (outs
    AnyType:$output
  );
  let attrs = (ins
    SI32Attr:$dim
  );
}

def OneFlow_DropoutGradOp : OneFlow_BaseOp<"dropout_grad", []> {
  let input = (ins
    AnyType:$dy,
    AnyType:$mask
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    F32Attr:$scale
  );
}

def OneFlow_DropoutOp : OneFlow_BaseOp<"dropout", []> {
  let input = (ins
    AnyType:$in,
    AnyType:$mask,
    AnyType:$_add_to_output
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    F32Attr:$scale
  );
}

def OneFlow_FusedBiasAddMaskScaleOp : OneFlow_BaseOp<"fused_bias_add_mask_scale", []> {
  let input = (ins
    AnyType:$a,
    AnyType:$b,
    AnyType:$mask,
    AnyType:$_add_to_output
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI32Attr:$axis,
    F32Attr:$scale
  );
}

def OneFlow_Expm1GradOp : OneFlow_BaseOp<"expm1_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_TensorBufferToListOfTensorsV2Op : OneFlow_BaseOp<"tensor_buffer_to_list_of_tensors_v2", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    ShapeArrayAttr:$out_shapes,
    DTArrayAttr:$out_dtypes,
    BoolAttr:$dynamic_out
  );
}

def OneFlow_UniformIntOp : OneFlow_BaseOp<"uniform_int", []> {
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI64Attr:$from,
    SI64Attr:$to,
    StrAttr:$dtype,
    AnyI64ElementsAttr:$shape,
    StrAttr:$nd_sbp
  );
}

def OneFlow_UnsortedBatchSegmentSumOp : OneFlow_BaseOp<"unsorted_batch_segment_sum", []> {
  let input = (ins
    AnyType:$data,
    AnyType:$segment_ids
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI64Attr:$num_segments
  );
}

def OneFlow_XdivyXGradOp : OneFlow_BaseOp<"xdivy_x_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$y,
    AnyType:$dz
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_AsinOp : OneFlow_BaseOp<"asin", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_DimScatterUpdateOp : OneFlow_BaseOp<"dim_scatter_update", []> {
  let input = (ins
    AnyType:$input,
    AnyType:$index,
    AnyType:$src
  );
  let output = (outs
    AnyType:$output
  );
  let attrs = (ins
    SI32Attr:$dim
  );
}

def OneFlow_TransposeOp : OneFlow_BaseOp<"transpose", []> {
  let input = (ins
    AnyType:$input
  );
  let output = (outs
    AnyType:$output
  );
  let attrs = (ins
    SI32ArrayAttr:$perm
  );
}

def OneFlow_PackOp : OneFlow_BaseOp<"pack", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI32Attr:$pack_num
  );
}

def OneFlow_DimGatherOp : OneFlow_BaseOp<"dim_gather", []> {
  let input = (ins
    AnyType:$input,
    AnyType:$index
  );
  let output = (outs
    AnyType:$output
  );
  let attrs = (ins
    SI32Attr:$dim
  );
}

def OneFlow_MlirJitOp : OneFlow_BaseOp<"mlir_jit", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    StrAttr:$mlir_assembly
  );
}

def OneFlow_ErfcOp : OneFlow_BaseOp<"erfc", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_DimScatterMulScalarOp : OneFlow_BaseOp<"dim_scatter_mul_scalar", []> {
  let input = (ins
    AnyType:$input,
    AnyType:$index
  );
  let output = (outs
    AnyType:$output
  );
  let attrs = (ins
    F32Attr:$src_scalar,
    SI32Attr:$dim
  );
}

def OneFlow_DiagGradOp : OneFlow_BaseOp<"diag_grad", []> {
  let input = (ins
    AnyType:$dy,
    AnyType:$in
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    SI32Attr:$diagonal
  );
}

def OneFlow_Deconv3dOp : OneFlow_BaseOp<"deconv3d", []> {
  let input = (ins
    AnyType:$in,
    AnyType:$weight
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI32Attr:$filters,
    SI32ArrayAttr:$padding_before,
    StrAttr:$data_format,
    SI32ArrayAttr:$kernel_size,
    SI32ArrayAttr:$output_padding,
    SI32ArrayAttr:$strides,
    SI32ArrayAttr:$dilation_rate,
    SI32Attr:$groups
  );
}

def OneFlow_MaxPool_1dGradOp : OneFlow_BaseOp<"max_pool_1d_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$y,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
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

def OneFlow_AbsOp : OneFlow_IdempotentBaseOp<"abs", []> {}

def OneFlow_FusedCastScaleOp : OneFlow_BaseOp<"fused_cast_scale", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$scale_by_tensor
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    F64Attr:$scale
  );
}

def OneFlow_OfrecordBytesDecoderOp : OneFlow_BaseOp<"ofrecord_bytes_decoder", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    StrAttr:$name
  );
}

def OneFlow_Deconv2dOp : OneFlow_BaseOp<"deconv2d", []> {
  let input = (ins
    AnyType:$in,
    AnyType:$weight
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI32Attr:$filters,
    SI32ArrayAttr:$padding_before,
    StrAttr:$data_format,
    SI32ArrayAttr:$kernel_size,
    SI32ArrayAttr:$output_padding,
    SI32ArrayAttr:$strides,
    SI32ArrayAttr:$dilation_rate,
    SI32Attr:$groups
  );
}

def OneFlow_CtcGreedyDecoderOp : OneFlow_BaseOp<"ctc_greedy_decoder", []> {
  let input = (ins
    AnyType:$log_probs,
    AnyType:$input_lengths
  );
  let output = (outs
    AnyType:$decoded,
    AnyType:$neg_sum_logits
  );
  let attrs = (ins
    BoolAttr:$merge_repeated
  );
}

def OneFlow_CopyOp : OneFlow_BaseOp<"copy", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    StrAttr:$device_type,
    SI64Attr:$device_id
  );
}

def OneFlow_OFRecordReaderOp : OneFlow_BaseOp<"OFRecordReader", []> {
  let output = (outs
    AnyType:$out
  );
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

def OneFlow_FusedBiasAddGeluOp : OneFlow_BaseOp<"fused_bias_add_gelu", []> {
  let input = (ins
    AnyType:$a,
    AnyType:$b
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI32Attr:$axis
  );
}

def OneFlow_LogGradOp : OneFlow_BaseOp<"log_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_Log1pGradOp : OneFlow_BaseOp<"log1p_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_ScalarLogicalEqualOp : OneFlow_BaseOp<"scalar_logical_equal", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    BoolAttr:$has_int_operand,
    BoolAttr:$has_float_operand,
    SI64Attr:$int_operand,
    F64Attr:$float_operand
  );
}

def OneFlow_ClipByScalarMaxOp : OneFlow_BaseOp<"clip_by_scalar_max", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    F64Attr:$floating_max,
    SI64Attr:$integral_max
  );
}

def OneFlow_CtcLossOp : OneFlow_BaseOp<"ctc_loss", []> {
  let input = (ins
    AnyType:$log_probs,
    AnyType:$targets,
    AnyType:$input_lengths,
    AnyType:$target_lengths
  );
  let output = (outs
    AnyType:$loss,
    AnyType:$alpha
  );
  let attrs = (ins
    SI64Attr:$max_target_length,
    SI32Attr:$blank,
    BoolAttr:$zero_infinity
  );
}

def OneFlow_BroadcastGreaterOp : OneFlow_BaseOp<"broadcast_greater", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
}

def OneFlow_ScatterNdLikeOp : OneFlow_BaseOp<"scatter_nd_like", []> {
  let input = (ins
    AnyType:$like,
    AnyType:$indices,
    AnyType:$updates
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_ExpandOp : OneFlow_BaseOp<"expand", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI32ArrayAttr:$logical_in_shape,
    SI32ArrayAttr:$logical_expand_shape
  );
}

def OneFlow_MultiCountNotFiniteOp : OneFlow_BaseOp<"multi_count_not_finite", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_FlattenOp : OneFlow_BaseOp<"flatten", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI32Attr:$start_dim,
    SI32Attr:$end_dim
  );
}

def OneFlow_BroadcastLogicalXorOp : OneFlow_BaseOp<"broadcast_logical_xor", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
}

def OneFlow_CountNotFiniteOp : OneFlow_BaseOp<"count_not_finite", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_AdaptiveAvgPool3dGradOp : OneFlow_BaseOp<"adaptive_avg_pool3d_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    SI64ArrayAttr:$output_size
  );
}

def OneFlow_UniformOp : OneFlow_BaseOp<"uniform", []> {
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    F64Attr:$from,
    F64Attr:$to,
    StrAttr:$dtype,
    AnyI64ElementsAttr:$shape,
    StrAttr:$nd_sbp
  );
}

def OneFlow_AdaptiveAvgPool1dGradOp : OneFlow_BaseOp<"adaptive_avg_pool1d_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    SI64ArrayAttr:$output_size
  );
}

def OneFlow_BernoulliOp : OneFlow_BaseOp<"bernoulli", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI64Attr:$seed,
    BoolAttr:$has_seed,
    StrAttr:$dtype
  );
}

def OneFlow_UpsampleBilinear_2dOp : OneFlow_BaseOp<"upsample_bilinear_2d", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    F32Attr:$height_scale,
    F32Attr:$width_scale,
    BoolAttr:$align_corners,
    StrAttr:$data_format
  );
}

def OneFlow_EagerNcclS2sOp : OneFlow_BaseOp<"eager_nccl_s2s", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI64Attr:$in_split_axis,
    SI64Attr:$out_split_axis,
    StrAttr:$parallel_conf
  );
}

def OneFlow_ReduceMinDeviceStageOp : OneFlow_BaseOp<"reduce_min_device_stage", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out,
    AnyType:$mask,
    AnyType:$count
  );
  let attrs = (ins
    SI32ArrayAttr:$axis
  );
}

def OneFlow_Avgpool_3dOp : OneFlow_BaseOp<"avgpool_3d", []> {
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
    BoolAttr:$ceil_mode,
    BoolAttr:$count_include_pad,
    SI64Attr:$divisor_override
  );
}

def OneFlow_AcoshOp : OneFlow_BaseOp<"acosh", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_SignOp : OneFlow_IdempotentBaseOp<"sign", []> {}

def OneFlow_DimScatterUpdateScalarOp : OneFlow_BaseOp<"dim_scatter_update_scalar", []> {
  let input = (ins
    AnyType:$input,
    AnyType:$index
  );
  let output = (outs
    AnyType:$output
  );
  let attrs = (ins
    F32Attr:$src_scalar,
    SI32Attr:$dim
  );
}

def OneFlow_CastOp : OneFlow_BaseOp<"cast", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    StrAttr:$dtype
  );
}

def OneFlow_ReduceSumLikeOp : OneFlow_BaseOp<"reduce_sum_like", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$like
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    SI32ArrayAttr:$axis
  );
}

def OneFlow_SqrtGradOp : OneFlow_BaseOp<"sqrt_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_NormalOp : OneFlow_BaseOp<"normal", []> {
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    F64Attr:$mean,
    F64Attr:$std,
    StrAttr:$dtype,
    AnyI64ElementsAttr:$shape,
    StrAttr:$nd_sbp
  );
}

def OneFlow_EagerNcclAllReduceOp : OneFlow_BaseOp<"eager_nccl_all_reduce", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    StrAttr:$parallel_conf,
    BoolAttr:$async_launch
  );
}

def OneFlow_CategoricalOrdinalEncodeOp : OneFlow_BaseOp<"CategoricalOrdinalEncode", []> {
  let input = (ins
    AnyType:$table,
    AnyType:$size,
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    BoolAttr:$hash_precomputed
  );
}

def OneFlow_Deconv1dOp : OneFlow_BaseOp<"deconv1d", []> {
  let input = (ins
    AnyType:$in,
    AnyType:$weight
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI32Attr:$filters,
    SI32ArrayAttr:$padding_before,
    StrAttr:$data_format,
    SI32ArrayAttr:$kernel_size,
    SI32ArrayAttr:$output_padding,
    SI32ArrayAttr:$strides,
    SI32ArrayAttr:$dilation_rate,
    SI32Attr:$groups
  );
}

def OneFlow_ClipByScalarMaxGradOp : OneFlow_BaseOp<"clip_by_scalar_max_grad", []> {
  let input = (ins
    AnyType:$dy,
    AnyType:$x
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    F64Attr:$floating_max,
    SI64Attr:$integral_max
  );
}

def OneFlow_SummaryWriteScalarOp : OneFlow_BaseOp<"summary_write_scalar", []> {
  let input = (ins
    AnyType:$in,
    AnyType:$step,
    AnyType:$tag
  );
}

def OneFlow_NarrowGradOp : OneFlow_BaseOp<"narrow_grad", []> {
  let input = (ins
    AnyType:$dy,
    AnyType:$like
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    SI64Attr:$dim,
    SI64Attr:$start,
    SI64Attr:$length
  );
}

def OneFlow_ClipByScalarOp : OneFlow_BaseOp<"clip_by_scalar", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    F64Attr:$floating_min,
    SI64Attr:$integral_min,
    F64Attr:$floating_max,
    SI64Attr:$integral_max
  );
}

def OneFlow_ScalarLogicalOrOp : OneFlow_BaseOp<"scalar_logical_or", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    BoolAttr:$has_int_operand,
    BoolAttr:$has_float_operand,
    SI64Attr:$int_operand,
    F64Attr:$float_operand
  );
}

def OneFlow_Avgpool_2dGradOp : OneFlow_BaseOp<"avgpool_2d_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$y,
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
    BoolAttr:$ceil_mode,
    BoolAttr:$count_include_pad,
    SI64Attr:$divisor_override
  );
}

def OneFlow_IdentityBufferOp : OneFlow_BaseOp<"identity_buffer", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI64Attr:$buffer_size
  );
}

def OneFlow_Avgpool_2dOp : OneFlow_BaseOp<"avgpool_2d", []> {
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
    BoolAttr:$ceil_mode,
    BoolAttr:$count_include_pad,
    SI64Attr:$divisor_override
  );
}

def OneFlow_BinaryCrossEntropyGradOp : OneFlow_BaseOp<"binary_cross_entropy_grad", []> {
  let input = (ins
    AnyType:$input,
    AnyType:$target,
    AnyType:$weight,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    StrAttr:$reduction
  );
}

def OneFlow_SqueezeOp : OneFlow_BaseOp<"squeeze", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI32ArrayAttr:$axes
  );
}

def OneFlow_SparseSoftmaxCrossEntropyMsGradOp : OneFlow_BaseOp<"sparse_softmax_cross_entropy_ms_grad", []> {
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

def OneFlow_OneHotOp : OneFlow_BaseOp<"one_hot", []> {
  let input = (ins
    AnyType:$indices
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI64Attr:$depth,
    F64Attr:$floating_on_value,
    SI64Attr:$integer_on_value,
    F64Attr:$floating_off_value,
    SI64Attr:$integer_off_value,
    StrAttr:$dtype
  );
}

def OneFlow__ncclLogical_2DSameDim1AllReduceOp : OneFlow_BaseOp<"_nccl_logical_2D_same_dim1_all_reduce", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_Expm1Op : OneFlow_BaseOp<"expm1", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_BatchMatmulOp : OneFlow_BaseOp<"batch_matmul", []> {
  let input = (ins
    AnyType:$a,
    AnyType:$b,
    AnyType:$_add_to_output
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    BoolAttr:$transpose_a,
    BoolAttr:$transpose_b,
    F64Attr:$alpha
  );
}

def OneFlow_NegativeGradOp : OneFlow_BaseOp<"negative_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_ConcatOp : OneFlow_BaseOp<"concat", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI64Attr:$axis,
    SI64Attr:$max_dim_size
  );
}

def OneFlow_ObjectSegmentationPolygonFlipOp : OneFlow_BaseOp<"object_segmentation_polygon_flip", []> {
  let input = (ins
    AnyType:$poly,
    AnyType:$image_size,
    AnyType:$flip_code
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_SigmoidV2GradOp : OneFlow_BaseOp<"sigmoid_v2_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_BroadcastLessEqualOp : OneFlow_BaseOp<"broadcast_less_equal", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
}

def OneFlow_AdaptiveAvgPool3dOp : OneFlow_BaseOp<"adaptive_avg_pool3d", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    SI64ArrayAttr:$output_size
  );
}

def OneFlow_BroadcastLikeOp : OneFlow_BaseOp<"broadcast_like", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$like
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    SI32ArrayAttr:$broadcast_axes
  );
}

def OneFlow_MultiSquareSumOp : OneFlow_BaseOp<"multi_square_sum", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_CudnnFusedNormalizationAddReluGradOp : OneFlow_BaseOp<"cudnn_fused_normalization_add_relu_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy,
    AnyType:$mean,
    AnyType:$inv_variance,
    AnyType:$gamma,
    AnyType:$beta,
    AnyType:$reserve_space,
    AnyType:$y
  );
  let output = (outs
    AnyType:$gamma_diff,
    AnyType:$beta_diff,
    AnyType:$dx,
    AnyType:$addend_diff
  );
  let attrs = (ins
    SI32Attr:$axis,
    F32Attr:$epsilon
  );
}

def OneFlow_CeluGradOp : OneFlow_BaseOp<"celu_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    F64Attr:$alpha
  );
}

def OneFlow_ConvDataGradOp : OneFlow_BaseOp<"conv_data_grad", []> {
  let input = (ins
    AnyType:$dy,
    AnyType:$filter,
    AnyType:$x_like,
    AnyType:$_add_to_output
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    SI32Attr:$num_spatial_dims,
    SI32ArrayAttr:$padding_before,
    StrAttr:$data_format,
    SI32ArrayAttr:$kernel_size,
    SI32ArrayAttr:$strides,
    SI32ArrayAttr:$dilation_rate,
    SI32Attr:$groups
  );
}

def OneFlow_AssignIfOp : OneFlow_BaseOp<"assign_if", []> {
  let input = (ins
    AnyType:$ref,
    AnyType:$value,
    AnyType:$condition
  );
}

def OneFlow_ScalarLogicalLessOp : OneFlow_BaseOp<"scalar_logical_less", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    BoolAttr:$has_int_operand,
    BoolAttr:$has_float_operand,
    SI64Attr:$int_operand,
    F64Attr:$float_operand
  );
}

def OneFlow_ScalarAddByTensorOp : OneFlow_BaseOp<"scalar_add_by_tensor", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$scalar
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_IndexedSlicesReduceSumOp : OneFlow_BaseOp<"indexed_slices_reduce_sum", []> {
  let input = (ins
    AnyType:$x_indices,
    AnyType:$x_values
  );
  let output = (outs
    AnyType:$y_indices,
    AnyType:$y_values,
    AnyType:$num_unique
  );
}

def OneFlow_ConstantPad1dOp : OneFlow_BaseOp<"constant_pad1d", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    SI64ArrayAttr:$padding,
    F64Attr:$floating_value,
    SI64Attr:$integral_value
  );
}

def OneFlow_ArgmaxOp : OneFlow_BaseOp<"argmax", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_EagerNaiveSToSOp : OneFlow_BaseOp<"eager_naive_s_to_s", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI64Attr:$in_split_axis,
    SI64Attr:$out_split_axis,
    StrAttr:$in_parallel_conf,
    StrAttr:$out_parallel_conf,
    AnyI64ElementsAttr:$shape
  );
}

def OneFlow_MishOp : OneFlow_BaseOp<"mish", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_BroadcastLogicalAndOp : OneFlow_BaseOp<"broadcast_logical_and", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
}

def OneFlow_ConvBiasGradOp : OneFlow_BaseOp<"conv_bias_grad", []> {
  let input = (ins
    AnyType:$dy
  );
  let output = (outs
    AnyType:$bias_diff
  );
  let attrs = (ins
    StrAttr:$data_format,
    SI32Attr:$num_spatial_dims
  );
}

def OneFlow_DimScatterAddOp : OneFlow_BaseOp<"dim_scatter_add", []> {
  let input = (ins
    AnyType:$input,
    AnyType:$index,
    AnyType:$src
  );
  let output = (outs
    AnyType:$output
  );
  let attrs = (ins
    SI32Attr:$dim
  );
}

def OneFlow_UpsampleGradOp : OneFlow_BaseOp<"upsample_grad", []> {
  let input = (ins
    AnyType:$dy,
    AnyType:$x
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    F32Attr:$height_scale,
    F32Attr:$width_scale,
    BoolAttr:$align_corners,
    StrAttr:$data_format,
    StrAttr:$interpolation
  );
}

def OneFlow_UnfoldTensorOp : OneFlow_BaseOp<"unfold_tensor", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    SI32Attr:$dimension,
    SI32Attr:$size,
    SI32Attr:$step
  );
}

def OneFlow_ClipByScalarMinOp : OneFlow_BaseOp<"clip_by_scalar_min", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    F64Attr:$floating_min,
    SI64Attr:$integral_min
  );
}

def OneFlow_BroadcastFloorModOp : OneFlow_BaseOp<"broadcast_floor_mod", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
}

def OneFlow_ScalarLogicalGreaterEqualOp : OneFlow_BaseOp<"scalar_logical_greater_equal", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    BoolAttr:$has_int_operand,
    BoolAttr:$has_float_operand,
    SI64Attr:$int_operand,
    F64Attr:$float_operand
  );
}

def OneFlow_RollOp : OneFlow_BaseOp<"roll", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI32ArrayAttr:$shifts,
    SI32ArrayAttr:$dims
  );
}

def OneFlow_BatchGatherOp : OneFlow_BaseOp<"batch_gather", []> {
  let input = (ins
    AnyType:$in,
    AnyType:$indices
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_FlushSummaryWriterOp : OneFlow_BaseOp<"flush_summary_writer", []> {
}

def OneFlow_EagerNcclReduceOp : OneFlow_BaseOp<"eager_nccl_reduce", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    StrAttr:$parallel_conf,
    SI64Attr:$root
  );
}

def OneFlow_SquareOp : OneFlow_BaseOp<"square", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_ConvFilterGradOp : OneFlow_BaseOp<"conv_filter_grad", []> {
  let input = (ins
    AnyType:$dy,
    AnyType:$x
  );
  let output = (outs
    AnyType:$filter_diff
  );
  let attrs = (ins
    SI32Attr:$num_spatial_dims,
    SI32ArrayAttr:$padding_before,
    StrAttr:$data_format,
    SI32ArrayAttr:$kernel_size,
    SI32ArrayAttr:$strides,
    SI32ArrayAttr:$dilation_rate,
    SI32Attr:$groups
  );
}

def OneFlow_SummaryWritePbOp : OneFlow_BaseOp<"summary_write_pb", []> {
  let input = (ins
    AnyType:$in,
    AnyType:$step
  );
}

def OneFlow_DimScatterAddLikeOp : OneFlow_BaseOp<"dim_scatter_add_like", []> {
  let input = (ins
    AnyType:$like,
    AnyType:$index,
    AnyType:$src
  );
  let output = (outs
    AnyType:$output
  );
  let attrs = (ins
    SI32Attr:$dim
  );
}

def OneFlow_ArgwhereOp : OneFlow_BaseOp<"argwhere", []> {
  let input = (ins
    AnyType:$input
  );
  let output = (outs
    AnyType:$output,
    AnyType:$output_size
  );
  let attrs = (ins
    StrAttr:$dtype
  );
}

def OneFlow_IndexedSlicesAdamUpdateOp : OneFlow_BaseOp<"indexed_slices_adam_update", []> {
  let input = (ins
    AnyType:$model,
    AnyType:$model_diff_indices,
    AnyType:$model_diff_values,
    AnyType:$learning_rate,
    AnyType:$bias_correction1,
    AnyType:$bias_correction2,
    AnyType:$m,
    AnyType:$v,
    AnyType:$max_v
  );
  let attrs = (ins
    F32Attr:$learning_rate_val,
    F32Attr:$beta1,
    F32Attr:$beta2,
    F32Attr:$epsilon,
    F32Attr:$weight_decay,
    BoolAttr:$amsgrad,
    BoolAttr:$do_bias_correction
  );
}

def OneFlow_AbsGradOp : OneFlow_BaseOp<"abs_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_GridSampleGradOp : OneFlow_BaseOp<"grid_sample_grad", []> {
  let input = (ins
    AnyType:$doutput,
    AnyType:$input,
    AnyType:$grid
  );
  let output = (outs
    AnyType:$dinput,
    AnyType:$dgrid
  );
  let attrs = (ins
    StrAttr:$interpolation_mode,
    StrAttr:$padding_mode,
    BoolAttr:$align_corners
  );
}

def OneFlow_ReciprocalGradOp : OneFlow_BaseOp<"reciprocal_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_CastToTickOp : OneFlow_BaseOp<"cast_to_tick", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_RoundOp : OneFlow_IdempotentBaseOp<"round", []> {}

def OneFlow_RandomMaskLikeOp : OneFlow_BaseOp<"random_mask_like", []> {
  let input = (ins
    AnyType:$like
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    F32Attr:$rate,
    SI64Attr:$seed
  );
}

def OneFlow_AvgPool_1dGradOp : OneFlow_BaseOp<"avg_pool_1d_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$y,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
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

def OneFlow_BroadcastMaximumOp : OneFlow_BaseOp<"broadcast_maximum", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
}

def OneFlow_COCOReaderOp : OneFlow_BaseOp<"COCOReader", []> {
  let output = (outs
    AnyType:$image,
    AnyType:$image_id,
    AnyType:$image_size,
    AnyType:$gt_bbox,
    AnyType:$gt_label,
    AnyType:$gt_segm,
    AnyType:$gt_segm_index
  );
  let attrs = (ins
    SI64Attr:$session_id,
    StrAttr:$annotation_file,
    StrAttr:$image_dir,
    SI64Attr:$batch_size,
    BoolAttr:$shuffle_after_epoch,
    SI64Attr:$random_seed,
    BoolAttr:$group_by_ratio,
    BoolAttr:$remove_images_without_annotations,
    BoolAttr:$stride_partition,
    StrArrayAttr:$nd_sbp
  );
}

def OneFlow_FusedSelfAttentionQueryMulKeyAndValueOp : OneFlow_BaseOp<"fused_self_attention_query_mul_key_and_value", []> {
  let input = (ins
    AnyType:$hidden_states
  );
  let output = (outs
    AnyType:$query_mul_key,
    AnyType:$value
  );
  let attrs = (ins
    SI64Attr:$head_size,
    F32Attr:$alpha
  );
}

def OneFlow_EagerSymmetricSToPOp : OneFlow_BaseOp<"eager_symmetric_s_to_p", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI64Attr:$in_split_axis,
    StrAttr:$parallel_conf
  );
}

def OneFlow_ImageDecodeOp : OneFlow_BaseOp<"image_decode", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    StrAttr:$color_space,
    StrAttr:$data_type
  );
}

def OneFlow_ClipByScalarMinGradOp : OneFlow_BaseOp<"clip_by_scalar_min_grad", []> {
  let input = (ins
    AnyType:$dy,
    AnyType:$x
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    F64Attr:$floating_min,
    SI64Attr:$integral_min
  );
}

def OneFlow_HardsigmoidGradOp : OneFlow_BaseOp<"hardsigmoid_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_RsqrtGradOp : OneFlow_BaseOp<"rsqrt_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_UpsampleBicubic_2dOp : OneFlow_BaseOp<"upsample_bicubic_2d", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    F32Attr:$height_scale,
    F32Attr:$width_scale,
    BoolAttr:$align_corners,
    StrAttr:$data_format
  );
}

def OneFlow_TupleIdentityOp : OneFlow_BaseOp<"tuple_identity", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_ReshapeLikeOp : OneFlow_BaseOp<"reshape_like", []> {
  let input = (ins
    AnyType:$in,
    AnyType:$like
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_EagerNcclAllGatherOp : OneFlow_BaseOp<"eager_nccl_all_gather", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    StrAttr:$parallel_conf
  );
}

def OneFlow_AddNOp : OneFlow_BaseOp<"add_n", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_ArgSortOp : OneFlow_BaseOp<"arg_sort", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    StrAttr:$direction
  );
}

def OneFlow_Avgpool_1dOp : OneFlow_BaseOp<"avgpool_1d", []> {
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
    BoolAttr:$ceil_mode,
    BoolAttr:$count_include_pad,
    SI64Attr:$divisor_override
  );
}

def OneFlow_BinaryCrossEntropyWithLogitsGradOp : OneFlow_BaseOp<"binary_cross_entropy_with_logits_grad", []> {
  let input = (ins
    AnyType:$input,
    AnyType:$target,
    AnyType:$weight,
    AnyType:$pos_weight,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    BoolAttr:$has_pos_weight,
    StrAttr:$reduction
  );
}

def OneFlow_AcosGradOp : OneFlow_BaseOp<"acos_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_TestMultiOutputOrderOp : OneFlow_BaseOp<"TestMultiOutputOrder", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out1,
    AnyType:$out2
  );
}

def OneFlow_MegatronGptMmapDataLoaderOp : OneFlow_BaseOp<"megatron_gpt_mmap_data_loader", []> {
  let input = (ins
    AnyType:$iteration
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    StrAttr:$data_file_prefix,
    SI64Attr:$seq_length,
    SI64Attr:$label_length,
    SI64Attr:$num_samples,
    SI64Attr:$batch_size,
    StrAttr:$dtype,
    SI64ArrayAttr:$split_sizes,
    SI64Attr:$split_index,
    BoolAttr:$shuffle,
    SI64Attr:$random_seed,
    StrArrayAttr:$nd_sbp
  );
}

def OneFlow_TestSourceOp : OneFlow_BaseOp<"TestSource", []> {
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_Maxpool_1dGradOp : OneFlow_BaseOp<"maxpool_1d_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$y,
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
    BoolAttr:$return_indices,
    BoolAttr:$ceil_mode
  );
}

def OneFlow_BroadcastPowYGradOp : OneFlow_BaseOp<"broadcast_pow_y_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$y,
    AnyType:$z,
    AnyType:$dz
  );
  let output = (outs
    AnyType:$dy
  );
}

def OneFlow_GeluGradOp : OneFlow_BaseOp<"gelu_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_ImageNormalizeOp : OneFlow_BaseOp<"image_normalize", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    F32ArrayAttr:$std,
    F32ArrayAttr:$mean
  );
}

def OneFlow_LogicalNotOp : OneFlow_BaseOp<"logical_not", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_CtcLossGradOp : OneFlow_BaseOp<"ctc_loss_grad", []> {
  let input = (ins
    AnyType:$grad_out,
    AnyType:$log_probs,
    AnyType:$targets,
    AnyType:$input_lengths,
    AnyType:$target_lengths,
    AnyType:$loss,
    AnyType:$alpha
  );
  let output = (outs
    AnyType:$grad
  );
  let attrs = (ins
    SI64Attr:$max_target_length,
    SI32Attr:$blank,
    BoolAttr:$zero_infinity
  );
}

def OneFlow_SliceUpdateOp : OneFlow_BaseOp<"slice_update", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$update
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    SI64ArrayAttr:$start,
    SI64ArrayAttr:$stop,
    SI64ArrayAttr:$step
  );
}

def OneFlow_BroadcastPowXGradOp : OneFlow_BaseOp<"broadcast_pow_x_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$y,
    AnyType:$z,
    AnyType:$dz
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_IndexedSlicesMomentumUpdateOp : OneFlow_BaseOp<"indexed_slices_momentum_update", []> {
  let input = (ins
    AnyType:$model,
    AnyType:$model_diff_indices,
    AnyType:$model_diff_values,
    AnyType:$learning_rate,
    AnyType:$momentum
  );
  let attrs = (ins
    F32Attr:$beta,
    F32Attr:$weight_decay
  );
}

def OneFlow_AdaptiveAvgPool1dOp : OneFlow_BaseOp<"adaptive_avg_pool1d", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    SI64ArrayAttr:$output_size
  );
}

def OneFlow_AccOp : OneFlow_BaseOp<"acc", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI32Attr:$max_acc_num
  );
}

def OneFlow_HardtanhOp : OneFlow_BaseOp<"hardtanh", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    F64Attr:$min_val,
    F64Attr:$max_val
  );
}

def OneFlow_CombinedMarginLossOp : OneFlow_BaseOp<"combined_margin_loss", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$label
  );
  let output = (outs
    AnyType:$y,
    AnyType:$theta
  );
  let attrs = (ins
    F32Attr:$m1,
    F32Attr:$m2,
    F32Attr:$m3,
    SI64Attr:$depth
  );
}

def OneFlow_AdaptiveAvgPool2dGradOp : OneFlow_BaseOp<"adaptive_avg_pool2d_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    SI64ArrayAttr:$output_size
  );
}

def OneFlow_ClipByScalarGradOp : OneFlow_BaseOp<"clip_by_scalar_grad", []> {
  let input = (ins
    AnyType:$dy,
    AnyType:$x
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    F64Attr:$floating_min,
    SI64Attr:$integral_min,
    F64Attr:$floating_max,
    SI64Attr:$integral_max
  );
}

def OneFlow_FusedTrilScaleSoftmaxMaskScaleGradOp : OneFlow_BaseOp<"fused_tril_scale_softmax_mask_scale_grad", []> {
  let input = (ins
    AnyType:$softmax_y,
    AnyType:$dy,
    AnyType:$mask
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    SI64Attr:$diagonal,
    F32Attr:$tril_scale_value,
    F32Attr:$mask_scale_value
  );
}

def OneFlow_ScalarPowOp : OneFlow_BaseOp<"scalar_pow", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    BoolAttr:$has_int_operand,
    BoolAttr:$has_float_operand,
    SI64Attr:$int_operand,
    F64Attr:$float_operand
  );
}

def OneFlow_AssignOp : OneFlow_BaseOp<"assign", []> {
  let input = (ins
    AnyType:$ref,
    AnyType:$value
  );
}

def OneFlow_Atan2XGradOp : OneFlow_BaseOp<"atan2_x_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$y,
    AnyType:$dz
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_ErfGradOp : OneFlow_BaseOp<"erf_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_InTopKOp : OneFlow_BaseOp<"in_top_k", []> {
  let input = (ins
    AnyType:$targets,
    AnyType:$predictions
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI32Attr:$k
  );
}

def OneFlow_Avgpool_1dGradOp : OneFlow_BaseOp<"avgpool_1d_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$y,
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
    BoolAttr:$ceil_mode,
    BoolAttr:$count_include_pad,
    SI64Attr:$divisor_override
  );
}

def OneFlow_CombinedMarginLossGradOp : OneFlow_BaseOp<"combined_margin_loss_grad", []> {
  let input = (ins
    AnyType:$dy,
    AnyType:$label,
    AnyType:$theta
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    F32Attr:$m1,
    F32Attr:$m2,
    F32Attr:$m3,
    SI64Attr:$depth
  );
}

def OneFlow_SquareSumOp : OneFlow_BaseOp<"square_sum", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_SoftmaxCrossEntropyOp : OneFlow_BaseOp<"softmax_cross_entropy", []> {
  let input = (ins
    AnyType:$prediction,
    AnyType:$label
  );
  let output = (outs
    AnyType:$prob,
    AnyType:$out
  );
}

def OneFlow_BroadcastGreaterEqualOp : OneFlow_BaseOp<"broadcast_greater_equal", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
}

def OneFlow_ArangeOp : OneFlow_BaseOp<"arange", []> {
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI64Attr:$integer_start,
    SI64Attr:$integer_delta,
    SI64Attr:$integer_limit,
    F64Attr:$float_start,
    F64Attr:$float_delta,
    F64Attr:$float_limit,
    StrAttr:$dtype,
    StrArrayAttr:$nd_sbp
  );
}

def OneFlow_Atan2Op : OneFlow_BaseOp<"atan2", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
}

def OneFlow_XlogyXGradOp : OneFlow_BaseOp<"xlogy_x_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$y,
    AnyType:$dz
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_EagerNcclBroadcastOp : OneFlow_BaseOp<"eager_nccl_broadcast", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    StrAttr:$parallel_conf,
    SI64Attr:$root
  );
}

def OneFlow_FloorOp : OneFlow_IdempotentBaseOp<"floor", []> {}

def OneFlow_ScalarFloordivOp : OneFlow_BaseOp<"scalar_floordiv", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    BoolAttr:$has_int_operand,
    BoolAttr:$has_float_operand,
    SI64Attr:$int_operand,
    F64Attr:$float_operand
  );
}

def OneFlow_AssignIfNotOp : OneFlow_BaseOp<"assign_if_not", []> {
  let input = (ins
    AnyType:$ref,
    AnyType:$value,
    AnyType:$condition
  );
}

def OneFlow_CropMirrorNormalizeFromUint8Op : OneFlow_BaseOp<"crop_mirror_normalize_from_uint8", []> {
  let input = (ins
    AnyType:$in,
    AnyType:$mirror
  );
  let output = (outs
    AnyType:$out
  );
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

def OneFlow_KlDivLossOp : OneFlow_BaseOp<"kl_div_loss", []> {
  let input = (ins
    AnyType:$input,
    AnyType:$target
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    StrAttr:$reduction,
    BoolAttr:$log_target
  );
}

def OneFlow_AsinhGradOp : OneFlow_BaseOp<"asinh_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_TensorScatterNdAddOp : OneFlow_BaseOp<"tensor_scatter_nd_add", []> {
  let input = (ins
    AnyType:$params,
    AnyType:$updates,
    AnyType:$indices
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_CastLikeOp : OneFlow_BaseOp<"cast_like", []> {
  let input = (ins
    AnyType:$in,
    AnyType:$dtype_like
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_Conv1dOp : OneFlow_BaseOp<"conv1d", []> {
  let input = (ins
    AnyType:$in,
    AnyType:$weight,
    AnyType:$bias,
    AnyType:$bias_multiplier
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI32Attr:$filters,
    SI32ArrayAttr:$padding_before,
    StrAttr:$data_format,
    SI32ArrayAttr:$kernel_size,
    SI32ArrayAttr:$strides,
    SI32ArrayAttr:$dilation_rate,
    SI32Attr:$groups
  );
}

def OneFlow_PreluOp : OneFlow_BaseOp<"prelu", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$alpha
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_ObjectBboxScaleOp : OneFlow_BaseOp<"object_bbox_scale", []> {
  let input = (ins
    AnyType:$bbox,
    AnyType:$scale
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_LeakyReluOp : OneFlow_BaseOp<"leaky_relu", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    F32Attr:$alpha
  );
}

def OneFlow_Conv3dOp : OneFlow_BaseOp<"conv3d", []> {
  let input = (ins
    AnyType:$in,
    AnyType:$weight,
    AnyType:$bias,
    AnyType:$bias_multiplier
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI32Attr:$filters,
    SI32ArrayAttr:$padding_before,
    StrAttr:$data_format,
    SI32ArrayAttr:$kernel_size,
    SI32ArrayAttr:$strides,
    SI32ArrayAttr:$dilation_rate,
    SI32Attr:$groups
  );
}

def OneFlow_HardtanhGradOp : OneFlow_BaseOp<"hardtanh_grad", []> {
  let input = (ins
    AnyType:$y,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    F64Attr:$min_val,
    F64Attr:$max_val
  );
}

def OneFlow_HierarchicalParallelCastLikeOp : OneFlow_BaseOp<"hierarchical_parallel_cast_like", []> {
  let input = (ins
    AnyType:$in,
    AnyType:$like
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_IdentityOp : OneFlow_BaseOp<"identity", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_HardswishOp : OneFlow_BaseOp<"hardswish", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_ImageBatchAlignOp : OneFlow_BaseOp<"image_batch_align", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    AnyI64ElementsAttr:$shape,
    StrAttr:$data_type,
    SI32Attr:$alignment,
    BoolAttr:$dynamic_out
  );
}

def OneFlow_ObjectBboxFlipOp : OneFlow_BaseOp<"object_bbox_flip", []> {
  let input = (ins
    AnyType:$bbox,
    AnyType:$image_size,
    AnyType:$flip_code
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_ReciprocalNoNanOp : OneFlow_BaseOp<"reciprocal_no_nan", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_ObjectSegmentationPolygonScaleOp : OneFlow_BaseOp<"object_segmentation_polygon_scale", []> {
  let input = (ins
    AnyType:$poly,
    AnyType:$scale
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_SigmoidCrossEntropyGradOp : OneFlow_BaseOp<"sigmoid_cross_entropy_grad", []> {
  let input = (ins
    AnyType:$prediction,
    AnyType:$loss_diff,
    AnyType:$label
  );
  let output = (outs
    AnyType:$prediction_diff
  );
}

def OneFlow_ScalarDivByTensorOp : OneFlow_BaseOp<"scalar_div_by_tensor", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$scalar
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_ObjectSegmentationPolygonToMaskOp : OneFlow_BaseOp<"object_segmentation_polygon_to_mask", []> {
  let input = (ins
    AnyType:$poly,
    AnyType:$poly_index,
    AnyType:$image_size
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_FloordivYGradOp : OneFlow_BaseOp<"floordiv_y_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$y,
    AnyType:$dz
  );
  let output = (outs
    AnyType:$dy
  );
}

def OneFlow_CoinFlipOp : OneFlow_BaseOp<"coin_flip", []> {
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    F32Attr:$probability,
    SI64Attr:$batch_size,
    SI64Attr:$seed,
    BoolAttr:$has_seed,
    StrArrayAttr:$nd_sbp
  );
}

def OneFlow_ImageTargetResizeOp : OneFlow_BaseOp<"image_target_resize", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out,
    AnyType:$size,
    AnyType:$scale
  );
  let attrs = (ins
    SI32Attr:$target_size,
    SI32Attr:$max_size
  );
}

def OneFlow_LeakyReluGradOp : OneFlow_BaseOp<"leaky_relu_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    F32Attr:$alpha
  );
}

def OneFlow_L2NormalizeGradOp : OneFlow_BaseOp<"l2_normalize_grad", []> {
  let input = (ins
    AnyType:$dy,
    AnyType:$y,
    AnyType:$square_x_sum
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    SI32Attr:$axis,
    F32Attr:$epsilon
  );
}

def OneFlow_LogOp : OneFlow_BaseOp<"log", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_LayerNormOp : OneFlow_BaseOp<"layer_norm", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$beta,
    AnyType:$gamma
  );
  let output = (outs
    AnyType:$y,
    AnyType:$mean,
    AnyType:$inv_variance,
    AnyType:$normalized
  );
  let attrs = (ins
    BoolAttr:$center,
    BoolAttr:$scale,
    SI64Attr:$begin_norm_axis,
    SI64Attr:$begin_params_axis,
    F64Attr:$epsilon
  );
}

def OneFlow_BroadcastSubOp : OneFlow_BaseOp<"broadcast_sub", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
}

def OneFlow_LayerNormGradOp : OneFlow_BaseOp<"layer_norm_grad", []> {
  let input = (ins
    AnyType:$dy,
    AnyType:$x,
    AnyType:$mean,
    AnyType:$inv_variance,
    AnyType:$_add_to_output
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    SI64Attr:$begin_norm_axis,
    F64Attr:$epsilon
  );
}

def OneFlow_AtanhOp : OneFlow_BaseOp<"atanh", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_LayerNormParamGradOp : OneFlow_BaseOp<"layer_norm_param_grad", []> {
  let input = (ins
    AnyType:$dy,
    AnyType:$normalized,
    AnyType:$gamma
  );
  let output = (outs
    AnyType:$normalized_diff,
    AnyType:$beta_diff,
    AnyType:$gamma_diff,
    AnyType:$reduce_buf
  );
  let attrs = (ins
    SI64Attr:$begin_params_axis
  );
}

def OneFlow_ScalarSubByTensorOp : OneFlow_BaseOp<"scalar_sub_by_tensor", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$scalar
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_LogSoftmaxOp : OneFlow_BaseOp<"log_softmax", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$prob
  );
}

def OneFlow_OneRecReaderOp : OneFlow_BaseOp<"OneRecReader", []> {
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    StrArrayAttr:$files,
    SI32Attr:$batch_size,
    BoolAttr:$random_shuffle,
    StrAttr:$shuffle_mode,
    SI64Attr:$seed,
    SI32Attr:$shuffle_buffer_size,
    BoolAttr:$shuffle_after_epoch,
    BoolAttr:$verify_example
  );
}

def OneFlow_MaskedFillOp : OneFlow_BaseOp<"masked_fill", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$mask
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    BoolAttr:$has_int_operand,
    BoolAttr:$has_float_operand,
    SI64Attr:$int_operand,
    F64Attr:$float_operand
  );
}

def OneFlow_CosOp : OneFlow_BaseOp<"cos", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_LogSigmoidGradOp : OneFlow_BaseOp<"log_sigmoid_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_BroadcastDivOp : OneFlow_BaseOp<"broadcast_div", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
}

def OneFlow_L2NormalizeOp : OneFlow_BaseOp<"l2_normalize", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y,
    AnyType:$square_x_sum
  );
  let attrs = (ins
    SI32Attr:$axis,
    F32Attr:$epsilon
  );
}

def OneFlow_BroadcastEqualOp : OneFlow_BaseOp<"broadcast_equal", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
}

def OneFlow_ErfOp : OneFlow_BaseOp<"erf", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_CreateSummaryWriterOp : OneFlow_BaseOp<"create_summary_writer", []> {
  let attrs = (ins
    StrAttr:$logdir
  );
}

def OneFlow_Log1pOp : OneFlow_BaseOp<"log1p", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_ScalarLogicalGreaterOp : OneFlow_BaseOp<"scalar_logical_greater", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    BoolAttr:$has_int_operand,
    BoolAttr:$has_float_operand,
    SI64Attr:$int_operand,
    F64Attr:$float_operand
  );
}

def OneFlow_BroadcastMinimumOp : OneFlow_BaseOp<"broadcast_minimum", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
}

def OneFlow_BroadcastFmodOp : OneFlow_BaseOp<"broadcast_fmod", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
}

def OneFlow_ScalarAddOp : OneFlow_BaseOp<"scalar_add", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    BoolAttr:$has_int_operand,
    BoolAttr:$has_float_operand,
    SI64Attr:$int_operand,
    F64Attr:$float_operand
  );
}

def OneFlow_OnerecDecoderOp : OneFlow_BaseOp<"onerec_decoder", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    StrAttr:$key,
    StrAttr:$data_type,
    AnyI64ElementsAttr:$static_shape,
    BoolAttr:$is_dynamic,
    BoolAttr:$has_reshape,
    AnyI64ElementsAttr:$reshape,
    BoolAttr:$has_batch_padding,
    AnyI64ElementsAttr:$batch_padding
  );
}

def OneFlow_BroadcastPowOp : OneFlow_BaseOp<"broadcast_pow", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
}

def OneFlow_BroadcastNotEqualOp : OneFlow_BaseOp<"broadcast_not_equal", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
}

def OneFlow_ReciprocalOp : OneFlow_InvolutionBaseOp<"reciprocal", []> {}

def OneFlow_BroadcastLessOp : OneFlow_BaseOp<"broadcast_less", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
}

def OneFlow_ImageRandomCropOp : OneFlow_BaseOp<"image_random_crop", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI32Attr:$num_attempts,
    SI64Attr:$seed,
    BoolAttr:$has_seed,
    F32ArrayAttr:$random_area,
    F32ArrayAttr:$random_aspect_ratio
  );
}

def OneFlow_BroadcastLogicalOrOp : OneFlow_BaseOp<"broadcast_logical_or", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
}

def OneFlow_PowOp : OneFlow_BaseOp<"pow", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
}

def OneFlow_CeilOp : OneFlow_IdempotentBaseOp<"ceil", []> {}

def OneFlow_Atan2YGradOp : OneFlow_BaseOp<"atan2_y_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$y,
    AnyType:$dz
  );
  let output = (outs
    AnyType:$dy
  );
}

def OneFlow_AvgPool_2dOp : OneFlow_BaseOp<"avg_pool_2d", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
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

def OneFlow_NormalizationOp : OneFlow_BaseOp<"normalization", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$moving_mean,
    AnyType:$moving_variance,
    AnyType:$gamma,
    AnyType:$beta,
    AnyType:$_add_to_output
  );
  let output = (outs
    AnyType:$y,
    AnyType:$mean,
    AnyType:$inv_variance
  );
  let attrs = (ins
    SI32Attr:$axis,
    F32Attr:$epsilon,
    BoolAttr:$training,
    F32Attr:$momentum
  );
}

def OneFlow_PowXGradOp : OneFlow_BaseOp<"pow_x_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$y,
    AnyType:$dz
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_FloordivOp : OneFlow_BaseOp<"floordiv", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
}

def OneFlow_FusedScaleTrilOp : OneFlow_BaseOp<"fused_scale_tril", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI64Attr:$diagonal,
    F64Attr:$floating_fill_value,
    SI64Attr:$integer_fill_value,
    BoolAttr:$is_floating_fill_value,
    F64Attr:$floating_scale_value,
    SI64Attr:$integer_scale_value,
    BoolAttr:$is_floating_scale_value
  );
}

def OneFlow_FloordivXGradOp : OneFlow_BaseOp<"floordiv_x_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$y,
    AnyType:$dz
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_XdivyYGradOp : OneFlow_BaseOp<"xdivy_y_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$y,
    AnyType:$dz
  );
  let output = (outs
    AnyType:$dy
  );
}

def OneFlow_SamePaddingGradOp : OneFlow_BaseOp<"same_padding_grad", []> {
  let input = (ins
    AnyType:$x_like,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    StrAttr:$padding,
    StrAttr:$data_format,
    SI32ArrayAttr:$kernel_size,
    SI32ArrayAttr:$strides,
    SI32ArrayAttr:$dilation_rate
  );
}

def OneFlow_XlogyYGradOp : OneFlow_BaseOp<"xlogy_y_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$y,
    AnyType:$dz
  );
  let output = (outs
    AnyType:$dy
  );
}

def OneFlow_LgammaOp : OneFlow_BaseOp<"lgamma", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_AcosOp : OneFlow_BaseOp<"acos", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_TensorToTensorBufferOp : OneFlow_BaseOp<"tensor_to_tensor_buffer", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI32Attr:$instance_dims
  );
}

def OneFlow_MaxPool_2dOp : OneFlow_BaseOp<"max_pool_2d", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
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

def OneFlow_AffineGridGradOp : OneFlow_BaseOp<"affine_grid_grad", []> {
  let input = (ins
    AnyType:$dgrid
  );
  let output = (outs
    AnyType:$dtheta
  );
  let attrs = (ins
    AnyI64ElementsAttr:$size,
    BoolAttr:$align_corners
  );
}

def OneFlow_LogSoftmaxGradOp : OneFlow_BaseOp<"log_softmax_grad", []> {
  let input = (ins
    AnyType:$prob,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_AcoshGradOp : OneFlow_BaseOp<"acosh_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_AsinhOp : OneFlow_BaseOp<"asinh", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_AsinGradOp : OneFlow_BaseOp<"asin_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_AtanOp : OneFlow_BaseOp<"atan", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_Conv2dOp : OneFlow_BaseOp<"conv2d", []> {
  let input = (ins
    AnyType:$in,
    AnyType:$weight,
    AnyType:$bias,
    AnyType:$bias_multiplier
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI32Attr:$filters,
    SI32ArrayAttr:$padding_before,
    StrAttr:$data_format,
    SI32ArrayAttr:$kernel_size,
    SI32ArrayAttr:$strides,
    SI32ArrayAttr:$dilation_rate,
    SI32Attr:$groups
  );
}

def OneFlow_ReciprocalNoNanGradOp : OneFlow_BaseOp<"reciprocal_no_nan_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_CeluOp : OneFlow_BaseOp<"celu", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    F64Attr:$alpha
  );
}

def OneFlow_BiasAddOp : OneFlow_BaseOp<"bias_add", []> {
  let input = (ins
    AnyType:$a,
    AnyType:$b
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI32Attr:$axis
  );
}

def OneFlow_XdivyOp : OneFlow_BaseOp<"xdivy", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
}

def OneFlow_AtanGradOp : OneFlow_BaseOp<"atan_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_DimScatterAddScalarOp : OneFlow_BaseOp<"dim_scatter_add_scalar", []> {
  let input = (ins
    AnyType:$input,
    AnyType:$index
  );
  let output = (outs
    AnyType:$output
  );
  let attrs = (ins
    F32Attr:$src_scalar,
    SI32Attr:$dim
  );
}

def OneFlow_AtanhGradOp : OneFlow_BaseOp<"atanh_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_HardsigmoidOp : OneFlow_BaseOp<"hardsigmoid", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_CeilGradOp : OneFlow_BaseOp<"ceil_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_TriuOp : OneFlow_BaseOp<"triu", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI64Attr:$diagonal
  );
}

def OneFlow_RintOp : OneFlow_IdempotentBaseOp<"rint", []> {}

def OneFlow_AdaptiveAvgPool2dOp : OneFlow_BaseOp<"adaptive_avg_pool2d", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    SI64ArrayAttr:$output_size
  );
}

def OneFlow_CoshOp : OneFlow_BaseOp<"cosh", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_UpsampleBilinear_2dGradOp : OneFlow_BaseOp<"upsample_bilinear_2d_grad", []> {
  let input = (ins
    AnyType:$dy,
    AnyType:$x
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    F32Attr:$height_scale,
    F32Attr:$width_scale,
    BoolAttr:$align_corners,
    StrAttr:$data_format
  );
}

def OneFlow_EagerBToSOp : OneFlow_BaseOp<"eager_b_to_s", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI64Attr:$out_split_axis,
    StrAttr:$in_parallel_conf,
    StrAttr:$out_parallel_conf,
    AnyI64ElementsAttr:$shape
  );
}

def OneFlow_ExpOp : OneFlow_BaseOp<"exp", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_NvtxEndOp : OneFlow_BaseOp<"nvtx_end", []> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    StrAttr:$mark_prefix
  );
}

def OneFlow_RoundGradOp : OneFlow_BaseOp<"round_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_WhereScalarXyOp : OneFlow_BaseOp<"where_scalar_xy", []> {
  let input = (ins
    AnyType:$condition
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    BoolAttr:$has_x_int_operand,
    BoolAttr:$has_x_float_operand,
    BoolAttr:$has_y_int_operand,
    BoolAttr:$has_y_float_operand,
    SI64Attr:$x_int_operand,
    F64Attr:$x_float_operand,
    SI64Attr:$y_int_operand,
    F64Attr:$y_float_operand
  );
}

def OneFlow_BinaryCrossEntropyOp : OneFlow_BaseOp<"binary_cross_entropy", []> {
  let input = (ins
    AnyType:$input,
    AnyType:$target,
    AnyType:$weight
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    StrAttr:$reduction
  );
}

def OneFlow_BroadcastMulOp : OneFlow_BaseOp<"broadcast_mul", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
}

def OneFlow_ExpGradOp : OneFlow_BaseOp<"exp_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_LogicalSliceOp : OneFlow_BaseOp<"logical_slice", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    SI64ArrayAttr:$start,
    SI64ArrayAttr:$stop,
    SI64ArrayAttr:$step
  );
}

def OneFlow_ErfcGradOp : OneFlow_BaseOp<"erfc_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_FloorGradOp : OneFlow_BaseOp<"floor_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_ReduceMaxDeviceStageGradOp : OneFlow_BaseOp<"reduce_max_device_stage_grad", []> {
  let input = (ins
    AnyType:$out_diff,
    AnyType:$mask,
    AnyType:$count
  );
  let output = (outs
    AnyType:$in_diff
  );
  let attrs = (ins
    SI32ArrayAttr:$axis
  );
}

def OneFlow_LgammaGradOp : OneFlow_BaseOp<"lgamma_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_Maxpool_3dOp : OneFlow_BaseOp<"maxpool_3d", []> {
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
    BoolAttr:$return_indices,
    BoolAttr:$ceil_mode
  );
}

def OneFlow_LogSigmoidOp : OneFlow_BaseOp<"log_sigmoid", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_RintGradOp : OneFlow_BaseOp<"rint_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_AffineGridOp : OneFlow_BaseOp<"affine_grid", []> {
  let input = (ins
    AnyType:$theta
  );
  let output = (outs
    AnyType:$grid
  );
  let attrs = (ins
    AnyI64ElementsAttr:$size,
    BoolAttr:$align_corners
  );
}

def OneFlow_RsqrtOp : OneFlow_BaseOp<"rsqrt", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_SigmoidV2Op : OneFlow_BaseOp<"sigmoid_v2", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_SignGradOp : OneFlow_BaseOp<"sign_grad", []> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_SinOp : OneFlow_BaseOp<"sin", []> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}


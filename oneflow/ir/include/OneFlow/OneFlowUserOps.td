#ifdef GET_ONEFLOW_ASSIGN_OP_DEFINITIONS

def OneFlow_AssignUserOp : OneFlow_BaseOp<"assign", [NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$ref,
    OneFlow_Tensor:$value
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_AssignIfOp : OneFlow_BaseOp<"assign_if", [NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$ref,
    OneFlow_Tensor:$value,
    OneFlow_Tensor:$condition
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_AssignIfNotOp : OneFlow_BaseOp<"assign_if_not", [NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$ref,
    OneFlow_Tensor:$value,
    OneFlow_Tensor:$condition
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

#endif // GET_ONEFLOW_ASSIGN_OP_DEFINITIONS


#ifdef GET_ONEFLOW_BASE_OP_DEFINITIONS



class OneFlow_NormalizationBaseOp<string mnemonic, list<Trait> traits = []> : OneFlow_BaseOp<mnemonic, !listconcat(traits, [NoMemoryEffect, AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>])> {
  let input = (ins
    OneFlow_Tensor:$x,
    Optional<OneFlow_Tensor>:$moving_mean,
    Optional<OneFlow_Tensor>:$moving_variance,
    OneFlow_Tensor:$gamma,
    OneFlow_Tensor:$beta,
    Optional<OneFlow_Tensor>:$_add_to_output
  );
  let output = (outs
    OneFlow_Tensor:$y,
    Optional<OneFlow_Tensor>:$mean,
    Optional<OneFlow_Tensor>:$inv_variance
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$axis,
    DefaultValuedAttr<F32Attr, "0.">:$epsilon,
    DefaultValuedAttr<BoolAttr, "false">:$training,
    DefaultValuedAttr<F32Attr, "0.">:$momentum
  );
  let trait_attrs = (ins
    DenseI32ArrayAttr:$operand_segment_sizes,
    DenseI32ArrayAttr:$result_segment_sizes
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

#endif // GET_ONEFLOW_BASE_OP_DEFINITIONS


#ifdef GET_ONEFLOW_BINARY_OP_DEFINITIONS

def OneFlow_BiasAddOp : OneFlow_BaseOp<"bias_add", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>, DeclareOpInterfaceMethods<NCHWCompatibleInterface>, DeclareOpInterfaceMethods<BiasAddCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$a,
    OneFlow_Tensor:$b
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$axis
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_CastLikeOp : OneFlow_BaseOp<"cast_like", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in,
    OneFlow_Tensor:$dtype_like
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_CeluGradOp : OneFlow_BaseOp<"celu_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$y,
    OneFlow_Tensor:$dy
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.">:$alpha
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_DiagGradOp : OneFlow_BaseOp<"diag_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$diagonal
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_DiagonalGradOp : OneFlow_BaseOp<"diagonal_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$offset
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_DotOp : OneFlow_BaseOp<"dot", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$y
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_DropoutGradOp : OneFlow_BaseOp<"dropout_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$mask
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$scale
  );
  let has_check_fn = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ElementwiseMaximumOp : OneFlow_BaseOp<"elementwise_maximum", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$y
  );
  let output = (outs
    OneFlow_Tensor:$z
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ElementwiseMinimumOp : OneFlow_BaseOp<"elementwise_minimum", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$y
  );
  let output = (outs
    OneFlow_Tensor:$z
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_EluGradOp : OneFlow_BaseOp<"elu_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$dy
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.">:$alpha
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FloordivOp : OneFlow_BaseOp<"floordiv", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$y
  );
  let output = (outs
    OneFlow_Tensor:$z
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_LerpOp : OneFlow_BaseOp<"lerp", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$start,
    OneFlow_Tensor:$end,
    OneFlow_Tensor:$weight
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_LerpGradOp : OneFlow_BaseOp<"lerp_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$start,
    OneFlow_Tensor:$end,
    OneFlow_Tensor:$weight,
    OneFlow_Tensor:$out_diff
  );
  let output = (outs
    OneFlow_Tensor:$start_diff,
    OneFlow_Tensor:$end_diff,
    OneFlow_Tensor:$weight_diff
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_TruncdivOp : OneFlow_BaseOp<"truncdiv", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$y
  );
  let output = (outs
    OneFlow_Tensor:$z
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_GeluGradOp : OneFlow_BaseOp<"gelu_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
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

def OneFlow_FastGeluGradOp : OneFlow_BaseOp<"fast_gelu_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
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

def OneFlow_QuickGeluGradOp : OneFlow_BaseOp<"quick_gelu_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
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

def OneFlow_GridSampleOp : OneFlow_BaseOp<"grid_sample", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$input,
    OneFlow_Tensor:$grid
  );
  let output = (outs
    OneFlow_Tensor:$output
  );
  let attrs = (ins
    StrAttr:$interpolation_mode,
    StrAttr:$padding_mode,
    DefaultValuedAttr<BoolAttr, "false">:$align_corners
  );
  let has_check_fn = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_HardsigmoidGradOp : OneFlow_BaseOp<"hardsigmoid_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
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

def OneFlow_HardShrinkGradOp : OneFlow_BaseOp<"hardshrink_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$y,
    OneFlow_Tensor:$dy
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.">:$lambd
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_HardswishGradOp : OneFlow_BaseOp<"hardswish_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
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

def OneFlow_L1L2RegularizeGradientOp : OneFlow_BaseOp<"l1_l2_regularize_gradient", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$model,
    OneFlow_Tensor:$model_diff
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$l1,
    DefaultValuedAttr<F32Attr, "0.">:$l2
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_LeakyReluGradOp : OneFlow_BaseOp<"leaky_relu_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$dy
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$alpha
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_MaskedFillOp : OneFlow_BaseOp<"masked_fill", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$mask
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$has_int_operand,
    DefaultValuedAttr<BoolAttr, "false">:$has_float_operand,
    DefaultValuedAttr<BoolAttr, "false">:$has_bool_operand,
    DefaultValuedAttr<SI64Attr, "0">:$int_operand,
    DefaultValuedAttr<F64Attr, "0.">:$float_operand,
    DefaultValuedAttr<BoolAttr, "0.">:$bool_operand
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_MishGradOp : OneFlow_BaseOp<"mish_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
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

def OneFlow_NarrowGradOp : OneFlow_BaseOp<"narrow_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$like
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$dim,
    DefaultValuedAttr<SI64Attr, "0">:$start,
    DefaultValuedAttr<SI64Attr, "0">:$length
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_PowOp : OneFlow_BaseOp<"pow", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$y
  );
  let output = (outs
    OneFlow_Tensor:$z
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FracOp : OneFlow_BaseOp<"frac", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_PreluOp : OneFlow_BaseOp<"prelu", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$alpha
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ReluGradOp : OneFlow_BaseOp<"relu_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$y,
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

def OneFlow_SeluGradOp : OneFlow_BaseOp<"selu_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
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

def OneFlow_SiluGradOp : OneFlow_BaseOp<"silu_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
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

def OneFlow_ThresholdGradOp : OneFlow_BaseOp<"threshold_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$dy
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.">:$threshold_val
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_SoftShrinkGradOp : OneFlow_BaseOp<"softshrink_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$y,
    OneFlow_Tensor:$dy
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.">:$alpha
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_TfPreluOp : OneFlow_BaseOp<"tf_prelu", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$alpha
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_UnfoldTensorGradOp : OneFlow_BaseOp<"unfold_tensor_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$dimension,
    DefaultValuedAttr<SI32Attr, "0">:$size,
    DefaultValuedAttr<SI32Attr, "0">:$step
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_XdivyOp : OneFlow_BaseOp<"xdivy", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$y
  );
  let output = (outs
    OneFlow_Tensor:$z
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_XlogyOp : OneFlow_BaseOp<"xlogy", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$y
  );
  let output = (outs
    OneFlow_Tensor:$z
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_BroadcastZetaOp : OneFlow_BaseOp<"broadcast_zeta", [NoGrad,NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$y
  );
  let output = (outs
    OneFlow_Tensor:$z
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

#endif // GET_ONEFLOW_BINARY_OP_DEFINITIONS


#ifdef GET_ONEFLOW_BROADCAST_OP_DEFINITIONS

def OneFlow_BroadcastAddOp : OneFlow_BaseOp<"broadcast_add", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>, DeclareOpInterfaceMethods<NCHWCompatibleInterface>, DeclareOpInterfaceMethods<BiasAddCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$y
  );
  let output = (outs
    OneFlow_Tensor:$z
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_BroadcastDivOp : OneFlow_BaseOp<"broadcast_div", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$y
  );
  let output = (outs
    OneFlow_Tensor:$z
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let hasFolder = 1;
}

def OneFlow_BroadcastDivGradOp : OneFlow_BaseOp<"broadcast_div_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$y,
    OneFlow_Tensor:$z,
    OneFlow_Tensor:$dz
  );
  let output = (outs
    OneFlow_Tensor:$dy
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_BroadcastEqualOp : OneFlow_BaseOp<"broadcast_equal", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$y
  );
  let output = (outs
    OneFlow_Tensor:$z
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_BroadcastFloorModOp : OneFlow_BaseOp<"broadcast_floor_mod", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$y
  );
  let output = (outs
    OneFlow_Tensor:$z
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_BroadcastFmodOp : OneFlow_BaseOp<"broadcast_fmod", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$y
  );
  let output = (outs
    OneFlow_Tensor:$z
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_BroadcastGreaterOp : OneFlow_BaseOp<"broadcast_greater", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$y
  );
  let output = (outs
    OneFlow_Tensor:$z
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_BroadCastInplaceGreaterOp : OneFlow_BaseOp<"broadcast_inplace_greater", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$y
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_BroadcastGreaterEqualOp : OneFlow_BaseOp<"broadcast_greater_equal", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$y
  );
  let output = (outs
    OneFlow_Tensor:$z
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_BroadcastLessOp : OneFlow_BaseOp<"broadcast_less", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$y
  );
  let output = (outs
    OneFlow_Tensor:$z
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_BroadcastLessEqualOp : OneFlow_BaseOp<"broadcast_less_equal", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$y
  );
  let output = (outs
    OneFlow_Tensor:$z
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_BroadcastIsCloseEqualNanOp : OneFlow_BaseOp<"broadcast_isclose_eq_nan", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$y
  );
  let output = (outs
    OneFlow_Tensor:$z
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "1e-08">:$atol,
    DefaultValuedAttr<F32Attr, "1e-05">:$rtol
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_BroadcastIsCloseNotEqualNanOp : OneFlow_BaseOp<"broadcast_isclose_neq_nan", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$y
  );
  let output = (outs
    OneFlow_Tensor:$z
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "1e-08">:$atol,
    DefaultValuedAttr<F32Attr, "1e-05">:$rtol
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_BroadcastLikeOp : OneFlow_BaseOp<"broadcast_like", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$like
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let attrs = (ins
    SI32ArrayAttr:$broadcast_axes
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_BroadcastLogicalAndOp : OneFlow_BaseOp<"broadcast_logical_and", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$y
  );
  let output = (outs
    OneFlow_Tensor:$z
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_BroadcastLogicalOrOp : OneFlow_BaseOp<"broadcast_logical_or", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$y
  );
  let output = (outs
    OneFlow_Tensor:$z
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_BroadcastLogicalXorOp : OneFlow_BaseOp<"broadcast_logical_xor", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$y
  );
  let output = (outs
    OneFlow_Tensor:$z
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_BroadcastMaximumOp : OneFlow_BaseOp<"broadcast_maximum", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$y
  );
  let output = (outs
    OneFlow_Tensor:$z
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_BroadcastMinimumOp : OneFlow_BaseOp<"broadcast_minimum", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$y
  );
  let output = (outs
    OneFlow_Tensor:$z
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_BroadcastMulOp : OneFlow_BaseOp<"broadcast_mul", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$y
  );
  let output = (outs
    OneFlow_Tensor:$z
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let hasFolder = 1;
}

def OneFlow_BroadcastNotEqualOp : OneFlow_BaseOp<"broadcast_not_equal", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$y
  );
  let output = (outs
    OneFlow_Tensor:$z
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_BroadcastPowOp : OneFlow_BaseOp<"broadcast_pow", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$y
  );
  let output = (outs
    OneFlow_Tensor:$z
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_BroadcastSubOp : OneFlow_BaseOp<"broadcast_sub", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$y
  );
  let output = (outs
    OneFlow_Tensor:$z
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let hasFolder = 1;
}

def OneFlow_BitwiseNotOp : OneFlow_BaseOp<"bitwise_not", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_BroadcastBitwiseAndOp : OneFlow_BaseOp<"broadcast_bitwise_and", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$y
  );
  let output = (outs
    OneFlow_Tensor:$z
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_BroadcastBitwiseOrOp : OneFlow_BaseOp<"broadcast_bitwise_or", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$y
  );
  let output = (outs
    OneFlow_Tensor:$z
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_BroadcastBitwiseXorOp : OneFlow_BaseOp<"broadcast_bitwise_xor", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$y
  );
  let output = (outs
    OneFlow_Tensor:$z
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

#endif // GET_ONEFLOW_BROADCAST_OP_DEFINITIONS


#ifdef GET_ONEFLOW_CONV_OP_DEFINITIONS

def OneFlow_Conv1DOp : OneFlow_ConvolutionBaseOp<"conv1d", [NoMemoryEffect, AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_Conv2DOp : OneFlow_ConvolutionBaseOp<"conv2d", [NoMemoryEffect, AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>, DeclareOpInterfaceMethods<NCHWCompatibleInterface>]> {}

def OneFlow_Conv3DOp : OneFlow_ConvolutionBaseOp<"conv3d", [NoMemoryEffect, AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_ConvBiasGradOp : OneFlow_BaseOp<"conv_bias_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy
  );
  let output = (outs
    OneFlow_Tensor:$bias_diff
  );
  let attrs = (ins
    StrAttr:$data_format,
    DefaultValuedAttr<SI32Attr, "0">:$num_spatial_dims
  );
  let has_check_fn = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_compute_complexity_fn = 1;
}

def OneFlow_ConvDataGradOp : OneFlow_BaseOp<"conv_data_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$filter,
    OneFlow_Tensor:$x_like,
    Optional<OneFlow_Tensor>:$_add_to_output
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$num_spatial_dims,
    SI32ArrayAttr:$padding_before,
    StrAttr:$data_format,
    SI32ArrayAttr:$kernel_size,
    SI32ArrayAttr:$strides,
    SI32ArrayAttr:$dilation_rate,
    DefaultValuedAttr<SI32Attr, "0">:$groups
  );
  let has_check_fn = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_compute_complexity_fn = 1;
}

def OneFlow_ConvFilterGradOp : OneFlow_BaseOp<"conv_filter_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$filter_diff
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$num_spatial_dims,
    SI32ArrayAttr:$padding_before,
    StrAttr:$data_format,
    SI32ArrayAttr:$kernel_size,
    SI32ArrayAttr:$strides,
    SI32ArrayAttr:$dilation_rate,
    DefaultValuedAttr<SI32Attr, "0">:$groups
  );
  let has_check_fn = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_compute_complexity_fn = 1;
}

def OneFlow_Deconv1DOp : OneFlow_BaseOp<"deconv1d", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in,
    OneFlow_Tensor:$weight
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$filters,
    SI32ArrayAttr:$padding_before,
    StrAttr:$data_format,
    SI32ArrayAttr:$kernel_size,
    SI32ArrayAttr:$output_padding,
    SI32ArrayAttr:$strides,
    SI32ArrayAttr:$dilation_rate,
    DefaultValuedAttr<SI32Attr, "1">:$groups
  );
  let has_check_fn = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_Deconv2DOp : OneFlow_BaseOp<"deconv2d", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in,
    OneFlow_Tensor:$weight
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$filters,
    SI32ArrayAttr:$padding_before,
    StrAttr:$data_format,
    SI32ArrayAttr:$kernel_size,
    SI32ArrayAttr:$output_padding,
    SI32ArrayAttr:$strides,
    SI32ArrayAttr:$dilation_rate,
    DefaultValuedAttr<SI32Attr, "1">:$groups
  );
  let has_check_fn = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_Deconv3DOp : OneFlow_BaseOp<"deconv3d", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in,
    OneFlow_Tensor:$weight
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$filters,
    SI32ArrayAttr:$padding_before,
    StrAttr:$data_format,
    SI32ArrayAttr:$kernel_size,
    SI32ArrayAttr:$output_padding,
    SI32ArrayAttr:$strides,
    SI32ArrayAttr:$dilation_rate,
    DefaultValuedAttr<SI32Attr, "1">:$groups
  );
  let has_check_fn = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}
def OneFlow_DeformConv2dOp : OneFlow_BaseOp<"deform_conv2d", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$input,
    OneFlow_Tensor:$offset,
    OneFlow_Tensor:$weight,
    Optional<OneFlow_Tensor>:$bias,
    OneFlow_Tensor:$mask
  );
  let output = (outs
    OneFlow_Tensor:$out
  );

  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$stride_h,
    DefaultValuedAttr<SI32Attr, "0">:$stride_w,
    DefaultValuedAttr<SI32Attr, "0">:$pad_h,
    DefaultValuedAttr<SI32Attr, "0">:$pad_w,
    DefaultValuedAttr<SI32Attr, "0">:$dilation_h,
    DefaultValuedAttr<SI32Attr, "0">:$dilation_w,
    DefaultValuedAttr<SI32Attr, "0">:$groups,
    DefaultValuedAttr<SI32Attr, "0">:$offset_groups,
    DefaultValuedAttr<BoolAttr, "false">:$use_mask
  );

  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_DeformConv2dInputGradOp : OneFlow_BaseOp<"deform_conv2d_input_grad", [NoMemoryEffect,
DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$output_grad,
    OneFlow_Tensor:$input,
    OneFlow_Tensor:$offset,
    OneFlow_Tensor:$weight,
    OneFlow_Tensor:$mask
  );
  let output = (outs
    OneFlow_Tensor:$input_grad,
    //OneFlow_Tensor:$weight_grad,
    OneFlow_Tensor:$offset_grad,
    OneFlow_Tensor:$mask_grad
  );

  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$stride_h,
    DefaultValuedAttr<SI32Attr, "0">:$stride_w,
    DefaultValuedAttr<SI32Attr, "0">:$pad_h,
    DefaultValuedAttr<SI32Attr, "0">:$pad_w,
    DefaultValuedAttr<SI32Attr, "0">:$dilation_h,
    DefaultValuedAttr<SI32Attr, "0">:$dilation_w,
    DefaultValuedAttr<SI32Attr, "0">:$groups,
    DefaultValuedAttr<SI32Attr, "0">:$offset_groups,
    DefaultValuedAttr<BoolAttr, "false">:$use_mask
  );

  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_DeformConv2dParamGradOp : OneFlow_BaseOp<"deform_conv2d_param_grad", [NoMemoryEffect,
DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$output_grad,
    OneFlow_Tensor:$input,
    OneFlow_Tensor:$offset,
    OneFlow_Tensor:$weight,
    OneFlow_Tensor:$mask
  );
  let output = (outs
    OneFlow_Tensor:$weight_grad
  );

  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$stride_h,
    DefaultValuedAttr<SI32Attr, "0">:$stride_w,
    DefaultValuedAttr<SI32Attr, "0">:$pad_h,
    DefaultValuedAttr<SI32Attr, "0">:$pad_w,
    DefaultValuedAttr<SI32Attr, "0">:$dilation_h,
    DefaultValuedAttr<SI32Attr, "0">:$dilation_w,
    DefaultValuedAttr<SI32Attr, "0">:$groups,
    DefaultValuedAttr<SI32Attr, "0">:$offset_groups,
    DefaultValuedAttr<BoolAttr, "false">:$use_mask
  );

  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

#endif // GET_ONEFLOW_CONV_OP_DEFINITIONS


#ifdef GET_ONEFLOW_CROSS_ENTROPY_OP_DEFINITIONS

def OneFlow_BinaryCrossEntropyOp : OneFlow_BaseOp<"binary_cross_entropy", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$input,
    OneFlow_Tensor:$target,
    Optional<OneFlow_Tensor>:$weight
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_BinaryCrossEntropyGradOp : OneFlow_BaseOp<"binary_cross_entropy_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$input,
    OneFlow_Tensor:$target,
    Optional<OneFlow_Tensor>:$weight,
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

def OneFlow_BinaryCrossEntropyWithLogitsOp : OneFlow_BaseOp<"binary_cross_entropy_with_logits", [NoMemoryEffect, AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$input,
    OneFlow_Tensor:$target,
    Optional<OneFlow_Tensor>:$weight,
    Optional<OneFlow_Tensor>:$pos_weight
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$has_pos_weight
  );
  let trait_attrs = (ins
    DenseI32ArrayAttr:$operand_segment_sizes
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_BinaryCrossEntropyWithLogitsGradOp : OneFlow_BaseOp<"binary_cross_entropy_with_logits_grad", [NoMemoryEffect, AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$input,
    OneFlow_Tensor:$target,
    Optional<OneFlow_Tensor>:$weight,
    Optional<OneFlow_Tensor>:$pos_weight,
    OneFlow_Tensor:$dy
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$has_pos_weight
  );
  let trait_attrs = (ins
    DenseI32ArrayAttr:$operand_segment_sizes
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_BinaryCrossEntropyWithLogitsReduceMeanOp : OneFlow_BaseOp<"binary_cross_entropy_with_logits_reduce_mean", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$input,
    OneFlow_Tensor:$target
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_BinaryCrossEntropyWithLogitsReduceMeanGradOp : OneFlow_BaseOp<"binary_cross_entropy_with_logits_reduce_mean_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$input,
    OneFlow_Tensor:$target,
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

def OneFlow_FusedBCEReduceMeanFwBwOp : OneFlow_BaseOp<"fused_bce_reduce_mean_fw_bw", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$input,
    OneFlow_Tensor:$target
  );
  let output = (outs
    OneFlow_Tensor:$out,
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    OneFlow_DataType:$out_dtype,
    DefaultValuedAttr<F64Attr, "0.">:$constant_value
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_SigmoidCrossEntropyOp : OneFlow_BaseOp<"sigmoid_cross_entropy", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$prediction,
    OneFlow_Tensor:$label
  );
  let output = (outs
    OneFlow_Tensor:$loss
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_SigmoidCrossEntropyGradOp : OneFlow_BaseOp<"sigmoid_cross_entropy_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$prediction,
    OneFlow_Tensor:$loss_diff,
    OneFlow_Tensor:$label
  );
  let output = (outs
    OneFlow_Tensor:$prediction_diff
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_SparseCrossEntropyOp : OneFlow_BaseOp<"sparse_cross_entropy", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$prediction,
    OneFlow_Tensor:$label
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$depth
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_SparseCrossEntropyGradOp : OneFlow_BaseOp<"sparse_cross_entropy_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$prediction,
    OneFlow_Tensor:$label,
    OneFlow_Tensor:$dy
  );
  let output = (outs
    OneFlow_Tensor:$prediction_diff
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$depth
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_SparseCrossEntropyMsOp : OneFlow_BaseOp<"sparse_cross_entropy_ms", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$prediction,
    OneFlow_Tensor:$label
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$depth
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_SparseCrossEntropyMsGradOp : OneFlow_BaseOp<"sparse_cross_entropy_ms_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$prediction,
    OneFlow_Tensor:$label,
    OneFlow_Tensor:$dy
  );
  let output = (outs
    OneFlow_Tensor:$prediction_diff
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$depth
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

#endif // GET_ONEFLOW_CROSS_ENTROPY_OP_DEFINITIONS


#ifdef GET_ONEFLOW_CUDA_OP_DEFINITIONS

def OneFlow_NvtxEndOp : OneFlow_BaseOp<"nvtx_end", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    StrAttr:$mark_prefix
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_NvtxStartOp : OneFlow_BaseOp<"nvtx_start", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    StrAttr:$mark_prefix
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

#endif // GET_ONEFLOW_CUDA_OP_DEFINITIONS


#ifdef GET_ONEFLOW_DATASET_OP_DEFINITIONS

def OneFlow_COCOReaderOp : OneFlow_BaseOp<"COCOReader", [NoMemoryEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let output = (outs
    OneFlow_TensorBufferTensor:$image,
    OneFlow_Tensor:$image_id,
    OneFlow_Tensor:$image_size,
    OneFlow_TensorBufferTensor:$gt_bbox,
    OneFlow_TensorBufferTensor:$gt_label,
    OneFlow_TensorBufferTensor:$gt_segm,
    OneFlow_TensorBufferTensor:$gt_segm_index
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$session_id,
    StrAttr:$annotation_file,
    StrAttr:$image_dir,
    DefaultValuedAttr<SI64Attr, "0">:$batch_size,
    DefaultValuedAttr<BoolAttr, "true">:$shuffle_after_epoch,
    DefaultValuedAttr<SI64Attr, "-1">:$random_seed,
    DefaultValuedAttr<BoolAttr, "true">:$group_by_ratio,
    DefaultValuedAttr<BoolAttr, "true">:$remove_images_without_annotations,
    DefaultValuedAttr<BoolAttr, "false">:$stride_partition,
    StrArrayAttr:$nd_sbp
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_output_arg_modify_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

def OneFlow_OFRecordReaderOp : OneFlow_BaseOp<"OFRecordReader", [NoMemoryEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    StrAttr:$data_dir,
    DefaultValuedAttr<SI32Attr, "0">:$data_part_num,
    DefaultValuedAttr<SI32Attr, "0">:$batch_size,
    DefaultValuedAttr<StrAttr, "\"part-\"">:$part_name_prefix,
    DefaultValuedAttr<SI32Attr, "-1">:$part_name_suffix_length,
    DefaultValuedAttr<BoolAttr, "false">:$random_shuffle,
    DefaultValuedAttr<SI64Attr, "-1">:$seed,
    DefaultValuedAttr<SI32Attr, "1024">:$shuffle_buffer_size,
    DefaultValuedAttr<BoolAttr, "false">:$shuffle_after_epoch,
    StrArrayAttr:$nd_sbp
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_output_arg_modify_fn = 1;
  let has_nd_sbp_infer_fn = 1;
  let has_get_nd_sbp_fn = 1;
  let has_compute_complexity_fn = 1;
}

def OneFlow_CtcGreedyDecoderOp : OneFlow_BaseOp<"ctc_greedy_decoder", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$log_probs,
    OneFlow_Tensor:$input_lengths
  );
  let output = (outs
    OneFlow_Tensor:$decoded,
    OneFlow_Tensor:$neg_sum_logits
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$merge_repeated
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_MegatronGptMmapDataLoaderOp : OneFlow_BaseOp<"megatron_gpt_mmap_data_loader", [NoMemoryEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    Optional<OneFlow_Tensor>:$iteration
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    StrAttr:$data_file_prefix,
    DefaultValuedAttr<SI64Attr, "0">:$seq_length,
    DefaultValuedAttr<SI64Attr, "1">:$label_length,
    DefaultValuedAttr<SI64Attr, "0">:$num_samples,
    DefaultValuedAttr<SI64Attr, "0">:$batch_size,
    OneFlow_DataType:$dtype,
    SI64ArrayAttr:$split_sizes,
    DefaultValuedAttr<SI64Attr, "0">:$split_index,
    DefaultValuedAttr<BoolAttr, "false">:$shuffle,
    DefaultValuedAttr<SI64Attr, "0">:$random_seed,
    StrArrayAttr:$nd_sbp
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

def OneFlow_OfrecordBytesDecoderOp : OneFlow_BaseOp<"ofrecord_bytes_decoder", [NoMemoryEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    StrAttr:$name
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_OfrecordImageClassificationReaderOp : OneFlow_BaseOp<"ofrecord_image_classification_reader", [NoMemoryEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let output = (outs
    OneFlow_Tensor:$image,
    OneFlow_Tensor:$label
  );
  let attrs = (ins
    StrAttr:$data_dir,
    DefaultValuedAttr<SI32Attr, "0">:$data_part_num,
    DefaultValuedAttr<SI32Attr, "0">:$batch_size,
    DefaultValuedAttr<StrAttr, "\"part-\"">:$part_name_prefix,
    DefaultValuedAttr<SI32Attr, "-1">:$part_name_suffix_length,
    DefaultValuedAttr<BoolAttr, "false">:$random_shuffle,
    DefaultValuedAttr<SI64Attr, "-1">:$seed,
    DefaultValuedAttr<SI32Attr, "1024">:$shuffle_buffer_size,
    DefaultValuedAttr<BoolAttr, "false">:$shuffle_after_epoch,
    DefaultValuedAttr<StrAttr, "\"BGR\"">:$color_space,
    DefaultValuedAttr<StrAttr, "\"encoded\"">:$image_feature_name,
    DefaultValuedAttr<StrAttr, "\"class/label\"">:$label_feature_name,
    DefaultValuedAttr<SI32Attr, "8">:$decode_buffer_size_per_thread,
    DefaultValuedAttr<SI32Attr, "0">:$num_decode_threads_per_machine
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_output_arg_modify_fn = 1;
}

def OneFlow_OfrecordImageDecoderOp : OneFlow_BaseOp<"ofrecord_image_decoder", [NoMemoryEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    StrAttr:$name,
    DefaultValuedAttr<StrAttr, "\"BGR\"">:$color_space
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_OfrecordImageDecoderRandomCropOp : OneFlow_BaseOp<"ofrecord_image_decoder_random_crop", [NoMemoryEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    StrAttr:$name,
    DefaultValuedAttr<StrAttr, "\"BGR\"">:$color_space,
    DefaultValuedAttr<SI32Attr, "10">:$num_attempts,
    DefaultValuedAttr<SI64Attr, "-1">:$seed,
    DefaultValuedAttr<BoolAttr, "false">:$has_seed,
    F32ArrayAttr:$random_area,
    F32ArrayAttr:$random_aspect_ratio
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_OfrecordRawDecoderOp : OneFlow_BaseOp<"ofrecord_raw_decoder", [NoMemoryEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    StrAttr:$name,
    ShapeAttr:$shape,
    OneFlow_DataType:$data_type,
    DefaultValuedAttr<BoolAttr, "false">:$dim1_varying_length,
    DefaultValuedAttr<BoolAttr, "false">:$truncate
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_RawReaderOp : OneFlow_BaseOp<"raw_reader", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    StrArrayAttr:$files,
    OneFlow_DataType:$data_type,
    ShapeAttr:$shape,
    SI64Attr:$batch_size,
    SI64Attr:$shuffle_block_size,
    DefaultValuedAttr<BoolAttr, "true">:$random_shuffle,
    DefaultValuedAttr<SI64Attr, "0">:$seed,
    StrArrayAttr:$nd_sbp
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

#endif // GET_ONEFLOW_DATASET_OP_DEFINITIONS


#ifdef GET_ONEFLOW_DETECTION_OP_DEFINITIONS

def OneFlow_InTopKOp : OneFlow_BaseOp<"in_top_k", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$targets,
    OneFlow_Tensor:$predictions
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$k
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_NmsOp : OneFlow_BaseOp<"nms", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$iou_threshold,
    DefaultValuedAttr<SI32Attr, "0">:$keep_n
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ObjectBboxFlipOp : OneFlow_BaseOp<"object_bbox_flip", [NoMemoryEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$bbox,
    OneFlow_Tensor:$image_size,
    OneFlow_Tensor:$flip_code
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ObjectBboxScaleOp : OneFlow_BaseOp<"object_bbox_scale", [NoMemoryEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$bbox,
    OneFlow_Tensor:$scale
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ObjectSegmentationPolygonFlipOp : OneFlow_BaseOp<"object_segmentation_polygon_flip", [NoMemoryEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$poly,
    OneFlow_Tensor:$image_size,
    OneFlow_Tensor:$flip_code
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ObjectSegmentationPolygonScaleOp : OneFlow_BaseOp<"object_segmentation_polygon_scale", [NoMemoryEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$poly,
    OneFlow_Tensor:$scale
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ObjectSegmentationPolygonToMaskOp : OneFlow_BaseOp<"object_segmentation_polygon_to_mask", [NoMemoryEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$poly,
    OneFlow_Tensor:$poly_index,
    OneFlow_Tensor:$image_size
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_RoiAlignOp : OneFlow_BaseOp<"roi_align", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$rois
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$pooled_h,
    DefaultValuedAttr<SI32Attr, "0">:$pooled_w,
    DefaultValuedAttr<F32Attr, "0.">:$spatial_scale,
    DefaultValuedAttr<SI32Attr, "0">:$sampling_ratio,
    DefaultValuedAttr<BoolAttr, "false">:$aligned
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_RoiAlignGradOp : OneFlow_BaseOp<"roi_align_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$x_like,
    OneFlow_Tensor:$rois
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$pooled_h,
    DefaultValuedAttr<SI32Attr, "0">:$pooled_w,
    DefaultValuedAttr<F32Attr, "0.">:$spatial_scale,
    DefaultValuedAttr<SI32Attr, "0">:$sampling_ratio,
    DefaultValuedAttr<BoolAttr, "false">:$aligned
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_TopKOp : OneFlow_BaseOp<"top_k", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$k,
    DefaultValuedAttr<BoolAttr, "false">:$sorted
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

#endif // GET_ONEFLOW_DETECTION_OP_DEFINITIONS


#ifdef GET_ONEFLOW_EAGER_OP_DEFINITIONS

def OneFlow_EagerBToSOp : OneFlow_BaseOp<"eager_b_to_s", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "-1">:$out_split_axis,
    StrAttr:$in_parallel_conf,
    StrAttr:$out_parallel_conf,
    ShapeAttr:$shape
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_device_and_stream_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

def OneFlow_EagerNaiveSToSOp : OneFlow_BaseOp<"eager_naive_s_to_s", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "-1">:$in_split_axis,
    DefaultValuedAttr<SI64Attr, "-1">:$out_split_axis,
    StrAttr:$in_parallel_conf,
    StrAttr:$out_parallel_conf,
    ShapeAttr:$shape
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_device_and_stream_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

def OneFlow_EagerCclAllGatherOp : OneFlow_BaseOp<"eager_ccl_all_gather", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    StrAttr:$parallel_conf,
    ShapeAttr:$output_shape,
    OneFlow_DataType:$output_dtype
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_device_and_stream_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

def OneFlow_EagerCclAllReduceOp : OneFlow_BaseOp<"eager_ccl_all_reduce", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    StrAttr:$parallel_conf
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_device_and_stream_infer_fn = 1;
}

def OneFlow_EagerCclBroadcastOp : OneFlow_BaseOp<"eager_ccl_broadcast", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    StrAttr:$parallel_conf,
    ShapeArrayAttr:$shape_list,
    DefaultValuedAttr<SI64Attr, "0">:$root,
    DefaultValuedAttr<BoolAttr, "true">:$async_launch
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_device_and_stream_infer_fn = 1;
}

def OneFlow_EagerCclTouchOp : OneFlow_BaseOp<"eager_ccl_touch", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    Variadic<OneFlow_Tensor>:$in
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "true">:$async_launch
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_device_and_stream_infer_fn = 1;
}

def OneFlow_EagerCclReduceOp : OneFlow_BaseOp<"eager_ccl_reduce", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    StrAttr:$parallel_conf,
    DefaultValuedAttr<SI64Attr, "0">:$root
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_device_and_stream_infer_fn = 1;
}

def OneFlow_EagerCclReduceScatterOp : OneFlow_BaseOp<"eager_ccl_reduce_scatter", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    StrAttr:$parallel_conf,
    ShapeAttr:$output_shape,
    OneFlow_DataType:$output_dtype,
    DefaultValuedAttr<StrAttr, "\"sum\"">:$op_type
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_device_and_stream_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

def OneFlow_EagerCclS2SOp : OneFlow_BaseOp<"eager_ccl_s2s", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "-1">:$in_split_axis,
    DefaultValuedAttr<SI64Attr, "-1">:$out_split_axis,
    StrAttr:$parallel_conf
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_device_and_stream_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

def OneFlow_EagerPToBOp : OneFlow_BaseOp<"eager_p_to_b", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    StrAttr:$in_parallel_conf,
    StrAttr:$out_parallel_conf,
    ShapeAttr:$shape
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_device_and_stream_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

def OneFlow_EagerPToSOp : OneFlow_BaseOp<"eager_p_to_s", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "-1">:$out_split_axis,
    StrAttr:$in_parallel_conf,
    StrAttr:$out_parallel_conf,
    ShapeAttr:$shape
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_device_and_stream_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

def OneFlow_EagerSToBOp : OneFlow_BaseOp<"eager_s_to_b", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "-1">:$in_split_axis,
    StrAttr:$in_parallel_conf,
    StrAttr:$out_parallel_conf,
    ShapeAttr:$shape
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_device_and_stream_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

def OneFlow_EagerSToPOp : OneFlow_BaseOp<"eager_s_to_p", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "-1">:$in_split_axis,
    StrAttr:$in_parallel_conf,
    StrAttr:$out_parallel_conf,
    ShapeAttr:$shape
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_device_and_stream_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

def OneFlow_EagerSymmetricSToPOp : OneFlow_BaseOp<"eager_symmetric_s_to_p", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "-1">:$in_split_axis,
    StrAttr:$parallel_conf
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_device_and_stream_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

#endif // GET_ONEFLOW_EAGER_OP_DEFINITIONS


#ifdef GET_ONEFLOW_FUSED_OP_DEFINITIONS

def OneFlow_FusedLstmCellOp : OneFlow_BaseOp<"fused_lstm_cell", [NoMemoryEffect, AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$input_gates,
    OneFlow_Tensor:$hidden_gates,
    OneFlow_Tensor:$cx,
    Optional<OneFlow_Tensor>:$input_bias,
    Optional<OneFlow_Tensor>:$hidden_bias
  );
  let output = (outs
    OneFlow_Tensor:$hy,
    OneFlow_Tensor:$cy,
    OneFlow_Tensor:$workspace
  );
  let trait_attrs = (ins
    DenseI32ArrayAttr:$operand_segment_sizes
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FusedLstmCellGradOp : OneFlow_BaseOp<"fused_lstm_cell_grad", [NoMemoryEffect, AttrSizedResultSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$grad_hy,
    OneFlow_Tensor:$grad_cy,
    OneFlow_Tensor:$cx,
    OneFlow_Tensor:$cy,
    OneFlow_Tensor:$workspace
  );
  let output = (outs
    OneFlow_Tensor:$grad_gates,
    Optional<OneFlow_Tensor>:$grad_cx,
    Optional<OneFlow_Tensor>:$grad_bias
  );
  let trait_attrs = (ins
    DenseI32ArrayAttr:$result_segment_sizes
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FusedGruCellOp : OneFlow_BaseOp<"fused_gru_cell", [NoMemoryEffect, AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$input_gates,
    OneFlow_Tensor:$hidden_gates,
    OneFlow_Tensor:$hx,
    Optional<OneFlow_Tensor>:$input_bias,
    Optional<OneFlow_Tensor>:$hidden_bias
  );
  let output = (outs
    OneFlow_Tensor:$hy,
    OneFlow_Tensor:$workspace
  );
  let trait_attrs = (ins
    DenseI32ArrayAttr:$operand_segment_sizes
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FusedGruCellGradOp : OneFlow_BaseOp<"fused_gru_cell_grad", [NoMemoryEffect, AttrSizedResultSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$grad_hy,
    OneFlow_Tensor:$workspace
  );
  let output = (outs
    OneFlow_Tensor:$grad_input_gates,
    OneFlow_Tensor:$grad_hidden_gates,
    Optional<OneFlow_Tensor>:$grad_hx,
    Optional<OneFlow_Tensor>:$grad_input_bias,
    Optional<OneFlow_Tensor>:$grad_hidden_bias
  );
  let trait_attrs = (ins
    DenseI32ArrayAttr:$result_segment_sizes
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_CudnnFusedNormalizationAddReluOp : OneFlow_BaseOp<"cudnn_fused_normalization_add_relu", [NoMemoryEffect, AttrSizedOperandSegments, AttrSizedResultSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    Optional<OneFlow_Tensor>:$addend,
    Optional<OneFlow_Tensor>:$moving_mean,
    Optional<OneFlow_Tensor>:$moving_variance,
    OneFlow_Tensor:$gamma,
    OneFlow_Tensor:$beta
  );
  let output = (outs
    OneFlow_Tensor:$y,
    OneFlow_Tensor:$reserve_space,
    Optional<OneFlow_Tensor>:$mean,
    Optional<OneFlow_Tensor>:$inv_variance
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$axis,
    DefaultValuedAttr<F32Attr, "0.">:$epsilon,
    DefaultValuedAttr<F32Attr, "0.">:$momentum
  );
  let trait_attrs = (ins
    DenseI32ArrayAttr:$operand_segment_sizes,
    DenseI32ArrayAttr:$result_segment_sizes
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_CudnnFusedNormalizationAddReluGradOp : OneFlow_BaseOp<"cudnn_fused_normalization_add_relu_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$mean,
    OneFlow_Tensor:$inv_variance,
    OneFlow_Tensor:$gamma,
    OneFlow_Tensor:$beta,
    OneFlow_Tensor:$reserve_space,
    OneFlow_Tensor:$y
  );
  let output = (outs
    OneFlow_Tensor:$gamma_diff,
    OneFlow_Tensor:$beta_diff,
    OneFlow_Tensor:$dx,
    Optional<OneFlow_Tensor>:$addend_diff
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$axis,
    DefaultValuedAttr<F32Attr, "0.">:$epsilon
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FusedBiasAddGeluOp : OneFlow_BaseOp<"fused_bias_add_gelu", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$a,
    OneFlow_Tensor:$b
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$axis
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FusedBiasAddGeluGradOp : OneFlow_BaseOp<"fused_bias_add_gelu_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$a,
    OneFlow_Tensor:$b,
    OneFlow_Tensor:$dy
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$axis
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FusedBiasAddMaskScaleOp : OneFlow_BaseOp<"fused_bias_add_mask_scale", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$a,
    OneFlow_Tensor:$b,
    OneFlow_Tensor:$mask,
    Optional<OneFlow_Tensor>:$_add_to_output
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$axis,
    DefaultValuedAttr<F32Attr, "1.">:$scale,
    DefaultValuedAttr<SI64Attr, "0">:$seed
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_FusedCastScaleOp : OneFlow_BaseOp<"fused_cast_scale", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$scale_by_tensor
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "1.">:$scale
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FusedScaleMaskSoftmaxOp : OneFlow_BaseOp<"fused_scale_mask_softmax", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$mask
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "1.">:$scale_value,
    DefaultValuedAttr<F32Attr, "0.">:$mask_fill_value
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_FusedScaleMaskSoftmaxDropoutOp : OneFlow_BaseOp<"fused_scale_mask_softmax_dropout", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$mask,
    OneFlow_Tensor:$dropout_mask
  );
  let output = (outs
    OneFlow_Tensor:$y,
    OneFlow_Tensor:$softmax_y
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "1.">:$scale_value,
    DefaultValuedAttr<F32Attr, "0.">:$mask_fill_value,
    DefaultValuedAttr<F32Attr, "1.">:$dropout_scale_value
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_FusedScaleMaskSoftmaxDropoutGradOp : OneFlow_BaseOp<"fused_scale_mask_softmax_dropout_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$softmax_y,
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$mask,
    OneFlow_Tensor:$dropout_mask
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$scale_value,
    DefaultValuedAttr<F32Attr, "0.">:$dropout_scale_value
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FusedBiasAddScaleMaskSoftmaxDropoutOp : OneFlow_BaseOp<"fused_bias_add_scale_mask_softmax_dropout", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$bias,
    OneFlow_Tensor:$mask,
    OneFlow_Tensor:$dropout_mask
  );
  let output = (outs
    OneFlow_Tensor:$y,
    OneFlow_Tensor:$softmax_y
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "1.">:$scale_value,
    DefaultValuedAttr<F32Attr, "0.">:$mask_fill_value,
    DefaultValuedAttr<F32Attr, "1.">:$dropout_scale_value
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_FusedScaleMaskSoftmaxGradOp : OneFlow_BaseOp<"fused_scale_mask_softmax_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$y,
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$mask
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$scale_value
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FusedScaleTrilOp : OneFlow_BaseOp<"fused_scale_tril", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$diagonal,
    DefaultValuedAttr<F64Attr, "0.">:$floating_fill_value,
    DefaultValuedAttr<SI64Attr, "0">:$integer_fill_value,
    DefaultValuedAttr<BoolAttr, "false">:$is_floating_fill_value,
    DefaultValuedAttr<F64Attr, "1.">:$floating_scale_value,
    DefaultValuedAttr<SI64Attr, "1">:$integer_scale_value,
    DefaultValuedAttr<BoolAttr, "false">:$is_floating_scale_value
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FusedSelfAttentionQueryMulKeyAndValueOp : OneFlow_BaseOp<"fused_self_attention_query_mul_key_and_value", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$hidden_states
  );
  let output = (outs
    OneFlow_Tensor:$query_mul_key,
    OneFlow_Tensor:$value
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$head_size,
    DefaultValuedAttr<F32Attr, "0.">:$alpha
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FusedSelfAttentionQueryMulKeyAndValueGradOp : OneFlow_BaseOp<"fused_self_attention_query_mul_key_and_value_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$query_mul_key_grad,
    OneFlow_Tensor:$value_grad,
    OneFlow_Tensor:$hidden_states
  );
  let output = (outs
    OneFlow_Tensor:$hidden_states_grad
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$alpha
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FusedTrilScaleSoftmaxMaskScaleOp : OneFlow_BaseOp<"fused_tril_scale_softmax_mask_scale", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$mask
  );
  let output = (outs
    OneFlow_Tensor:$y,
    OneFlow_Tensor:$softmax_y
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$diagonal,
    DefaultValuedAttr<F32Attr, "0.">:$tril_fill_value,
    DefaultValuedAttr<F32Attr, "1.">:$tril_scale_value,
    DefaultValuedAttr<F32Attr, "1.">:$mask_scale_value
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_FusedTrilScaleSoftmaxMaskScaleGradOp : OneFlow_BaseOp<"fused_tril_scale_softmax_mask_scale_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$softmax_y,
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$mask
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$diagonal,
    DefaultValuedAttr<F32Attr, "0.">:$tril_scale_value,
    DefaultValuedAttr<F32Attr, "0.">:$mask_scale_value
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}


def OneFlow_NormalizationAddReluGradOp : OneFlow_BaseOp<"normalization_add_relu_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$mean,
    OneFlow_Tensor:$inv_variance,
    OneFlow_Tensor:$gamma,
    OneFlow_Tensor:$beta,
    OneFlow_Tensor:$reserve_space,
    OneFlow_Tensor:$y
  );
  let output = (outs
    OneFlow_Tensor:$gamma_diff,
    OneFlow_Tensor:$beta_diff,
    OneFlow_Tensor:$dx,
    Optional<OneFlow_Tensor>:$addend_diff
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$axis,
    DefaultValuedAttr<F32Attr, "0.">:$epsilon
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}


def OneFlow_FusedDotFeatureInteractionOp : OneFlow_BaseOp<"fused_dot_feature_interaction", [NoMemoryEffect, AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    Variadic<OneFlow_Tensor>:$features,
    Optional<OneFlow_Tensor>:$output_concat,
    Optional<OneFlow_Tensor>:$num_valid_sparse_feature,
    Optional<OneFlow_Tensor>:$sparse_feature,
    Optional<OneFlow_Tensor>:$sparse_indices
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$self_interaction,
    DefaultValuedAttr<BoolAttr, "false">:$has_output_concat,
    DefaultValuedAttr<SI32Attr, "0">:$output_padding,
    DefaultValuedAttr<StrAttr, "\"none\"">:$pooling
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FusedDotFeatureInteractionGradOp : OneFlow_BaseOp<"fused_dot_feature_interaction_grad", [NoMemoryEffect, AttrSizedOperandSegments, AttrSizedResultSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    Variadic<OneFlow_Tensor>:$features,
    Optional<OneFlow_Tensor>:$num_valid_sparse_feature,
    Optional<OneFlow_Tensor>:$sparse_feature,
    Optional<OneFlow_Tensor>:$sparse_indices
  );
  let output = (outs
    Variadic<OneFlow_Tensor>:$features_grad,
    Optional<OneFlow_Tensor>:$output_concat_grad,
    Optional<OneFlow_Tensor>:$sparse_feature_grad
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$self_interaction,
    DefaultValuedAttr<SI32Attr, "0">:$output_concat_grad_dim,
    DefaultValuedAttr<StrAttr, "\"none\"">:$pooling
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FusedCrossFeatureInteractionOp : OneFlow_BaseOp<"fused_cross_feature_interaction", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$weight,
    OneFlow_Tensor:$bias,
    OneFlow_Tensor:$x0
  );
  let output = (outs
    OneFlow_Tensor:$out,
    OneFlow_Tensor:$matmul_result
  );
  let attrs = (ins
    StrAttr:$interaction_mode
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}


def OneFlow_FusedCrossFeatureInteractionV1GradOp : OneFlow_BaseOp<"fused_cross_feature_interaction_v1_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>, NoGrad]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$weight,
    OneFlow_Tensor:$x0,
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$matmul_result
  );
  let output = (outs
    OneFlow_Tensor:$dx0,
    OneFlow_Tensor:$dw,
    OneFlow_Tensor:$dx,
    OneFlow_Tensor:$dbias
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FusedCrossFeatureInteractionV2GradOp : OneFlow_BaseOp<"fused_cross_feature_interaction_v2_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>, NoGrad]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$weight,
    OneFlow_Tensor:$bias,
    OneFlow_Tensor:$x0,
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$matmul_result
  );
  let output = (outs
    OneFlow_Tensor:$dx0,
    OneFlow_Tensor:$dw,
    OneFlow_Tensor:$dx,
    OneFlow_Tensor:$dbias
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FusedMultiHeadAttentionInferenceOp : OneFlow_BaseOp<"fused_multi_head_attention_inference", [NoMemoryEffect, AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$query,
    OneFlow_Tensor:$key,
    OneFlow_Tensor:$value,
    Optional<OneFlow_Tensor>:$attn_bias,
    Optional<OneFlow_Tensor>:$query_seq_start,
    Optional<OneFlow_Tensor>:$key_seq_start,
    Optional<OneFlow_Tensor>:$key_seq_len
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    SI64Attr:$query_head_size,
    DefaultValuedAttr<SI64Attr, "0">:$query_max_seq_len,
    DefaultValuedAttr<SI64Attr, "0">:$key_max_seq_len,
    F64Attr:$scale,
    DefaultValuedAttr<SI64Attr, "0">:$causal_diagonal_offset,
    DefaultValuedAttr<StrAttr, "\"none\"">:$attn_mask_type,
    StrAttr:$query_layout,
    StrAttr:$key_layout,
    StrAttr:$value_layout,
    StrAttr:$output_layout
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FusedAttentionConcatPastKeyValueOp : OneFlow_BaseOp<"fused_attention_concat_past_key_value", [NoMemoryEffect, AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$key,
    OneFlow_Tensor:$value,
    Optional<OneFlow_Tensor>:$past_key,
    Optional<OneFlow_Tensor>:$past_value
  );
  let output = (outs
    OneFlow_Tensor:$output_key,
    OneFlow_Tensor:$output_value
  );
  let attrs = (ins
    StrAttr:$past_key_layout,
    StrAttr:$past_value_layout,
    StrAttr:$key_layout,
    StrAttr:$value_layout,
    SI64Attr:$key_head_size
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FusedFastGeluMulOp : OneFlow_BaseOp<"fused_fast_gelu_mul", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in,
    OneFlow_Tensor:$multiplier
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FusedFastGeluMulGradOp : OneFlow_BaseOp<"fused_fast_gelu_mul_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$out_diff,
    OneFlow_Tensor:$in,
    OneFlow_Tensor:$multiplier
  );
  let output = (outs
    OneFlow_Tensor:$in_diff,
    OneFlow_Tensor:$multiplier_diff
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FusedGetBounddingBoxesCoordOp : OneFlow_BaseOp<"fused_get_boundding_boxes_coord", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x1,
    OneFlow_Tensor:$y1,
    OneFlow_Tensor:$w1,
    OneFlow_Tensor:$h1,
    OneFlow_Tensor:$x2,
    OneFlow_Tensor:$y2,
    OneFlow_Tensor:$w2,
    OneFlow_Tensor:$h2
  );
  let output = (outs
    OneFlow_Tensor:$b1_x1,
    OneFlow_Tensor:$b1_x2,
    OneFlow_Tensor:$b1_y1,
    OneFlow_Tensor:$b1_y2,
    OneFlow_Tensor:$b2_x1,
    OneFlow_Tensor:$b2_x2,
    OneFlow_Tensor:$b2_y1,
    OneFlow_Tensor:$b2_y2
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FusedGetBounddingBoxesCoordGradOp : OneFlow_BaseOp<"fused_get_boundding_boxes_coord_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$b1_x1_diff,
    OneFlow_Tensor:$b1_x2_diff,
    OneFlow_Tensor:$b1_y1_diff,
    OneFlow_Tensor:$b1_y2_diff,
    OneFlow_Tensor:$b2_x1_diff,
    OneFlow_Tensor:$b2_x2_diff,
    OneFlow_Tensor:$b2_y1_diff,
    OneFlow_Tensor:$b2_y2_diff
  );
  let output = (outs
    OneFlow_Tensor:$x1_diff,
    OneFlow_Tensor:$y1_diff,
    OneFlow_Tensor:$w1_diff,
    OneFlow_Tensor:$h1_diff,
    OneFlow_Tensor:$x2_diff,
    OneFlow_Tensor:$y2_diff,
    OneFlow_Tensor:$w2_diff,
    OneFlow_Tensor:$h2_diff
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FusedGetCiouResultOp : OneFlow_BaseOp<"fused_get_ciou_result", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$v,
    OneFlow_Tensor:$iou,
    OneFlow_Tensor:$rho2,
    OneFlow_Tensor:$c2
  );
  let output = (outs
    OneFlow_Tensor:$y,
    OneFlow_Tensor:$alpha
  );
  let attrs = (ins
    F32Attr: $eps
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FusedGetCiouResultGradOp : OneFlow_BaseOp<"fused_get_ciou_result_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$alpha,
    OneFlow_Tensor:$rho2,
    OneFlow_Tensor:$c2
  );
  let output = (outs
    OneFlow_Tensor:$dv,
    OneFlow_Tensor:$diou,
    OneFlow_Tensor:$drho2,
    OneFlow_Tensor:$dc2
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FusedGetIouOp : OneFlow_BaseOp<"fused_get_iou", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$w1,
    OneFlow_Tensor:$h1,
    OneFlow_Tensor:$w2,
    OneFlow_Tensor:$h2,
    OneFlow_Tensor:$inter
  );
  let output = (outs
    OneFlow_Tensor:$iou
  );
  let attrs = (ins
    F32Attr: $eps
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FusedGetIouGradOp : OneFlow_BaseOp<"fused_get_iou_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$diou,
    OneFlow_Tensor:$w1,
    OneFlow_Tensor:$h1,
    OneFlow_Tensor:$w2,
    OneFlow_Tensor:$h2,
    OneFlow_Tensor:$inter
  );
  let attrs = (ins
    F32Attr: $eps
  );
  let output = (outs
    OneFlow_Tensor:$dw1,
    OneFlow_Tensor:$dh1,
    OneFlow_Tensor:$dinter
  );
  let attrs = (ins
    F32Attr: $eps
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FusedCenterOp : OneFlow_BaseOp<"fused_get_center_dist", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$b1_x1,
    OneFlow_Tensor:$b1_x2,
    OneFlow_Tensor:$b2_x1,
    OneFlow_Tensor:$b2_x2,
    OneFlow_Tensor:$b1_y1,
    OneFlow_Tensor:$b1_y2,
    OneFlow_Tensor:$b2_y1,
    OneFlow_Tensor:$b2_y2
  );
  let output = (outs
    OneFlow_Tensor:$rho2
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FusedCenterGradOp : OneFlow_BaseOp<"fused_get_center_dist_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$b1_x1,
    OneFlow_Tensor:$b1_x2,
    OneFlow_Tensor:$b2_x1,
    OneFlow_Tensor:$b2_x2,
    OneFlow_Tensor:$b1_y1,
    OneFlow_Tensor:$b1_y2,
    OneFlow_Tensor:$b2_y1,
    OneFlow_Tensor:$b2_y2,
    OneFlow_Tensor:$rho2_diff
  );
  let output = (outs
    OneFlow_Tensor:$b1_x1_diff,
    OneFlow_Tensor:$b1_x2_diff,
    OneFlow_Tensor:$b2_x1_diff,
    OneFlow_Tensor:$b2_x2_diff,
    OneFlow_Tensor:$b1_y1_diff,
    OneFlow_Tensor:$b1_y2_diff,
    OneFlow_Tensor:$b2_y1_diff,
    OneFlow_Tensor:$b2_y2_diff
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FusedGetCiouDiagonalAngleOp : OneFlow_BaseOp<"fused_get_ciou_diagonal_angle", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$w1,
    OneFlow_Tensor:$h1,
    OneFlow_Tensor:$w2,
    OneFlow_Tensor:$h2
  );
  let output = (outs
    OneFlow_Tensor:$v
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "1e-08">:$eps
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FusedGetCiouDiagonalAngleGradOp : OneFlow_BaseOp<"fused_get_ciou_diagonal_angle_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$w1,
    OneFlow_Tensor:$h1,
    OneFlow_Tensor:$w2,
    OneFlow_Tensor:$h2,
    OneFlow_Tensor:$v_diff
  );
  let output = (outs
    OneFlow_Tensor:$w1_diff,
    OneFlow_Tensor:$h1_diff,
    OneFlow_Tensor:$w2_diff,
    OneFlow_Tensor:$h2_diff
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "1e-08">:$eps
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FusedGetIntersectionAreaOp : OneFlow_BaseOp<"fused_get_intersection_area", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$b1_x1,
    OneFlow_Tensor:$b1_x2,
    OneFlow_Tensor:$b2_x1,
    OneFlow_Tensor:$b2_x2,
    OneFlow_Tensor:$b1_y1,
    OneFlow_Tensor:$b1_y2,
    OneFlow_Tensor:$b2_y1,
    OneFlow_Tensor:$b2_y2
  );
  let output = (outs
    OneFlow_Tensor:$inter
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FusedGetIntersectionAreaGradOp : OneFlow_BaseOp<"fused_get_intersection_area_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$b1_x1,
    OneFlow_Tensor:$b1_x2,
    OneFlow_Tensor:$b2_x1,
    OneFlow_Tensor:$b2_x2,
    OneFlow_Tensor:$b1_y1,
    OneFlow_Tensor:$b1_y2,
    OneFlow_Tensor:$b2_y1,
    OneFlow_Tensor:$b2_y2,
    OneFlow_Tensor:$inter_diff
  );
  let output = (outs
    OneFlow_Tensor:$b1_x1_diff,
    OneFlow_Tensor:$b1_x2_diff,
    OneFlow_Tensor:$b2_x1_diff,
    OneFlow_Tensor:$b2_x2_diff,
    OneFlow_Tensor:$b1_y1_diff,
    OneFlow_Tensor:$b1_y2_diff,
    OneFlow_Tensor:$b2_y1_diff,
    OneFlow_Tensor:$b2_y2_diff
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}


def OneFlow_FusedGetConvexDiagonalSquaredOp : OneFlow_BaseOp<"fused_get_convex_diagonal_squared", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$b1_x1,
    OneFlow_Tensor:$b1_x2,
    OneFlow_Tensor:$b2_x1,
    OneFlow_Tensor:$b2_x2,
    OneFlow_Tensor:$b1_y1,
    OneFlow_Tensor:$b1_y2,
    OneFlow_Tensor:$b2_y1,
    OneFlow_Tensor:$b2_y2
  );
  let output = (outs
    OneFlow_Tensor:$c2
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "1e-08">:$eps
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FusedGetConvexDiagonalSquaredGradOp : OneFlow_BaseOp<"fused_get_convex_diagonal_squared_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$c2_diff,
    OneFlow_Tensor:$b1_x1,
    OneFlow_Tensor:$b1_x2,
    OneFlow_Tensor:$b2_x1,
    OneFlow_Tensor:$b2_x2,
    OneFlow_Tensor:$b1_y1,
    OneFlow_Tensor:$b1_y2,
    OneFlow_Tensor:$b2_y1,
    OneFlow_Tensor:$b2_y2
  );
  let output = (outs
    OneFlow_Tensor:$b1_x1_diff,
    OneFlow_Tensor:$b1_x2_diff,
    OneFlow_Tensor:$b2_x1_diff,
    OneFlow_Tensor:$b2_x2_diff,
    OneFlow_Tensor:$b1_y1_diff,
    OneFlow_Tensor:$b1_y2_diff,
    OneFlow_Tensor:$b2_y1_diff,
    OneFlow_Tensor:$b2_y2_diff
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "1e-08">:$eps
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FusedScaleMaskBiasSoftmaxGradOp : OneFlow_BaseOp<"fused_scale_mask_bias_softmax_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$y,
    OneFlow_Tensor:$dy
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.35355">:$scale
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FusedScaleMaskBiasSoftmaxOp : OneFlow_BaseOp<"fused_scale_mask_bias_softmax", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$mask,
    Optional<OneFlow_Tensor>:$bias
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.125">:$scale,
    DefaultValuedAttr<BoolAttr, "false">:$inplace
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FusedCodegeexQkvReshapeOp : OneFlow_BaseOp<"fused_codegeex_qkv_reshape", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$query,
    OneFlow_Tensor:$key,
    OneFlow_Tensor:$value
  );
  let output = (outs
    OneFlow_Tensor:$new_query,
    OneFlow_Tensor:$new_key,
    OneFlow_Tensor:$new_value
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "1">:$num_attention_heads
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_NonContiguousBinaryOp : OneFlow_BaseOp<"noncontiguous_binary_op", [NoMemoryEffect, SupportNonContiguous, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$lhs,
    OneFlow_Tensor:$rhs
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let attrs = (ins
    DefaultValuedAttr<StrAttr, "\"add\"">:$op,
    DefaultValuedAttr<BoolAttr, "false">:$inplace
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_NonContiguousBinaryOpGrad : OneFlow_BaseOp<"noncontiguous_binary_op_grad", [NoMemoryEffect, SupportNonContiguous, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$lhs,
    OneFlow_Tensor:$rhs
  );
  let output = (outs
    OneFlow_Tensor:$dlhs,
    OneFlow_Tensor:$drhs
  );
  let attrs = (ins
    DefaultValuedAttr<StrAttr, "\"add\"">:$op,
    DefaultValuedAttr<BoolAttr, "false">:$inplace
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

#endif // GET_ONEFLOW_FUSED_OP_DEFINITIONS


#ifdef GET_ONEFLOW_IDEMPOTENT_OP_DEFINITIONS

def OneFlow_AbsOp : OneFlow_IdempotentBaseOp<"abs", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_CeilOp : OneFlow_IdempotentBaseOp<"ceil", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_FloorOp : OneFlow_IdempotentBaseOp<"floor", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_OnesLikeOp : OneFlow_IdempotentBaseOp<"ones_like", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let same_output_regst_num = 1;
  let has_nd_sbp_infer_fn = 1;
  let input = (ins AnyType:$like);
  let output = (outs AnyType:$out);
}

def OneFlow_ReluOp : OneFlow_IdempotentBaseOp<"relu", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>, DeclareOpInterfaceMethods<NCHWCompatibleInterface>]> {}

def OneFlow_RintOp : OneFlow_IdempotentBaseOp<"rint", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_RoundOp : OneFlow_IdempotentBaseOp<"round", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_SignOp : OneFlow_IdempotentBaseOp<"sign", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

#endif // GET_ONEFLOW_IDEMPOTENT_OP_DEFINITIONS


#ifdef GET_ONEFLOW_IDENTITY_OP_DEFINITIONS

def OneFlow_AmpWhiteIdentityOp : OneFlow_BaseOp<"amp_white_identity", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_AmpBlackIdentityOp : OneFlow_BaseOp<"amp_black_identity", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_IdentityOp : OneFlow_BaseOp<"identity", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_IdentityBufferOp : OneFlow_BaseOp<"identity_buffer", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$buffer_size
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_TupleIdentityOp : OneFlow_BaseOp<"tuple_identity", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    Variadic<OneFlow_Tensor>:$in
  );
  let output = (outs
    Variadic<OneFlow_Tensor>:$out
  );
  let has_check_fn = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_sbp_signature_infer_fn = 1;
}

def OneFlow_PinnedIdentityOp : OneFlow_BaseOp<"pinned_identity", [DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let summary = "mark defining op of operand can't be erased";
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

#endif // GET_ONEFLOW_IDENTITY_OP_DEFINITIONS


#ifdef GET_ONEFLOW_IMAGE_OP_DEFINITIONS

def OneFlow_ImageBatchAlignOp : OneFlow_BaseOp<"image_batch_align", [NoMemoryEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    ShapeAttr:$shape,
    OneFlow_DataType:$data_type,
    DefaultValuedAttr<SI32Attr, "0">:$alignment,
    DefaultValuedAttr<BoolAttr, "false">:$dynamic_out
  );
  let has_check_fn = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_output_arg_modify_fn = 1;
}

def OneFlow_ImageDecodeOp : OneFlow_BaseOp<"image_decode", [NoMemoryEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<StrAttr, "\"BGR\"">:$color_space,
    OneFlow_DataType:$data_type
  );
  let has_check_fn = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ImageFlipOp : OneFlow_BaseOp<"image_flip", [NoMemoryEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in,
    OneFlow_Tensor:$flip_code
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ImageRandomCropOp : OneFlow_BaseOp<"image_random_crop", [NoMemoryEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "10">:$num_attempts,
    DefaultValuedAttr<SI64Attr, "-1">:$seed,
    DefaultValuedAttr<BoolAttr, "false">:$has_seed,
    F32ArrayAttr:$random_area,
    F32ArrayAttr:$random_aspect_ratio
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_ImageResizeKeepAspectRatioOp : OneFlow_BaseOp<"image_resize_keep_aspect_ratio", [NoMemoryEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out,
    OneFlow_Tensor:$size,
    OneFlow_Tensor:$scale
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$target_size,
    DefaultValuedAttr<SI32Attr, "0">:$min_size,
    DefaultValuedAttr<SI32Attr, "0">:$max_size,
    DefaultValuedAttr<BoolAttr, "false">:$resize_longer,
    DefaultValuedAttr<StrAttr, "\"bilinear\"">:$interpolation_type
  );
  let has_check_fn = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ImageResizeToFixedOp : OneFlow_BaseOp<"image_resize_to_fixed", [NoMemoryEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out,
    OneFlow_Tensor:$scale
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$target_width,
    DefaultValuedAttr<SI64Attr, "0">:$target_height,
    DefaultValuedAttr<SI64Attr, "3">:$channels,
    OneFlow_DataType:$data_type,
    DefaultValuedAttr<StrAttr, "\"bilinear\"">:$interpolation_type
  );
  let has_check_fn = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

#endif // GET_ONEFLOW_IMAGE_OP_DEFINITIONS


#ifdef GET_ONEFLOW_INDICES_OP_DEFINITIONS

def OneFlow_ArgSortOp : OneFlow_BaseOp<"arg_sort", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    StrAttr:$direction
  );
  let has_check_fn = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ArgmaxOp : OneFlow_BaseOp<"argmax", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ArgwhereOp : OneFlow_BaseOp<"argwhere", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$input
  );
  let output = (outs
    OneFlow_Tensor:$output,
    OneFlow_Tensor:$output_size
  );
  let attrs = (ins
    OneFlow_DataType:$dtype
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_BatchGatherOp : OneFlow_BaseOp<"batch_gather", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in,
    OneFlow_Tensor:$indices
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_DimGatherOp : OneFlow_BaseOp<"dim_gather", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$input,
    OneFlow_Tensor:$index
  );
  let output = (outs
    OneFlow_Tensor:$output
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$dim
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_DimScatterAddOp : OneFlow_BaseOp<"dim_scatter_add", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$input,
    OneFlow_Tensor:$index,
    OneFlow_Tensor:$src
  );
  let output = (outs
    OneFlow_Tensor:$output
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$dim
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_DimScatterAddLikeOp : OneFlow_BaseOp<"dim_scatter_add_like", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$like,
    OneFlow_Tensor:$index,
    OneFlow_Tensor:$src
  );
  let output = (outs
    OneFlow_Tensor:$output
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$dim
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_DimScatterAddScalarOp : OneFlow_BaseOp<"dim_scatter_add_scalar", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$input,
    OneFlow_Tensor:$index
  );
  let output = (outs
    OneFlow_Tensor:$output
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$src_scalar,
    DefaultValuedAttr<SI32Attr, "0">:$dim
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_DimScatterMulOp : OneFlow_BaseOp<"dim_scatter_mul", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$input,
    OneFlow_Tensor:$index,
    OneFlow_Tensor:$src
  );
  let output = (outs
    OneFlow_Tensor:$output
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$dim
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_DimScatterMulScalarOp : OneFlow_BaseOp<"dim_scatter_mul_scalar", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$input,
    OneFlow_Tensor:$index
  );
  let output = (outs
    OneFlow_Tensor:$output
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$src_scalar,
    DefaultValuedAttr<SI32Attr, "0">:$dim
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_DimScatterUpdateOp : OneFlow_BaseOp<"dim_scatter_update", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$input,
    OneFlow_Tensor:$index,
    OneFlow_Tensor:$src
  );
  let output = (outs
    OneFlow_Tensor:$output
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$dim
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_DimScatterUpdateScalarOp : OneFlow_BaseOp<"dim_scatter_update_scalar", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$input,
    OneFlow_Tensor:$index
  );
  let output = (outs
    OneFlow_Tensor:$output
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$src_scalar,
    DefaultValuedAttr<SI32Attr, "0">:$dim
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_EmbeddingRenormOp : OneFlow_BaseOp<"embedding_renorm", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in,
    OneFlow_Tensor:$indices
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.">:$max_norm,
    DefaultValuedAttr<F64Attr, "2.">:$norm_type
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_EmbeddingOp : OneFlow_BaseOp<"embedding", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$weight,
    OneFlow_Tensor:$indices
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "-1">:$padding_idx,
    DefaultValuedAttr<BoolAttr, "false">:$scale_grad_by_freq
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_FusedApplyRotaryEmbOp : OneFlow_BaseOp<"fused_apply_rotary_emb", [NoMemoryEffect, AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    Optional<OneFlow_Tensor>:$cos,
    Optional<OneFlow_Tensor>:$sin,
    Optional<OneFlow_Tensor>:$position_ids
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<StrAttr, "\"BHMK\"">:$x_layout,
    DefaultValuedAttr<StrAttr, "\"BHMK\"">:$output_layout,
    DefaultValuedAttr<StrAttr, "\"plane\"">:$mode,
    DefaultValuedAttr<SI64Attr, "0">:$tensor_index,
    DefaultValuedAttr<F32Attr, "1e4">:$base,
    DefaultValuedAttr<SI64Attr, "0">:$k_size,
    DefaultValuedAttr<SI64Attr, "0">:$rotary_size
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_EmbeddingGradOp : OneFlow_BaseOp<"embedding_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$weight,
    OneFlow_Tensor:$indices
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "-1">:$padding_idx,
    DefaultValuedAttr<BoolAttr, "false">:$scale_grad_by_freq
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_GatherOp : OneFlow_BaseOp<"gather", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in,
    OneFlow_Tensor:$indices
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$axis
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_GatherNdOp : OneFlow_BaseOp<"gather_nd", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$params,
    OneFlow_Tensor:$indices
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_GenerateRandomBatchPermutationIndicesOp : OneFlow_BaseOp<"generate_random_batch_permutation_indices", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$seed
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ImageTargetResizeOp : OneFlow_BaseOp<"image_target_resize", [NoMemoryEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out,
    OneFlow_Tensor:$size,
    OneFlow_Tensor:$scale
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$target_size,
    DefaultValuedAttr<SI32Attr, "0">:$max_size
  );
  let has_check_fn = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_SliceOp : OneFlow_BaseOp<"slice", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let attrs = (ins
    SI64ArrayAttr:$start,
    SI64ArrayAttr:$stop,
    SI64ArrayAttr:$step
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_SliceUpdateOp : OneFlow_BaseOp<"slice_update", [SupportNonContiguous, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$ref,
    OneFlow_Tensor:$value
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let attrs = (ins
    SI64ArrayAttr:$start,
    SI64ArrayAttr:$stop,
    SI64ArrayAttr:$step
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_SliceGradOp : OneFlow_BaseOp<"slice_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    ShapeAttr:$like_shape,
    SI64ArrayAttr:$start,
    SI64ArrayAttr:$stop,
    SI64ArrayAttr:$step
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_ScatterNdOp : OneFlow_BaseOp<"scatter_nd", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$indices,
    OneFlow_Tensor:$updates
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    ShapeAttr:$shape
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_ScatterNdLikeOp : OneFlow_BaseOp<"scatter_nd_like", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$like,
    OneFlow_Tensor:$indices,
    OneFlow_Tensor:$updates
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_TensorScatterNdAddOp : OneFlow_BaseOp<"tensor_scatter_nd_add", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$params,
    OneFlow_Tensor:$updates,
    OneFlow_Tensor:$indices
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_TensorScatterNdUpdateOp : OneFlow_BaseOp<"tensor_scatter_nd_update", [NoMemoryEffect, SupportNonContiguous, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$params,
    OneFlow_Tensor:$updates,
    OneFlow_Tensor:$indices
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_UnsortedBatchSegmentSumOp : OneFlow_BaseOp<"unsorted_batch_segment_sum", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$data,
    OneFlow_Tensor:$segment_ids
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$num_segments
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_UnsortedSegmentSumOp : OneFlow_BaseOp<"unsorted_segment_sum", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$data,
    OneFlow_Tensor:$segment_ids
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$axis,
    DefaultValuedAttr<SI64Attr, "0">:$num_segments
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_UnsortedSegmentSumLikeOp : OneFlow_BaseOp<"unsorted_segment_sum_like", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$data,
    OneFlow_Tensor:$segment_ids,
    OneFlow_Tensor:$like
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$axis
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_WhereOp : OneFlow_BaseOp<"where", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$condition,
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$y
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_MedianOp : OneFlow_BaseOp<"median", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$input
  );
  let output = (outs
    OneFlow_Tensor:$output
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_MedianWithIndicesOp : OneFlow_BaseOp<"median_with_indices", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$input
  );
  let output = (outs
    OneFlow_Tensor:$values,
    OneFlow_Tensor:$indices
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_SearchSortedOp : OneFlow_BaseOp<"searchsorted", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$sorted_sequence,
    OneFlow_Tensor:$values
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$out_int32,
    DefaultValuedAttr<BoolAttr, "false">:$right
  );
  let has_check_fn = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_SearchSortedScalarOp : OneFlow_BaseOp<"searchsorted_scalar", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$sorted_sequence
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$out_int32,
    DefaultValuedAttr<BoolAttr, "false">:$right,
    DefaultValuedAttr<F32Attr, "0.">:$values
  );
  let has_check_fn = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ModeOp: OneFlow_BaseOp<"mode", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$input
  );
  let output = (outs
    OneFlow_Tensor:$values,
    OneFlow_Tensor:$indices
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

#endif // GET_ONEFLOW_INDICES_OP_DEFINITIONS


#ifdef GET_ONEFLOW_INVOLUTION_OP_DEFINITIONS

def OneFlow_NegativeOp : OneFlow_InvolutionBaseOp<"negative", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_ReciprocalOp : OneFlow_InvolutionBaseOp<"reciprocal", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

#endif // GET_ONEFLOW_INVOLUTION_OP_DEFINITIONS


#ifdef GET_ONEFLOW_LOSS_OP_DEFINITIONS

def OneFlow_CombinedMarginLossOp : OneFlow_BaseOp<"combined_margin_loss", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$label
  );
  let output = (outs
    OneFlow_Tensor:$y,
    OneFlow_Tensor:$theta
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$m1,
    DefaultValuedAttr<F32Attr, "0.">:$m2,
    DefaultValuedAttr<F32Attr, "0.">:$m3,
    DefaultValuedAttr<SI64Attr, "0">:$depth
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_CombinedMarginLossGradOp : OneFlow_BaseOp<"combined_margin_loss_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$label,
    OneFlow_Tensor:$theta
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$m1,
    DefaultValuedAttr<F32Attr, "0.">:$m2,
    DefaultValuedAttr<F32Attr, "0.">:$m3,
    DefaultValuedAttr<SI64Attr, "0">:$depth
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_CtcLossOp : OneFlow_BaseOp<"ctc_loss", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$log_probs,
    OneFlow_Tensor:$targets,
    OneFlow_Tensor:$input_lengths,
    OneFlow_Tensor:$target_lengths
  );
  let output = (outs
    OneFlow_Tensor:$loss,
    OneFlow_Tensor:$alpha
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$max_target_length,
    DefaultValuedAttr<SI32Attr, "0">:$blank,
    DefaultValuedAttr<BoolAttr, "false">:$zero_infinity
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_CtcLossGradOp : OneFlow_BaseOp<"ctc_loss_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$grad_out,
    OneFlow_Tensor:$log_probs,
    OneFlow_Tensor:$targets,
    OneFlow_Tensor:$input_lengths,
    OneFlow_Tensor:$target_lengths,
    OneFlow_Tensor:$loss,
    OneFlow_Tensor:$alpha
  );
  let output = (outs
    OneFlow_Tensor:$grad
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$max_target_length,
    DefaultValuedAttr<SI32Attr, "0">:$blank,
    DefaultValuedAttr<BoolAttr, "false">:$zero_infinity
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_DynamicLossScaleScheduleOp : OneFlow_BaseOp<"dynamic_loss_scale_schedule", [DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$count_not_finite,
    OneFlow_Tensor:$loss_scale,
    OneFlow_Tensor:$good_step_counter
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "2000">:$increment_period,
    DefaultValuedAttr<F32Attr, "2.">:$multiplier
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_KlDivLossOp : OneFlow_BaseOp<"kl_div_loss", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$input,
    OneFlow_Tensor:$target
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$log_target
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_KlDivLossGradOp : OneFlow_BaseOp<"kl_div_loss_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$input,
    OneFlow_Tensor:$target,
    OneFlow_Tensor:$dy
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$log_target
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_SmoothL1LossOp : OneFlow_BaseOp<"smooth_l1_loss", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$input,
    OneFlow_Tensor:$target
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$beta
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_SmoothL1LossGradOp : OneFlow_BaseOp<"smooth_l1_loss_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$input,
    OneFlow_Tensor:$target,
    OneFlow_Tensor:$dy
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$beta
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

#endif // GET_ONEFLOW_LOSS_OP_DEFINITIONS


#ifdef GET_ONEFLOW_MATH_OP_DEFINITIONS

def OneFlow_AbsGradOp : OneFlow_BaseOp<"abs_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
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

def OneFlow_ErfOp : OneFlow_BaseOp<"erf", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ErfGradOp : OneFlow_BaseOp<"erf_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
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

def OneFlow_ExpOp : OneFlow_BaseOp<"exp", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ExpGradOp : OneFlow_BaseOp<"exp_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
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

def OneFlow_Exp2Op : OneFlow_BaseOp<"exp2", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_Exp2GradOp : OneFlow_BaseOp<"exp2_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
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

def OneFlow_Expm1Op : OneFlow_BaseOp<"expm1", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_Expm1GradOp : OneFlow_BaseOp<"expm1_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
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


def OneFlow_FloordivXGradOp : OneFlow_BaseOp<"floordiv_x_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$y,
    OneFlow_Tensor:$dz
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FloordivYGradOp : OneFlow_BaseOp<"floordiv_y_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$y,
    OneFlow_Tensor:$dz
  );
  let output = (outs
    OneFlow_Tensor:$dy
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_TruncdivXGradOp : OneFlow_BaseOp<"truncdiv_x_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$y,
    OneFlow_Tensor:$dz
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_TruncdivYGradOp : OneFlow_BaseOp<"truncdiv_y_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$y,
    OneFlow_Tensor:$dz
  );
  let output = (outs
    OneFlow_Tensor:$dy
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_LgammaOp : OneFlow_BaseOp<"lgamma", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_LgammaGradOp : OneFlow_BaseOp<"lgamma_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
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


def OneFlow_DigammaOp : OneFlow_BaseOp<"digamma", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_DigammaGradOp : OneFlow_BaseOp<"digamma_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
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

def OneFlow_TrigammaOp : OneFlow_BaseOp<"trigamma", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_LogOp : OneFlow_BaseOp<"log", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_Log1pOp : OneFlow_BaseOp<"log1p", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_Log1pGradOp : OneFlow_BaseOp<"log1p_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
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

def OneFlow_Log2GradOp : OneFlow_BaseOp<"log2_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
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

def OneFlow_Log10GradOp : OneFlow_BaseOp<"log10_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
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

def OneFlow_LogGradOp : OneFlow_BaseOp<"log_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
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

def OneFlow_LogSigmoidOp : OneFlow_BaseOp<"log_sigmoid", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_LogSigmoidGradOp : OneFlow_BaseOp<"log_sigmoid_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
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

def OneFlow_ReciprocalGradOp : OneFlow_BaseOp<"reciprocal_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
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

def OneFlow_ReciprocalNoNanOp : OneFlow_BaseOp<"reciprocal_no_nan", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ReciprocalNoNanGradOp : OneFlow_BaseOp<"reciprocal_no_nan_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
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

def OneFlow_RsqrtOp : OneFlow_BaseOp<"rsqrt", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_RsqrtGradOp : OneFlow_BaseOp<"rsqrt_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
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



def OneFlow_SigmoidOp : OneFlow_BaseOp<"sigmoid", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_SigmoidGradOp : OneFlow_BaseOp<"sigmoid_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$y,
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

def OneFlow_SoftplusOp : OneFlow_BaseOp<"softplus", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "1.">:$beta,
    DefaultValuedAttr<F64Attr, "20.">:$threshold
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_SoftplusGradOp : OneFlow_BaseOp<"softplus_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$dy
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "1.">:$beta,
    DefaultValuedAttr<F64Attr, "20.">:$threshold
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_SoftsignGradOp : OneFlow_BaseOp<"softsign_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
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

def OneFlow_VarOp : OneFlow_BaseOp<"var", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$input
  );
  let output = (outs
    OneFlow_Tensor:$output
  );
  let attrs = (ins
    SI32ArrayAttr:$dim,
    DefaultValuedAttr<BoolAttr, "true">:$unbiased,
    DefaultValuedAttr<BoolAttr, "false">:$keepdim,
    OneFlow_DataType:$dtype
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}


def OneFlow_SqrtOp : OneFlow_BaseOp<"sqrt", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let hasFolder = 1;
}

def OneFlow_SqrtGradOp : OneFlow_BaseOp<"sqrt_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
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

def OneFlow_SquareOp : OneFlow_BaseOp<"square", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_SquareGradOp : OneFlow_BaseOp<"square_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
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

def OneFlow_XlogyXGradOp : OneFlow_BaseOp<"xlogy_x_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$y,
    OneFlow_Tensor:$dz
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_XlogyYGradOp : OneFlow_BaseOp<"xlogy_y_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$y,
    OneFlow_Tensor:$dz
  );
  let output = (outs
    OneFlow_Tensor:$dy
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_CumsumOp : OneFlow_BaseOp<"cumsum", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let attrs = (ins
    SI64Attr:$dim
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_CumProdOp : OneFlow_BaseOp<"cumprod", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let attrs = (ins
    SI64Attr:$dim
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_CumProdGradOp : OneFlow_BaseOp<"cumprod_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$output,
    OneFlow_Tensor:$input
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    SI64Attr:$dim
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ErfInvOp : OneFlow_BaseOp<"erfinv", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FftC2COp : OneFlow_BaseOp<"fft_c2c", [SupportNonContiguous, NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$input
  );
  let output = (outs
    OneFlow_Tensor:$out
  );

  let attrs = (ins
    SI64ArrayAttr:$dims,
    BoolAttr:$forward,
    SI32Attr:$norm_mode,
    F64Attr:$norm_fct
  );

  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FftR2COp : OneFlow_BaseOp<"fft_r2c", [SupportNonContiguous, NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$input
  );
  let output = (outs
    OneFlow_Tensor:$out
  );

  let attrs = (ins
    SI64ArrayAttr:$dims,
    SI32Attr:$norm_mode,
    F64Attr:$norm_fct,
    BoolAttr:$onesided
  );

  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FftC2ROp : OneFlow_BaseOp<"fft_c2r", [SupportNonContiguous, NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$input
  );
  let output = (outs
    OneFlow_Tensor:$out
  );

  let attrs = (ins
    SI64ArrayAttr:$dims,
    SI32Attr:$norm_mode,
    F64Attr:$norm_fct,
    SI64Attr:$last_dim_size
  );

  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_StftOp : OneFlow_BaseOp<"stft", [SupportNonContiguous, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$input,
    Optional<OneFlow_Tensor>:$window
  );
  let output = (outs
    OneFlow_Tensor:$out
  );

  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$n_fft,
    DefaultValuedAttr<SI32Attr, "0">:$hop_length,
    DefaultValuedAttr<SI32Attr, "0">:$win_length,
    DefaultValuedAttr<BoolAttr, "true">:$center,
    DefaultValuedAttr<StrAttr, "\"reflect\"">:$pad_mode,
    DefaultValuedAttr<BoolAttr, "false">:$normalized,
    DefaultValuedAttr<BoolAttr, "false">:$onesided,
    DefaultValuedAttr<BoolAttr, "false">:$return_complex
  );

  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_compute_complexity_fn = 1;
}
#endif // GET_ONEFLOW_MATH_OP_DEFINITIONS


#ifdef GET_ONEFLOW_MATMUL_OP_DEFINITIONS

def OneFlow_BatchMatmulOp : OneFlow_BaseOp<"batch_matmul", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$a,
    OneFlow_Tensor:$b,
    Optional<OneFlow_Tensor>:$_add_to_output
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$transpose_a,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_b,
    DefaultValuedAttr<F64Attr, "1.">:$alpha
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_compute_complexity_fn = 1;
}

def OneFlow_BroadcastMatmulOp : OneFlow_BaseOp<"broadcast_matmul", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>, DeclareOpInterfaceMethods<MatMulCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$a,
    OneFlow_Tensor:$b,
    Optional<OneFlow_Tensor>:$_add_to_output
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$transpose_a,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_b,
    DefaultValuedAttr<F64Attr, "1.">:$alpha
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_compute_complexity_fn = 1;
}

def OneFlow_BroadcastMatmulGradBOp : OneFlow_BaseOp<"broadcast_matmul_grad_b", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$a,
    OneFlow_Tensor:$b,
    Optional<OneFlow_Tensor>:$_add_to_output
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "1.">:$alpha
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_compute_complexity_fn = 1;
}

def OneFlow_DistributedPartialFcSampleOp : OneFlow_BaseOp<"distributed_partial_fc_sample", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$weight,
    OneFlow_Tensor:$label
  );
  let output = (outs
    OneFlow_Tensor:$mapped_label,
    OneFlow_Tensor:$sampled_label,
    OneFlow_Tensor:$sampled_weight
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$num_sample,
    DefaultValuedAttr<SI64Attr, "-1">:$seed
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_DistributedPartialFcSampleDisableBoxingOp : OneFlow_BaseOp<"distributed_partial_fc_sample_disable_boxing", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$sampled_weight_diff,
    OneFlow_Tensor:$sampled_label
  );
  let output = (outs
    OneFlow_Tensor:$boxing_disabled_sampled_weight_diff,
    OneFlow_Tensor:$boxing_disabled_sampled_label
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ErfcOp : OneFlow_BaseOp<"erfc", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ErfcGradOp : OneFlow_BaseOp<"erfc_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
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

def OneFlow_MatmulOp : OneFlow_BaseOp<"matmul", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>, DeclareOpInterfaceMethods<MatMulCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$a,
    OneFlow_Tensor:$b,
    Optional<OneFlow_Tensor>:$_add_to_output
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$transpose_a,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_b,
    DefaultValuedAttr<F64Attr, "1.">:$alpha
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_compute_complexity_fn = 1;
}

def OneFlow_MatrixVectorProductOp : OneFlow_BaseOp<"matrix_vector_product", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$a,
    OneFlow_Tensor:$b
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_MatrixVectorProductGradAOp : OneFlow_BaseOp<"matrix_vector_product_grad_a", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$b
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_MatrixVectorProductGradBOp : OneFlow_BaseOp<"matrix_vector_product_grad_b", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$a
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_VectorMatrixProductOp : OneFlow_BaseOp<"vector_matrix_product", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$a,
    OneFlow_Tensor:$b
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_VectorMatrixProductGradAOp : OneFlow_BaseOp<"vector_matrix_product_grad_a", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$b
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_VectorMatrixProductGradBOp : OneFlow_BaseOp<"vector_matrix_product_grad_b", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$a
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}


def OneFlow_CublasFusedMLPOp : OneFlow_BaseOp<"cublas_fused_mlp", [NoMemoryEffect, AttrSizedOperandSegments, AttrSizedResultSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    Variadic<OneFlow_Tensor>:$weights,
    Variadic<OneFlow_Tensor>:$biases
  );
  let output = (outs
    OneFlow_Tensor:$out,
    Variadic<OneFlow_Tensor>:$cublas_aux,
    Variadic<OneFlow_Tensor>:$hidden
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$skip_final_activation
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_CublasFusedMLPGradOp : OneFlow_BaseOp<"cublas_fused_mlp_grad", [NoMemoryEffect, NoGrad, AttrSizedOperandSegments, AttrSizedResultSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$x,
    Variadic<OneFlow_Tensor>:$weights,
    Variadic<OneFlow_Tensor>:$cublas_aux,
    Variadic<OneFlow_Tensor>:$hidden
  );
  let output = (outs
    OneFlow_Tensor:$d_x,
    Variadic<OneFlow_Tensor>:$d_biases,
    Variadic<OneFlow_Tensor>:$d_weights
  );
  let attrs = (ins
    F32ArrayAttr:$alpha_list
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_CublasBiasAddReluMatmulGradOp : OneFlow_BaseOp<"cublas_bias_add_relu_matmul_grad", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$weight,
    OneFlow_Tensor:$aux
  );
  let output = (outs
    OneFlow_Tensor:$d_grad,
    OneFlow_Tensor:$d_bias
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "1.">:$alpha
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_CublasMatmulBiasAddGradOp : OneFlow_BaseOp<"cublas_matmul_bias_add_grad", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$w_grad,
    OneFlow_Tensor:$b_grad
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FusedMatmulBiasOp : OneFlow_BaseOp<"fused_matmul_bias", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>, DeclareOpInterfaceMethods<MatMulCompatibleInterface>, DeclareOpInterfaceMethods<BiasAddCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$weight,
    OneFlow_Tensor:$bias,
    Optional<OneFlow_Tensor>:$_add_to_output
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "1.">:$alpha,
    DefaultValuedAttr<F64Attr, "1.">:$beta
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_get_sbp_fn = 1;
}

def OneFlow_FusedMatmulBiasAddReluDropoutOp : OneFlow_BaseOp<"fused_matmul_bias_add_relu_dropout", [NoMemoryEffect, AttrSizedOperandSegments, AttrSizedResultSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    Variadic<OneFlow_Tensor>:$weights,
    Variadic<OneFlow_Tensor>:$biases
  );
  let output = (outs
    OneFlow_Tensor:$out,
    Variadic<OneFlow_Tensor>:$cublas_aux,
    Variadic<OneFlow_Tensor>:$hidden
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$skip_final_activation,
    DefaultValuedAttr<SI64Attr, "0">:$seed,
    F32ArrayAttr:$dropout_rate_list
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FusedReluDropoutGradOp : OneFlow_BaseOp<"fused_relu_dropout_grad", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$mask
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$scale
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FusedGluOp : OneFlow_BaseOp<"fused_glu", [NoMemoryEffect, AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$w,
    Optional<OneFlow_Tensor>:$b,
    Optional<OneFlow_Tensor>:$v,
    Optional<OneFlow_Tensor>:$c
  );
  let attrs = (ins
    DefaultValuedAttr<StrAttr, "\"none\"">:$activation,
    DefaultValuedAttr<BoolAttr, "false">:$has_bias,
    DefaultValuedAttr<BoolAttr, "false">:$is_split
  );
  let trait_attrs = (ins
    DenseI32ArrayAttr:$operand_segment_sizes
  );
  let output = (outs
    OneFlow_Tensor:$y,
    OneFlow_Tensor:$matmul_wx,
    Optional<OneFlow_Tensor>:$matmul_vx
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FusedGluWithoutLinearGradOp : OneFlow_BaseOp<"fused_glu_without_linear_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$matmul_wx,
    Optional<OneFlow_Tensor>:$matmul_vx
  );
  let attrs = (ins
    DefaultValuedAttr<StrAttr, "\"none\"">:$activation
  );
  let output = (outs
    OneFlow_Tensor:$d_matmul_wx,
    Optional<OneFlow_Tensor>:$d_matmul_vx
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_GroupedMatmulBiasOp : OneFlow_BaseOp<"grouped_matmul_bias", [NoMemoryEffect, AttrSizedOperandSegments,  DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    Variadic<OneFlow_Tensor>:$xs,
    Variadic<OneFlow_Tensor>:$weights,
    Variadic<OneFlow_Tensor>:$biases
  );
  let output = (outs
    Variadic<OneFlow_Tensor>:$ys
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

#endif // GET_ONEFLOW_MATMUL_OP_DEFINITIONS


#ifdef GET_ONEFLOW_MISC_OP_DEFINITIONS

def OneFlow_CategoricalOrdinalEncodeOp : OneFlow_BaseOp<"CategoricalOrdinalEncode", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$table,
    OneFlow_Tensor:$size,
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$hash_precomputed
  );
  let has_check_fn = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_AddNOp : OneFlow_BaseOp<"add_n", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    Variadic<OneFlow_Tensor>:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let hasCanonicalizer = 1;
  let has_check_fn = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ArangeOp : OneFlow_BaseOp<"arange", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$integer_start,
    DefaultValuedAttr<SI64Attr, "0">:$integer_delta,
    DefaultValuedAttr<SI64Attr, "0">:$integer_limit,
    DefaultValuedAttr<F64Attr, "0.">:$float_start,
    DefaultValuedAttr<F64Attr, "0.">:$float_delta,
    DefaultValuedAttr<F64Attr, "0.">:$float_limit,
    OneFlow_DataType:$dtype,
    StrArrayAttr:$nd_sbp
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

def OneFlow_BinCountOp : OneFlow_BaseOp<"bincount", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in,
    Optional<OneFlow_Tensor>:$weight
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$size
  );
  let has_data_type_infer_fn = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
}

def OneFlow_CoinFlipOp : OneFlow_BaseOp<"coin_flip", [NoMemoryEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.5">:$probability,
    DefaultValuedAttr<SI64Attr, "0">:$batch_size,
    DefaultValuedAttr<SI64Attr, "-1">:$seed,
    DefaultValuedAttr<BoolAttr, "false">:$has_seed,
    StrArrayAttr:$nd_sbp
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

def OneFlow_ConcatOp : OneFlow_BaseOp<"cat", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>, DeclareOpInterfaceMethods<NCHWCompatibleInterface>]> {
  let input = (ins
    Variadic<OneFlow_Tensor>:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$axis,
    DefaultValuedAttr<SI64Attr, "0">:$max_dim_size
  );
  let has_check_fn = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_TensorConstantOp : OneFlow_BaseOp<"tensor_constant", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    OneFlow_DataType:$dtype,
    ShapeAttr:$shape,
    StrArrayAttr:$nd_sbp
  );
  let same_output_regst_num = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

def OneFlow_ConstantOp : OneFlow_BaseOp<"constant", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    ComplexDoubleAttr:$complex_value,
    DefaultValuedAttr<F64Attr, "0.">:$floating_value,
    DefaultValuedAttr<SI64Attr, "0">:$integer_value,
    DefaultValuedAttr<BoolAttr, "false">:$is_floating_value,
    DefaultValuedAttr<BoolAttr, "false">:$is_complex_value,
    OneFlow_DataType:$dtype,
    ShapeAttr:$shape,
    StrArrayAttr:$nd_sbp
  );
  let same_output_regst_num = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

def OneFlow_DropoutOp : OneFlow_BaseOp<"dropout", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in,
    Optional<OneFlow_Tensor>:$_add_to_output
  );
  let output = (outs
    OneFlow_Tensor:$out,
    OneFlow_Tensor:$mask
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$rate,
    DefaultValuedAttr<SI64Attr, "0">:$seed
  );
  let has_check_fn = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ElementwiseMaximumBackwardOp : OneFlow_BaseOp<"elementwise_maximum_backward", [NoMemoryEffect, AttrSizedResultSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dz,
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$y
  );
  let output = (outs
    Optional<OneFlow_Tensor>:$dx,
    Optional<OneFlow_Tensor>:$dy
  );
  let trait_attrs = (ins
    DenseI32ArrayAttr:$result_segment_sizes
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ElementwiseMinimumBackwardOp : OneFlow_BaseOp<"elementwise_minimum_backward", [NoMemoryEffect, AttrSizedResultSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dz,
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$y
  );
  let output = (outs
    Optional<OneFlow_Tensor>:$dx,
    Optional<OneFlow_Tensor>:$dy
  );
  let trait_attrs = (ins
    DenseI32ArrayAttr:$result_segment_sizes
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_EmptyOp : OneFlow_BaseOp<"empty", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    OneFlow_DataType:$dtype,
    ShapeAttr:$shape,
    StrArrayAttr:$nd_sbp,
    DefaultValuedAttr<BoolAttr, "false">:$pin_memory,
    StrAttr:$device_type,
    DefaultValuedAttr<SI64Attr, "0">:$device_id
  );
  let same_output_regst_num = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
  let has_device_and_stream_infer_fn = 1;
}

def OneFlow_EyeOp : OneFlow_BaseOp<"eye", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$rows,
    DefaultValuedAttr<SI64Attr, "0">:$cols,
    OneFlow_DataType:$dtype,
    StrArrayAttr:$nd_sbp
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_GridSampleGradOp : OneFlow_BaseOp<"grid_sample_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$doutput,
    OneFlow_Tensor:$input,
    OneFlow_Tensor:$grid
  );
  let output = (outs
    OneFlow_Tensor:$dinput,
    OneFlow_Tensor:$dgrid
  );
  let attrs = (ins
    StrAttr:$interpolation_mode,
    StrAttr:$padding_mode,
    DefaultValuedAttr<BoolAttr, "false">:$align_corners
  );
  let has_check_fn = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_MultiCountNotFiniteOp : OneFlow_BaseOp<"multi_count_not_finite", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    Variadic<OneFlow_Tensor>:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_check_fn = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_MultiSquareSumOp : OneFlow_BaseOp<"multi_square_sum", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    Variadic<OneFlow_Tensor>:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_check_fn = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_MultiReduceSumPowAbsOp : OneFlow_BaseOp<"multi_reduce_sum_pow_abs", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    Variadic<OneFlow_Tensor>:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0">:$p
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_get_sbp_fn = 1;
}

def OneFlow_MultiReduceMaxAbsOp : OneFlow_BaseOp<"multi_reduce_max_abs", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    Variadic<OneFlow_Tensor>:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_get_sbp_fn = 1;
}

def OneFlow_MultiReduceMinAbsOp : OneFlow_BaseOp<"multi_reduce_min_abs", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    Variadic<OneFlow_Tensor>:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_get_sbp_fn = 1;
}

def OneFlow_LocalMultiReduceMaxAbsOp : OneFlow_BaseOp<"local_multi_reduce_max_abs", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    Variadic<OneFlow_Tensor>:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_get_sbp_fn = 1;
}

def OneFlow_LocalMultiReduceMinAbsOp : OneFlow_BaseOp<"local_multi_reduce_min_abs", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    Variadic<OneFlow_Tensor>:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_get_sbp_fn = 1;
}

def OneFlow_NLLOp : OneFlow_BaseOp<"nll", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$input,
    OneFlow_Tensor:$target,
    Optional<OneFlow_Tensor>:$weight
  );
  let output = (outs
    OneFlow_Tensor:$output,
    OneFlow_Tensor:$out_weight
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$ignore_index
  );
  let has_data_type_infer_fn = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_NLLGradOp : OneFlow_BaseOp<"nll_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$out_grad,
    OneFlow_Tensor:$input,
    OneFlow_Tensor:$target,
    Optional<OneFlow_Tensor>:$weight
  );
  let output = (outs
    OneFlow_Tensor:$in_grad
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$ignore_index
  );
  let has_data_type_infer_fn = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
}

def OneFlow_PowXGradOp : OneFlow_BaseOp<"pow_x_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$y,
    OneFlow_Tensor:$dz
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_PowYGradOp : OneFlow_BaseOp<"pow_y_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$y,
    OneFlow_Tensor:$dz
  );
  let output = (outs
    OneFlow_Tensor:$dy
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_PreluGradOp : OneFlow_BaseOp<"prelu_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$alpha
  );
  let output = (outs
    OneFlow_Tensor:$dx,
    OneFlow_Tensor:$alpha_diff
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "true">:$alpha_requires_grad
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_RandpermOp : OneFlow_BaseOp<"randperm", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$n,
    DefaultValuedAttr<SI64Attr, "0">:$seed,
    StrArrayAttr:$nd_sbp
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

def OneFlow_RecvOp : OneFlow_BaseOp<"recv", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$src_process_id,
    OneFlow_DataType:$dtype,
    ShapeAttr:$shape,
    StrAttr:$device_type,
    DefaultValuedAttr<SI64Attr, "0">:$device_id
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_device_and_stream_infer_fn = 1;
}

def OneFlow_SendOp : OneFlow_BaseOp<"send", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$dst_process_id
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_device_and_stream_infer_fn = 1;
}

def OneFlow_SplitLikeOp : OneFlow_BaseOp<"split_like", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in,
    Variadic<OneFlow_Tensor>:$like
  );
  let output = (outs
    Variadic<OneFlow_Tensor>:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$axis
  );
  let has_check_fn = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_SspVariableProxyOp : OneFlow_BaseOp<"ssp_variable_proxy", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$var
  );
  let output = (outs
    OneFlow_Tensor:$ref,
    OneFlow_Tensor:$value
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "1">:$buffer_size
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_output_arg_modify_fn = 1;
}

def OneFlow_TfPreluGradOp : OneFlow_BaseOp<"tf_prelu_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$alpha
  );
  let output = (outs
    OneFlow_Tensor:$dx,
    OneFlow_Tensor:$alpha_diff
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_UniformOp : OneFlow_BaseOp<"uniform", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.">:$from,
    DefaultValuedAttr<F64Attr, "1.">:$to,
    DefaultValuedAttr<SI64Attr, "0">:$seed,
    OneFlow_DataType:$dtype,
    ShapeAttr:$shape,
    StrArrayAttr:$nd_sbp
  );
  let same_output_regst_num = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
  let has_dump_nd_sbp_signature_for_op_conf_fn = 1;
}

def OneFlow_UniformIntOp : OneFlow_BaseOp<"uniform_int", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$from,
    DefaultValuedAttr<SI64Attr, "1">:$to,
    DefaultValuedAttr<SI64Attr, "0">:$seed,
    OneFlow_DataType:$dtype,
    ShapeAttr:$shape,
    StrArrayAttr:$nd_sbp
  );
  let same_output_regst_num = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

def OneFlow_ExponentialOp : OneFlow_BaseOp<"exponential", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$seed,
    DefaultValuedAttr<F32Attr, "1.0">:$lambd,
    OneFlow_DataType:$dtype,
    ShapeAttr:$out_shape,
    StrArrayAttr:$nd_sbp
  );
  let same_output_regst_num = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

def OneFlow_MultinomialWithReplacementOp : OneFlow_BaseOp<"multinomial_with_replacement", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    Optional<OneFlow_Tensor>:$prefix_sum
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$seed,
    DefaultValuedAttr<SI32Attr, "1">:$num_samples
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_UniqueOp : OneFlow_BaseOp<"unique", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y,
    OneFlow_Tensor:$idx,
    OneFlow_Tensor:$num_unique
  );
  let attrs = (ins
    OneFlow_DataType:$out_idx,
    DefaultValuedAttr<BoolAttr, "true">:$sorted
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_UniqueWithCountsOp : OneFlow_BaseOp<"unique_with_counts", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y,
    OneFlow_Tensor:$idx,
    OneFlow_Tensor:$num_unique,
    OneFlow_Tensor:$count
  );
  let attrs = (ins
    OneFlow_DataType:$out_idx,
    DefaultValuedAttr<BoolAttr, "true">:$sorted
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_XdivyXGradOp : OneFlow_BaseOp<"xdivy_x_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$y,
    OneFlow_Tensor:$dz
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_XdivyYGradOp : OneFlow_BaseOp<"xdivy_y_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$y,
    OneFlow_Tensor:$dz
  );
  let output = (outs
    OneFlow_Tensor:$dy
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_StackOp : OneFlow_BaseOp<"stack", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    Variadic<OneFlow_Tensor>:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$axis,
    DefaultValuedAttr<SI64Attr, "0">:$max_dim_size
  );
  let has_check_fn = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_StackGradOp : OneFlow_BaseOp<"stack_grad", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in,
    Variadic<OneFlow_Tensor>:$like
  );
  let output = (outs
    Variadic<OneFlow_Tensor>:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$axis
  );
  let has_check_fn = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_FusedWeightedSumOp : OneFlow_BaseOp<"fused_weighted_sum", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    Variadic<OneFlow_Tensor>:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    F32ArrayAttr:$weights,
    DefaultValuedAttr<F32Attr, "1.0">:$alpha
  );
  let has_check_fn = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_DependOp : OneFlow_BaseOp<"depend", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in,
    OneFlow_Tensor:$depend_tensor
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

#endif // GET_ONEFLOW_MISC_OP_DEFINITIONS


#ifdef GET_ONEFLOW_NCCL_OP_DEFINITIONS

def OneFlow__ncclLogical_2DSameDim0All2allOp : OneFlow_BaseOp<"_nccl_logical_2D_same_dim0_all2all", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    StrArrayAttr:$src_reduced_nd_sbp,
    StrArrayAttr:$dst_reduced_nd_sbp
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_device_and_stream_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

def OneFlow__ncclLogical_2DSameDim0AllGatherOp : OneFlow_BaseOp<"_nccl_logical_2D_same_dim0_all_gather", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    StrArrayAttr:$src_reduced_nd_sbp,
    StrArrayAttr:$dst_reduced_nd_sbp
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_device_and_stream_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

def OneFlow__ncclLogical_2DSameDim0AllGatherNoncontinuousOp : OneFlow_BaseOp<"_nccl_logical_2D_same_dim0_all_gather_noncontinuous", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    StrArrayAttr:$src_reduced_nd_sbp,
    StrArrayAttr:$dst_reduced_nd_sbp
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_device_and_stream_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

def OneFlow__ncclLogical_2DSameDim0AllReduceOp : OneFlow_BaseOp<"_nccl_logical_2D_same_dim0_all_reduce", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    StrArrayAttr:$src_reduced_nd_sbp,
    StrArrayAttr:$dst_reduced_nd_sbp
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_device_and_stream_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

def OneFlow__ncclLogical_2DSameDim1AllReduceOp : OneFlow_BaseOp<"_nccl_logical_2D_same_dim1_all_reduce", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    StrArrayAttr:$src_reduced_nd_sbp,
    StrArrayAttr:$dst_reduced_nd_sbp
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_device_and_stream_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

def OneFlow__ncclLogicalAllGatherOp : OneFlow_BaseOp<"_nccl_logical_all_gather", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    StrArrayAttr:$src_reduced_nd_sbp,
    StrArrayAttr:$dst_reduced_nd_sbp
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_device_and_stream_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

def OneFlow__ncclLogicalAllGatherNoncontinuousOp : OneFlow_BaseOp<"_nccl_logical_all_gather_noncontinuous", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    StrArrayAttr:$src_reduced_nd_sbp,
    StrArrayAttr:$dst_reduced_nd_sbp
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_device_and_stream_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

def OneFlow__ncclLogicalAllReduceOp : OneFlow_BaseOp<"_nccl_logical_all_reduce", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    StrArrayAttr:$src_reduced_nd_sbp,
    StrArrayAttr:$dst_reduced_nd_sbp
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_device_and_stream_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

def OneFlow__ncclLogicalReduceScatterOp : OneFlow_BaseOp<"_nccl_logical_reduce_scatter", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    StrArrayAttr:$src_reduced_nd_sbp,
    StrArrayAttr:$dst_reduced_nd_sbp
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_device_and_stream_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

def OneFlow__ncclLogicalReduceScatterNoncontinuousOp : OneFlow_BaseOp<"_nccl_logical_reduce_scatter_noncontinuous", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    StrArrayAttr:$src_reduced_nd_sbp,
    StrArrayAttr:$dst_reduced_nd_sbp
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_device_and_stream_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

def OneFlow__ncclLogicalS2sOp : OneFlow_BaseOp<"_nccl_logical_s2s", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    StrArrayAttr:$src_reduced_nd_sbp,
    StrArrayAttr:$dst_reduced_nd_sbp
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_device_and_stream_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

def OneFlow__ncclLogicalSendRecvOp : OneFlow_BaseOp<"_nccl_logical_send_recv", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    StrArrayAttr:$src_reduced_nd_sbp,
    StrArrayAttr:$dst_reduced_nd_sbp
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_device_and_stream_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

def OneFlow__ncclLogicalFusionOp : OneFlow_BaseOp<"_nccl_logical_fusion", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    Variadic<OneFlow_Tensor>:$in
  );
  let output = (outs
    Variadic<OneFlow_Tensor>:$out
  );
  let attrs = (ins
    StrArrayAttr:$src_nd_sbp_str_list,
    StrArrayAttr:$dst_nd_sbp_str_list,
    StrArrayAttr:$nccl_type_list
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_device_and_stream_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

#endif // GET_ONEFLOW_NCCL_OP_DEFINITIONS


#ifdef GET_ONEFLOW_NORMALIZATION_OP_DEFINITIONS

def OneFlow_NormalizationAddReluOp : OneFlow_BaseOp<"normalization_add_relu", [NoMemoryEffect, AttrSizedOperandSegments, AttrSizedResultSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>, DeclareOpInterfaceMethods<InferTypeOpInterface, ["refineReturnTypes"]>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    Optional<OneFlow_Tensor>:$addend,
    Optional<OneFlow_Tensor>:$moving_mean,
    Optional<OneFlow_Tensor>:$moving_variance,
    OneFlow_Tensor:$gamma,
    OneFlow_Tensor:$beta
  );
  let output = (outs
    OneFlow_Tensor:$y,
    OneFlow_Tensor:$reserve_space,
    Optional<OneFlow_Tensor>:$mean,
    Optional<OneFlow_Tensor>:$inv_variance
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$axis,
    DefaultValuedAttr<F32Attr, "0.">:$epsilon,
    DefaultValuedAttr<BoolAttr, "false">:$training,
    DefaultValuedAttr<F32Attr, "0.">:$momentum
  );
  let trait_attrs = (ins
    DenseI32ArrayAttr:$operand_segment_sizes,
    DenseI32ArrayAttr:$result_segment_sizes
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_BatchNormStatsOp : OneFlow_BaseOp<"batch_norm_stats", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$input
  );
  let output = (outs
    OneFlow_Tensor:$mean,
    OneFlow_Tensor:$invstd
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "1">:$axis,
    DefaultValuedAttr<F32Attr, "0.00001">:$eps
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_BatchNormGatherStatsWithCountsOp : OneFlow_BaseOp<"batch_norm_gather_stats_with_counts", [NoMemoryEffect, AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$input,
    OneFlow_Tensor:$mean,
    OneFlow_Tensor:$invstd,
    OneFlow_Tensor:$counts,
    Optional<OneFlow_Tensor>:$running_mean,
    Optional<OneFlow_Tensor>:$running_var
  );
  let output = (outs
    OneFlow_Tensor:$global_mean,
    OneFlow_Tensor:$global_invstd
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.00001">:$eps,
    DefaultValuedAttr<F32Attr, "0.9">:$momentum
  );
  let trait_attrs = (ins
    DenseI32ArrayAttr:$operand_segment_sizes
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_BatchNormElemtOp : OneFlow_BaseOp<"batch_norm_elemt", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$input,
    OneFlow_Tensor:$weight,
    OneFlow_Tensor:$bias,
    OneFlow_Tensor:$mean,
    OneFlow_Tensor:$invstd
  );
  let output = (outs
    OneFlow_Tensor:$output
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "1">:$axis,
    DefaultValuedAttr<F32Attr, "0.00001">:$eps
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_BatchNormBackwardReduceOp : OneFlow_BaseOp<"batch_norm_backward_reduce", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$grad_out,
    OneFlow_Tensor:$input,
    OneFlow_Tensor:$mean,
    OneFlow_Tensor:$invstd
  );
  let output = (outs
    OneFlow_Tensor:$sum_dy,
    OneFlow_Tensor:$sum_dy_xmu,
    OneFlow_Tensor:$grad_weight,
    OneFlow_Tensor:$grad_bias
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "1">:$axis
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_BatchNormBackwardElemtOp : OneFlow_BaseOp<"batch_norm_backward_elemt", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$grad_out,
    OneFlow_Tensor:$input,
    OneFlow_Tensor:$mean,
    OneFlow_Tensor:$invstd,
    OneFlow_Tensor:$weight,
    OneFlow_Tensor:$sum_dy,
    OneFlow_Tensor:$sum_dy_xmu,
    OneFlow_Tensor:$count
  );
  let output = (outs
    OneFlow_Tensor:$grad_in
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "1">:$axis
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_CropMirrorNormalizeFromTensorbufferOp : OneFlow_BaseOp<"crop_mirror_normalize_from_tensorbuffer", [NoMemoryEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in,
    Optional<OneFlow_Tensor>:$mirror
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<StrAttr, "\"BGR\"">:$color_space,
    DefaultValuedAttr<StrAttr, "\"NCHW\"">:$output_layout,
    F32ArrayAttr:$mean,
    F32ArrayAttr:$std,
    DefaultValuedAttr<SI64Attr, "0">:$crop_h,
    DefaultValuedAttr<SI64Attr, "0">:$crop_w,
    DefaultValuedAttr<F32Attr, "0.5">:$crop_pos_x,
    DefaultValuedAttr<F32Attr, "0.5">:$crop_pos_y,
    OneFlow_DataType:$output_dtype
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_CropMirrorNormalizeFromUint8Op : OneFlow_BaseOp<"crop_mirror_normalize_from_uint8", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in,
    Optional<OneFlow_Tensor>:$mirror
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<StrAttr, "\"BGR\"">:$color_space,
    DefaultValuedAttr<StrAttr, "\"NCHW\"">:$output_layout,
    F32ArrayAttr:$mean,
    F32ArrayAttr:$std,
    DefaultValuedAttr<SI64Attr, "0">:$crop_h,
    DefaultValuedAttr<SI64Attr, "0">:$crop_w,
    DefaultValuedAttr<F32Attr, "0.5">:$crop_pos_x,
    DefaultValuedAttr<F32Attr, "0.5">:$crop_pos_y,
    OneFlow_DataType:$output_dtype
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ImageNormalizeOp : OneFlow_BaseOp<"image_normalize", [NoMemoryEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    F32ArrayAttr:$std,
    F32ArrayAttr:$mean
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_L2NormalizeOp : OneFlow_BaseOp<"l2_normalize", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y,
    OneFlow_Tensor:$square_x_sum
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$axis,
    DefaultValuedAttr<F32Attr, "0.">:$epsilon
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_L2NormalizeGradOp : OneFlow_BaseOp<"l2_normalize_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$y,
    OneFlow_Tensor:$square_x_sum
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$axis,
    DefaultValuedAttr<F32Attr, "0.">:$epsilon
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_LayerNormOp : OneFlow_BaseOp<"layer_norm", [NoMemoryEffect, AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    Optional<OneFlow_Tensor>:$beta,
    Optional<OneFlow_Tensor>:$gamma
  );
  let output = (outs
    OneFlow_Tensor:$y,
    OneFlow_Tensor:$mean,
    OneFlow_Tensor:$inv_variance
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$center,
    DefaultValuedAttr<BoolAttr, "false">:$scale,
    DefaultValuedAttr<SI64Attr, "0">:$begin_norm_axis,
    DefaultValuedAttr<SI64Attr, "0">:$begin_params_axis,
    DefaultValuedAttr<F64Attr, "0.">:$epsilon
  );
  let trait_attrs = (ins
    DenseI32ArrayAttr:$operand_segment_sizes
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_SkipLayerNormOp : OneFlow_BaseOp<"skip_layer_norm", [NoMemoryEffect, AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    Optional<OneFlow_Tensor>:$gamma,
    Optional<OneFlow_Tensor>:$beta,
    Optional<OneFlow_Tensor>:$bias,
    Optional<OneFlow_Tensor>:$skip
  );
  let output = (outs
    OneFlow_Tensor:$y,
    OneFlow_Tensor:$mean,
    OneFlow_Tensor:$inv_variance
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.00001">:$epsilon,
    DefaultValuedAttr<F64Attr, "1.0">:$alpha
  );
  let trait_attrs = (ins
    DenseI32ArrayAttr:$operand_segment_sizes
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_LayerNormGradOp : OneFlow_BaseOp<"layer_norm_grad", [NoMemoryEffect, AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$mean,
    OneFlow_Tensor:$inv_variance,
    Optional<OneFlow_Tensor>:$gamma,
    Optional<OneFlow_Tensor>:$_add_to_output
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$begin_norm_axis,
    DefaultValuedAttr<F64Attr, "0.">:$epsilon
  );
  let trait_attrs = (ins
    DenseI32ArrayAttr:$operand_segment_sizes
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_LayerNormParamGradOp : OneFlow_BaseOp<"layer_norm_param_grad", [NoMemoryEffect, AttrSizedResultSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$mean,
    OneFlow_Tensor:$inv_variance
  );
  let output = (outs
    Optional<OneFlow_Tensor>:$beta_diff,
    Optional<OneFlow_Tensor>:$gamma_diff
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$begin_params_axis
  );
  let trait_attrs = (ins
    DenseI32ArrayAttr:$result_segment_sizes
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_NormalOp : OneFlow_BaseOp<"normal", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.">:$mean,
    DefaultValuedAttr<F64Attr, "1.">:$std,
    DefaultValuedAttr<SI64Attr, "0">:$seed,
    OneFlow_DataType:$dtype,
    ShapeAttr:$shape,
    StrArrayAttr:$nd_sbp
  );
  let same_output_regst_num = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

def OneFlow_NormalizationOp : OneFlow_NormalizationBaseOp<"normalization", [AttrSizedResultSegments, DeclareOpInterfaceMethods<NCHWCompatibleInterface>]> {
  let hasCanonicalizer = 1;
}

def OneFlow_NormalizationGradOp : OneFlow_BaseOp<"normalization_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$mean,
    OneFlow_Tensor:$inv_variance,
    OneFlow_Tensor:$gamma
  );
  let output = (outs
    OneFlow_Tensor:$gamma_diff,
    OneFlow_Tensor:$beta_diff,
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$axis,
    DefaultValuedAttr<F32Attr, "0.">:$epsilon
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_GroupNormOp : OneFlow_BaseOp<"group_norm", [NoMemoryEffect, AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>, DeclareOpInterfaceMethods<NCHWCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    Optional<OneFlow_Tensor>:$beta,
    Optional<OneFlow_Tensor>:$gamma
  );
  let output = (outs
    OneFlow_Tensor:$y,
    OneFlow_Tensor:$mean,
    OneFlow_Tensor:$inv_variance
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$affine,
    DefaultValuedAttr<SI32Attr, "0">:$num_groups,
    DefaultValuedAttr<F64Attr, "0.">:$epsilon,
    DefaultValuedAttr<StrAttr, "\"channels_first\"">:$data_format,
    DefaultValuedAttr<StrAttr, "\"none\"">:$activation
  );
  let trait_attrs = (ins
    DenseI32ArrayAttr:$operand_segment_sizes
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_GroupNormGradOp : OneFlow_BaseOp<"group_norm_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$mean,
    OneFlow_Tensor:$inv_variance,
    Optional<OneFlow_Tensor>:$gamma
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$num_groups,
    DefaultValuedAttr<F64Attr, "0.">:$epsilon
  );
  let trait_attrs = (ins
    DenseI32ArrayAttr:$operand_segment_sizes
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_GroupNormParamGradOp : OneFlow_BaseOp<"group_norm_param_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$mean,
    OneFlow_Tensor:$inv_variance
  );
  let output = (outs
    OneFlow_Tensor:$dgamma,
    OneFlow_Tensor:$dbeta
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_RmsNormOp : OneFlow_BaseOp<"rms_norm", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    Optional<OneFlow_Tensor>:$weight
  );
  let output = (outs
    OneFlow_Tensor:$y,
    OneFlow_Tensor:$inv_rms
  );
  let attrs = (ins
    ShapeAttr:$normalized_shape,
    DefaultValuedAttr<F32Attr, "0.00001">:$epsilon
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_RmsNormParamGradOp : OneFlow_BaseOp<"rms_norm_param_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$inv_rms
  );
  let output = (outs
    OneFlow_Tensor:$weight_grad
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_RmsNormGradOp : OneFlow_BaseOp<"rms_norm_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$inv_rms,
    Optional<OneFlow_Tensor>:$weight
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_SkipRmsNormOp : OneFlow_BaseOp<"skip_rms_norm", [NoMemoryEffect, AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    Optional<OneFlow_Tensor>:$weight,
    Optional<OneFlow_Tensor>:$bias,
    Optional<OneFlow_Tensor>:$skip
  );
  let output = (outs
    OneFlow_Tensor:$y,
    OneFlow_Tensor:$inv_rms
  );
  let attrs = (ins
    ShapeAttr:$normalized_shape,
    DefaultValuedAttr<F32Attr, "0.00001">:$epsilon,
    DefaultValuedAttr<F32Attr, "1.0">:$alpha
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

#endif // GET_ONEFLOW_NORMALIZATION_OP_DEFINITIONS


#ifdef GET_ONEFLOW_OPTIMIZER_OP_DEFINITIONS

def OneFlow_AdagradUpdateOp : OneFlow_BaseOp<"adagrad_update", [NoGrad, AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$model,
    OneFlow_Tensor:$model_diff,
    Optional<OneFlow_Tensor>:$learning_rate,
    Optional<OneFlow_Tensor>:$scale_by_tensor,
    Optional<OneFlow_Tensor>:$skip_if,
    Optional<OneFlow_Tensor>:$train_step,
    OneFlow_Tensor:$sum
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$train_step_val,
    DefaultValuedAttr<F32Attr, "0.">:$learning_rate_val,
    DefaultValuedAttr<F32Attr, "1.">:$learning_rate_scale,
    DefaultValuedAttr<F64Attr, "1.">:$scale,
    DefaultValuedAttr<F32Attr, "0.">:$l1,
    DefaultValuedAttr<F32Attr, "0.">:$l2,
    DefaultValuedAttr<F32Attr, "0.">:$lr_decay,
    DefaultValuedAttr<F32Attr, "0.">:$weight_decay,
    DefaultValuedAttr<F32Attr, "0.">:$epsilon
  );
  let trait_attrs = (ins
    DenseI32ArrayAttr:$operand_segment_sizes
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_AdamBiasCorrectionFactorOp : OneFlow_BaseOp<"adam_bias_correction_factor", [NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$train_step
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.9">:$beta
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_AdamUpdateOp : OneFlow_BaseOp<"adam_update", [NoGrad, AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$model,
    OneFlow_Tensor:$model_diff,
    Optional<OneFlow_Tensor>:$model_copy,
    Optional<OneFlow_Tensor>:$learning_rate,
    Optional<OneFlow_Tensor>:$scale_by_tensor,
    Optional<OneFlow_Tensor>:$skip_if,
    Optional<OneFlow_Tensor>:$bias_correction1,
    Optional<OneFlow_Tensor>:$bias_correction2,
    OneFlow_Tensor:$m,
    OneFlow_Tensor:$v,
    Optional<OneFlow_Tensor>:$max_v
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$learning_rate_val,
    DefaultValuedAttr<F32Attr, "1.">:$learning_rate_scale,
    DefaultValuedAttr<F32Attr, "1.">:$bias_correction1_val,
    DefaultValuedAttr<F32Attr, "1.">:$bias_correction2_val,
    DefaultValuedAttr<F64Attr, "1.">:$scale,
    DefaultValuedAttr<F32Attr, "0.">:$l1,
    DefaultValuedAttr<F32Attr, "0.">:$l2,
    DefaultValuedAttr<F32Attr, "0.9">:$beta1,
    DefaultValuedAttr<F32Attr, "0.999">:$beta2,
    DefaultValuedAttr<F32Attr, "0.">:$epsilon,
    DefaultValuedAttr<F32Attr, "0.">:$weight_decay,
    DefaultValuedAttr<BoolAttr, "false">:$amsgrad,
    DefaultValuedAttr<BoolAttr, "true">:$do_bias_correction
  );
  let trait_attrs = (ins
    DenseI32ArrayAttr:$operand_segment_sizes
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_IndexedSlicesAdamUpdateOp : OneFlow_BaseOp<"indexed_slices_adam_update", [NoGrad, AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$model,
    OneFlow_Tensor:$model_diff_indices,
    OneFlow_Tensor:$model_diff_values,
    OneFlow_Tensor:$learning_rate,
    Optional<OneFlow_Tensor>:$bias_correction1,
    Optional<OneFlow_Tensor>:$bias_correction2,
    OneFlow_Tensor:$m,
    OneFlow_Tensor:$v,
    Optional<OneFlow_Tensor>:$max_v
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$learning_rate_val,
    DefaultValuedAttr<F32Attr, "1.">:$learning_rate_scale,
    DefaultValuedAttr<F32Attr, "0.9">:$beta1,
    DefaultValuedAttr<F32Attr, "0.999">:$beta2,
    DefaultValuedAttr<F32Attr, "0.">:$epsilon,
    DefaultValuedAttr<F32Attr, "0.">:$weight_decay,
    DefaultValuedAttr<BoolAttr, "false">:$amsgrad,
    DefaultValuedAttr<BoolAttr, "true">:$do_bias_correction
  );
  let trait_attrs = (ins
    DenseI32ArrayAttr:$operand_segment_sizes
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_IndexedSlicesMomentumUpdateOp : OneFlow_BaseOp<"indexed_slices_momentum_update", [NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$model,
    OneFlow_Tensor:$model_diff_indices,
    OneFlow_Tensor:$model_diff_values,
    OneFlow_Tensor:$learning_rate,
    OneFlow_Tensor:$momentum
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "1.">:$learning_rate_scale,
    DefaultValuedAttr<F32Attr, "0.9">:$beta,
    DefaultValuedAttr<F32Attr, "0.0">:$dampening,
    DefaultValuedAttr<BoolAttr, "false">:$nesterov,
    DefaultValuedAttr<BoolAttr, "false">:$maximize,
    DefaultValuedAttr<F32Attr, "0.">:$weight_decay
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_IndexedSlicesSgdUpdateOp : OneFlow_BaseOp<"indexed_slices_sgd_update", [NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$model,
    OneFlow_Tensor:$model_diff_indices,
    OneFlow_Tensor:$model_diff_values,
    OneFlow_Tensor:$learning_rate
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "1.">:$learning_rate_scale,
    DefaultValuedAttr<F32Attr, "0.">:$weight_decay
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_LambUpdateOp : OneFlow_BaseOp<"lamb_update", [NoGrad, AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$model,
    OneFlow_Tensor:$model_diff,
    Optional<OneFlow_Tensor>:$learning_rate,
    Optional<OneFlow_Tensor>:$scale_by_tensor,
    Optional<OneFlow_Tensor>:$skip_if,
    Optional<OneFlow_Tensor>:$bias_correction1,
    Optional<OneFlow_Tensor>:$bias_correction2,
    OneFlow_Tensor:$m,
    OneFlow_Tensor:$v
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$learning_rate_val,
    DefaultValuedAttr<F32Attr, "1.">:$learning_rate_scale,
    DefaultValuedAttr<F32Attr, "1.">:$bias_correction1_val,
    DefaultValuedAttr<F32Attr, "1.">:$bias_correction2_val,
    DefaultValuedAttr<F64Attr, "1.">:$scale,
    DefaultValuedAttr<F32Attr, "0.">:$l1,
    DefaultValuedAttr<F32Attr, "0.">:$l2,
    DefaultValuedAttr<F32Attr, "0.9">:$beta1,
    DefaultValuedAttr<F32Attr, "0.999">:$beta2,
    DefaultValuedAttr<F32Attr, "0.">:$epsilon,
    DefaultValuedAttr<F32Attr, "0.">:$weight_decay,
    DefaultValuedAttr<BoolAttr, "true">:$do_bias_correction
  );
  let trait_attrs = (ins
    DenseI32ArrayAttr:$operand_segment_sizes
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_LarsUpdateOp : OneFlow_BaseOp<"lars_update", [NoGrad, AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$model,
    OneFlow_Tensor:$model_diff,
    OneFlow_Tensor:$learning_rate,
    OneFlow_Tensor:$momentum,
    Optional<OneFlow_Tensor>:$scale_by_tensor,
    Optional<OneFlow_Tensor>:$skip_if
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "1.">:$learning_rate_scale,
    DefaultValuedAttr<F64Attr, "1.">:$scale,
    DefaultValuedAttr<F32Attr, "0.">:$l1,
    DefaultValuedAttr<F32Attr, "0.">:$l2,
    DefaultValuedAttr<F32Attr, "0.9">:$momentum_beta,
    DefaultValuedAttr<F32Attr, "0.">:$epsilon,
    DefaultValuedAttr<F32Attr, "0.0001">:$lars_coefficient,
    DefaultValuedAttr<F32Attr, "0.">:$weight_decay
  );
  let trait_attrs = (ins
    DenseI32ArrayAttr:$operand_segment_sizes
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_MomentumUpdateOp : OneFlow_BaseOp<"momentum_update", [NoGrad, AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$model,
    OneFlow_Tensor:$model_diff,
    OneFlow_Tensor:$momentum,
    Optional<OneFlow_Tensor>:$learning_rate,
    Optional<OneFlow_Tensor>:$scale_by_tensor,
    Optional<OneFlow_Tensor>:$skip_if
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$learning_rate_val,
    DefaultValuedAttr<F32Attr, "1.">:$learning_rate_scale,
    DefaultValuedAttr<F64Attr, "1.">:$scale,
    DefaultValuedAttr<F32Attr, "0.">:$l1,
    DefaultValuedAttr<F32Attr, "0.">:$l2,
    DefaultValuedAttr<F32Attr, "0.9">:$beta,
    DefaultValuedAttr<F32Attr, "0.0">:$dampening,
    DefaultValuedAttr<BoolAttr, "false">:$nesterov,
    DefaultValuedAttr<BoolAttr, "false">:$maximize,
    DefaultValuedAttr<F32Attr, "0.">:$weight_decay
  );
  let trait_attrs = (ins
    DenseI32ArrayAttr:$operand_segment_sizes
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_RmspropUpdateOp : OneFlow_BaseOp<"rmsprop_update", [NoGrad, AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$model,
    OneFlow_Tensor:$model_diff,
    Optional<OneFlow_Tensor>:$learning_rate,
    Optional<OneFlow_Tensor>:$scale_by_tensor,
    Optional<OneFlow_Tensor>:$skip_if,
    OneFlow_Tensor:$mean_square,
    Optional<OneFlow_Tensor>:$mean_gradient
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$learning_rate_val,
    DefaultValuedAttr<F32Attr, "1.">:$learning_rate_scale,
    DefaultValuedAttr<F64Attr, "1.">:$scale,
    DefaultValuedAttr<F32Attr, "0.">:$l1,
    DefaultValuedAttr<F32Attr, "0.">:$l2,
    DefaultValuedAttr<BoolAttr, "false">:$centered,
    DefaultValuedAttr<F32Attr, "0.">:$epsilon,
    DefaultValuedAttr<F32Attr, "0.99">:$decay_rate,
    DefaultValuedAttr<F32Attr, "0.">:$weight_decay
  );
  let trait_attrs = (ins
    DenseI32ArrayAttr:$operand_segment_sizes
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_SgdUpdateOp : OneFlow_BaseOp<"sgd_update", [NoGrad, AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$model,
    OneFlow_Tensor:$model_diff,
    Optional<OneFlow_Tensor>:$model_copy,
    Optional<OneFlow_Tensor>:$learning_rate,
    Optional<OneFlow_Tensor>:$scale_by_tensor,
    Optional<OneFlow_Tensor>:$skip_if
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$learning_rate_val,
    DefaultValuedAttr<F32Attr, "1.">:$learning_rate_scale,
    DefaultValuedAttr<F64Attr, "1.">:$scale,
    DefaultValuedAttr<F32Attr, "0.">:$l1,
    DefaultValuedAttr<F32Attr, "0.">:$l2,
    DefaultValuedAttr<F32Attr, "0.">:$weight_decay
  );
  let trait_attrs = (ins
    DenseI32ArrayAttr:$operand_segment_sizes
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_FtrlUpdateOp : OneFlow_BaseOp<"ftrl_update", [NoGrad, AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$model,
    OneFlow_Tensor:$model_diff,
    Optional<OneFlow_Tensor>:$learning_rate,
    Optional<OneFlow_Tensor>:$skip_if,
    OneFlow_Tensor:$accumulate,
    OneFlow_Tensor:$z
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$learning_rate_val,
    DefaultValuedAttr<F32Attr, "1.">:$learning_rate_scale,
    DefaultValuedAttr<F64Attr, "1.">:$scale,
    DefaultValuedAttr<F32Attr, "0.">:$l1,
    DefaultValuedAttr<F32Attr, "0.">:$l2,
    DefaultValuedAttr<F32Attr, "0.">:$weight_decay,
    DefaultValuedAttr<F32Attr, "0.">:$lr_power,
    DefaultValuedAttr<F32Attr, "0.">:$lambda1,
    DefaultValuedAttr<F32Attr, "0.">:$lambda2,
    DefaultValuedAttr<F32Attr, "0.">:$beta
  );
  let trait_attrs = (ins
    DenseI32ArrayAttr:$operand_segment_sizes
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_AdadeltaUpdateOp : OneFlow_BaseOp<"adadelta_update", [NoGrad, AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$model,
    OneFlow_Tensor:$model_diff,
    Optional<OneFlow_Tensor>:$learning_rate,
    Optional<OneFlow_Tensor>:$skip_if,
    OneFlow_Tensor:$square_avgs,
    OneFlow_Tensor:$acc_deltas
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$learning_rate_val,
    DefaultValuedAttr<F32Attr, "1.">:$learning_rate_scale,
    DefaultValuedAttr<F64Attr, "1.">:$scale,
    DefaultValuedAttr<F32Attr, "0.">:$l1,
    DefaultValuedAttr<F32Attr, "0.">:$l2,
    DefaultValuedAttr<F32Attr, "0.">:$weight_decay,
    DefaultValuedAttr<F32Attr, "0.9">:$rho,
    DefaultValuedAttr<F32Attr, "0.">:$epsilon,
    DefaultValuedAttr<BoolAttr, "false">:$maximize
  );
  let trait_attrs = (ins
    DenseI32ArrayAttr:$operand_segment_sizes
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_MultiTensorSgdUpdateOp : OneFlow_BaseOp<"multi_tensor_sgd_update", [NoGrad, AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    Variadic<OneFlow_Tensor>:$model,
    Variadic<OneFlow_Tensor>:$model_diff,
    Optional<OneFlow_Tensor>:$learning_rate,
    Optional<OneFlow_Tensor>:$scale_by_tensor,
    Optional<OneFlow_Tensor>:$skip_if
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$learning_rate_val,
    DefaultValuedAttr<F32Attr, "1.">:$learning_rate_scale,
    DefaultValuedAttr<F64Attr, "1.">:$scale,
    DefaultValuedAttr<F32Attr, "0.">:$l1,
    DefaultValuedAttr<F32Attr, "0.">:$l2,
    DefaultValuedAttr<F32Attr, "0.">:$weight_decay
  );
  let trait_attrs = (ins
    DenseI32ArrayAttr:$operand_segment_sizes
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_MultiTensorMomentumUpdateOp : OneFlow_BaseOp<"multi_tensor_momentum_update", [NoGrad, AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    Variadic<OneFlow_Tensor>:$model,
    Variadic<OneFlow_Tensor>:$model_diff,
    Variadic<OneFlow_Tensor>:$momentum_buf,
    Optional<OneFlow_Tensor>:$learning_rate,
    Optional<OneFlow_Tensor>:$scale_by_tensor,
    Optional<OneFlow_Tensor>:$skip_if
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$learning_rate_val,
    DefaultValuedAttr<F32Attr, "1.">:$learning_rate_scale,
    DefaultValuedAttr<F64Attr, "1.">:$scale,
    DefaultValuedAttr<F32Attr, "0.">:$l1,
    DefaultValuedAttr<F32Attr, "0.">:$l2,
    DefaultValuedAttr<F32Attr, "0.">:$weight_decay,
    DefaultValuedAttr<F32Attr, "0.">:$momentum,
    DefaultValuedAttr<F32Attr, "0.">:$dampening,
    DefaultValuedAttr<BoolAttr, "false">:$nesterov,
    DefaultValuedAttr<BoolAttr, "false">:$maximize
  );
  let trait_attrs = (ins
    DenseI32ArrayAttr:$operand_segment_sizes
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_MultiTensorAdamUpdateOp : OneFlow_BaseOp<"multi_tensor_adam_update", [NoGrad, AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    Variadic<OneFlow_Tensor>:$model,
    Variadic<OneFlow_Tensor>:$model_diff,
    Optional<OneFlow_Tensor>:$learning_rate,
    Optional<OneFlow_Tensor>:$scale_by_tensor,
    Optional<OneFlow_Tensor>:$skip_if,
    Optional<OneFlow_Tensor>:$bias_correction1,
    Optional<OneFlow_Tensor>:$bias_correction2,
    Variadic<OneFlow_Tensor>:$m,
    Variadic<OneFlow_Tensor>:$v
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$learning_rate_val,
    DefaultValuedAttr<F32Attr, "1.">:$learning_rate_scale,
    DefaultValuedAttr<F32Attr, "1.">:$bias_correction1_val,
    DefaultValuedAttr<F32Attr, "1.">:$bias_correction2_val,
    DefaultValuedAttr<F64Attr, "1.">:$scale,
    DefaultValuedAttr<F32Attr, "0.">:$l1,
    DefaultValuedAttr<F32Attr, "0.">:$l2,
    DefaultValuedAttr<F32Attr, "0.9">:$beta1,
    DefaultValuedAttr<F32Attr, "0.999">:$beta2,
    DefaultValuedAttr<F32Attr, "0.00001">:$epsilon,
    DefaultValuedAttr<F32Attr, "0.">:$weight_decay,
    DefaultValuedAttr<BoolAttr, "false">:$amsgrad,
    DefaultValuedAttr<BoolAttr, "true">:$do_bias_correction
  );
  let trait_attrs = (ins
    DenseI32ArrayAttr:$operand_segment_sizes
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_MultiTensorSgdUpdateWithCastOp : OneFlow_BaseOp<"multi_tensor_sgd_update_with_cast", [NoGrad, AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    Variadic<OneFlow_Tensor>:$model,
    Variadic<OneFlow_Tensor>:$model_diff,
    Variadic<OneFlow_Tensor>:$model_copy,
    Optional<OneFlow_Tensor>:$learning_rate,
    Optional<OneFlow_Tensor>:$scale_by_tensor,
    Optional<OneFlow_Tensor>:$skip_if
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$learning_rate_val,
    DefaultValuedAttr<F32Attr, "1.">:$learning_rate_scale,
    DefaultValuedAttr<F64Attr, "1.">:$scale,
    DefaultValuedAttr<F32Attr, "0.">:$l1,
    DefaultValuedAttr<F32Attr, "0.">:$l2,
    DefaultValuedAttr<F32Attr, "0.">:$weight_decay
  );
  let trait_attrs = (ins
    DenseI32ArrayAttr:$operand_segment_sizes
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_MultiTensorMomentumUpdateWithCastOp : OneFlow_BaseOp<"multi_tensor_momentum_update_with_cast", [NoGrad, AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    Variadic<OneFlow_Tensor>:$model,
    Variadic<OneFlow_Tensor>:$model_diff,
    Variadic<OneFlow_Tensor>:$model_copy,
    Variadic<OneFlow_Tensor>:$momentum_buf,
    Optional<OneFlow_Tensor>:$learning_rate,
    Optional<OneFlow_Tensor>:$scale_by_tensor,
    Optional<OneFlow_Tensor>:$skip_if
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$learning_rate_val,
    DefaultValuedAttr<F32Attr, "1.">:$learning_rate_scale,
    DefaultValuedAttr<F64Attr, "1.">:$scale,
    DefaultValuedAttr<F32Attr, "0.">:$l1,
    DefaultValuedAttr<F32Attr, "0.">:$l2,
    DefaultValuedAttr<F32Attr, "0.">:$weight_decay,
    DefaultValuedAttr<F32Attr, "0.">:$momentum,
    DefaultValuedAttr<F32Attr, "0.">:$dampening,
    DefaultValuedAttr<BoolAttr, "false">:$nesterov,
    DefaultValuedAttr<BoolAttr, "false">:$maximize
  );
  let trait_attrs = (ins
    DenseI32ArrayAttr:$operand_segment_sizes
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_MultiTensorAdamUpdateWithCastOp : OneFlow_BaseOp<"multi_tensor_adam_update_with_cast", [NoGrad, AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    Variadic<OneFlow_Tensor>:$model,
    Variadic<OneFlow_Tensor>:$model_diff,
    Variadic<OneFlow_Tensor>:$model_copy,
    Optional<OneFlow_Tensor>:$learning_rate,
    Optional<OneFlow_Tensor>:$scale_by_tensor,
    Optional<OneFlow_Tensor>:$skip_if,
    Optional<OneFlow_Tensor>:$bias_correction1,
    Optional<OneFlow_Tensor>:$bias_correction2,
    Variadic<OneFlow_Tensor>:$m,
    Variadic<OneFlow_Tensor>:$v
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$learning_rate_val,
    DefaultValuedAttr<F32Attr, "1.">:$learning_rate_scale,
    DefaultValuedAttr<F32Attr, "1.">:$bias_correction1_val,
    DefaultValuedAttr<F32Attr, "1.">:$bias_correction2_val,
    DefaultValuedAttr<F64Attr, "1.">:$scale,
    DefaultValuedAttr<F32Attr, "0.">:$l1,
    DefaultValuedAttr<F32Attr, "0.">:$l2,
    DefaultValuedAttr<F32Attr, "0.9">:$beta1,
    DefaultValuedAttr<F32Attr, "0.999">:$beta2,
    DefaultValuedAttr<F32Attr, "0.00001">:$epsilon,
    DefaultValuedAttr<F32Attr, "0.">:$weight_decay,
    DefaultValuedAttr<BoolAttr, "false">:$amsgrad,
    DefaultValuedAttr<BoolAttr, "true">:$do_bias_correction
  );
  let trait_attrs = (ins
    DenseI32ArrayAttr:$operand_segment_sizes
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_MultiTensorYoloV5WeightUpdateOp : OneFlow_BaseOp<"multi_tensor_yolov5_weight_update", [NoGrad, AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    Variadic<OneFlow_Tensor>:$model,
    Variadic<OneFlow_Tensor>:$model_update
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$d
  );
  let trait_attrs = (ins
    DenseI32ArrayAttr:$operand_segment_sizes
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

#endif // GET_ONEFLOW_OPTIMIZER_OP_DEFINITIONS


#ifdef GET_ONEFLOW_PADDING_OP_DEFINITIONS


def OneFlow_PadOp : OneFlow_BaseOp<"pad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let attrs = (ins
    SI64ArrayAttr:$padding_before,
    SI64ArrayAttr:$padding_after,
    SI64ArrayAttr:$padding,
    DefaultValuedAttr<F64Attr, "0.">:$floating_constant_value,
    DefaultValuedAttr<SI64Attr, "0">:$integral_constant_value
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ReplicationPad1DOp : OneFlow_BaseOp<"replication_pad1d", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let attrs = (ins
    SI64ArrayAttr:$padding
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_ReplicationPad1DGradOp : OneFlow_BaseOp<"replication_pad1d_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    SI64ArrayAttr:$padding
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ReflectionPad1DOp : OneFlow_BaseOp<"reflection_pad1d", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let attrs = (ins
    SI64ArrayAttr:$padding
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_ReflectionPad1DGradOp : OneFlow_BaseOp<"reflection_pad1d_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    SI64ArrayAttr:$padding
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ReflectionPad2DOp : OneFlow_BaseOp<"reflection_pad2d", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let attrs = (ins
    SI64ArrayAttr:$padding
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_ReflectionPad2DGradOp : OneFlow_BaseOp<"reflection_pad2d_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    SI64ArrayAttr:$padding
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ReplicationPad2DOp : OneFlow_BaseOp<"replication_pad2d", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let attrs = (ins
    SI64ArrayAttr:$padding
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_ReplicationPad2DGradOp : OneFlow_BaseOp<"replication_pad2d_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    SI64ArrayAttr:$padding
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_SamePaddingOp : OneFlow_BaseOp<"same_padding", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let attrs = (ins
    StrAttr:$padding,
    StrAttr:$data_format,
    SI32ArrayAttr:$kernel_size,
    SI32ArrayAttr:$strides,
    SI32ArrayAttr:$dilation_rate
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_SamePaddingGradOp : OneFlow_BaseOp<"same_padding_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x_like,
    OneFlow_Tensor:$dy
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    StrAttr:$padding,
    StrAttr:$data_format,
    SI32ArrayAttr:$kernel_size,
    SI32ArrayAttr:$strides,
    SI32ArrayAttr:$dilation_rate
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

#endif // GET_ONEFLOW_PADDING_OP_DEFINITIONS


#ifdef GET_ONEFLOW_PARALLEL_CAST_OP_DEFINITIONS

def OneFlow_HierarchicalParallelCastOp : OneFlow_BaseOp<"hierarchical_parallel_cast", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    StrArrayAttr:$nd_sbp,
    StrAttr:$grad_mode,
    StrArrayAttr:$grad_nd_sbp
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
  let has_get_nd_sbp_fn = 1;
}

def OneFlow_HierarchicalParallelCastLikeOp : OneFlow_BaseOp<"hierarchical_parallel_cast_like", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in,
    OneFlow_Tensor:$like
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

def OneFlow_ParallelCastOp : OneFlow_BaseOp<"parallel_cast", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    StrAttr:$sbp_parallel,
    StrAttr:$grad_sbp_parallel
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_sbp_signature_infer_fn = 1;
}

#endif // GET_ONEFLOW_PARALLEL_CAST_OP_DEFINITIONS


#ifdef GET_ONEFLOW_POOL_OP_DEFINITIONS

def OneFlow_AdaptiveAvgPool1DOp : OneFlow_AdaptivePoolBaseOp<"adaptive_avg_pool1d", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_AdaptiveAvgPool1DGradOp : OneFlow_AdaptivePoolGradBaseOp<"adaptive_avg_pool1d_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_AdaptiveAvgPool2DOp : OneFlow_AdaptivePoolBaseOp<"adaptive_avg_pool2d", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_AdaptiveAvgPool2DGradOp : OneFlow_AdaptivePoolGradBaseOp<"adaptive_avg_pool2d_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_AdaptiveAvgPool3DOp : OneFlow_AdaptivePoolBaseOp<"adaptive_avg_pool3d", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_AdaptiveAvgPool3DGradOp : OneFlow_AdaptivePoolGradBaseOp<"adaptive_avg_pool3d_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_AdaptiveMaxPool1DOp : OneFlow_AdaptiveMaxPoolBaseOp<"adaptive_max_pool1d", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}
def OneFlow_AdaptiveMaxPool1DGradOp : OneFlow_AdaptiveMaxPoolGradBaseOp<"adaptive_max_pool1d_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_AdaptiveMaxPool2DOp : OneFlow_AdaptiveMaxPoolBaseOp<"adaptive_max_pool2d", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}
def OneFlow_AdaptiveMaxPool2DGradOp : OneFlow_AdaptiveMaxPoolGradBaseOp<"adaptive_max_pool2d_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_AdaptiveMaxPool3DOp : OneFlow_AdaptiveMaxPoolBaseOp<"adaptive_max_pool3d", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}
def OneFlow_AdaptiveMaxPool3DGradOp : OneFlow_AdaptiveMaxPoolGradBaseOp<"adaptive_max_pool3d_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_AvgPool1DOp : OneFlow_AvgPoolBaseOp<"avg_pool_1d", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_AvgPool1DGradOp : OneFlow_AvgPoolGradBaseOp<"avg_pool_1d_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_AvgPool2DOp : OneFlow_AvgPoolBaseOp<"avg_pool_2d", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_AvgPool2DGradOp : OneFlow_AvgPoolGradBaseOp<"avg_pool_2d_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_AvgPool3DOp : OneFlow_AvgPoolBaseOp<"avg_pool_3d", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_AvgPool3DGradOp : OneFlow_AvgPoolGradBaseOp<"avg_pool_3d_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_MaxPool1DOp : OneFlow_MaxPoolBaseOp<"max_pool_1d", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_MaxPool1DGradOp : OneFlow_MaxPoolGradBaseOp<"max_pool_1d_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_MaxPool2DOp : OneFlow_MaxPoolBaseOp<"max_pool_2d", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>, DeclareOpInterfaceMethods<NCHWCompatibleInterface>]> {}

def OneFlow_MaxPool2DGradOp : OneFlow_MaxPoolGradBaseOp<"max_pool_2d_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_MaxPool3DOp : OneFlow_MaxPoolBaseOp<"max_pool_3d", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_MaxPool3DGradOp : OneFlow_MaxPoolGradBaseOp<"max_pool_3d_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_MaxUnpool1DOp : OneFlow_MaxUnpoolBaseOp<"max_unpool_1d", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_MaxUnpool2DOp : OneFlow_MaxUnpoolBaseOp<"max_unpool_2d", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_MaxUnpool3DOp : OneFlow_MaxUnpoolBaseOp<"max_unpool_3d", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_MaxUnpool1DGradOp : OneFlow_MaxUnpoolGradBaseOp<"max_unpool_1d_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_MaxUnpool2DGradOp : OneFlow_MaxUnpoolGradBaseOp<"max_unpool_2d_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_MaxUnpool3DGradOp : OneFlow_MaxUnpoolGradBaseOp<"max_unpool_3d_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_TfAvgPool1DOp : OneFlow_TFPoolBaseOp<"tf_avg_pool_1d", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_TfAvgPool1DGradOp : OneFlow_TFPoolGradBaseOp<"tf_avg_pool_1d_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_TfAvgPool2DOp : OneFlow_TFPoolBaseOp<"tf_avg_pool_2d", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_TfAvgPool2DGradOp : OneFlow_TFPoolGradBaseOp<"tf_avg_pool_2d_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_TfAvgPool3DOp : OneFlow_TFPoolBaseOp<"tf_avg_pool_3d", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_TfAvgPool3DGradOp : OneFlow_TFPoolGradBaseOp<"tf_avg_pool_3d_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_TfMaxPool1DOp : OneFlow_TFPoolBaseOp<"tf_max_pool_1d", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_TfMaxPool1DGradOp : OneFlow_TFPoolGradBaseOp<"tf_max_pool_1d_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_TfMaxPool2DOp : OneFlow_TFPoolBaseOp<"tf_max_pool_2d", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_TfMaxPool2DGradOp : OneFlow_TFPoolGradBaseOp<"tf_max_pool_2d_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_TfMaxPool3DOp : OneFlow_TFPoolBaseOp<"tf_max_pool_3d", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_TfMaxPool3DGradOp : OneFlow_TFPoolGradBaseOp<"tf_max_pool_3d_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

#endif // GET_ONEFLOW_POOL_OP_DEFINITIONS


#ifdef GET_ONEFLOW_QUANTIZATION_OP_DEFINITIONS

def OneFlow_FakeQuantizationOp : OneFlow_BaseOp<"fake_quantization", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in,
    OneFlow_Tensor:$scale,
    OneFlow_Tensor:$zero_point
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<StrAttr, "\"google\"">:$quantization_formula,
    DefaultValuedAttr<SI32Attr, "8">:$quantization_bit,
    DefaultValuedAttr<StrAttr, "\"symmetric\"">:$quantization_scheme
  );
  let has_check_fn = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_MinMaxObserverOp : OneFlow_BaseOp<"min_max_observer", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$scale,
    OneFlow_Tensor:$zero_point
  );
  let attrs = (ins
    DefaultValuedAttr<StrAttr, "\"google\"">:$quantization_formula,
    DefaultValuedAttr<SI32Attr, "8">:$quantization_bit,
    DefaultValuedAttr<StrAttr, "\"symmetric\"">:$quantization_scheme,
    DefaultValuedAttr<BoolAttr, "true">:$per_layer_quantization
  );
  let has_check_fn = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_MovingAverageMinMaxObserverOp : OneFlow_BaseOp<"moving_average_min_max_observer", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in,
    OneFlow_Tensor:$current_train_step,
    OneFlow_Tensor:$moving_max,
    OneFlow_Tensor:$moving_min
  );
  let output = (outs
    OneFlow_Tensor:$scale,
    OneFlow_Tensor:$zero_point
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$training,
    DefaultValuedAttr<StrAttr, "\"google\"">:$quantization_formula,
    DefaultValuedAttr<SI64Attr, "0">:$stop_update_after_iters,
    DefaultValuedAttr<SI32Attr, "8">:$quantization_bit,
    DefaultValuedAttr<StrAttr, "\"symmetric\"">:$quantization_scheme,
    DefaultValuedAttr<F32Attr, "0.95">:$momentum
  );
  let has_check_fn = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_QuantizationOp : OneFlow_BaseOp<"quantization", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in,
    OneFlow_Tensor:$scale,
    OneFlow_Tensor:$zero_point
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<StrAttr, "\"google\"">:$quantization_formula,
    DefaultValuedAttr<SI32Attr, "8">:$quantization_bit,
    DefaultValuedAttr<StrAttr, "\"symmetric\"">:$quantization_scheme
  );
  let has_check_fn = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_GroupwiseDequantizeOp : OneFlow_BaseOp<"groupwise_dequantize", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in,
    OneFlow_Tensor:$scale,
    Optional<OneFlow_Tensor>:$zero
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "8">:$num_bits,
    DefaultValuedAttr<BoolAttr, "true">:$symmetric,
    SI64Attr:$group_dim,
    SI64Attr:$group_size
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FusedLinearWithGroupwiseQuantizedWeightOp : OneFlow_BaseOp<"fused_linear_with_groupwise_quantized_weight", [NoMemoryEffect, AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$w,
    OneFlow_Tensor:$w_scale,
    Optional<OneFlow_Tensor>:$w_zero,
    Optional<OneFlow_Tensor>:$b
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "8">:$num_bits,
    DefaultValuedAttr<BoolAttr, "true">:$symmetric,
    SI64Attr:$group_dim,
    SI64Attr:$group_size
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

#endif // GET_ONEFLOW_QUANTIZATION_OP_DEFINITIONS


#ifdef GET_ONEFLOW_REDUCE_OP_DEFINITIONS

def OneFlow_IndexedSlicesReduceSumOp : OneFlow_BaseOp<"indexed_slices_reduce_sum", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x_indices,
    OneFlow_Tensor:$x_values
  );
  let output = (outs
    OneFlow_Tensor:$y_indices,
    OneFlow_Tensor:$y_values,
    OneFlow_Tensor:$num_unique
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ReduceAllOp : OneFlow_BaseOp<"reduce_all", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$input_tensor
  );
  let output = (outs
    OneFlow_Tensor:$output_tensor
  );
  let attrs = (ins
    SI32ArrayAttr:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$keepdims
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ReduceAnyOp : OneFlow_BaseOp<"reduce_any", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$input_tensor
  );
  let output = (outs
    OneFlow_Tensor:$output_tensor
  );
  let attrs = (ins
    SI32ArrayAttr:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$keepdims
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ReduceMaxOp : OneFlow_BaseOp<"reduce_max", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$input_tensor
  );
  let output = (outs
    OneFlow_Tensor:$output_tensor
  );
  let attrs = (ins
    SI32ArrayAttr:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$keepdims
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ReduceMaxDeviceStageOp : OneFlow_BaseOp<"reduce_max_device_stage", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out,
    OneFlow_Tensor:$mask,
    OneFlow_Tensor:$count
  );
  let attrs = (ins
    SI32ArrayAttr:$axis
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ReduceMaxDeviceStageGradOp : OneFlow_BaseOp<"reduce_max_device_stage_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$out_diff,
    OneFlow_Tensor:$mask,
    OneFlow_Tensor:$count
  );
  let output = (outs
    OneFlow_Tensor:$in_diff
  );
  let attrs = (ins
    SI32ArrayAttr:$axis
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ReduceMaxGlobalStageOp : OneFlow_BaseOp<"reduce_max_global_stage", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in,
    OneFlow_Tensor:$device_count
  );
  let output = (outs
    OneFlow_Tensor:$out,
    OneFlow_Tensor:$mask
  );
  let attrs = (ins
    SI32ArrayAttr:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$keepdims
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_ReduceMaxGlobalStageGradOp : OneFlow_BaseOp<"reduce_max_global_stage_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$out_diff,
    OneFlow_Tensor:$mask,
    OneFlow_Tensor:$device_count
  );
  let output = (outs
    OneFlow_Tensor:$in_diff
  );
  let attrs = (ins
    SI32ArrayAttr:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$keepdims
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ReduceMinOp : OneFlow_BaseOp<"reduce_min", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$input_tensor
  );
  let output = (outs
    OneFlow_Tensor:$output_tensor
  );
  let attrs = (ins
    SI32ArrayAttr:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$keepdims
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ReduceMinDeviceStageOp : OneFlow_BaseOp<"reduce_min_device_stage", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out,
    OneFlow_Tensor:$mask,
    OneFlow_Tensor:$count
  );
  let attrs = (ins
    SI32ArrayAttr:$axis
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ReduceMinDeviceStageGradOp : OneFlow_BaseOp<"reduce_min_device_stage_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$out_diff,
    OneFlow_Tensor:$mask,
    OneFlow_Tensor:$count
  );
  let output = (outs
    OneFlow_Tensor:$in_diff
  );
  let attrs = (ins
    SI32ArrayAttr:$axis
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ReduceMinGlobalStageOp : OneFlow_BaseOp<"reduce_min_global_stage", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in,
    OneFlow_Tensor:$device_count
  );
  let output = (outs
    OneFlow_Tensor:$out,
    OneFlow_Tensor:$mask
  );
  let attrs = (ins
    SI32ArrayAttr:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$keepdims
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_ReduceMinGlobalStageGradOp : OneFlow_BaseOp<"reduce_min_global_stage_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$out_diff,
    OneFlow_Tensor:$mask,
    OneFlow_Tensor:$device_count
  );
  let output = (outs
    OneFlow_Tensor:$in_diff
  );
  let attrs = (ins
    SI32ArrayAttr:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$keepdims
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ReduceProdOp : OneFlow_BaseOp<"reduce_prod", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$input_tensor
  );
  let output = (outs
    OneFlow_Tensor:$output_tensor
  );
  let attrs = (ins
    SI32ArrayAttr:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$keepdims
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ReduceSumOp : OneFlow_BaseOp<"reduce_sum", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$input_tensor
  );
  let output = (outs
    OneFlow_Tensor:$output_tensor
  );
  let attrs = (ins
    SI32ArrayAttr:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$keepdims
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ReduceNanSumOp : OneFlow_BaseOp<"reduce_nansum", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$input_tensor
  );
  let output = (outs
    OneFlow_Tensor:$output_tensor
  );
  let attrs = (ins
    SI32ArrayAttr:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$keepdims
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ReduceSumLikeOp : OneFlow_BaseOp<"reduce_sum_like", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$like
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let attrs = (ins
    SI32ArrayAttr:$axis
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

#endif // GET_ONEFLOW_REDUCE_OP_DEFINITIONS


#ifdef GET_ONEFLOW_RESHAPE_OP_DEFINITIONS

def OneFlow_ReshapeOp : OneFlow_BaseOp<"reshape", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    ShapeAttr:$shape
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_enumerate_nd_sbp_signatures_fn = 1;
  let has_data_type_infer_fn = 1;
  let hasFolder = 1;
}

def OneFlow_ReshapeLikeOp : OneFlow_BaseOp<"reshape_like", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in,
    OneFlow_Tensor:$like
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

#endif // GET_ONEFLOW_RESHAPE_OP_DEFINITIONS


#ifdef GET_ONEFLOW_SCALAR_OP_DEFINITIONS

def OneFlow_ClipByScalarOp : OneFlow_BaseOp<"clip_by_scalar", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.">:$floating_min,
    DefaultValuedAttr<SI64Attr, "0">:$integral_min,
    DefaultValuedAttr<F64Attr, "0.">:$floating_max,
    DefaultValuedAttr<SI64Attr, "0">:$integral_max
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ClipByScalarGradOp : OneFlow_BaseOp<"clip_by_scalar_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.">:$floating_min,
    DefaultValuedAttr<SI64Attr, "0">:$integral_min,
    DefaultValuedAttr<F64Attr, "0.">:$floating_max,
    DefaultValuedAttr<SI64Attr, "0">:$integral_max
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ClipByScalarMaxOp : OneFlow_BaseOp<"clip_by_scalar_max", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.">:$floating_max,
    DefaultValuedAttr<SI64Attr, "0">:$integral_max
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ClipByScalarMaxGradOp : OneFlow_BaseOp<"clip_by_scalar_max_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.">:$floating_max,
    DefaultValuedAttr<SI64Attr, "0">:$integral_max
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ClipByScalarMinOp : OneFlow_BaseOp<"clip_by_scalar_min", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.">:$floating_min,
    DefaultValuedAttr<SI64Attr, "0">:$integral_min
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ClipByScalarMinGradOp : OneFlow_BaseOp<"clip_by_scalar_min_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.">:$floating_min,
    DefaultValuedAttr<SI64Attr, "0">:$integral_min
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ScalarAddOp : OneFlow_BaseOp<"scalar_add", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$has_int_operand,
    DefaultValuedAttr<BoolAttr, "false">:$has_float_operand,
    DefaultValuedAttr<SI64Attr, "0">:$int_operand,
    DefaultValuedAttr<F64Attr, "0.">:$float_operand
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let hasFolder = 1;
}

def OneFlow_ScalarAddByTensorOp : OneFlow_BaseOp<"scalar_add_by_tensor", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$scalar
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

// host_scalar_add_by_tensor op just for test host memory input
def OneFlow_HostScalarAddByTensorOp : OneFlow_BaseOp<"host_scalar_add_by_tensor", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$scalar
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ScalarDivByTensorOp : OneFlow_BaseOp<"scalar_div_by_tensor", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$scalar
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ScalarFloordivOp : OneFlow_BaseOp<"scalar_floordiv", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$has_int_operand,
    DefaultValuedAttr<BoolAttr, "false">:$has_float_operand,
    DefaultValuedAttr<SI64Attr, "0">:$int_operand,
    DefaultValuedAttr<F64Attr, "0.">:$float_operand
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ScalarTruncdivOp : OneFlow_BaseOp<"scalar_truncdiv", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$has_int_operand,
    DefaultValuedAttr<BoolAttr, "false">:$has_float_operand,
    DefaultValuedAttr<SI64Attr, "0">:$int_operand,
    DefaultValuedAttr<F64Attr, "0.">:$float_operand
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ScalarFmodOp : OneFlow_BaseOp<"scalar_fmod", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$has_int_operand,
    DefaultValuedAttr<BoolAttr, "false">:$has_float_operand,
    DefaultValuedAttr<SI64Attr, "0">:$int_operand,
    DefaultValuedAttr<F64Attr, "0.">:$float_operand
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ScalarLogicalAndOp : OneFlow_BaseOp<"scalar_logical_and", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$has_int_operand,
    DefaultValuedAttr<BoolAttr, "false">:$has_float_operand,
    DefaultValuedAttr<SI64Attr, "0">:$int_operand,
    DefaultValuedAttr<F64Attr, "0.">:$float_operand
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ScalarLogicalEqualOp : OneFlow_BaseOp<"scalar_logical_equal", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$has_int_operand,
    DefaultValuedAttr<BoolAttr, "false">:$has_float_operand,
    DefaultValuedAttr<SI64Attr, "0">:$int_operand,
    DefaultValuedAttr<F64Attr, "0.">:$float_operand
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ScalarLogicalGreaterOp : OneFlow_BaseOp<"scalar_logical_greater", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$has_int_operand,
    DefaultValuedAttr<BoolAttr, "false">:$has_float_operand,
    DefaultValuedAttr<SI64Attr, "0">:$int_operand,
    DefaultValuedAttr<F64Attr, "0.">:$float_operand
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ScalarLogicalGreaterEqualOp : OneFlow_BaseOp<"scalar_logical_greater_equal", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$has_int_operand,
    DefaultValuedAttr<BoolAttr, "false">:$has_float_operand,
    DefaultValuedAttr<SI64Attr, "0">:$int_operand,
    DefaultValuedAttr<F64Attr, "0.">:$float_operand
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ScalarLogicalInplaceGreaterOp : OneFlow_BaseOp<"scalar_logical_inplace_greater", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$has_int_operand,
    DefaultValuedAttr<BoolAttr, "false">:$has_float_operand,
    DefaultValuedAttr<SI64Attr, "0">:$int_operand,
    DefaultValuedAttr<F64Attr, "0.">:$float_operand
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ScalarLogicalLessOp : OneFlow_BaseOp<"scalar_logical_less", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$has_int_operand,
    DefaultValuedAttr<BoolAttr, "false">:$has_float_operand,
    DefaultValuedAttr<SI64Attr, "0">:$int_operand,
    DefaultValuedAttr<F64Attr, "0.">:$float_operand
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ScalarLogicalLessEqualOp : OneFlow_BaseOp<"scalar_logical_less_equal", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$has_int_operand,
    DefaultValuedAttr<BoolAttr, "false">:$has_float_operand,
    DefaultValuedAttr<SI64Attr, "0">:$int_operand,
    DefaultValuedAttr<F64Attr, "0.">:$float_operand
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ScalarLogicalNotEqualOp : OneFlow_BaseOp<"scalar_logical_not_equal", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$has_int_operand,
    DefaultValuedAttr<BoolAttr, "false">:$has_float_operand,
    DefaultValuedAttr<SI64Attr, "0">:$int_operand,
    DefaultValuedAttr<F64Attr, "0.">:$float_operand
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ScalarLogicalOrOp : OneFlow_BaseOp<"scalar_logical_or", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$has_int_operand,
    DefaultValuedAttr<BoolAttr, "false">:$has_float_operand,
    DefaultValuedAttr<SI64Attr, "0">:$int_operand,
    DefaultValuedAttr<F64Attr, "0.">:$float_operand
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ScalarLogicalXorOp : OneFlow_BaseOp<"scalar_logical_xor", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$has_int_operand,
    DefaultValuedAttr<BoolAttr, "false">:$has_float_operand,
    DefaultValuedAttr<SI64Attr, "0">:$int_operand,
    DefaultValuedAttr<F64Attr, "0.">:$float_operand
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ScalarMulOp : OneFlow_BaseOp<"scalar_mul", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$has_int_operand,
    DefaultValuedAttr<BoolAttr, "false">:$has_float_operand,
    DefaultValuedAttr<SI64Attr, "0">:$int_operand,
    DefaultValuedAttr<F64Attr, "0.">:$float_operand
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ScalarMulByTensorOp : OneFlow_BaseOp<"scalar_mul_by_tensor", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$scalar
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ScalarDivOp : OneFlow_BaseOp<"scalar_div", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>, DeclareOpInterfaceMethods<NCHWCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$has_int_operand,
    DefaultValuedAttr<BoolAttr, "false">:$has_float_operand,
    DefaultValuedAttr<SI64Attr, "0">:$int_operand,
    DefaultValuedAttr<F64Attr, "0.">:$float_operand
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ScalarPowOp : OneFlow_BaseOp<"scalar_pow", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$has_int_operand,
    DefaultValuedAttr<BoolAttr, "false">:$has_float_operand,
    DefaultValuedAttr<SI64Attr, "0">:$int_operand,
    DefaultValuedAttr<F64Attr, "0.">:$float_operand
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ScalarPowGradOp : OneFlow_BaseOp<"scalar_pow_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$dy
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$has_int_operand,
    DefaultValuedAttr<BoolAttr, "false">:$has_float_operand,
    DefaultValuedAttr<SI64Attr, "0">:$int_operand,
    DefaultValuedAttr<F64Attr, "0.">:$float_operand
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ScalarReversePowOp : OneFlow_BaseOp<"scalar_reverse_pow", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$has_int_operand,
    DefaultValuedAttr<BoolAttr, "false">:$has_float_operand,
    DefaultValuedAttr<SI64Attr, "0">:$int_operand,
    DefaultValuedAttr<F64Attr, "0.">:$float_operand
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ScalarReversePowGradOp : OneFlow_BaseOp<"scalar_reverse_pow_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$dy
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$has_int_operand,
    DefaultValuedAttr<BoolAttr, "false">:$has_float_operand,
    DefaultValuedAttr<SI64Attr, "0">:$int_operand,
    DefaultValuedAttr<F64Attr, "0.">:$float_operand
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ScalarSubByTensorOp : OneFlow_BaseOp<"scalar_sub_by_tensor", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$scalar
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ScalarLerpOp : OneFlow_BaseOp<"scalar_lerp", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$start,
    OneFlow_Tensor:$end
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$has_int_operand,
    DefaultValuedAttr<BoolAttr, "false">:$has_float_operand,
    DefaultValuedAttr<SI64Attr, "0">:$int_operand,
    DefaultValuedAttr<F64Attr, "0.">:$float_operand
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ScalarLerpGradOp : OneFlow_BaseOp<"scalar_lerp_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$start,
    OneFlow_Tensor:$end,
    OneFlow_Tensor:$out_diff
  );
  let output = (outs
    OneFlow_Tensor:$start_diff,
    OneFlow_Tensor:$end_diff
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$has_int_operand,
    DefaultValuedAttr<BoolAttr, "false">:$has_float_operand,
    DefaultValuedAttr<SI64Attr, "0">:$int_operand,
    DefaultValuedAttr<F64Attr, "0.">:$float_operand
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ScalarBitwiseAndOp : OneFlow_BaseOp<"scalar_bitwise_and", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$operand
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ScalarBitwiseOrOp : OneFlow_BaseOp<"scalar_bitwise_or", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$operand
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ScalarBitwiseXorOp : OneFlow_BaseOp<"scalar_bitwise_xor", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$operand
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}
#endif // GET_ONEFLOW_SCALAR_OP_DEFINITIONS


#ifdef GET_ONEFLOW_SOFTMAX_OP_DEFINITIONS

def OneFlow_LogSoftmaxOp : OneFlow_BaseOp<"log_softmax", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$prob
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_LogSoftmaxGradOp : OneFlow_BaseOp<"log_softmax_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$prob,
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

def OneFlow_SoftmaxOp : OneFlow_BaseOp<"softmax", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_compute_complexity_fn = 1;
}

def OneFlow_SoftmaxCrossEntropyOp : OneFlow_BaseOp<"softmax_cross_entropy", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$prediction,
    OneFlow_Tensor:$label
  );
  let output = (outs
    OneFlow_Tensor:$prob,
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_SoftmaxCrossEntropyGradOp : OneFlow_BaseOp<"softmax_cross_entropy_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$label,
    OneFlow_Tensor:$prob
  );
  let output = (outs
    OneFlow_Tensor:$prediction_diff
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_SoftmaxGradOp : OneFlow_BaseOp<"softmax_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$y,
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

def OneFlow_SparseSoftmaxCrossEntropyOp : OneFlow_BaseOp<"sparse_softmax_cross_entropy", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$prediction,
    OneFlow_Tensor:$label
  );
  let output = (outs
    OneFlow_Tensor:$prob,
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$depth
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_SparseSoftmaxCrossEntropyGradOp : OneFlow_BaseOp<"sparse_softmax_cross_entropy_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$label,
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$prob
  );
  let output = (outs
    OneFlow_Tensor:$prediction_diff
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$depth
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_SparseSoftmaxCrossEntropyMsOp : OneFlow_BaseOp<"sparse_softmax_cross_entropy_ms", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$prediction,
    OneFlow_Tensor:$label
  );
  let output = (outs
    OneFlow_Tensor:$prob,
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$depth
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_SparseSoftmaxCrossEntropyMsGradOp : OneFlow_BaseOp<"sparse_softmax_cross_entropy_ms_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$label,
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$prob
  );
  let output = (outs
    OneFlow_Tensor:$prediction_diff
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$depth
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

#endif // GET_ONEFLOW_SOFTMAX_OP_DEFINITIONS


#ifdef GET_ONEFLOW_SUMMARY_OP_DEFINITIONS

def OneFlow_CreateSummaryWriterOp : OneFlow_BaseOp<"create_summary_writer", [NoMemoryEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let attrs = (ins
    StrAttr:$logdir
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FlushSummaryWriterOp : OneFlow_BaseOp<"flush_summary_writer", [NoMemoryEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_SummaryWriteHistogramOp : OneFlow_BaseOp<"summary_write_histogram", [NoMemoryEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in,
    OneFlow_Tensor:$step,
    OneFlow_Tensor:$tag
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_SummaryWriteImageOp : OneFlow_BaseOp<"summary_write_image", [NoMemoryEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in,
    OneFlow_Tensor:$step,
    OneFlow_Tensor:$tag
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_SummaryWritePbOp : OneFlow_BaseOp<"summary_write_pb", [NoMemoryEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in,
    OneFlow_Tensor:$step
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_SummaryWriteScalarOp : OneFlow_BaseOp<"summary_write_scalar", [NoMemoryEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in,
    OneFlow_Tensor:$step,
    OneFlow_Tensor:$tag
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

#endif // GET_ONEFLOW_SUMMARY_OP_DEFINITIONS


#ifdef GET_ONEFLOW_TENSOR_BUFFER_OP_DEFINITIONS

def OneFlow_GenTensorBufferOp : OneFlow_BaseOp<"gen_tensor_buffer", [NoMemoryEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    ShapeAttr:$shape,
    ShapeArrayAttr:$shape_list,
    F32ArrayAttr:$value_list,
    OneFlow_DataType:$data_type,
    DefaultValuedAttr<BoolAttr, "false">:$dynamic_out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_TensorBufferToListOfTensorsOp : OneFlow_BaseOp<"tensor_buffer_to_list_of_tensors", [NoMemoryEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    Variadic<OneFlow_Tensor>:$out
  );
  let attrs = (ins
    ShapeAttr:$out_shape,
    OneFlow_DataType:$out_dtype,
    DefaultValuedAttr<BoolAttr, "false">:$dynamic_out
  );
  let has_check_fn = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_output_arg_modify_fn = 1;
}

def OneFlow_TensorBufferToListOfTensorsV2Op : OneFlow_BaseOp<"tensor_buffer_to_list_of_tensors_v2", [NoMemoryEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    Variadic<OneFlow_Tensor>:$out
  );
  let attrs = (ins
    ShapeArrayAttr:$out_shapes,
    DTArrayAttr:$out_dtypes,
    DefaultValuedAttr<BoolAttr, "false">:$dynamic_out
  );
  let has_check_fn = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_output_arg_modify_fn = 1;
}

def OneFlow_TensorBufferToTensorOp : OneFlow_BaseOp<"tensor_buffer_to_tensor", [NoMemoryEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    ShapeAttr:$instance_shape,
    OneFlow_DataType:$dtype
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_TensorToTensorBufferOp : OneFlow_BaseOp<"tensor_to_tensor_buffer", [NoMemoryEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$instance_dims
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

#endif // GET_ONEFLOW_TENSOR_BUFFER_OP_DEFINITIONS


#ifdef GET_ONEFLOW_TEST_OP_DEFINITIONS

def OneFlow_ThrowErrorOp : OneFlow_BaseOp<"throw_error", [NoMemoryEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

#endif // GET_ONEFLOW_TEST_OP_DEFINITIONS

#ifdef GET_ONEFLOW_TRIGONOMETRIC_OP_DEFINITIONS

def OneFlow_AcosOp : OneFlow_BaseOp<"acos", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_AcosGradOp : OneFlow_BaseOp<"acos_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
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

def OneFlow_AcoshOp : OneFlow_BaseOp<"acosh", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_AcoshGradOp : OneFlow_BaseOp<"acosh_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
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

def OneFlow_AsinOp : OneFlow_BaseOp<"asin", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_AsinGradOp : OneFlow_BaseOp<"asin_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
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

def OneFlow_AsinhOp : OneFlow_BaseOp<"asinh", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_AsinhGradOp : OneFlow_BaseOp<"asinh_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
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

def OneFlow_AtanOp : OneFlow_BaseOp<"atan", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_Atan2Op : OneFlow_BaseOp<"atan2", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$y
  );
  let output = (outs
    OneFlow_Tensor:$z
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_Atan2XGradOp : OneFlow_BaseOp<"atan2_x_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$y,
    OneFlow_Tensor:$dz
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_Atan2YGradOp : OneFlow_BaseOp<"atan2_y_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$y,
    OneFlow_Tensor:$dz
  );
  let output = (outs
    OneFlow_Tensor:$dy
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_AtanGradOp : OneFlow_BaseOp<"atan_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
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

def OneFlow_AtanhOp : OneFlow_BaseOp<"atanh", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_AtanhGradOp : OneFlow_BaseOp<"atanh_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
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

def OneFlow_CosOp : OneFlow_BaseOp<"cos", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_CosGradOp : OneFlow_BaseOp<"cos_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
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

def OneFlow_CoshOp : OneFlow_BaseOp<"cosh", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_CoshGradOp : OneFlow_BaseOp<"cosh_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
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

def OneFlow_HardtanhOp : OneFlow_BaseOp<"hardtanh", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.">:$min_val,
    DefaultValuedAttr<F64Attr, "0.">:$max_val
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_HardtanhGradOp : OneFlow_BaseOp<"hardtanh_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$y,
    OneFlow_Tensor:$dy
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.">:$min_val,
    DefaultValuedAttr<F64Attr, "0.">:$max_val
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_SinOp : OneFlow_BaseOp<"sin", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_SinGradOp : OneFlow_BaseOp<"sin_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
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

def OneFlow_SinhOp : OneFlow_BaseOp<"sinh", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_SinhGradOp : OneFlow_BaseOp<"sinh_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
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

def OneFlow_TanOp : OneFlow_BaseOp<"tan", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_TanGradOp : OneFlow_BaseOp<"tan_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
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

def OneFlow_TanhOp : OneFlow_BaseOp<"tanh", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_TanhGradOp : OneFlow_BaseOp<"tanh_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$y,
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

def OneFlow_NotEqualZeroOp : OneFlow_BaseOp<"not_equal_zero", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

#endif // GET_ONEFLOW_TRIGONOMETRIC_OP_DEFINITIONS


#ifdef GET_ONEFLOW_UNARY_OP_DEFINITIONS

def OneFlow_AccOp : OneFlow_BaseOp<"acc", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$max_acc_num
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_output_blob_time_shape_infer_fn = 1;
}

def OneFlow_AccCtrlTickOp : OneFlow_BaseOp<"acc_ctrl_tick", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$max_acc_num
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
  let has_output_blob_time_shape_infer_fn = 1;
}

def OneFlow_AffineGridOp : OneFlow_BaseOp<"affine_grid", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$theta
  );
  let output = (outs
    OneFlow_Tensor:$grid
  );
  let attrs = (ins
    ShapeAttr:$size,
    DefaultValuedAttr<BoolAttr, "false">:$align_corners
  );
  let has_check_fn = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_AffineGridGradOp : OneFlow_BaseOp<"affine_grid_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dgrid
  );
  let output = (outs
    OneFlow_Tensor:$dtheta
  );
  let attrs = (ins
    ShapeAttr:$size,
    DefaultValuedAttr<BoolAttr, "false">:$align_corners
  );
  let has_check_fn = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_BernoulliOp : OneFlow_BaseOp<"bernoulli", [NoMemoryEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    OneFlow_DataType:$dtype,
    DefaultValuedAttr<SI64Attr, "-1">:$seed,
    DefaultValuedAttr<F64Attr, "0.">:$p
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_CastOp : OneFlow_BaseOp<"cast", [NoMemoryEffect, SupportNonContiguous, DeclareOpInterfaceMethods<UserOpCompatibleInterface>, DeclareOpInterfaceMethods<NCHWCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    OneFlow_DataType:$dtype,
    DefaultValuedAttr<BoolAttr, "false">:$pin_memory
  );
  let has_device_and_stream_infer_fn = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_MutableCastOnceOp : OneFlow_BaseOp<"mutable_cast_once", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    OneFlow_DataType:$dtype
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let same_output_regst_num = 1;
}

def OneFlow_CastToStaticShapeOp : OneFlow_BaseOp<"cast_to_static_shape", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$input
  );
  let output = (outs
    OneFlow_Tensor:$output
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_CastToTickOp : OneFlow_BaseOp<"cast_to_tick", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

def OneFlow_CeluOp : OneFlow_BaseOp<"celu", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.">:$alpha
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_CopyOp : OneFlow_BaseOp<"copy", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    StrAttr:$device_type,
    DefaultValuedAttr<SI64Attr, "0">:$device_id,
    DefaultValuedAttr<BoolAttr, "false">:$pin_memory
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_device_and_stream_infer_fn = 1;
}

def OneFlow_CountNotFiniteOp : OneFlow_BaseOp<"count_not_finite", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_DiagOp : OneFlow_BaseOp<"diag", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$diagonal
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_DiagonalOp : OneFlow_BaseOp<"diagonal", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$offset
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_EluOp : OneFlow_BaseOp<"elu", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.">:$alpha
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ExpandOp : OneFlow_BaseOp<"expand", [NoMemoryEffect, SupportNonContiguous, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    ShapeAttr:$expand_shape
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ExpandDimsOp : OneFlow_BaseOp<"expand_dims", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$axis
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FlipOp : OneFlow_BaseOp<"flip", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let attrs = (ins
    SI32ArrayAttr:$dims
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FoldOp : OneFlow_BaseOp<"fold", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let attrs = (ins
    StrAttr:$data_format,
    SI32ArrayAttr:$output_size,
    SI32ArrayAttr:$kernel_size,
    SI32ArrayAttr:$strides,
    SI32ArrayAttr:$padding,
    SI32ArrayAttr:$dilation_rate
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_GeluOp : OneFlow_BaseOp<"gelu", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FastGeluOp : OneFlow_BaseOp<"fast_gelu", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_QuickGeluOp : OneFlow_BaseOp<"quick_gelu", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_HardsigmoidOp : OneFlow_BaseOp<"hardsigmoid", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_HardShrinkOp : OneFlow_BaseOp<"hardshrink", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.">:$lambd
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_HardswishOp : OneFlow_BaseOp<"hardswish", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_LeakyReluOp : OneFlow_BaseOp<"leaky_relu", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$alpha
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_RReluOp : OneFlow_BaseOp<"rrelu", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$output,
    OneFlow_Tensor:$noise_data
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$seed,
    DefaultValuedAttr<F32Attr, "0.125">:$lower,
    DefaultValuedAttr<F32Attr, "0.3333333333333333">:$upper,
    DefaultValuedAttr<BoolAttr, "false">:$training
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_Log2Op : OneFlow_BaseOp<"log2", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_Log10Op : OneFlow_BaseOp<"log10", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_LogicalNotOp : OneFlow_BaseOp<"logical_not", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_MishOp : OneFlow_BaseOp<"mish", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_NarrowOp : OneFlow_BaseOp<"narrow", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$dim,
    DefaultValuedAttr<SI64Attr, "0">:$start,
    DefaultValuedAttr<SI64Attr, "0">:$length
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_OneHotOp : OneFlow_BaseOp<"one_hot", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$indices
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$depth,
    DefaultValuedAttr<F64Attr, "0.">:$floating_on_value,
    DefaultValuedAttr<SI64Attr, "0">:$integer_on_value,
    DefaultValuedAttr<F64Attr, "0.">:$floating_off_value,
    DefaultValuedAttr<SI64Attr, "0">:$integer_off_value,
    OneFlow_DataType:$dtype
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_PackOp : OneFlow_BaseOp<"pack", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$pack_num
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_output_blob_time_shape_infer_fn = 1;
}

def OneFlow_RandomMaskLikeOp : OneFlow_BaseOp<"random_mask_like", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$like
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$rate,
    DefaultValuedAttr<SI64Attr, "0">:$seed
  );
  let has_check_fn = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let hasCanonicalizer = 1;
}

def OneFlow_RepeatOp : OneFlow_BaseOp<"repeat", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$repeat_num
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_output_blob_time_shape_infer_fn = 1;
}

def OneFlow_Repeat_InterLeaveOp : OneFlow_BaseOp<"repeat_interleave", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in,
    OneFlow_Tensor:$cumsum
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$repeat_num
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_RollOp : OneFlow_BaseOp<"roll", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    SI32ArrayAttr:$shifts,
    SI32ArrayAttr:$dims
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_SeluOp : OneFlow_BaseOp<"selu", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_SiluOp : OneFlow_BaseOp<"silu", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>, DeclareOpInterfaceMethods<NCHWCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_SoftShrinkOp: OneFlow_BaseOp<"softshrink", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.">:$alpha
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_SoftsignOp : OneFlow_BaseOp<"softsign", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_SortOp : OneFlow_BaseOp<"sort", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    StrAttr:$direction
  );
  let has_check_fn = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_SquareSumOp : OneFlow_BaseOp<"square_sum", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_SqrtSquareSumOp : OneFlow_BaseOp<"sqrt_square_sum", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_SqueezeOp : OneFlow_BaseOp<"squeeze", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    SI32ArrayAttr:$axes
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ThresholdOp : OneFlow_BaseOp<"threshold", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.">:$threshold_val,
    DefaultValuedAttr<F64Attr, "0.">:$value
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_TransposeOp : OneFlow_BaseOp<"transpose", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$input
  );
  let output = (outs
    OneFlow_Tensor:$output
  );
  let attrs = (ins
    SI32ArrayAttr:$perm
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let hasFolder = 1;
}

def OneFlow_AsStridedOp : OneFlow_BaseOp<"as_strided", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$input
  );
  let output = (outs
    OneFlow_Tensor:$output
  );
  let attrs = (ins
    SI64ArrayAttr:$size,
    SI64ArrayAttr:$stride,
    DefaultValuedAttr<SI64Attr, "0">:$storage_offset
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_IndexAddOp : OneFlow_BaseOp<"index_add", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$input,
    OneFlow_Tensor:$index,
    OneFlow_Tensor:$source
  );
  let output = (outs
    OneFlow_Tensor:$output
  );
  let attrs = (ins
    SI64Attr: $dim,
    F32Attr: $alpha
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_AsStridedGradOp : OneFlow_BaseOp<"as_strided_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$input
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    SI64ArrayAttr:$size,
    SI64ArrayAttr:$stride,
    DefaultValuedAttr<SI64Attr, "0">:$storage_offset
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_TrilOp : OneFlow_BaseOp<"tril", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$diagonal,
    DefaultValuedAttr<F64Attr, "0.">:$floating_fill_value,
    DefaultValuedAttr<SI64Attr, "0">:$integer_fill_value,
    DefaultValuedAttr<BoolAttr, "false">:$is_floating_fill_value
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_TriuOp : OneFlow_BaseOp<"triu", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$diagonal
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_TruncOp : OneFlow_BaseOp<"trunc", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_TruncGradOp : OneFlow_BaseOp<"trunc_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
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

def OneFlow_UnfoldOp : OneFlow_BaseOp<"unfold", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let attrs = (ins
    StrAttr:$data_format,
    SI32ArrayAttr:$kernel_size,
    SI32ArrayAttr:$padding,
    SI32ArrayAttr:$strides,
    SI32ArrayAttr:$dilation_rate
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_UnfoldTensorOp : OneFlow_BaseOp<"unfold_tensor", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$dimension,
    DefaultValuedAttr<SI32Attr, "0">:$size,
    DefaultValuedAttr<SI32Attr, "0">:$step
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_UnpackOp : OneFlow_BaseOp<"unpack", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$unpack_num
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_output_blob_time_shape_infer_fn = 1;
}

def OneFlow_ZeroLikeOp : OneFlow_BaseOp<"zero_like", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$like
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let same_output_regst_num = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

def OneFlow_ToContiguousOp : OneFlow_BaseOp<"to_contiguous", [NoMemoryEffect, SupportNonContiguous, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ConvertMemoryFormatOp : OneFlow_BaseOp<"convert_memory_format", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    OneFlow_MemoryFormat:$memory_format
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_IsNanOp : OneFlow_BaseOp<"isnan", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_IsInfOp : OneFlow_BaseOp<"isinf", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_IsFiniteOp : OneFlow_BaseOp<"isfinite", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_RealOp : OneFlow_BaseOp<"real", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_RealGradOp : OneFlow_BaseOp<"real_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dout
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ImagOp : OneFlow_BaseOp<"imag", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ImagGradOp : OneFlow_BaseOp<"imag_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dout
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ConjPhysicalOp : OneFlow_BaseOp<"conj_physical", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

#endif // GET_ONEFLOW_UNARY_OP_DEFINITIONS


#ifdef GET_ONEFLOW_UPSAMPLE_OP_DEFINITIONS

def OneFlow_UpsampleBicubic2DOp : OneFlow_BaseOp<"upsample_bicubic_2d", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.">:$height_scale,
    DefaultValuedAttr<F64Attr, "0.">:$width_scale,
    DefaultValuedAttr<BoolAttr, "false">:$align_corners,
    SI64ArrayAttr:$output_size,
    StrAttr:$data_format
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_UpsampleBicubic2DGradOp : OneFlow_BaseOp<"upsample_bicubic_2d_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.">:$height_scale,
    DefaultValuedAttr<F64Attr, "0.">:$width_scale,
    DefaultValuedAttr<BoolAttr, "false">:$align_corners,
    SI64ArrayAttr:$output_size,
    StrAttr:$data_format
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_UpsampleBilinear2DOp : OneFlow_BaseOp<"upsample_bilinear_2d", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.">:$height_scale,
    DefaultValuedAttr<F64Attr, "0.">:$width_scale,
    DefaultValuedAttr<BoolAttr, "false">:$align_corners,
    SI64ArrayAttr:$output_size,
    StrAttr:$data_format
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_UpsampleBilinear2DGradOp : OneFlow_BaseOp<"upsample_bilinear_2d_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.">:$height_scale,
    DefaultValuedAttr<F64Attr, "0.">:$width_scale,
    DefaultValuedAttr<BoolAttr, "false">:$align_corners,
    SI64ArrayAttr:$output_size,
    StrAttr:$data_format
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_UpsampleLinear1DOp : OneFlow_BaseOp<"upsample_linear_1d", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.">:$scale_factor,
    DefaultValuedAttr<BoolAttr, "false">:$align_corners,
    SI64ArrayAttr:$output_size,
    StrAttr:$data_format
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_UpsampleLinear1DGradOp : OneFlow_BaseOp<"upsample_linear_1d_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.">:$scale_factor,
    DefaultValuedAttr<BoolAttr, "false">:$align_corners,
    SI64ArrayAttr:$output_size,
    StrAttr:$data_format
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_UpsampleNearest1DOp : OneFlow_BaseOp<"upsample_nearest_1d", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.">:$scale_factor,
    SI64ArrayAttr:$output_size,
    StrAttr:$data_format
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_UpsampleNearest1DGradOp : OneFlow_BaseOp<"upsample_nearest_1d_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.">:$scale_factor,
    SI64ArrayAttr:$output_size,
    StrAttr:$data_format
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_UpsampleNearest2DOp : OneFlow_BaseOp<"upsample_nearest_2d", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.">:$height_scale,
    DefaultValuedAttr<F64Attr, "0.">:$width_scale,
    SI64ArrayAttr:$output_size,
    StrAttr:$data_format
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_UpsampleNearest2DGradOp : OneFlow_BaseOp<"upsample_nearest_2d_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.">:$height_scale,
    DefaultValuedAttr<F64Attr, "0.">:$width_scale,
    SI64ArrayAttr:$output_size,
    StrAttr:$data_format
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_UpsampleNearest3DOp : OneFlow_BaseOp<"upsample_nearest_3d", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.">:$depth_scale,
    DefaultValuedAttr<F64Attr, "0.">:$height_scale,
    DefaultValuedAttr<F64Attr, "0.">:$width_scale,
    SI64ArrayAttr:$output_size,
    StrAttr:$data_format
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_UpsampleNearest3DGradOp : OneFlow_BaseOp<"upsample_nearest_3d_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.">:$depth_scale,
    DefaultValuedAttr<F64Attr, "0.">:$height_scale,
    DefaultValuedAttr<F64Attr, "0.">:$width_scale,
    SI64ArrayAttr:$output_size,
    StrAttr:$data_format
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_UpsampleTrilinear3DOp : OneFlow_BaseOp<"upsample_trilinear_3d", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.">:$depth_scale,
    DefaultValuedAttr<F64Attr, "0.">:$height_scale,
    DefaultValuedAttr<F64Attr, "0.">:$width_scale,
    DefaultValuedAttr<BoolAttr, "false">:$align_corners,
    SI64ArrayAttr:$output_size,
    StrAttr:$data_format
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_UpsampleTrilinear3DGradOp : OneFlow_BaseOp<"upsample_trilinear_3d_grad", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.">:$depth_scale,
    DefaultValuedAttr<F64Attr, "0.">:$height_scale,
    DefaultValuedAttr<F64Attr, "0.">:$width_scale,
    DefaultValuedAttr<BoolAttr, "false">:$align_corners,
    SI64ArrayAttr:$output_size,
    StrAttr:$data_format
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}
#endif // GET_ONEFLOW_UPSAMPLE_OP_DEFINITIONS


#ifdef GET_ONEFLOW_ONE_EMBEDDING_OP_DEFINITIONS

def OneFlow_OneEmbeddingFusedLookupOp : OneFlow_BaseOp<"one_embedding_fused_lookup", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$shadow,
    OneFlow_Tensor:$ids,
    Optional<OneFlow_Tensor>:$table_ids
  );
  let output = (outs
    OneFlow_Tensor:$embeddings
  );
  let attrs = (ins
    OneFlow_DataType:$dtype,
    StrAttr:$embedding_name,
    DefaultValuedAttr<SI64Attr, "0">:$line_size,
    DefaultValuedAttr<SI64Attr, "0">:$embedding_size,
    DefaultValuedAttr<BoolAttr, "false">:$is_full_cache,
    DefaultValuedAttr<SI32Attr, "1">:$num_tables,
    DefaultValuedAttr<SI64Attr, "-1">:$padding_idx,
    DefaultValuedAttr<BoolAttr, "false">:$has_padding_idx,
    StrAttr:$embedding_tables,
    DefaultValuedAttr<SI64Attr, "0">:$seed
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_OneEmbeddingFusedLookupGradOp : OneFlow_BaseOp<"one_embedding_fused_lookup_grad", [DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$ids,
    OneFlow_Tensor:$embedding_grad
  );
  let attrs = (ins
    StrAttr:$embedding_name,
    DefaultValuedAttr<SI64Attr, "0">:$line_size,
    DefaultValuedAttr<SI64Attr, "0">:$embedding_size
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_UniqueKeyValuePairOp : OneFlow_BaseOp<"unique_key_value_pair", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$keys,
    Optional<OneFlow_Tensor>:$values
  );
  let output = (outs
    OneFlow_Tensor:$num_unique,
    OneFlow_Tensor:$unique_keys,
    OneFlow_Tensor:$unique_values,
    OneFlow_Tensor:$inverse_indices
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "1">:$num_tables,
    DefaultValuedAttr<SI64Attr, "-1">:$padding_idx,
    DefaultValuedAttr<BoolAttr, "false">:$has_padding_idx,
    StrAttr:$embedding_name
  );
  let same_output_regst_num = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_IdShuffleOp : OneFlow_BaseOp<"id_shuffle", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$ids,
    Optional<OneFlow_Tensor>:$table_ids
  );
  let output = (outs
    OneFlow_Tensor:$num_unique_matrix,
    OneFlow_Tensor:$inverse_unique_partition_indices,
    OneFlow_Tensor:$cur_rank_num_unique,
    OneFlow_Tensor:$cur_rank_unique_ids,
    OneFlow_Tensor:$cur_rank_unique_table_ids,
    OneFlow_Tensor:$cur_rank_inverse_indices
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "1">:$num_tables,
    DefaultValuedAttr<SI64Attr, "-1">:$padding_idx,
    DefaultValuedAttr<BoolAttr, "false">:$has_padding_idx,
    StrAttr:$embedding_name
  );
  let same_output_regst_num = 2;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_IdShuffleCopyOutOp : OneFlow_BaseOp<"id_shuffle_copy_out", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$num_unique_matrix,
    OneFlow_Tensor:$inverse_unique_partition_indices,
    OneFlow_Tensor:$cur_rank_num_unique,
    OneFlow_Tensor:$cur_rank_unique_ids,
    OneFlow_Tensor:$cur_rank_unique_table_ids,
    OneFlow_Tensor:$cur_rank_inverse_indices
  );
  let output = (outs
    OneFlow_Tensor:$out_num_unique_matrix,
    OneFlow_Tensor:$out_inverse_unique_partition_indices,
    OneFlow_Tensor:$out_cur_rank_num_unique,
    OneFlow_Tensor:$out_cur_rank_unique_ids,
    OneFlow_Tensor:$out_cur_rank_unique_table_ids,
    OneFlow_Tensor:$out_cur_rank_inverse_indices
  );
  let attrs = (ins
    StrAttr:$embedding_name
  );
  let same_output_regst_num = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_OneEmbeddingGatherOp : OneFlow_BaseOp<"one_embedding_gather", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in,
    OneFlow_Tensor:$indices
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$embedding_size,
    StrAttr:$embedding_name
  );
  let same_output_regst_num = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_EmbeddingShuffleOp : OneFlow_BaseOp<"embedding_shuffle", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$cur_rank_embeddings,
    OneFlow_Tensor:$num_unique_matrix,
    OneFlow_Tensor:$cur_rank_inverse_indices,
    OneFlow_Tensor:$inverse_unique_partition_indices
  );
  let output = (outs
    OneFlow_Tensor:$embeddings
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$embedding_size,
    DefaultValuedAttr<BoolAttr, "false">:$skip_last_gather,
    DefaultValuedAttr<BoolAttr, "false">:$is_train,
    StrAttr:$embedding_name
  );
  let same_output_regst_num = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_EmbeddingGradientShuffleOp : OneFlow_BaseOp<"embedding_gradient_shuffle", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$embedding_grad,
    OneFlow_Tensor:$num_unique_matrix,
    OneFlow_Tensor:$cur_rank_inverse_indices,
    OneFlow_Tensor:$inverse_unique_partition_indices
  );
  let output = (outs
    OneFlow_Tensor:$cur_rank_unique_embedding_grad
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$embedding_size,
    DefaultValuedAttr<BoolAttr, "false">:$only_zero_valid_grad,
    DefaultValuedAttr<BoolAttr, "false">:$skip_first_scatter,
    StrAttr:$embedding_name
  );
  let same_output_regst_num = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_EmbeddingPrefetchOp : OneFlow_BaseOp<"embedding_prefetch", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$num_unique_ids,
    OneFlow_Tensor:$unique_ids,
    OneFlow_Tensor:$table_ids
  );
  let output = (outs
    OneFlow_Tensor:$context //no practical sense, control lookup run after prefetch.
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$line_size,
    DefaultValuedAttr<SI64Attr, "0">:$embedding_size,
    StrAttr:$embedding_name,
    StrAttr:$embedding_tables,
    StrAttr:$state_initializer,
    DefaultValuedAttr<SI64Attr, "0">:$seed
  );
  let same_output_regst_num = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_EmbeddingLookupOp : OneFlow_BaseOp<"embedding_lookup", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$num_unique_ids,
    OneFlow_Tensor:$unique_ids,
    OneFlow_Tensor:$table_ids,
    Optional<OneFlow_Tensor>:$context
  );
  let output = (outs
    OneFlow_Tensor:$unique_values,
    Optional<OneFlow_Tensor>:$embeddings
  );
  let attrs = (ins
    OneFlow_DataType:$dtype,
    OneFlow_DataType:$embeddings_dtype,
    DefaultValuedAttr<SI64Attr, "0">:$line_size,
    DefaultValuedAttr<SI64Attr, "0">:$embedding_size,
    StrAttr:$embedding_name,
    StrAttr:$embedding_tables,
    StrAttr:$state_initializer,
    DefaultValuedAttr<SI64Attr, "0">:$seed
  );
  let same_output_regst_num = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_OneEmbeddingFusedSgdUpdatePutOp : OneFlow_BaseOp<"one_embedding_fused_sgd_update_put", [DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$num_unique_ids,
    OneFlow_Tensor:$unique_ids,
    OneFlow_Tensor:$unique_embeddings,
    OneFlow_Tensor:$embedding_grad,
    OneFlow_Tensor:$learning_rate
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "1.">:$scale,
    DefaultValuedAttr<SI64Attr, "0">:$line_size,
    DefaultValuedAttr<SI64Attr, "0">:$embedding_size,
    StrAttr:$embedding_name
  );
  let same_output_regst_num = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_OneEmbeddingSgdUpdateOp : OneFlow_BaseOp<"one_embedding_sgd_update", [AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$num_unique_ids,
    OneFlow_Tensor:$unique_embeddings,
    OneFlow_Tensor:$embedding_grad,
    Optional<OneFlow_Tensor>:$learning_rate,
    Optional<OneFlow_Tensor>:$scale_by_tensor,
    Optional<OneFlow_Tensor>:$down_scale_by_tensor,
    Optional<OneFlow_Tensor>:$skip_if
  );
  let output = (outs
    OneFlow_Tensor:$updated_unique_embeddings
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$learning_rate_val,
    DefaultValuedAttr<F64Attr, "1.">:$scale,
    DefaultValuedAttr<F32Attr, "0.">:$l1,
    DefaultValuedAttr<F32Attr, "0.">:$l2,
    DefaultValuedAttr<F32Attr, "0.">:$weight_decay,
    DefaultValuedAttr<SI64Attr, "0">:$line_size,
    DefaultValuedAttr<SI64Attr, "0">:$embedding_size,
    StrAttr:$embedding_name
  );
  let same_output_regst_num = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_OneEmbeddingMomentumUpdateOp : OneFlow_BaseOp<"one_embedding_momentum_update", [AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$num_unique_ids,
    OneFlow_Tensor:$unique_embeddings,
    OneFlow_Tensor:$embedding_grad,
    Optional<OneFlow_Tensor>:$learning_rate,
    Optional<OneFlow_Tensor>:$scale_by_tensor,
    Optional<OneFlow_Tensor>:$down_scale_by_tensor,
    Optional<OneFlow_Tensor>:$skip_if
  );
  let output = (outs
    OneFlow_Tensor:$updated_unique_embeddings
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$learning_rate_val,
    DefaultValuedAttr<F64Attr, "1.">:$scale,
    DefaultValuedAttr<F32Attr, "0.">:$l1,
    DefaultValuedAttr<F32Attr, "0.">:$l2,
    DefaultValuedAttr<F32Attr, "0.">:$weight_decay,
    DefaultValuedAttr<F32Attr, "0.9">:$beta,
    DefaultValuedAttr<SI64Attr, "0">:$line_size,
    DefaultValuedAttr<SI64Attr, "0">:$embedding_size,
    StrAttr:$embedding_name
  );
  let same_output_regst_num = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_OneEmbeddingAdamUpdateOp : OneFlow_BaseOp<"one_embedding_adam_update", [AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$num_unique_ids,
    OneFlow_Tensor:$unique_embeddings,
    OneFlow_Tensor:$embedding_grad,
    Optional<OneFlow_Tensor>:$learning_rate,
    Optional<OneFlow_Tensor>:$scale_by_tensor,
    Optional<OneFlow_Tensor>:$down_scale_by_tensor,
    Optional<OneFlow_Tensor>:$skip_if,
    Optional<OneFlow_Tensor>:$bias_correction1,
    Optional<OneFlow_Tensor>:$bias_correction2
  );
  let output = (outs
    OneFlow_Tensor:$updated_unique_embeddings
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$learning_rate_val,
    DefaultValuedAttr<F32Attr, "1.">:$bias_correction1_val,
    DefaultValuedAttr<F32Attr, "1.">:$bias_correction2_val,
    DefaultValuedAttr<F64Attr, "1.">:$scale,
    DefaultValuedAttr<F32Attr, "0.">:$l1,
    DefaultValuedAttr<F32Attr, "0.">:$l2,
    DefaultValuedAttr<F32Attr, "0.">:$weight_decay,
    DefaultValuedAttr<F32Attr, "0.9">:$beta1,
    DefaultValuedAttr<F32Attr, "0.999">:$beta2,
    DefaultValuedAttr<F32Attr, "0.">:$epsilon,
    DefaultValuedAttr<BoolAttr, "true">:$do_bias_correction,
    DefaultValuedAttr<SI64Attr, "0">:$line_size,
    DefaultValuedAttr<SI64Attr, "0">:$embedding_size,
    StrAttr:$embedding_name
  );
  let same_output_regst_num = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_OneEmbeddingSmartDecaySparseAdamUpdateOp : OneFlow_BaseOp<"one_embedding_smart_decay_sparse_adam_update", [AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$num_unique_ids,
    OneFlow_Tensor:$unique_embeddings,
    OneFlow_Tensor:$embedding_grad,
    Optional<OneFlow_Tensor>:$train_step,
    Optional<OneFlow_Tensor>:$learning_rate,
    Optional<OneFlow_Tensor>:$scale_by_tensor,
    Optional<OneFlow_Tensor>:$down_scale_by_tensor,
    Optional<OneFlow_Tensor>:$skip_if
  );
  let output = (outs
    OneFlow_Tensor:$updated_unique_embeddings
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$train_step_val,
    DefaultValuedAttr<F32Attr, "0.">:$learning_rate_val,
    DefaultValuedAttr<F64Attr, "1.">:$scale,
    DefaultValuedAttr<F32Attr, "0.">:$l1,
    DefaultValuedAttr<F32Attr, "0.">:$l2,
    DefaultValuedAttr<F32Attr, "0.">:$weight_decay,
    DefaultValuedAttr<F32Attr, "0.9">:$beta1,
    DefaultValuedAttr<F32Attr, "0.999">:$beta2,
    DefaultValuedAttr<F32Attr, "0.">:$epsilon,
    DefaultValuedAttr<BoolAttr, "true">:$do_bias_correction,
    DefaultValuedAttr<SI64Attr, "0">:$line_size,
    DefaultValuedAttr<SI64Attr, "0">:$embedding_size,
    StrAttr:$embedding_name
  );
  let same_output_regst_num = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_OneEmbeddingAdagradUpdateOp : OneFlow_BaseOp<"one_embedding_adagrad_update", [AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$num_unique_ids,
    OneFlow_Tensor:$unique_embeddings,
    OneFlow_Tensor:$embedding_grad,
    Optional<OneFlow_Tensor>:$train_step,
    Optional<OneFlow_Tensor>:$learning_rate,
    Optional<OneFlow_Tensor>:$scale_by_tensor,
    Optional<OneFlow_Tensor>:$down_scale_by_tensor,
    Optional<OneFlow_Tensor>:$skip_if
  );
  let output = (outs
    OneFlow_Tensor:$updated_unique_embeddings
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$train_step_val,
    DefaultValuedAttr<F32Attr, "0.">:$learning_rate_val,
    DefaultValuedAttr<F64Attr, "1.">:$scale,
    DefaultValuedAttr<F32Attr, "0.">:$l1,
    DefaultValuedAttr<F32Attr, "0.">:$l2,
    DefaultValuedAttr<F32Attr, "0.">:$weight_decay,
    DefaultValuedAttr<F32Attr, "0.">:$lr_decay,
    DefaultValuedAttr<F32Attr, "0.">:$epsilon,
    DefaultValuedAttr<SI64Attr, "0">:$line_size,
    DefaultValuedAttr<SI64Attr, "0">:$embedding_size,
    StrAttr:$embedding_name
  );
  let same_output_regst_num = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_EmbeddingPutOp : OneFlow_BaseOp<"embedding_put", [DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$num_unique_ids,
    OneFlow_Tensor:$unique_ids,
    OneFlow_Tensor:$unique_embeddings
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$line_size,
    StrAttr:$embedding_name
  );
  let same_output_regst_num = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_OneEmbeddingFtrlUpdateOp : OneFlow_BaseOp<"one_embedding_ftrl_update", [AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$num_unique_ids,
    OneFlow_Tensor:$unique_embeddings,
    OneFlow_Tensor:$embedding_grad,
    Optional<OneFlow_Tensor>:$learning_rate,
    Optional<OneFlow_Tensor>:$scale_by_tensor,
    Optional<OneFlow_Tensor>:$down_scale_by_tensor,
    Optional<OneFlow_Tensor>:$skip_if
  );
  let output = (outs
    OneFlow_Tensor:$updated_unique_embeddings
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$learning_rate_val,
    DefaultValuedAttr<F64Attr, "1.">:$scale,
    DefaultValuedAttr<F32Attr, "0.">:$l1,
    DefaultValuedAttr<F32Attr, "0.">:$l2,
    DefaultValuedAttr<F32Attr, "0.">:$weight_decay,
    DefaultValuedAttr<F32Attr, "0.">:$lr_power,
    DefaultValuedAttr<F32Attr, "0.">:$lambda1,
    DefaultValuedAttr<F32Attr, "0.">:$lambda2,
    DefaultValuedAttr<F32Attr, "0.">:$beta,
    DefaultValuedAttr<SI64Attr, "0">:$line_size,
    DefaultValuedAttr<SI64Attr, "0">:$embedding_size,
    StrAttr:$embedding_name
  );
  let same_output_regst_num = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_RocAucScoreOp : OneFlow_BaseOp<"roc_auc_score", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$label,
    OneFlow_Tensor:$pred
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FillOp : OneFlow_BaseOp<"fill_", [NoMemoryEffect, SupportNonContiguous, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.">:$floating_value,
    DefaultValuedAttr<SI64Attr, "0.">:$integral_value,
    DefaultValuedAttr<BoolAttr, "false">:$is_floating_value
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FillTensorOp : OneFlow_BaseOp<"fill_tensor_", [NoMemoryEffect, SupportNonContiguous, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in,
    OneFlow_Tensor:$value
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

#endif // GET_ONEFLOW_ONE_EMBEDDING_OP_DEFINITIONS


#ifdef GET_ONEFLOW_LINEAR_ALGEBRA_OP_DEFINITIONS

def OneFlow_InvOp : OneFlow_BaseOp<"inv", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_LinalgCrossOp : OneFlow_BaseOp<"linalg_cross", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$input,
    OneFlow_Tensor:$other
  );
  let attrs = (ins
    SI64Attr:$dim
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_DetOp : OneFlow_BaseOp<"det", [NoMemoryEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_LUDecompositionOp : OneFlow_BaseOp<"lu_decomposition", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$LU,
    OneFlow_Tensor:$pivot
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

#endif // GET_ONEFLOW_LINEAR_ALGEBRA_OP_DEFINITIONS


#ifdef GET_ONEFLOW_SYSTEM_OP_DEFINITIONS

def OneFlow_CopyH2DOp : OneFlow_BaseOp<"copy_h2d", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
      OneFlow_Tensor:$in
  );
  let output = (outs
      OneFlow_Tensor:$out
  );

  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_CopyD2HOp : OneFlow_BaseOp<"copy_d2h", [NoMemoryEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
      OneFlow_Tensor:$in
  );
  let output = (outs
      OneFlow_Tensor:$out
  );

  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

#endif // GET_ONEFLOW_SYSTEM_OP_DEFINITIONS

include "mlir/Interfaces/CallInterfaces.td"

class OneFlow_JITLikeOp <string mnemonic> : OneFlow_BaseOp<mnemonic, [CallOpInterface, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins Variadic<AnyType>:$in);
  let output = (outs Variadic<AnyType>:$out);
  let attrs = (ins
    FlatSymbolRefAttr:$callee,
    BytesAttr:$mlir_assembly
  );
  let builders = [
    OpBuilder<(ins "func::FuncOp":$callee,
      "NamedAttrList":$attributes,
      CArg<"ValueRange", "{}">:$in), [{
      $_state.addOperands(in);
      $_state.addAttributes(attributes);
      $_state.addAttribute("callee", SymbolRefAttr::get(callee));
      $_state.addTypes(callee.getFunctionType().getResults());
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
    
    void setCalleeFromCallable(CallInterfaceCallable callee) {
      (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
    }
  }];
  let assemblyFormat = [{
    $callee `(` $in `)` attr-dict `:` functional-type($in, results)
  }];
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}


#ifdef GET_ONEFLOW_MLIR_JIT_OP_DEFINITIONS

def OneFlow_MlirJitOp : OneFlow_JITLikeOp<"mlir_jit"> {}

def OneFlow_KernelLaunchOp : OneFlow_JITLikeOp<"kernel_launch"> {}

#endif // GET_ONEFLOW_MLIR_JIT_OP_DEFINITIONS

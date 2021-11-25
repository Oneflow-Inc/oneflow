// assign, assign_if, assign_if_not, logical_slice_assign
// Total: 4
#ifndef GET_ONEFLOW_ASSIGN_OP_DEFINITIONS
#define GET_ONEFLOW_ASSIGN_OP_DEFINITIONS

def OneFlow_AssignOp : OneFlow_BaseOp<"assign", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$ref,
    AnyType:$value
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"ref", "value"}; }
    static std::vector<std::string> outputOrder() { return {}; }
  }];
}

def OneFlow_AssignIfOp : OneFlow_BaseOp<"assign_if", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$ref,
    AnyType:$value,
    AnyType:$condition
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"ref", "value", "condition"}; }
    static std::vector<std::string> outputOrder() { return {}; }
  }];
}

def OneFlow_AssignIfNotOp : OneFlow_BaseOp<"assign_if_not", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$ref,
    AnyType:$value,
    AnyType:$condition
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"ref", "value", "condition"}; }
    static std::vector<std::string> outputOrder() { return {}; }
  }];
}

def OneFlow_LogicalSliceAssignOp : OneFlow_BaseOp<"logical_slice_assign", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$ref,
    AnyType:$value
  );
  let attrs = (ins
    SI64ArrayAttr:$start,
    SI64ArrayAttr:$stop,
    SI64ArrayAttr:$step
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"ref", "value"}; }
    static std::vector<std::string> outputOrder() { return {}; }
  }];
}

#endif  // GET_ONEFLOW_ASSIGN_OP_DEFINITIONS

// broadcast_add, broadcast_div, broadcast_div_grad, broadcast_equal, broadcast_floor_mod, broadcast_fmod, broadcast_greater, broadcast_greater_equal, broadcast_less, broadcast_less_equal, broadcast_like, broadcast_logical_and, broadcast_logical_or, broadcast_logical_xor, broadcast_maximum, broadcast_minimum, broadcast_mul, broadcast_not_equal, broadcast_pow, broadcast_pow_x_grad, broadcast_pow_y_grad, broadcast_sub
// Total: 22
#ifndef GET_ONEFLOW_BROADCAST_OP_DEFINITIONS
#define GET_ONEFLOW_BROADCAST_OP_DEFINITIONS

def OneFlow_BroadcastAddOp : OneFlow_BaseOp<"broadcast_add", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "y"}; }
    static std::vector<std::string> outputOrder() { return {"z"}; }
  }];
}

def OneFlow_BroadcastDivOp : OneFlow_BaseOp<"broadcast_div", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "y"}; }
    static std::vector<std::string> outputOrder() { return {"z"}; }
  }];
}

def OneFlow_BroadcastDivGradOp : OneFlow_BaseOp<"broadcast_div_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$y,
    AnyType:$z,
    AnyType:$dz
  );
  let output = (outs
    AnyType:$dy
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"y", "z", "dz"}; }
    static std::vector<std::string> outputOrder() { return {"dy"}; }
  }];
}

def OneFlow_BroadcastEqualOp : OneFlow_BaseOp<"broadcast_equal", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "y"}; }
    static std::vector<std::string> outputOrder() { return {"z"}; }
  }];
}

def OneFlow_BroadcastFloorModOp : OneFlow_BaseOp<"broadcast_floor_mod", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "y"}; }
    static std::vector<std::string> outputOrder() { return {"z"}; }
  }];
}

def OneFlow_BroadcastFmodOp : OneFlow_BaseOp<"broadcast_fmod", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "y"}; }
    static std::vector<std::string> outputOrder() { return {"z"}; }
  }];
}

def OneFlow_BroadcastGreaterOp : OneFlow_BaseOp<"broadcast_greater", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "y"}; }
    static std::vector<std::string> outputOrder() { return {"z"}; }
  }];
}

def OneFlow_BroadcastGreaterEqualOp : OneFlow_BaseOp<"broadcast_greater_equal", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "y"}; }
    static std::vector<std::string> outputOrder() { return {"z"}; }
  }];
}

def OneFlow_BroadcastLessOp : OneFlow_BaseOp<"broadcast_less", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "y"}; }
    static std::vector<std::string> outputOrder() { return {"z"}; }
  }];
}

def OneFlow_BroadcastLessEqualOp : OneFlow_BaseOp<"broadcast_less_equal", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "y"}; }
    static std::vector<std::string> outputOrder() { return {"z"}; }
  }];
}

def OneFlow_BroadcastLikeOp : OneFlow_BaseOp<"broadcast_like", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "like"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_BroadcastLogicalAndOp : OneFlow_BaseOp<"broadcast_logical_and", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "y"}; }
    static std::vector<std::string> outputOrder() { return {"z"}; }
  }];
}

def OneFlow_BroadcastLogicalOrOp : OneFlow_BaseOp<"broadcast_logical_or", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "y"}; }
    static std::vector<std::string> outputOrder() { return {"z"}; }
  }];
}

def OneFlow_BroadcastLogicalXorOp : OneFlow_BaseOp<"broadcast_logical_xor", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "y"}; }
    static std::vector<std::string> outputOrder() { return {"z"}; }
  }];
}

def OneFlow_BroadcastMaximumOp : OneFlow_BaseOp<"broadcast_maximum", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "y"}; }
    static std::vector<std::string> outputOrder() { return {"z"}; }
  }];
}

def OneFlow_BroadcastMinimumOp : OneFlow_BaseOp<"broadcast_minimum", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "y"}; }
    static std::vector<std::string> outputOrder() { return {"z"}; }
  }];
}

def OneFlow_BroadcastMulOp : OneFlow_BaseOp<"broadcast_mul", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "y"}; }
    static std::vector<std::string> outputOrder() { return {"z"}; }
  }];
}

def OneFlow_BroadcastNotEqualOp : OneFlow_BaseOp<"broadcast_not_equal", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "y"}; }
    static std::vector<std::string> outputOrder() { return {"z"}; }
  }];
}

def OneFlow_BroadcastPowOp : OneFlow_BaseOp<"broadcast_pow", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "y"}; }
    static std::vector<std::string> outputOrder() { return {"z"}; }
  }];
}

def OneFlow_BroadcastPowXGradOp : OneFlow_BaseOp<"broadcast_pow_x_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y,
    AnyType:$z,
    AnyType:$dz
  );
  let output = (outs
    AnyType:$dx
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "y", "z", "dz"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_BroadcastPowYGradOp : OneFlow_BaseOp<"broadcast_pow_y_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y,
    AnyType:$z,
    AnyType:$dz
  );
  let output = (outs
    AnyType:$dy
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "y", "z", "dz"}; }
    static std::vector<std::string> outputOrder() { return {"dy"}; }
  }];
}

def OneFlow_BroadcastSubOp : OneFlow_BaseOp<"broadcast_sub", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "y"}; }
    static std::vector<std::string> outputOrder() { return {"z"}; }
  }];
}

#endif  // GET_ONEFLOW_BROADCAST_OP_DEFINITIONS

// conv1d, conv2d, conv3d, conv_bias_grad, conv_data_grad, conv_filter_grad, deconv1d, deconv2d, deconv3d
// Total: 9
#ifndef GET_ONEFLOW_CONV_OP_DEFINITIONS
#define GET_ONEFLOW_CONV_OP_DEFINITIONS

def OneFlow_Conv1DOp : OneFlow_ConvolutionBaseOp<"conv1d", [AttrSizedOperandSegments, DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in,
    AnyType:$weight,
    Optional<AnyType>:$bias,
    Optional<AnyType>:$bias_multiplier
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
  let trait_attrs = (ins
    I32ElementsAttr:$operand_segment_sizes
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in", "weight", "bias", "bias_multiplier"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_Conv2DOp : OneFlow_ConvolutionBaseOp<"conv2d", [AttrSizedOperandSegments, DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in,
    AnyType:$weight,
    Optional<AnyType>:$bias,
    Optional<AnyType>:$bias_multiplier
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
  let trait_attrs = (ins
    I32ElementsAttr:$operand_segment_sizes
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in", "weight", "bias", "bias_multiplier"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_Conv3DOp : OneFlow_ConvolutionBaseOp<"conv3d", [AttrSizedOperandSegments, DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in,
    AnyType:$weight,
    Optional<AnyType>:$bias,
    Optional<AnyType>:$bias_multiplier
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
  let trait_attrs = (ins
    I32ElementsAttr:$operand_segment_sizes
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in", "weight", "bias", "bias_multiplier"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_ConvBiasGradOp : OneFlow_BaseOp<"conv_bias_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"dy"}; }
    static std::vector<std::string> outputOrder() { return {"bias_diff"}; }
  }];
}

def OneFlow_ConvDataGradOp : OneFlow_BaseOp<"conv_data_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$dy,
    AnyType:$filter,
    AnyType:$x_like,
    Optional<AnyType>:$_add_to_output
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"dy", "filter", "x_like", "_add_to_output"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_ConvFilterGradOp : OneFlow_BaseOp<"conv_filter_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"dy", "x"}; }
    static std::vector<std::string> outputOrder() { return {"filter_diff"}; }
  }];
}

def OneFlow_Deconv1DOp : OneFlow_BaseOp<"deconv1d", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in", "weight"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_Deconv2DOp : OneFlow_BaseOp<"deconv2d", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in", "weight"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_Deconv3DOp : OneFlow_BaseOp<"deconv3d", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in", "weight"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

#endif  // GET_ONEFLOW_CONV_OP_DEFINITIONS

// binary_cross_entropy, binary_cross_entropy_grad, binary_cross_entropy_with_logits, binary_cross_entropy_with_logits_grad, sigmoid_cross_entropy, sigmoid_cross_entropy_grad, sparse_cross_entropy, sparse_cross_entropy_grad, sparse_cross_entropy_ms, sparse_cross_entropy_ms_grad
// Total: 10
#ifndef GET_ONEFLOW_CROSS_ENTROPY_OP_DEFINITIONS
#define GET_ONEFLOW_CROSS_ENTROPY_OP_DEFINITIONS

def OneFlow_BinaryCrossEntropyOp : OneFlow_BaseOp<"binary_cross_entropy", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$input,
    AnyType:$target,
    Optional<AnyType>:$weight
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    StrAttr:$reduction
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"input", "target", "weight"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_BinaryCrossEntropyGradOp : OneFlow_BaseOp<"binary_cross_entropy_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$input,
    AnyType:$target,
    Optional<AnyType>:$weight,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    StrAttr:$reduction
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"input", "target", "weight", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_BinaryCrossEntropyWithLogitsOp : OneFlow_BaseOp<"binary_cross_entropy_with_logits", [AttrSizedOperandSegments, DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$input,
    AnyType:$target,
    Optional<AnyType>:$weight,
    Optional<AnyType>:$pos_weight
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    BoolAttr:$has_pos_weight,
    StrAttr:$reduction
  );
  let trait_attrs = (ins
    I32ElementsAttr:$operand_segment_sizes
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"input", "target", "weight", "pos_weight"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_BinaryCrossEntropyWithLogitsGradOp : OneFlow_BaseOp<"binary_cross_entropy_with_logits_grad", [AttrSizedOperandSegments, DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$input,
    AnyType:$target,
    Optional<AnyType>:$weight,
    Optional<AnyType>:$pos_weight,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    BoolAttr:$has_pos_weight,
    StrAttr:$reduction
  );
  let trait_attrs = (ins
    I32ElementsAttr:$operand_segment_sizes
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"input", "target", "weight", "pos_weight", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_SigmoidCrossEntropyOp : OneFlow_BaseOp<"sigmoid_cross_entropy", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$prediction,
    AnyType:$label
  );
  let output = (outs
    AnyType:$loss
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"prediction", "label"}; }
    static std::vector<std::string> outputOrder() { return {"loss"}; }
  }];
}

def OneFlow_SigmoidCrossEntropyGradOp : OneFlow_BaseOp<"sigmoid_cross_entropy_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$prediction,
    AnyType:$loss_diff,
    AnyType:$label
  );
  let output = (outs
    AnyType:$prediction_diff
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"prediction", "loss_diff", "label"}; }
    static std::vector<std::string> outputOrder() { return {"prediction_diff"}; }
  }];
}

def OneFlow_SparseCrossEntropyOp : OneFlow_BaseOp<"sparse_cross_entropy", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"prediction", "label"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_SparseCrossEntropyGradOp : OneFlow_BaseOp<"sparse_cross_entropy_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"prediction", "label", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"prediction_diff"}; }
  }];
}

def OneFlow_SparseCrossEntropyMsOp : OneFlow_BaseOp<"sparse_cross_entropy_ms", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"prediction", "label"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_SparseCrossEntropyMsGradOp : OneFlow_BaseOp<"sparse_cross_entropy_ms_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"prediction", "label", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"prediction_diff"}; }
  }];
}

#endif  // GET_ONEFLOW_CROSS_ENTROPY_OP_DEFINITIONS

// COCOReader, OFRecordReader, OneRecReader, ctc_greedy_decoder, megatron_gpt_mmap_data_loader, ofrecord_bytes_decoder, ofrecord_image_classification_reader, ofrecord_image_decoder, ofrecord_image_decoder_random_crop, ofrecord_raw_decoder, onerec_decoder
// Total: 11
#ifndef GET_ONEFLOW_DATASET_OP_DEFINITIONS
#define GET_ONEFLOW_DATASET_OP_DEFINITIONS

def OneFlow_COCOReaderOp : OneFlow_BaseOp<"COCOReader", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {}; }
    static std::vector<std::string> outputOrder() { return {"image", "image_id", "image_size", "gt_bbox", "gt_label", "gt_segm", "gt_segm_index"}; }
  }];
}

def OneFlow_OFRecordReaderOp : OneFlow_BaseOp<"OFRecordReader", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_OneRecReaderOp : OneFlow_BaseOp<"OneRecReader", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_CtcGreedyDecoderOp : OneFlow_BaseOp<"ctc_greedy_decoder", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"log_probs", "input_lengths"}; }
    static std::vector<std::string> outputOrder() { return {"decoded", "neg_sum_logits"}; }
  }];
}

def OneFlow_MegatronGptMmapDataLoaderOp : OneFlow_BaseOp<"megatron_gpt_mmap_data_loader", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    Optional<AnyType>:$iteration
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"iteration"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_OfrecordBytesDecoderOp : OneFlow_BaseOp<"ofrecord_bytes_decoder", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    StrAttr:$name
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_OfrecordImageClassificationReaderOp : OneFlow_BaseOp<"ofrecord_image_classification_reader", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {}; }
    static std::vector<std::string> outputOrder() { return {"image", "label"}; }
  }];
}

def OneFlow_OfrecordImageDecoderOp : OneFlow_BaseOp<"ofrecord_image_decoder", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_OfrecordImageDecoderRandomCropOp : OneFlow_BaseOp<"ofrecord_image_decoder_random_crop", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_OfrecordRawDecoderOp : OneFlow_BaseOp<"ofrecord_raw_decoder", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_OnerecDecoderOp : OneFlow_BaseOp<"onerec_decoder", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

#endif  // GET_ONEFLOW_DATASET_OP_DEFINITIONS

// in_top_k, nms, object_bbox_flip, object_bbox_scale, object_segmentation_polygon_flip, object_segmentation_polygon_scale, object_segmentation_polygon_to_mask, top_k
// Total: 8
#ifndef GET_ONEFLOW_DETECTION_OP_DEFINITIONS
#define GET_ONEFLOW_DETECTION_OP_DEFINITIONS

def OneFlow_InTopKOp : OneFlow_BaseOp<"in_top_k", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"targets", "predictions"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_NmsOp : OneFlow_BaseOp<"nms", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_ObjectBboxFlipOp : OneFlow_BaseOp<"object_bbox_flip", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$bbox,
    AnyType:$image_size,
    AnyType:$flip_code
  );
  let output = (outs
    AnyType:$out
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"bbox", "image_size", "flip_code"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_ObjectBboxScaleOp : OneFlow_BaseOp<"object_bbox_scale", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$bbox,
    AnyType:$scale
  );
  let output = (outs
    AnyType:$out
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"bbox", "scale"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_ObjectSegmentationPolygonFlipOp : OneFlow_BaseOp<"object_segmentation_polygon_flip", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$poly,
    AnyType:$image_size,
    AnyType:$flip_code
  );
  let output = (outs
    AnyType:$out
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"poly", "image_size", "flip_code"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_ObjectSegmentationPolygonScaleOp : OneFlow_BaseOp<"object_segmentation_polygon_scale", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$poly,
    AnyType:$scale
  );
  let output = (outs
    AnyType:$out
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"poly", "scale"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_ObjectSegmentationPolygonToMaskOp : OneFlow_BaseOp<"object_segmentation_polygon_to_mask", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$poly,
    AnyType:$poly_index,
    AnyType:$image_size
  );
  let output = (outs
    AnyType:$out
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"poly", "poly_index", "image_size"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_TopKOp : OneFlow_BaseOp<"top_k", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

#endif  // GET_ONEFLOW_DETECTION_OP_DEFINITIONS

// eager_b_to_s, eager_naive_s_to_s, eager_nccl_all_gather, eager_nccl_all_reduce, eager_nccl_broadcast, eager_nccl_reduce, eager_nccl_reduce_scatter, eager_nccl_s2s, eager_p_to_b, eager_p_to_s, eager_s_to_b, eager_symmetric_s_to_p
// Total: 12
#ifndef GET_ONEFLOW_EAGER_OP_DEFINITIONS
#define GET_ONEFLOW_EAGER_OP_DEFINITIONS

def OneFlow_EagerBToSOp : OneFlow_BaseOp<"eager_b_to_s", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_EagerNaiveSToSOp : OneFlow_BaseOp<"eager_naive_s_to_s", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_EagerNcclAllGatherOp : OneFlow_BaseOp<"eager_nccl_all_gather", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    StrAttr:$parallel_conf
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_EagerNcclAllReduceOp : OneFlow_BaseOp<"eager_nccl_all_reduce", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_EagerNcclBroadcastOp : OneFlow_BaseOp<"eager_nccl_broadcast", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_EagerNcclReduceOp : OneFlow_BaseOp<"eager_nccl_reduce", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_EagerNcclReduceScatterOp : OneFlow_BaseOp<"eager_nccl_reduce_scatter", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_EagerNcclS2sOp : OneFlow_BaseOp<"eager_nccl_s2s", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_EagerPToBOp : OneFlow_BaseOp<"eager_p_to_b", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_EagerPToSOp : OneFlow_BaseOp<"eager_p_to_s", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_EagerSToBOp : OneFlow_BaseOp<"eager_s_to_b", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_EagerSymmetricSToPOp : OneFlow_BaseOp<"eager_symmetric_s_to_p", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

#endif  // GET_ONEFLOW_EAGER_OP_DEFINITIONS

// cudnn_fused_normalization_add_relu, cudnn_fused_normalization_add_relu_grad, fused_bias_add_gelu, fused_bias_add_gelu_grad, fused_bias_add_mask_scale, fused_cast_scale, fused_scale_tril, fused_self_attention_query_mul_key_and_value, fused_self_attention_query_mul_key_and_value_grad, fused_tril_scale_softmax_mask_scale, fused_tril_scale_softmax_mask_scale_grad, normalization_add_relu, normalization_add_relu_grad
// Total: 13
#ifndef GET_ONEFLOW_FUSED_OP_DEFINITIONS
#define GET_ONEFLOW_FUSED_OP_DEFINITIONS

def OneFlow_CudnnFusedNormalizationAddReluOp : OneFlow_BaseOp<"cudnn_fused_normalization_add_relu", [AttrSizedOperandSegments, AttrSizedResultSegments, DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    Optional<AnyType>:$addend,
    Optional<AnyType>:$moving_mean,
    Optional<AnyType>:$moving_variance,
    AnyType:$gamma,
    AnyType:$beta
  );
  let output = (outs
    AnyType:$y,
    AnyType:$reserve_space,
    Optional<AnyType>:$mean,
    Optional<AnyType>:$inv_variance
  );
  let attrs = (ins
    SI32Attr:$axis,
    F32Attr:$epsilon,
    F32Attr:$momentum
  );
  let trait_attrs = (ins
    I32ElementsAttr:$operand_segment_sizes,
    I32ElementsAttr:$result_segment_sizes
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "addend", "moving_mean", "moving_variance", "gamma", "beta"}; }
    static std::vector<std::string> outputOrder() { return {"y", "reserve_space", "mean", "inv_variance"}; }
  }];
}

def OneFlow_CudnnFusedNormalizationAddReluGradOp : OneFlow_BaseOp<"cudnn_fused_normalization_add_relu_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
    Optional<AnyType>:$addend_diff
  );
  let attrs = (ins
    SI32Attr:$axis,
    F32Attr:$epsilon
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "dy", "mean", "inv_variance", "gamma", "beta", "reserve_space", "y"}; }
    static std::vector<std::string> outputOrder() { return {"gamma_diff", "beta_diff", "dx", "addend_diff"}; }
  }];
}

def OneFlow_FusedBiasAddGeluOp : OneFlow_BaseOp<"fused_bias_add_gelu", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"a", "b"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_FusedBiasAddGeluGradOp : OneFlow_BaseOp<"fused_bias_add_gelu_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"a", "b", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_FusedBiasAddMaskScaleOp : OneFlow_BaseOp<"fused_bias_add_mask_scale", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$a,
    AnyType:$b,
    AnyType:$mask,
    Optional<AnyType>:$_add_to_output
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI32Attr:$axis,
    F32Attr:$scale
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"a", "b", "mask", "_add_to_output"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_FusedCastScaleOp : OneFlow_BaseOp<"fused_cast_scale", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "scale_by_tensor"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_FusedScaleTrilOp : OneFlow_BaseOp<"fused_scale_tril", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_FusedSelfAttentionQueryMulKeyAndValueOp : OneFlow_BaseOp<"fused_self_attention_query_mul_key_and_value", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"hidden_states"}; }
    static std::vector<std::string> outputOrder() { return {"query_mul_key", "value"}; }
  }];
}

def OneFlow_FusedSelfAttentionQueryMulKeyAndValueGradOp : OneFlow_BaseOp<"fused_self_attention_query_mul_key_and_value_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"query_mul_key_grad", "value_grad", "hidden_states"}; }
    static std::vector<std::string> outputOrder() { return {"hidden_states_grad"}; }
  }];
}

def OneFlow_FusedTrilScaleSoftmaxMaskScaleOp : OneFlow_BaseOp<"fused_tril_scale_softmax_mask_scale", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "mask"}; }
    static std::vector<std::string> outputOrder() { return {"y", "softmax_y"}; }
  }];
}

def OneFlow_FusedTrilScaleSoftmaxMaskScaleGradOp : OneFlow_BaseOp<"fused_tril_scale_softmax_mask_scale_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"softmax_y", "dy", "mask"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

class OneFlow_NormalizationAddReluBaseOp : OneFlow_BaseOp<"normalization_add_relu", [AttrSizedOperandSegments, AttrSizedResultSegments, DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    Optional<AnyType>:$addend,
    Optional<AnyType>:$moving_mean,
    Optional<AnyType>:$moving_variance,
    AnyType:$gamma,
    AnyType:$beta
  );
  let output = (outs
    AnyType:$y,
    AnyType:$reserve_space,
    Optional<AnyType>:$mean,
    Optional<AnyType>:$inv_variance
  );
  let attrs = (ins
    SI32Attr:$axis,
    F32Attr:$epsilon,
    BoolAttr:$training,
    F32Attr:$momentum
  );
  let trait_attrs = (ins
    I32ElementsAttr:$operand_segment_sizes,
    I32ElementsAttr:$result_segment_sizes
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "addend", "moving_mean", "moving_variance", "gamma", "beta"}; }
    static std::vector<std::string> outputOrder() { return {"y", "reserve_space", "mean", "inv_variance"}; }
  }];
}

def OneFlow_NormalizationAddReluGradOp : OneFlow_BaseOp<"normalization_add_relu_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
    Optional<AnyType>:$addend_diff
  );
  let attrs = (ins
    SI32Attr:$axis,
    F32Attr:$epsilon
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "dy", "mean", "inv_variance", "gamma", "beta", "reserve_space", "y"}; }
    static std::vector<std::string> outputOrder() { return {"gamma_diff", "beta_diff", "dx", "addend_diff"}; }
  }];
}

#endif  // GET_ONEFLOW_FUSED_OP_DEFINITIONS

// abs, ceil, floor, ones_like, relu, rint, round, sign
// Total: 8
#ifndef GET_ONEFLOW_IDEMPOTENT_OP_DEFINITIONS
#define GET_ONEFLOW_IDEMPOTENT_OP_DEFINITIONS

def OneFlow_AbsOp : OneFlow_IdempotentBaseOp<"abs", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_CeilOp : OneFlow_IdempotentBaseOp<"ceil", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_FloorOp : OneFlow_IdempotentBaseOp<"floor", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_OnesLikeOp : OneFlow_IdempotentBaseOp<"ones_like", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$like
  );
  let output = (outs
    AnyType:$out
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"like"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_ReluOp : OneFlow_IdempotentBaseOp<"relu", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_RintOp : OneFlow_IdempotentBaseOp<"rint", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_RoundOp : OneFlow_IdempotentBaseOp<"round", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_SignOp : OneFlow_IdempotentBaseOp<"sign", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

#endif  // GET_ONEFLOW_IDEMPOTENT_OP_DEFINITIONS

// amp_white_identity, identity, identity_buffer, tuple_identity
// Total: 4
#ifndef GET_ONEFLOW_IDENTITY_OP_DEFINITIONS
#define GET_ONEFLOW_IDENTITY_OP_DEFINITIONS

def OneFlow_AmpWhiteIdentityOp : OneFlow_BaseOp<"amp_white_identity", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_IdentityOp : OneFlow_BaseOp<"identity", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_IdentityBufferOp : OneFlow_BaseOp<"identity_buffer", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI64Attr:$buffer_size
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_TupleIdentityOp : OneFlow_BaseOp<"tuple_identity", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    Variadic<AnyType>:$in
  );
  let output = (outs
    Variadic<AnyType>:$out
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

#endif  // GET_ONEFLOW_IDENTITY_OP_DEFINITIONS

// image_batch_align, image_decode, image_flip, image_random_crop, image_resize_keep_aspect_ratio, image_resize_to_fixed
// Total: 6
#ifndef GET_ONEFLOW_IMAGE_OP_DEFINITIONS
#define GET_ONEFLOW_IMAGE_OP_DEFINITIONS

def OneFlow_ImageBatchAlignOp : OneFlow_BaseOp<"image_batch_align", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_ImageDecodeOp : OneFlow_BaseOp<"image_decode", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_ImageFlipOp : OneFlow_BaseOp<"image_flip", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in,
    AnyType:$flip_code
  );
  let output = (outs
    AnyType:$out
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in", "flip_code"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_ImageRandomCropOp : OneFlow_BaseOp<"image_random_crop", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_ImageResizeKeepAspectRatioOp : OneFlow_BaseOp<"image_resize_keep_aspect_ratio", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out", "size", "scale"}; }
  }];
}

def OneFlow_ImageResizeToFixedOp : OneFlow_BaseOp<"image_resize_to_fixed", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out", "scale"}; }
  }];
}

#endif  // GET_ONEFLOW_IMAGE_OP_DEFINITIONS

// arg_sort, argmax, argwhere, batch_gather, dim_gather, dim_scatter_add, dim_scatter_add_like, dim_scatter_add_scalar, dim_scatter_mul, dim_scatter_mul_scalar, dim_scatter_update, dim_scatter_update_scalar, gather, gather_nd, image_target_resize, logical_slice, scatter_nd, scatter_nd_like, slice, slice_grad, tensor_scatter_nd_add, tensor_scatter_nd_update, unsorted_batch_segment_sum, unsorted_segment_sum, unsorted_segment_sum_like, where, where_scalar_x, where_scalar_xy, where_scalar_y
// Total: 29
#ifndef GET_ONEFLOW_INDICES_OP_DEFINITIONS
#define GET_ONEFLOW_INDICES_OP_DEFINITIONS

def OneFlow_ArgSortOp : OneFlow_BaseOp<"arg_sort", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    StrAttr:$direction
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_ArgmaxOp : OneFlow_BaseOp<"argmax", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_ArgwhereOp : OneFlow_BaseOp<"argwhere", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"input"}; }
    static std::vector<std::string> outputOrder() { return {"output", "output_size"}; }
  }];
}

def OneFlow_BatchGatherOp : OneFlow_BaseOp<"batch_gather", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in,
    AnyType:$indices
  );
  let output = (outs
    AnyType:$out
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in", "indices"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_DimGatherOp : OneFlow_BaseOp<"dim_gather", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"input", "index"}; }
    static std::vector<std::string> outputOrder() { return {"output"}; }
  }];
}

def OneFlow_DimScatterAddOp : OneFlow_BaseOp<"dim_scatter_add", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"input", "index", "src"}; }
    static std::vector<std::string> outputOrder() { return {"output"}; }
  }];
}

def OneFlow_DimScatterAddLikeOp : OneFlow_BaseOp<"dim_scatter_add_like", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"like", "index", "src"}; }
    static std::vector<std::string> outputOrder() { return {"output"}; }
  }];
}

def OneFlow_DimScatterAddScalarOp : OneFlow_BaseOp<"dim_scatter_add_scalar", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"input", "index"}; }
    static std::vector<std::string> outputOrder() { return {"output"}; }
  }];
}

def OneFlow_DimScatterMulOp : OneFlow_BaseOp<"dim_scatter_mul", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"input", "index", "src"}; }
    static std::vector<std::string> outputOrder() { return {"output"}; }
  }];
}

def OneFlow_DimScatterMulScalarOp : OneFlow_BaseOp<"dim_scatter_mul_scalar", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"input", "index"}; }
    static std::vector<std::string> outputOrder() { return {"output"}; }
  }];
}

def OneFlow_DimScatterUpdateOp : OneFlow_BaseOp<"dim_scatter_update", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"input", "index", "src"}; }
    static std::vector<std::string> outputOrder() { return {"output"}; }
  }];
}

def OneFlow_DimScatterUpdateScalarOp : OneFlow_BaseOp<"dim_scatter_update_scalar", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"input", "index"}; }
    static std::vector<std::string> outputOrder() { return {"output"}; }
  }];
}

def OneFlow_GatherOp : OneFlow_BaseOp<"gather", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in", "indices"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_GatherNdOp : OneFlow_BaseOp<"gather_nd", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$params,
    AnyType:$indices
  );
  let output = (outs
    AnyType:$out
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"params", "indices"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_ImageTargetResizeOp : OneFlow_BaseOp<"image_target_resize", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out", "size", "scale"}; }
  }];
}

def OneFlow_LogicalSliceOp : OneFlow_BaseOp<"logical_slice", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_ScatterNdOp : OneFlow_BaseOp<"scatter_nd", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"indices", "updates"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_ScatterNdLikeOp : OneFlow_BaseOp<"scatter_nd_like", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$like,
    AnyType:$indices,
    AnyType:$updates
  );
  let output = (outs
    AnyType:$out
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"like", "indices", "updates"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_SliceOp : OneFlow_BaseOp<"slice", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_SliceGradOp : OneFlow_BaseOp<"slice_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"dy", "like"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_TensorScatterNdAddOp : OneFlow_BaseOp<"tensor_scatter_nd_add", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$params,
    AnyType:$updates,
    AnyType:$indices
  );
  let output = (outs
    AnyType:$out
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"params", "updates", "indices"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_TensorScatterNdUpdateOp : OneFlow_BaseOp<"tensor_scatter_nd_update", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$params,
    AnyType:$updates,
    AnyType:$indices
  );
  let output = (outs
    AnyType:$out
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"params", "updates", "indices"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_UnsortedBatchSegmentSumOp : OneFlow_BaseOp<"unsorted_batch_segment_sum", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"data", "segment_ids"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_UnsortedSegmentSumOp : OneFlow_BaseOp<"unsorted_segment_sum", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"data", "segment_ids"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_UnsortedSegmentSumLikeOp : OneFlow_BaseOp<"unsorted_segment_sum_like", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"data", "segment_ids", "like"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_WhereOp : OneFlow_BaseOp<"where", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$condition,
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$out
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"condition", "x", "y"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_WhereScalarXOp : OneFlow_BaseOp<"where_scalar_x", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"condition", "y"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_WhereScalarXyOp : OneFlow_BaseOp<"where_scalar_xy", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"condition"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_WhereScalarYOp : OneFlow_BaseOp<"where_scalar_y", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"condition", "x"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

#endif  // GET_ONEFLOW_INDICES_OP_DEFINITIONS

// negative, reciprocal
// Total: 2
#ifndef GET_ONEFLOW_INVOLUTION_OP_DEFINITIONS
#define GET_ONEFLOW_INVOLUTION_OP_DEFINITIONS

def OneFlow_NegativeOp : OneFlow_InvolutionBaseOp<"negative", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_ReciprocalOp : OneFlow_InvolutionBaseOp<"reciprocal", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

#endif  // GET_ONEFLOW_INVOLUTION_OP_DEFINITIONS

// combined_margin_loss, combined_margin_loss_grad, ctc_loss, ctc_loss_grad, dynamic_loss_scale_schedule, kl_div_loss, kl_div_loss_grad, smooth_l1_loss, smooth_l1_loss_grad
// Total: 9
#ifndef GET_ONEFLOW_LOSS_OP_DEFINITIONS
#define GET_ONEFLOW_LOSS_OP_DEFINITIONS

def OneFlow_CombinedMarginLossOp : OneFlow_BaseOp<"combined_margin_loss", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "label"}; }
    static std::vector<std::string> outputOrder() { return {"y", "theta"}; }
  }];
}

def OneFlow_CombinedMarginLossGradOp : OneFlow_BaseOp<"combined_margin_loss_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"dy", "label", "theta"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_CtcLossOp : OneFlow_BaseOp<"ctc_loss", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"log_probs", "targets", "input_lengths", "target_lengths"}; }
    static std::vector<std::string> outputOrder() { return {"loss", "alpha"}; }
  }];
}

def OneFlow_CtcLossGradOp : OneFlow_BaseOp<"ctc_loss_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"grad_out", "log_probs", "targets", "input_lengths", "target_lengths", "loss", "alpha"}; }
    static std::vector<std::string> outputOrder() { return {"grad"}; }
  }];
}

def OneFlow_DynamicLossScaleScheduleOp : OneFlow_BaseOp<"dynamic_loss_scale_schedule", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$count_not_finite,
    AnyType:$loss_scale,
    AnyType:$good_step_counter
  );
  let attrs = (ins
    SI64Attr:$increment_period,
    F32Attr:$multiplier
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"count_not_finite", "loss_scale", "good_step_counter"}; }
    static std::vector<std::string> outputOrder() { return {}; }
  }];
}

def OneFlow_KlDivLossOp : OneFlow_BaseOp<"kl_div_loss", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"input", "target"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_KlDivLossGradOp : OneFlow_BaseOp<"kl_div_loss_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"input", "target", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_SmoothL1LossOp : OneFlow_BaseOp<"smooth_l1_loss", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"input", "target"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_SmoothL1LossGradOp : OneFlow_BaseOp<"smooth_l1_loss_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"input", "target", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

#endif  // GET_ONEFLOW_LOSS_OP_DEFINITIONS

// abs_grad, ceil_grad, erf, erf_grad, exp, exp_grad, expand_grad, expm1, expm1_grad, floor_grad, floordiv_x_grad, floordiv_y_grad, lgamma, lgamma_grad, log, log1p, log1p_grad, log_grad, log_sigmoid, log_sigmoid_grad, negative_grad, reciprocal_grad, reciprocal_no_nan, reciprocal_no_nan_grad, rint_grad, round_grad, rsqrt, rsqrt_grad, sigmoid_v2, sigmoid_v2_grad, sign_grad, softplus, softplus_grad, softsign_grad, sqrt, sqrt_grad, square, square_grad, xlogy_x_grad, xlogy_y_grad
// Total: 40
#ifndef GET_ONEFLOW_MATH_OP_DEFINITIONS
#define GET_ONEFLOW_MATH_OP_DEFINITIONS

def OneFlow_AbsGradOp : OneFlow_BaseOp<"abs_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_CeilGradOp : OneFlow_BaseOp<"ceil_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_ErfOp : OneFlow_BaseOp<"erf", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_ErfGradOp : OneFlow_BaseOp<"erf_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_ExpOp : OneFlow_BaseOp<"exp", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_ExpGradOp : OneFlow_BaseOp<"exp_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_ExpandGradOp : OneFlow_BaseOp<"expand_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_Expm1Op : OneFlow_BaseOp<"expm1", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_Expm1GradOp : OneFlow_BaseOp<"expm1_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_FloorGradOp : OneFlow_BaseOp<"floor_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_FloordivXGradOp : OneFlow_BaseOp<"floordiv_x_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y,
    AnyType:$dz
  );
  let output = (outs
    AnyType:$dx
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "y", "dz"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_FloordivYGradOp : OneFlow_BaseOp<"floordiv_y_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y,
    AnyType:$dz
  );
  let output = (outs
    AnyType:$dy
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "y", "dz"}; }
    static std::vector<std::string> outputOrder() { return {"dy"}; }
  }];
}

def OneFlow_LgammaOp : OneFlow_BaseOp<"lgamma", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_LgammaGradOp : OneFlow_BaseOp<"lgamma_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_LogOp : OneFlow_BaseOp<"log", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_Log1pOp : OneFlow_BaseOp<"log1p", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_Log1pGradOp : OneFlow_BaseOp<"log1p_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_LogGradOp : OneFlow_BaseOp<"log_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_LogSigmoidOp : OneFlow_BaseOp<"log_sigmoid", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_LogSigmoidGradOp : OneFlow_BaseOp<"log_sigmoid_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_NegativeGradOp : OneFlow_BaseOp<"negative_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_ReciprocalGradOp : OneFlow_BaseOp<"reciprocal_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_ReciprocalNoNanOp : OneFlow_BaseOp<"reciprocal_no_nan", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_ReciprocalNoNanGradOp : OneFlow_BaseOp<"reciprocal_no_nan_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_RintGradOp : OneFlow_BaseOp<"rint_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_RoundGradOp : OneFlow_BaseOp<"round_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_RsqrtOp : OneFlow_BaseOp<"rsqrt", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_RsqrtGradOp : OneFlow_BaseOp<"rsqrt_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_SigmoidV2Op : OneFlow_BaseOp<"sigmoid_v2", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_SigmoidV2GradOp : OneFlow_BaseOp<"sigmoid_v2_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_SignGradOp : OneFlow_BaseOp<"sign_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_SoftplusOp : OneFlow_BaseOp<"softplus", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_SoftplusGradOp : OneFlow_BaseOp<"softplus_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_SoftsignGradOp : OneFlow_BaseOp<"softsign_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_SqrtOp : OneFlow_BaseOp<"sqrt", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_SqrtGradOp : OneFlow_BaseOp<"sqrt_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_SquareOp : OneFlow_BaseOp<"square", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_SquareGradOp : OneFlow_BaseOp<"square_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_XlogyXGradOp : OneFlow_BaseOp<"xlogy_x_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y,
    AnyType:$dz
  );
  let output = (outs
    AnyType:$dx
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "y", "dz"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_XlogyYGradOp : OneFlow_BaseOp<"xlogy_y_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y,
    AnyType:$dz
  );
  let output = (outs
    AnyType:$dy
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "y", "dz"}; }
    static std::vector<std::string> outputOrder() { return {"dy"}; }
  }];
}

#endif  // GET_ONEFLOW_MATH_OP_DEFINITIONS

// batch_matmul, broadcast_matmul, broadcast_matmul_grad_b, distributed_partial_fc_sample, distributed_partial_fc_sample_disable_boxing, erfc, erfc_grad, matmul
// Total: 8
#ifndef GET_ONEFLOW_MATMUL_OP_DEFINITIONS
#define GET_ONEFLOW_MATMUL_OP_DEFINITIONS

def OneFlow_BatchMatmulOp : OneFlow_BaseOp<"batch_matmul", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$a,
    AnyType:$b,
    Optional<AnyType>:$_add_to_output
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    BoolAttr:$transpose_a,
    BoolAttr:$transpose_b,
    F64Attr:$alpha
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"a", "b", "_add_to_output"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_BroadcastMatmulOp : OneFlow_BaseOp<"broadcast_matmul", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$a,
    AnyType:$b,
    Optional<AnyType>:$_add_to_output
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    BoolAttr:$transpose_a,
    BoolAttr:$transpose_b,
    F64Attr:$alpha
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"a", "b", "_add_to_output"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_BroadcastMatmulGradBOp : OneFlow_BaseOp<"broadcast_matmul_grad_b", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$a,
    AnyType:$b,
    Optional<AnyType>:$_add_to_output
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    F64Attr:$alpha
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"a", "b", "_add_to_output"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_DistributedPartialFcSampleOp : OneFlow_BaseOp<"distributed_partial_fc_sample", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"weight", "label"}; }
    static std::vector<std::string> outputOrder() { return {"mapped_label", "sampled_label", "sampled_weight"}; }
  }];
}

def OneFlow_DistributedPartialFcSampleDisableBoxingOp : OneFlow_BaseOp<"distributed_partial_fc_sample_disable_boxing", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$sampled_weight_diff,
    AnyType:$sampled_label
  );
  let output = (outs
    AnyType:$boxing_disabled_sampled_weight_diff,
    AnyType:$boxing_disabled_sampled_label
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"sampled_weight_diff", "sampled_label"}; }
    static std::vector<std::string> outputOrder() { return {"boxing_disabled_sampled_weight_diff", "boxing_disabled_sampled_label"}; }
  }];
}

def OneFlow_ErfcOp : OneFlow_BaseOp<"erfc", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_ErfcGradOp : OneFlow_BaseOp<"erfc_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_MatmulOp : OneFlow_BaseOp<"matmul", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$a,
    AnyType:$b,
    Optional<AnyType>:$_add_to_output
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    BoolAttr:$transpose_a,
    BoolAttr:$transpose_b,
    F64Attr:$alpha
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"a", "b", "_add_to_output"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

#endif  // GET_ONEFLOW_MATMUL_OP_DEFINITIONS

// CategoricalOrdinalEncode, acc, add_n, affine_grid, affine_grid_grad, arange, bernoulli, bias_add, cast, cast_like, cast_to_static_shape, cast_to_tick, celu, celu_grad, coin_flip, concat, constant, copy, count_not_finite, diag, diag_grad, dropout, dropout_grad, elementwise_maximum, elementwise_maximum_backward, elementwise_minimum, elementwise_minimum_backward, elu, elu_grad, empty, expand, expand_dims, eye, flatten, flip, flip_grad, floordiv, fold, gelu, gelu_grad, generate_random_batch_permutation_indices, grid_sample, grid_sample_grad, hardsigmoid, hardsigmoid_grad, hardswish, hardswish_grad, hierarchical_parallel_cast, hierarchical_parallel_cast_like, l1_l2_regularize_gradient, leaky_relu, leaky_relu_grad, logical_not, masked_fill, mish, mish_grad, multi_count_not_finite, multi_square_sum, multiply, narrow, narrow_grad, nll, nll_grad, nvtx_end, nvtx_start, one_hot, pack, parallel_cast, pow, pow_x_grad, pow_y_grad, prelu, prelu_grad, random_mask_like, randperm, recv, relu_grad, repeat, roll, selu, selu_grad, send, sigmoid, sigmoid_grad, silu, silu_grad, softsign, sort, split_like, square_sum, squeeze, ssp_variable_proxy, tf_prelu, tf_prelu_grad, transpose, tril, triu, unfold, unfold_tensor, unfold_tensor_grad, uniform, uniform_int, unique_with_counts, unpack, xdivy, xdivy_x_grad, xdivy_y_grad, xlogy, zero_like
// Total: 109
#ifndef GET_ONEFLOW_MISC_OP_DEFINITIONS
#define GET_ONEFLOW_MISC_OP_DEFINITIONS

def OneFlow_CategoricalOrdinalEncodeOp : OneFlow_BaseOp<"CategoricalOrdinalEncode", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"table", "size", "in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_AccOp : OneFlow_BaseOp<"acc", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI32Attr:$max_acc_num
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_AddNOp : OneFlow_BaseOp<"add_n", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    Variadic<AnyType>:$in
  );
  let output = (outs
    AnyType:$out
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
  let hasCanonicalizer = 1;
}

def OneFlow_AffineGridOp : OneFlow_BaseOp<"affine_grid", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"theta"}; }
    static std::vector<std::string> outputOrder() { return {"grid"}; }
  }];
}

def OneFlow_AffineGridGradOp : OneFlow_BaseOp<"affine_grid_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"dgrid"}; }
    static std::vector<std::string> outputOrder() { return {"dtheta"}; }
  }];
}

def OneFlow_ArangeOp : OneFlow_BaseOp<"arange", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_BernoulliOp : OneFlow_BaseOp<"bernoulli", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_BiasAddOp : OneFlow_BaseOp<"bias_add", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"a", "b"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_CastOp : OneFlow_BaseOp<"cast", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    StrAttr:$dtype
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_CastLikeOp : OneFlow_BaseOp<"cast_like", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in,
    AnyType:$dtype_like
  );
  let output = (outs
    AnyType:$out
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in", "dtype_like"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_CastToStaticShapeOp : OneFlow_BaseOp<"cast_to_static_shape", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$input
  );
  let output = (outs
    AnyType:$output
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"input"}; }
    static std::vector<std::string> outputOrder() { return {"output"}; }
  }];
}

def OneFlow_CastToTickOp : OneFlow_BaseOp<"cast_to_tick", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_CeluOp : OneFlow_BaseOp<"celu", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    F64Attr:$alpha
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_CeluGradOp : OneFlow_BaseOp<"celu_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_CoinFlipOp : OneFlow_BaseOp<"coin_flip", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_ConcatOp : OneFlow_BaseOp<"concat", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    Variadic<AnyType>:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI64Attr:$axis,
    SI64Attr:$max_dim_size
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_ConstantOp : OneFlow_BaseOp<"constant", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_CopyOp : OneFlow_BaseOp<"copy", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_CountNotFiniteOp : OneFlow_BaseOp<"count_not_finite", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_DiagOp : OneFlow_BaseOp<"diag", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI32Attr:$diagonal
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_DiagGradOp : OneFlow_BaseOp<"diag_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"dy", "in"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_DropoutOp : OneFlow_BaseOp<"dropout", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in,
    AnyType:$mask,
    Optional<AnyType>:$_add_to_output
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    F32Attr:$scale
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in", "mask", "_add_to_output"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_DropoutGradOp : OneFlow_BaseOp<"dropout_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"dy", "mask"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_ElementwiseMaximumOp : OneFlow_BaseOp<"elementwise_maximum", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "y"}; }
    static std::vector<std::string> outputOrder() { return {"z"}; }
  }];
}

def OneFlow_ElementwiseMaximumBackwardOp : OneFlow_BaseOp<"elementwise_maximum_backward", [AttrSizedResultSegments, DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$dz,
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    Optional<AnyType>:$dx,
    Optional<AnyType>:$dy
  );
  let trait_attrs = (ins
    I32ElementsAttr:$result_segment_sizes
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"dz", "x", "y"}; }
    static std::vector<std::string> outputOrder() { return {"dx", "dy"}; }
  }];
}

def OneFlow_ElementwiseMinimumOp : OneFlow_BaseOp<"elementwise_minimum", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "y"}; }
    static std::vector<std::string> outputOrder() { return {"z"}; }
  }];
}

def OneFlow_ElementwiseMinimumBackwardOp : OneFlow_BaseOp<"elementwise_minimum_backward", [AttrSizedResultSegments, DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$dz,
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    Optional<AnyType>:$dx,
    Optional<AnyType>:$dy
  );
  let trait_attrs = (ins
    I32ElementsAttr:$result_segment_sizes
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"dz", "x", "y"}; }
    static std::vector<std::string> outputOrder() { return {"dx", "dy"}; }
  }];
}

def OneFlow_EluOp : OneFlow_BaseOp<"elu", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    F64Attr:$alpha
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_EluGradOp : OneFlow_BaseOp<"elu_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_EmptyOp : OneFlow_BaseOp<"empty", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    StrAttr:$dtype,
    AnyI64ElementsAttr:$shape,
    StrArrayAttr:$nd_sbp
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_ExpandOp : OneFlow_BaseOp<"expand", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_ExpandDimsOp : OneFlow_BaseOp<"expand_dims", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI32Attr:$axis
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_EyeOp : OneFlow_BaseOp<"eye", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI64Attr:$rows,
    SI64Attr:$cols,
    StrAttr:$dtype,
    StrArrayAttr:$nd_sbp
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_FlattenOp : OneFlow_BaseOp<"flatten", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_FlipOp : OneFlow_BaseOp<"flip", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    SI32ArrayAttr:$dims
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_FlipGradOp : OneFlow_BaseOp<"flip_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    SI32ArrayAttr:$dims
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_FloordivOp : OneFlow_BaseOp<"floordiv", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "y"}; }
    static std::vector<std::string> outputOrder() { return {"z"}; }
  }];
}

def OneFlow_FoldOp : OneFlow_BaseOp<"fold", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_GeluOp : OneFlow_BaseOp<"gelu", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_GeluGradOp : OneFlow_BaseOp<"gelu_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_GenerateRandomBatchPermutationIndicesOp : OneFlow_BaseOp<"generate_random_batch_permutation_indices", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    SI64Attr:$seed
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_GridSampleOp : OneFlow_BaseOp<"grid_sample", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"input", "grid"}; }
    static std::vector<std::string> outputOrder() { return {"output"}; }
  }];
}

def OneFlow_GridSampleGradOp : OneFlow_BaseOp<"grid_sample_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"doutput", "input", "grid"}; }
    static std::vector<std::string> outputOrder() { return {"dinput", "dgrid"}; }
  }];
}

def OneFlow_HardsigmoidOp : OneFlow_BaseOp<"hardsigmoid", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_HardsigmoidGradOp : OneFlow_BaseOp<"hardsigmoid_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_HardswishOp : OneFlow_BaseOp<"hardswish", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_HardswishGradOp : OneFlow_BaseOp<"hardswish_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_HierarchicalParallelCastOp : OneFlow_BaseOp<"hierarchical_parallel_cast", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_HierarchicalParallelCastLikeOp : OneFlow_BaseOp<"hierarchical_parallel_cast_like", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in,
    AnyType:$like
  );
  let output = (outs
    AnyType:$out
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in", "like"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_L1L2RegularizeGradientOp : OneFlow_BaseOp<"l1_l2_regularize_gradient", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"model", "model_diff"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_LeakyReluOp : OneFlow_BaseOp<"leaky_relu", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    F32Attr:$alpha
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_LeakyReluGradOp : OneFlow_BaseOp<"leaky_relu_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_LogicalNotOp : OneFlow_BaseOp<"logical_not", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_MaskedFillOp : OneFlow_BaseOp<"masked_fill", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "mask"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_MishOp : OneFlow_BaseOp<"mish", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_MishGradOp : OneFlow_BaseOp<"mish_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_MultiCountNotFiniteOp : OneFlow_BaseOp<"multi_count_not_finite", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    Variadic<AnyType>:$x
  );
  let output = (outs
    AnyType:$y
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_MultiSquareSumOp : OneFlow_BaseOp<"multi_square_sum", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    Variadic<AnyType>:$x
  );
  let output = (outs
    AnyType:$y
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_MultiplyOp : OneFlow_BaseOp<"multiply", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$out
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "y"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_NarrowOp : OneFlow_BaseOp<"narrow", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_NarrowGradOp : OneFlow_BaseOp<"narrow_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"dy", "like"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_NllOp : OneFlow_BaseOp<"nll", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$input,
    AnyType:$target,
    Optional<AnyType>:$weight
  );
  let output = (outs
    AnyType:$out,
    AnyType:$total_weight
  );
  let attrs = (ins
    SI64Attr:$ignore_index,
    StrAttr:$reduction
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"input", "target", "weight"}; }
    static std::vector<std::string> outputOrder() { return {"out", "total_weight"}; }
  }];
}

def OneFlow_NllGradOp : OneFlow_BaseOp<"nll_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$input,
    AnyType:$target,
    AnyType:$total_weight,
    Optional<AnyType>:$weight,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    SI64Attr:$ignore_index,
    StrAttr:$reduction
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"input", "target", "total_weight", "weight", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_NvtxEndOp : OneFlow_BaseOp<"nvtx_end", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    StrAttr:$mark_prefix
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_NvtxStartOp : OneFlow_BaseOp<"nvtx_start", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    StrAttr:$mark_prefix
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_OneHotOp : OneFlow_BaseOp<"one_hot", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"indices"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_PackOp : OneFlow_BaseOp<"pack", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI32Attr:$pack_num
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_ParallelCastOp : OneFlow_BaseOp<"parallel_cast", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_PowOp : OneFlow_BaseOp<"pow", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "y"}; }
    static std::vector<std::string> outputOrder() { return {"z"}; }
  }];
}

def OneFlow_PowXGradOp : OneFlow_BaseOp<"pow_x_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y,
    AnyType:$dz
  );
  let output = (outs
    AnyType:$dx
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "y", "dz"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_PowYGradOp : OneFlow_BaseOp<"pow_y_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y,
    AnyType:$dz
  );
  let output = (outs
    AnyType:$dy
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "y", "dz"}; }
    static std::vector<std::string> outputOrder() { return {"dy"}; }
  }];
}

def OneFlow_PreluOp : OneFlow_BaseOp<"prelu", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$alpha
  );
  let output = (outs
    AnyType:$y
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "alpha"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_PreluGradOp : OneFlow_BaseOp<"prelu_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$dy,
    AnyType:$x,
    AnyType:$alpha
  );
  let output = (outs
    AnyType:$dx,
    AnyType:$alpha_diff
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"dy", "x", "alpha"}; }
    static std::vector<std::string> outputOrder() { return {"dx", "alpha_diff"}; }
  }];
}

def OneFlow_RandomMaskLikeOp : OneFlow_BaseOp<"random_mask_like", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"like"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_RandpermOp : OneFlow_BaseOp<"randperm", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI32Attr:$n,
    StrAttr:$nd_sbp
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_RecvOp : OneFlow_BaseOp<"recv", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_ReluGradOp : OneFlow_BaseOp<"relu_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$y,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"y", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_RepeatOp : OneFlow_BaseOp<"repeat", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI32Attr:$repeat_num
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_RollOp : OneFlow_BaseOp<"roll", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_SeluOp : OneFlow_BaseOp<"selu", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_SeluGradOp : OneFlow_BaseOp<"selu_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_SendOp : OneFlow_BaseOp<"send", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let attrs = (ins
    SI64Attr:$dst_process_id
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {}; }
  }];
}

def OneFlow_SigmoidOp : OneFlow_BaseOp<"sigmoid", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_SigmoidGradOp : OneFlow_BaseOp<"sigmoid_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$y,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"y", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_SiluOp : OneFlow_BaseOp<"silu", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_SiluGradOp : OneFlow_BaseOp<"silu_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_SoftsignOp : OneFlow_BaseOp<"softsign", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_SortOp : OneFlow_BaseOp<"sort", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    StrAttr:$direction
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_SplitLikeOp : OneFlow_BaseOp<"split_like", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in,
    Variadic<AnyType>:$like
  );
  let output = (outs
    Variadic<AnyType>:$out
  );
  let attrs = (ins
    SI64Attr:$axis
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in", "like"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_SquareSumOp : OneFlow_BaseOp<"square_sum", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_SqueezeOp : OneFlow_BaseOp<"squeeze", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI32ArrayAttr:$axes
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_SspVariableProxyOp : OneFlow_BaseOp<"ssp_variable_proxy", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"var"}; }
    static std::vector<std::string> outputOrder() { return {"ref", "value"}; }
  }];
}

def OneFlow_TfPreluOp : OneFlow_BaseOp<"tf_prelu", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$alpha
  );
  let output = (outs
    AnyType:$y
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "alpha"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_TfPreluGradOp : OneFlow_BaseOp<"tf_prelu_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$dy,
    AnyType:$x,
    AnyType:$alpha
  );
  let output = (outs
    AnyType:$dx,
    AnyType:$alpha_diff
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"dy", "x", "alpha"}; }
    static std::vector<std::string> outputOrder() { return {"dx", "alpha_diff"}; }
  }];
}

def OneFlow_TransposeOp : OneFlow_BaseOp<"transpose", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$input
  );
  let output = (outs
    AnyType:$output
  );
  let attrs = (ins
    SI32ArrayAttr:$perm
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"input"}; }
    static std::vector<std::string> outputOrder() { return {"output"}; }
  }];
}

def OneFlow_TrilOp : OneFlow_BaseOp<"tril", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_TriuOp : OneFlow_BaseOp<"triu", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI64Attr:$diagonal
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_UnfoldOp : OneFlow_BaseOp<"unfold", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_UnfoldTensorOp : OneFlow_BaseOp<"unfold_tensor", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_UnfoldTensorGradOp : OneFlow_BaseOp<"unfold_tensor_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"dy", "x"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_UniformOp : OneFlow_BaseOp<"uniform", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_UniformIntOp : OneFlow_BaseOp<"uniform_int", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_UniqueWithCountsOp : OneFlow_BaseOp<"unique_with_counts", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y", "idx", "count", "num_unique"}; }
  }];
}

def OneFlow_UnpackOp : OneFlow_BaseOp<"unpack", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI32Attr:$unpack_num
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_XdivyOp : OneFlow_BaseOp<"xdivy", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "y"}; }
    static std::vector<std::string> outputOrder() { return {"z"}; }
  }];
}

def OneFlow_XdivyXGradOp : OneFlow_BaseOp<"xdivy_x_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y,
    AnyType:$dz
  );
  let output = (outs
    AnyType:$dx
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "y", "dz"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_XdivyYGradOp : OneFlow_BaseOp<"xdivy_y_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y,
    AnyType:$dz
  );
  let output = (outs
    AnyType:$dy
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "y", "dz"}; }
    static std::vector<std::string> outputOrder() { return {"dy"}; }
  }];
}

def OneFlow_XlogyOp : OneFlow_BaseOp<"xlogy", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "y"}; }
    static std::vector<std::string> outputOrder() { return {"z"}; }
  }];
}

def OneFlow_ZeroLikeOp : OneFlow_BaseOp<"zero_like", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$like
  );
  let output = (outs
    AnyType:$out
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"like"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

#endif  // GET_ONEFLOW_MISC_OP_DEFINITIONS

// _nccl_logical_2D_same_dim0_all2all, _nccl_logical_2D_same_dim0_all_gather, _nccl_logical_2D_same_dim0_all_gather_noncontinuous, _nccl_logical_2D_same_dim0_all_reduce, _nccl_logical_2D_same_dim1_all_reduce, _nccl_logical_all_gather, _nccl_logical_all_gather_noncontinuous, _nccl_logical_all_reduce, _nccl_logical_reduce_scatter, _nccl_logical_s2s
// Total: 10
#ifndef GET_ONEFLOW_NCCL_OP_DEFINITIONS
#define GET_ONEFLOW_NCCL_OP_DEFINITIONS

def OneFlow__ncclLogical_2DSameDim0All2allOp : OneFlow_BaseOp<"_nccl_logical_2D_same_dim0_all2all", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow__ncclLogical_2DSameDim0AllGatherOp : OneFlow_BaseOp<"_nccl_logical_2D_same_dim0_all_gather", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow__ncclLogical_2DSameDim0AllGatherNoncontinuousOp : OneFlow_BaseOp<"_nccl_logical_2D_same_dim0_all_gather_noncontinuous", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI64Attr:$in_dim1_split_axis
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow__ncclLogical_2DSameDim0AllReduceOp : OneFlow_BaseOp<"_nccl_logical_2D_same_dim0_all_reduce", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow__ncclLogical_2DSameDim1AllReduceOp : OneFlow_BaseOp<"_nccl_logical_2D_same_dim1_all_reduce", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow__ncclLogicalAllGatherOp : OneFlow_BaseOp<"_nccl_logical_all_gather", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow__ncclLogicalAllGatherNoncontinuousOp : OneFlow_BaseOp<"_nccl_logical_all_gather_noncontinuous", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI64Attr:$in_split_axis
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow__ncclLogicalAllReduceOp : OneFlow_BaseOp<"_nccl_logical_all_reduce", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow__ncclLogicalReduceScatterOp : OneFlow_BaseOp<"_nccl_logical_reduce_scatter", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow__ncclLogicalS2sOp : OneFlow_BaseOp<"_nccl_logical_s2s", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

#endif  // GET_ONEFLOW_NCCL_OP_DEFINITIONS

// crop_mirror_normalize_from_tensorbuffer, crop_mirror_normalize_from_uint8, image_normalize, l2_normalize, l2_normalize_grad, layer_norm, layer_norm_grad, layer_norm_param_grad, normal, normalization, normalization_grad
// Total: 11
#ifndef GET_ONEFLOW_NORMALIZATION_OP_DEFINITIONS
#define GET_ONEFLOW_NORMALIZATION_OP_DEFINITIONS

def OneFlow_CropMirrorNormalizeFromTensorbufferOp : OneFlow_BaseOp<"crop_mirror_normalize_from_tensorbuffer", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in,
    Optional<AnyType>:$mirror
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in", "mirror"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_CropMirrorNormalizeFromUint8Op : OneFlow_BaseOp<"crop_mirror_normalize_from_uint8", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in,
    Optional<AnyType>:$mirror
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in", "mirror"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_ImageNormalizeOp : OneFlow_BaseOp<"image_normalize", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_L2NormalizeOp : OneFlow_BaseOp<"l2_normalize", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y", "square_x_sum"}; }
  }];
}

def OneFlow_L2NormalizeGradOp : OneFlow_BaseOp<"l2_normalize_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"dy", "y", "square_x_sum"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_LayerNormOp : OneFlow_BaseOp<"layer_norm", [AttrSizedOperandSegments, DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    Optional<AnyType>:$beta,
    Optional<AnyType>:$gamma
  );
  let output = (outs
    AnyType:$y,
    AnyType:$mean,
    AnyType:$inv_variance,
    Optional<AnyType>:$normalized
  );
  let attrs = (ins
    BoolAttr:$center,
    BoolAttr:$scale,
    SI64Attr:$begin_norm_axis,
    SI64Attr:$begin_params_axis,
    F64Attr:$epsilon
  );
  let trait_attrs = (ins
    I32ElementsAttr:$operand_segment_sizes
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "beta", "gamma"}; }
    static std::vector<std::string> outputOrder() { return {"y", "mean", "inv_variance", "normalized"}; }
  }];
}

def OneFlow_LayerNormGradOp : OneFlow_BaseOp<"layer_norm_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$dy,
    AnyType:$x,
    AnyType:$mean,
    AnyType:$inv_variance,
    Optional<AnyType>:$_add_to_output
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    SI64Attr:$begin_norm_axis,
    F64Attr:$epsilon
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"dy", "x", "mean", "inv_variance", "_add_to_output"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_LayerNormParamGradOp : OneFlow_BaseOp<"layer_norm_param_grad", [AttrSizedOperandSegments, AttrSizedResultSegments, DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$dy,
    Optional<AnyType>:$normalized,
    Optional<AnyType>:$gamma
  );
  let output = (outs
    Optional<AnyType>:$normalized_diff,
    Optional<AnyType>:$beta_diff,
    Optional<AnyType>:$gamma_diff,
    Optional<AnyType>:$reduce_buf
  );
  let attrs = (ins
    SI64Attr:$begin_params_axis
  );
  let trait_attrs = (ins
    I32ElementsAttr:$operand_segment_sizes,
    I32ElementsAttr:$result_segment_sizes
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"dy", "normalized", "gamma"}; }
    static std::vector<std::string> outputOrder() { return {"normalized_diff", "beta_diff", "gamma_diff", "reduce_buf"}; }
  }];
}

def OneFlow_NormalOp : OneFlow_BaseOp<"normal", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_NormalizationOp : OneFlow_BaseOp<"normalization", [AttrSizedOperandSegments, AttrSizedResultSegments, DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    Optional<AnyType>:$moving_mean,
    Optional<AnyType>:$moving_variance,
    AnyType:$gamma,
    AnyType:$beta,
    Optional<AnyType>:$_add_to_output
  );
  let output = (outs
    AnyType:$y,
    Optional<AnyType>:$mean,
    Optional<AnyType>:$inv_variance
  );
  let attrs = (ins
    SI32Attr:$axis,
    F32Attr:$epsilon,
    BoolAttr:$training,
    F32Attr:$momentum
  );
  let trait_attrs = (ins
    I32ElementsAttr:$operand_segment_sizes,
    I32ElementsAttr:$result_segment_sizes
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "moving_mean", "moving_variance", "gamma", "beta", "_add_to_output"}; }
    static std::vector<std::string> outputOrder() { return {"y", "mean", "inv_variance"}; }
  }];
}

def OneFlow_NormalizationGradOp : OneFlow_BaseOp<"normalization_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "dy", "mean", "inv_variance", "gamma"}; }
    static std::vector<std::string> outputOrder() { return {"gamma_diff", "beta_diff", "dx"}; }
  }];
}

#endif  // GET_ONEFLOW_NORMALIZATION_OP_DEFINITIONS

// adagrad_update, adam_bias_correction_factor, adam_update, indexed_slices_adam_update, indexed_slices_momentum_update, indexed_slices_sgd_update, lamb_update, lars_update, momentum_update, rmsprop_update, sgd_update, slice_update
// Total: 12
#ifndef GET_ONEFLOW_OPTIMIZER_OP_DEFINITIONS
#define GET_ONEFLOW_OPTIMIZER_OP_DEFINITIONS

def OneFlow_AdagradUpdateOp : OneFlow_BaseOp<"adagrad_update", [AttrSizedOperandSegments, DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$model,
    AnyType:$model_diff,
    Optional<AnyType>:$learning_rate,
    Optional<AnyType>:$scale_by_tensor,
    Optional<AnyType>:$skip_if,
    Optional<AnyType>:$train_step,
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
  let trait_attrs = (ins
    I32ElementsAttr:$operand_segment_sizes
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"model", "model_diff", "learning_rate", "scale_by_tensor", "skip_if", "train_step", "sum"}; }
    static std::vector<std::string> outputOrder() { return {}; }
  }];
}

def OneFlow_AdamBiasCorrectionFactorOp : OneFlow_BaseOp<"adam_bias_correction_factor", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$train_step
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    F32Attr:$beta
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"train_step"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_AdamUpdateOp : OneFlow_BaseOp<"adam_update", [AttrSizedOperandSegments, DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$model,
    AnyType:$model_diff,
    Optional<AnyType>:$learning_rate,
    Optional<AnyType>:$scale_by_tensor,
    Optional<AnyType>:$skip_if,
    Optional<AnyType>:$bias_correction1,
    Optional<AnyType>:$bias_correction2,
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
  let trait_attrs = (ins
    I32ElementsAttr:$operand_segment_sizes
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"model", "model_diff", "learning_rate", "scale_by_tensor", "skip_if", "bias_correction1", "bias_correction2", "m", "v", "max_v"}; }
    static std::vector<std::string> outputOrder() { return {}; }
  }];
}

def OneFlow_IndexedSlicesAdamUpdateOp : OneFlow_BaseOp<"indexed_slices_adam_update", [AttrSizedOperandSegments, DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$model,
    AnyType:$model_diff_indices,
    AnyType:$model_diff_values,
    AnyType:$learning_rate,
    Optional<AnyType>:$bias_correction1,
    Optional<AnyType>:$bias_correction2,
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
  let trait_attrs = (ins
    I32ElementsAttr:$operand_segment_sizes
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"model", "model_diff_indices", "model_diff_values", "learning_rate", "bias_correction1", "bias_correction2", "m", "v", "max_v"}; }
    static std::vector<std::string> outputOrder() { return {}; }
  }];
}

def OneFlow_IndexedSlicesMomentumUpdateOp : OneFlow_BaseOp<"indexed_slices_momentum_update", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"model", "model_diff_indices", "model_diff_values", "learning_rate", "momentum"}; }
    static std::vector<std::string> outputOrder() { return {}; }
  }];
}

def OneFlow_IndexedSlicesSgdUpdateOp : OneFlow_BaseOp<"indexed_slices_sgd_update", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$model,
    AnyType:$model_diff_indices,
    AnyType:$model_diff_values,
    AnyType:$learning_rate
  );
  let attrs = (ins
    F32Attr:$weight_decay
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"model", "model_diff_indices", "model_diff_values", "learning_rate"}; }
    static std::vector<std::string> outputOrder() { return {}; }
  }];
}

def OneFlow_LambUpdateOp : OneFlow_BaseOp<"lamb_update", [AttrSizedOperandSegments, DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$m,
    AnyType:$v,
    AnyType:$beta1_t,
    AnyType:$beta2_t,
    AnyType:$model,
    AnyType:$model_diff,
    AnyType:$learning_rate,
    Optional<AnyType>:$scale_by_tensor,
    Optional<AnyType>:$skip_if
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
  let trait_attrs = (ins
    I32ElementsAttr:$operand_segment_sizes
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"m", "v", "beta1_t", "beta2_t", "model", "model_diff", "learning_rate", "scale_by_tensor", "skip_if"}; }
    static std::vector<std::string> outputOrder() { return {}; }
  }];
}

def OneFlow_LarsUpdateOp : OneFlow_BaseOp<"lars_update", [AttrSizedOperandSegments, DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$model,
    AnyType:$model_diff,
    AnyType:$learning_rate,
    AnyType:$momentum,
    Optional<AnyType>:$scale_by_tensor,
    Optional<AnyType>:$skip_if
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
  let trait_attrs = (ins
    I32ElementsAttr:$operand_segment_sizes
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"model", "model_diff", "learning_rate", "momentum", "scale_by_tensor", "skip_if"}; }
    static std::vector<std::string> outputOrder() { return {}; }
  }];
}

def OneFlow_MomentumUpdateOp : OneFlow_BaseOp<"momentum_update", [AttrSizedOperandSegments, DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$model,
    AnyType:$model_diff,
    AnyType:$momentum,
    Optional<AnyType>:$learning_rate,
    Optional<AnyType>:$scale_by_tensor,
    Optional<AnyType>:$skip_if
  );
  let attrs = (ins
    F32Attr:$learning_rate_val,
    F64Attr:$scale,
    F32Attr:$l1,
    F32Attr:$l2,
    F32Attr:$beta,
    F32Attr:$weight_decay
  );
  let trait_attrs = (ins
    I32ElementsAttr:$operand_segment_sizes
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"model", "model_diff", "momentum", "learning_rate", "scale_by_tensor", "skip_if"}; }
    static std::vector<std::string> outputOrder() { return {}; }
  }];
}

def OneFlow_RmspropUpdateOp : OneFlow_BaseOp<"rmsprop_update", [AttrSizedOperandSegments, DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$model,
    AnyType:$model_diff,
    Optional<AnyType>:$learning_rate,
    Optional<AnyType>:$scale_by_tensor,
    Optional<AnyType>:$skip_if,
    AnyType:$mean_square,
    Optional<AnyType>:$mean_gradient
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
  let trait_attrs = (ins
    I32ElementsAttr:$operand_segment_sizes
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"model", "model_diff", "learning_rate", "scale_by_tensor", "skip_if", "mean_square", "mean_gradient"}; }
    static std::vector<std::string> outputOrder() { return {}; }
  }];
}

def OneFlow_SgdUpdateOp : OneFlow_BaseOp<"sgd_update", [AttrSizedOperandSegments, DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"model", "model_diff", "learning_rate", "scale_by_tensor", "skip_if"}; }
    static std::vector<std::string> outputOrder() { return {}; }
  }];
}

def OneFlow_SliceUpdateOp : OneFlow_BaseOp<"slice_update", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "update"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

#endif  // GET_ONEFLOW_OPTIMIZER_OP_DEFINITIONS

// constant_pad1d, constant_pad1d_grad, constant_pad2d, constant_pad2d_grad, constant_pad3d, constant_pad3d_grad, pad, pad_grad, reflection_pad2d, reflection_pad2d_grad, replication_pad2d, replication_pad2d_grad, same_padding, same_padding_grad
// Total: 14
#ifndef GET_ONEFLOW_PADDING_OP_DEFINITIONS
#define GET_ONEFLOW_PADDING_OP_DEFINITIONS

def OneFlow_ConstantPad1DOp : OneFlow_BaseOp<"constant_pad1d", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_ConstantPad1DGradOp : OneFlow_BaseOp<"constant_pad1d_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_ConstantPad2DOp : OneFlow_BaseOp<"constant_pad2d", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_ConstantPad2DGradOp : OneFlow_BaseOp<"constant_pad2d_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_ConstantPad3DOp : OneFlow_BaseOp<"constant_pad3d", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_ConstantPad3DGradOp : OneFlow_BaseOp<"constant_pad3d_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_PadOp : OneFlow_BaseOp<"pad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_PadGradOp : OneFlow_BaseOp<"pad_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_ReflectionPad2DOp : OneFlow_BaseOp<"reflection_pad2d", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    SI64ArrayAttr:$padding
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_ReflectionPad2DGradOp : OneFlow_BaseOp<"reflection_pad2d_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    SI64ArrayAttr:$padding
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_ReplicationPad2DOp : OneFlow_BaseOp<"replication_pad2d", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    SI64ArrayAttr:$padding
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_ReplicationPad2DGradOp : OneFlow_BaseOp<"replication_pad2d_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    SI64ArrayAttr:$padding
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_SamePaddingOp : OneFlow_BaseOp<"same_padding", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_SamePaddingGradOp : OneFlow_BaseOp<"same_padding_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x_like", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

#endif  // GET_ONEFLOW_PADDING_OP_DEFINITIONS

// adaptive_avg_pool1d, adaptive_avg_pool1d_grad, adaptive_avg_pool2d, adaptive_avg_pool2d_grad, adaptive_avg_pool3d, adaptive_avg_pool3d_grad, avg_pool_1d, avg_pool_1d_grad, avg_pool_2d, avg_pool_2d_grad, avg_pool_3d, avg_pool_3d_grad, avgpool_1d, avgpool_1d_grad, avgpool_2d, avgpool_2d_grad, avgpool_3d, avgpool_3d_grad, max_pool_1d, max_pool_1d_grad, max_pool_2d, max_pool_2d_grad, max_pool_3d, max_pool_3d_grad, maxpool_1d, maxpool_1d_grad, maxpool_2d, maxpool_2d_grad, maxpool_3d, maxpool_3d_grad
// Total: 30
#ifndef GET_ONEFLOW_POOL_OP_DEFINITIONS
#define GET_ONEFLOW_POOL_OP_DEFINITIONS

def OneFlow_AdaptiveAvgPool1DOp : OneFlow_BaseOp<"adaptive_avg_pool1d", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    SI64ArrayAttr:$output_size
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_AdaptiveAvgPool1DGradOp : OneFlow_BaseOp<"adaptive_avg_pool1d_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_AdaptiveAvgPool2DOp : OneFlow_BaseOp<"adaptive_avg_pool2d", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    SI64ArrayAttr:$output_size
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_AdaptiveAvgPool2DGradOp : OneFlow_BaseOp<"adaptive_avg_pool2d_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_AdaptiveAvgPool3DOp : OneFlow_BaseOp<"adaptive_avg_pool3d", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    SI64ArrayAttr:$output_size
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_AdaptiveAvgPool3DGradOp : OneFlow_BaseOp<"adaptive_avg_pool3d_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_LazyAvgPool1DOp : OneFlow_LazyPoolBaseOp<"avg_pool_1d", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_LazyAvgPool1DGradOp : OneFlow_LazyPoolGradBaseOp<"avg_pool_1d_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "y", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_LazyAvgPool2DOp : OneFlow_LazyPoolBaseOp<"avg_pool_2d", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_LazyAvgPool2DGradOp : OneFlow_LazyPoolGradBaseOp<"avg_pool_2d_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "y", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_LazyAvgPool3DOp : OneFlow_LazyPoolBaseOp<"avg_pool_3d", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_LazyAvgPool3DGradOp : OneFlow_LazyPoolGradBaseOp<"avg_pool_3d_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "y", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_EagerAvgPool1DOp : OneFlow_EagerPoolBaseOp<"avgpool_1d", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_EagerAvgPool1DGradOp : OneFlow_EagerPoolGradBaseOp<"avgpool_1d_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "y", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_EagerAvgPool2DOp : OneFlow_EagerPoolBaseOp<"avgpool_2d", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_EagerAvgPool2DGradOp : OneFlow_EagerPoolGradBaseOp<"avgpool_2d_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "y", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_EagerAvgPool3DOp : OneFlow_EagerPoolBaseOp<"avgpool_3d", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_EagerAvgPool3DGradOp : OneFlow_EagerPoolGradBaseOp<"avgpool_3d_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "y", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_LazyMaxPool1DOp : OneFlow_LazyPoolBaseOp<"max_pool_1d", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_LazyMaxPool1DGradOp : OneFlow_LazyPoolGradBaseOp<"max_pool_1d_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "y", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_LazyMaxPool2DOp : OneFlow_LazyPoolBaseOp<"max_pool_2d", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_LazyMaxPool2DGradOp : OneFlow_LazyPoolGradBaseOp<"max_pool_2d_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "y", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_LazyMaxPool3DOp : OneFlow_LazyPoolBaseOp<"max_pool_3d", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_LazyMaxPool3DGradOp : OneFlow_LazyPoolGradBaseOp<"max_pool_3d_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "y", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_EagerMaxPool1DOp : OneFlow_EagerPoolBaseOp<"maxpool_1d", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y", "indice"}; }
  }];
}

def OneFlow_EagerMaxPool1DGradOp : OneFlow_EagerPoolGradBaseOp<"maxpool_1d_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "y", "indice", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_EagerMaxPool2DOp : OneFlow_EagerPoolBaseOp<"maxpool_2d", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y", "indice"}; }
  }];
}

def OneFlow_EagerMaxPool2DGradOp : OneFlow_EagerPoolGradBaseOp<"maxpool_2d_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "y", "indice", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_EagerMaxPool3DOp : OneFlow_EagerPoolBaseOp<"maxpool_3d", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y", "indice"}; }
  }];
}

def OneFlow_EagerMaxPool3DGradOp : OneFlow_EagerPoolGradBaseOp<"maxpool_3d_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "y", "indice", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

#endif  // GET_ONEFLOW_POOL_OP_DEFINITIONS

// fake_quantization, min_max_observer, moving_average_min_max_observer, quantization
// Total: 4
#ifndef GET_ONEFLOW_QUANTIZATION_OP_DEFINITIONS
#define GET_ONEFLOW_QUANTIZATION_OP_DEFINITIONS

def OneFlow_FakeQuantizationOp : OneFlow_BaseOp<"fake_quantization", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in", "scale", "zero_point"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_MinMaxObserverOp : OneFlow_BaseOp<"min_max_observer", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"scale", "zero_point"}; }
  }];
}

def OneFlow_MovingAverageMinMaxObserverOp : OneFlow_BaseOp<"moving_average_min_max_observer", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in", "current_train_step", "moving_max", "moving_min"}; }
    static std::vector<std::string> outputOrder() { return {"scale", "zero_point"}; }
  }];
}

def OneFlow_QuantizationOp : OneFlow_BaseOp<"quantization", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in", "scale", "zero_point"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

#endif  // GET_ONEFLOW_QUANTIZATION_OP_DEFINITIONS

// indexed_slices_reduce_sum, reduce_all, reduce_any, reduce_max, reduce_max_device_stage, reduce_max_device_stage_grad, reduce_max_global_stage, reduce_max_global_stage_grad, reduce_min, reduce_min_device_stage, reduce_min_device_stage_grad, reduce_min_global_stage, reduce_min_global_stage_grad, reduce_prod, reduce_sum, reduce_sum_like
// Total: 16
#ifndef GET_ONEFLOW_REDUCE_OP_DEFINITIONS
#define GET_ONEFLOW_REDUCE_OP_DEFINITIONS

def OneFlow_IndexedSlicesReduceSumOp : OneFlow_BaseOp<"indexed_slices_reduce_sum", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x_indices,
    AnyType:$x_values
  );
  let output = (outs
    AnyType:$y_indices,
    AnyType:$y_values,
    AnyType:$num_unique
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x_indices", "x_values"}; }
    static std::vector<std::string> outputOrder() { return {"y_indices", "y_values", "num_unique"}; }
  }];
}

def OneFlow_ReduceAllOp : OneFlow_BaseOp<"reduce_all", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"input_tensor"}; }
    static std::vector<std::string> outputOrder() { return {"output_tensor"}; }
  }];
}

def OneFlow_ReduceAnyOp : OneFlow_BaseOp<"reduce_any", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"input_tensor"}; }
    static std::vector<std::string> outputOrder() { return {"output_tensor"}; }
  }];
}

def OneFlow_ReduceMaxOp : OneFlow_BaseOp<"reduce_max", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"input_tensor"}; }
    static std::vector<std::string> outputOrder() { return {"output_tensor"}; }
  }];
}

def OneFlow_ReduceMaxDeviceStageOp : OneFlow_BaseOp<"reduce_max_device_stage", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out", "mask", "count"}; }
  }];
}

def OneFlow_ReduceMaxDeviceStageGradOp : OneFlow_BaseOp<"reduce_max_device_stage_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"out_diff", "mask", "count"}; }
    static std::vector<std::string> outputOrder() { return {"in_diff"}; }
  }];
}

def OneFlow_ReduceMaxGlobalStageOp : OneFlow_BaseOp<"reduce_max_global_stage", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in", "device_count"}; }
    static std::vector<std::string> outputOrder() { return {"out", "mask"}; }
  }];
}

def OneFlow_ReduceMaxGlobalStageGradOp : OneFlow_BaseOp<"reduce_max_global_stage_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"out_diff", "mask", "device_count"}; }
    static std::vector<std::string> outputOrder() { return {"in_diff"}; }
  }];
}

def OneFlow_ReduceMinOp : OneFlow_BaseOp<"reduce_min", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"input_tensor"}; }
    static std::vector<std::string> outputOrder() { return {"output_tensor"}; }
  }];
}

def OneFlow_ReduceMinDeviceStageOp : OneFlow_BaseOp<"reduce_min_device_stage", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out", "mask", "count"}; }
  }];
}

def OneFlow_ReduceMinDeviceStageGradOp : OneFlow_BaseOp<"reduce_min_device_stage_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"out_diff", "mask", "count"}; }
    static std::vector<std::string> outputOrder() { return {"in_diff"}; }
  }];
}

def OneFlow_ReduceMinGlobalStageOp : OneFlow_BaseOp<"reduce_min_global_stage", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in", "device_count"}; }
    static std::vector<std::string> outputOrder() { return {"out", "mask"}; }
  }];
}

def OneFlow_ReduceMinGlobalStageGradOp : OneFlow_BaseOp<"reduce_min_global_stage_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"out_diff", "mask", "device_count"}; }
    static std::vector<std::string> outputOrder() { return {"in_diff"}; }
  }];
}

def OneFlow_ReduceProdOp : OneFlow_BaseOp<"reduce_prod", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"input_tensor"}; }
    static std::vector<std::string> outputOrder() { return {"output_tensor"}; }
  }];
}

def OneFlow_ReduceSumOp : OneFlow_BaseOp<"reduce_sum", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"input_tensor"}; }
    static std::vector<std::string> outputOrder() { return {"output_tensor"}; }
  }];
}

def OneFlow_ReduceSumLikeOp : OneFlow_BaseOp<"reduce_sum_like", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "like"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

#endif  // GET_ONEFLOW_REDUCE_OP_DEFINITIONS

// reshape, reshape_like
// Total: 2
#ifndef GET_ONEFLOW_RESHAPE_OP_DEFINITIONS
#define GET_ONEFLOW_RESHAPE_OP_DEFINITIONS

def OneFlow_ReshapeOp : OneFlow_BaseOp<"reshape", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    AnyI64ElementsAttr:$shape
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_ReshapeLikeOp : OneFlow_BaseOp<"reshape_like", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in,
    AnyType:$like
  );
  let output = (outs
    AnyType:$out
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in", "like"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

#endif  // GET_ONEFLOW_RESHAPE_OP_DEFINITIONS

// clip_by_scalar, clip_by_scalar_grad, clip_by_scalar_max, clip_by_scalar_max_grad, clip_by_scalar_min, clip_by_scalar_min_grad, scalar_add, scalar_add_by_tensor, scalar_div_by_tensor, scalar_floordiv, scalar_fmod, scalar_logical_and, scalar_logical_equal, scalar_logical_greater, scalar_logical_greater_equal, scalar_logical_less, scalar_logical_less_equal, scalar_logical_not_equal, scalar_logical_or, scalar_logical_xor, scalar_mul, scalar_mul_by_tensor, scalar_pow, scalar_pow_grad, scalar_sub_by_tensor
// Total: 25
#ifndef GET_ONEFLOW_SCALAR_OP_DEFINITIONS
#define GET_ONEFLOW_SCALAR_OP_DEFINITIONS

def OneFlow_ClipByScalarOp : OneFlow_BaseOp<"clip_by_scalar", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_ClipByScalarGradOp : OneFlow_BaseOp<"clip_by_scalar_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"dy", "x"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_ClipByScalarMaxOp : OneFlow_BaseOp<"clip_by_scalar_max", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_ClipByScalarMaxGradOp : OneFlow_BaseOp<"clip_by_scalar_max_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"dy", "x"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_ClipByScalarMinOp : OneFlow_BaseOp<"clip_by_scalar_min", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_ClipByScalarMinGradOp : OneFlow_BaseOp<"clip_by_scalar_min_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"dy", "x"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_ScalarAddOp : OneFlow_BaseOp<"scalar_add", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_ScalarAddByTensorOp : OneFlow_BaseOp<"scalar_add_by_tensor", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$scalar
  );
  let output = (outs
    AnyType:$y
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "scalar"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_ScalarDivByTensorOp : OneFlow_BaseOp<"scalar_div_by_tensor", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$scalar
  );
  let output = (outs
    AnyType:$y
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "scalar"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_ScalarFloordivOp : OneFlow_BaseOp<"scalar_floordiv", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_ScalarFmodOp : OneFlow_BaseOp<"scalar_fmod", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_ScalarLogicalAndOp : OneFlow_BaseOp<"scalar_logical_and", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_ScalarLogicalEqualOp : OneFlow_BaseOp<"scalar_logical_equal", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_ScalarLogicalGreaterOp : OneFlow_BaseOp<"scalar_logical_greater", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_ScalarLogicalGreaterEqualOp : OneFlow_BaseOp<"scalar_logical_greater_equal", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_ScalarLogicalLessOp : OneFlow_BaseOp<"scalar_logical_less", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_ScalarLogicalLessEqualOp : OneFlow_BaseOp<"scalar_logical_less_equal", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_ScalarLogicalNotEqualOp : OneFlow_BaseOp<"scalar_logical_not_equal", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_ScalarLogicalOrOp : OneFlow_BaseOp<"scalar_logical_or", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_ScalarLogicalXorOp : OneFlow_BaseOp<"scalar_logical_xor", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_ScalarMulOp : OneFlow_BaseOp<"scalar_mul", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_ScalarMulByTensorOp : OneFlow_BaseOp<"scalar_mul_by_tensor", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$scalar
  );
  let output = (outs
    AnyType:$y
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "scalar"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_ScalarPowOp : OneFlow_BaseOp<"scalar_pow", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_ScalarPowGradOp : OneFlow_BaseOp<"scalar_pow_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_ScalarSubByTensorOp : OneFlow_BaseOp<"scalar_sub_by_tensor", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$scalar
  );
  let output = (outs
    AnyType:$y
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "scalar"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

#endif  // GET_ONEFLOW_SCALAR_OP_DEFINITIONS

// log_softmax, log_softmax_grad, softmax, softmax_cross_entropy, softmax_cross_entropy_grad, softmax_grad, sparse_softmax_cross_entropy, sparse_softmax_cross_entropy_grad, sparse_softmax_cross_entropy_ms, sparse_softmax_cross_entropy_ms_grad
// Total: 10
#ifndef GET_ONEFLOW_SOFTMAX_OP_DEFINITIONS
#define GET_ONEFLOW_SOFTMAX_OP_DEFINITIONS

def OneFlow_LogSoftmaxOp : OneFlow_BaseOp<"log_softmax", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$prob
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"prob"}; }
  }];
}

def OneFlow_LogSoftmaxGradOp : OneFlow_BaseOp<"log_softmax_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$prob,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"prob", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_SoftmaxOp : OneFlow_BaseOp<"softmax", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_SoftmaxCrossEntropyOp : OneFlow_BaseOp<"softmax_cross_entropy", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$prediction,
    AnyType:$label
  );
  let output = (outs
    AnyType:$prob,
    AnyType:$out
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"prediction", "label"}; }
    static std::vector<std::string> outputOrder() { return {"prob", "out"}; }
  }];
}

def OneFlow_SoftmaxCrossEntropyGradOp : OneFlow_BaseOp<"softmax_cross_entropy_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$dy,
    AnyType:$label,
    AnyType:$prob
  );
  let output = (outs
    AnyType:$prediction_diff
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"dy", "label", "prob"}; }
    static std::vector<std::string> outputOrder() { return {"prediction_diff"}; }
  }];
}

def OneFlow_SoftmaxGradOp : OneFlow_BaseOp<"softmax_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$y,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"y", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_SparseSoftmaxCrossEntropyOp : OneFlow_BaseOp<"sparse_softmax_cross_entropy", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"prediction", "label"}; }
    static std::vector<std::string> outputOrder() { return {"prob", "out"}; }
  }];
}

def OneFlow_SparseSoftmaxCrossEntropyGradOp : OneFlow_BaseOp<"sparse_softmax_cross_entropy_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"label", "dy", "prob"}; }
    static std::vector<std::string> outputOrder() { return {"prediction_diff"}; }
  }];
}

def OneFlow_SparseSoftmaxCrossEntropyMsOp : OneFlow_BaseOp<"sparse_softmax_cross_entropy_ms", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"prediction", "label"}; }
    static std::vector<std::string> outputOrder() { return {"prob", "out"}; }
  }];
}

def OneFlow_SparseSoftmaxCrossEntropyMsGradOp : OneFlow_BaseOp<"sparse_softmax_cross_entropy_ms_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"label", "dy", "prob"}; }
    static std::vector<std::string> outputOrder() { return {"prediction_diff"}; }
  }];
}

#endif  // GET_ONEFLOW_SOFTMAX_OP_DEFINITIONS

// create_summary_writer, flush_summary_writer, summary_write_histogram, summary_write_image, summary_write_pb, summary_write_scalar
// Total: 6
#ifndef GET_ONEFLOW_SUMMARY_OP_DEFINITIONS
#define GET_ONEFLOW_SUMMARY_OP_DEFINITIONS

def OneFlow_CreateSummaryWriterOp : OneFlow_BaseOp<"create_summary_writer", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let attrs = (ins
    StrAttr:$logdir
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {}; }
    static std::vector<std::string> outputOrder() { return {}; }
  }];
}

def OneFlow_FlushSummaryWriterOp : OneFlow_BaseOp<"flush_summary_writer", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {}; }
    static std::vector<std::string> outputOrder() { return {}; }
  }];
}

def OneFlow_SummaryWriteHistogramOp : OneFlow_BaseOp<"summary_write_histogram", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in,
    AnyType:$step,
    AnyType:$tag
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in", "step", "tag"}; }
    static std::vector<std::string> outputOrder() { return {}; }
  }];
}

def OneFlow_SummaryWriteImageOp : OneFlow_BaseOp<"summary_write_image", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in,
    AnyType:$step,
    AnyType:$tag
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in", "step", "tag"}; }
    static std::vector<std::string> outputOrder() { return {}; }
  }];
}

def OneFlow_SummaryWritePbOp : OneFlow_BaseOp<"summary_write_pb", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in,
    AnyType:$step
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in", "step"}; }
    static std::vector<std::string> outputOrder() { return {}; }
  }];
}

def OneFlow_SummaryWriteScalarOp : OneFlow_BaseOp<"summary_write_scalar", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in,
    AnyType:$step,
    AnyType:$tag
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in", "step", "tag"}; }
    static std::vector<std::string> outputOrder() { return {}; }
  }];
}

#endif  // GET_ONEFLOW_SUMMARY_OP_DEFINITIONS

// gen_tensor_buffer, tensor_buffer_to_list_of_tensors, tensor_buffer_to_list_of_tensors_v2, tensor_buffer_to_tensor, tensor_to_tensor_buffer
// Total: 5
#ifndef GET_ONEFLOW_TENSOR_BUFFER_OP_DEFINITIONS
#define GET_ONEFLOW_TENSOR_BUFFER_OP_DEFINITIONS

def OneFlow_GenTensorBufferOp : OneFlow_BaseOp<"gen_tensor_buffer", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_TensorBufferToListOfTensorsOp : OneFlow_BaseOp<"tensor_buffer_to_list_of_tensors", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    Variadic<AnyType>:$out
  );
  let attrs = (ins
    AnyI64ElementsAttr:$out_shape,
    StrAttr:$out_dtype,
    BoolAttr:$dynamic_out
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_TensorBufferToListOfTensorsV2Op : OneFlow_BaseOp<"tensor_buffer_to_list_of_tensors_v2", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    Variadic<AnyType>:$out
  );
  let attrs = (ins
    ShapeArrayAttr:$out_shapes,
    DTArrayAttr:$out_dtypes,
    BoolAttr:$dynamic_out
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_TensorBufferToTensorOp : OneFlow_BaseOp<"tensor_buffer_to_tensor", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_TensorToTensorBufferOp : OneFlow_BaseOp<"tensor_to_tensor_buffer", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI32Attr:$instance_dims
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

#endif  // GET_ONEFLOW_TENSOR_BUFFER_OP_DEFINITIONS

// TestDataTypeAttr, TestDynamicSource, TestListDataTypeAndListShapeAndListStringAttr, TestMultiInput, TestMultiInputGrad, TestMultiOutputOrder, TestRandomSource, TestReshape, TestSource, TestSourceMultiGpuFixedOutNum, ccrelu, ccrelu_grad, cpu_only_relu_test, test_user_op_attr_auto_type
// Total: 14
#ifndef GET_ONEFLOW_TEST_OP_DEFINITIONS
#define GET_ONEFLOW_TEST_OP_DEFINITIONS

def OneFlow_TestDataTypeAttrOp : OneFlow_BaseOp<"TestDataTypeAttr", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    StrAttr:$output_type
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_TestDynamicSourceOp : OneFlow_BaseOp<"TestDynamicSource", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let output = (outs
    AnyType:$out
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_TestListDataTypeAndListShapeAndListStringAttrOp : OneFlow_BaseOp<"TestListDataTypeAndListShapeAndListStringAttr", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_TestMultiInputOp : OneFlow_BaseOp<"TestMultiInput", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x1,
    AnyType:$x2
  );
  let output = (outs
    AnyType:$y
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x1", "x2"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_TestMultiInputGradOp : OneFlow_BaseOp<"TestMultiInputGrad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x1,
    AnyType:$x2,
    AnyType:$y_diff
  );
  let output = (outs
    AnyType:$x1_diff,
    AnyType:$x2_diff
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x1", "x2", "y_diff"}; }
    static std::vector<std::string> outputOrder() { return {"x1_diff", "x2_diff"}; }
  }];
}

def OneFlow_TestMultiOutputOrderOp : OneFlow_BaseOp<"TestMultiOutputOrder", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out1,
    AnyType:$out2
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out1", "out2"}; }
  }];
}

def OneFlow_TestRandomSourceOp : OneFlow_BaseOp<"TestRandomSource", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI64Attr:$seed
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_TestReshapeOp : OneFlow_BaseOp<"TestReshape", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    AnyI64ElementsAttr:$shape
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_TestSourceOp : OneFlow_BaseOp<"TestSource", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let output = (outs
    AnyType:$out
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_TestSourceMultiGpuFixedOutNumOp : OneFlow_BaseOp<"TestSourceMultiGpuFixedOutNum", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    SI64Attr:$out_num
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_CcreluOp : OneFlow_BaseOp<"ccrelu", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_CcreluGradOp : OneFlow_BaseOp<"ccrelu_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$y,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"y", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_CpuOnlyReluTestOp : OneFlow_BaseOp<"cpu_only_relu_test", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_TestUserOpAttrAutoTypeOp : OneFlow_BaseOp<"test_user_op_attr_auto_type", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

#endif  // GET_ONEFLOW_TEST_OP_DEFINITIONS

// acos, acos_grad, acosh, acosh_grad, asin, asin_grad, asinh, asinh_grad, atan, atan2, atan2_x_grad, atan2_y_grad, atan_grad, atanh, atanh_grad, cos, cos_grad, cosh, cosh_grad, hardtanh, hardtanh_grad, sin, sin_grad, sinh, sinh_grad, tan, tan_grad, tanh, tanh_grad
// Total: 29
#ifndef GET_ONEFLOW_TRIGONOMETRIC_OP_DEFINITIONS
#define GET_ONEFLOW_TRIGONOMETRIC_OP_DEFINITIONS

def OneFlow_AcosOp : OneFlow_BaseOp<"acos", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_AcosGradOp : OneFlow_BaseOp<"acos_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_AcoshOp : OneFlow_BaseOp<"acosh", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_AcoshGradOp : OneFlow_BaseOp<"acosh_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_AsinOp : OneFlow_BaseOp<"asin", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_AsinGradOp : OneFlow_BaseOp<"asin_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_AsinhOp : OneFlow_BaseOp<"asinh", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_AsinhGradOp : OneFlow_BaseOp<"asinh_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_AtanOp : OneFlow_BaseOp<"atan", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_Atan2Op : OneFlow_BaseOp<"atan2", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "y"}; }
    static std::vector<std::string> outputOrder() { return {"z"}; }
  }];
}

def OneFlow_Atan2XGradOp : OneFlow_BaseOp<"atan2_x_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y,
    AnyType:$dz
  );
  let output = (outs
    AnyType:$dx
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "y", "dz"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_Atan2YGradOp : OneFlow_BaseOp<"atan2_y_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y,
    AnyType:$dz
  );
  let output = (outs
    AnyType:$dy
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "y", "dz"}; }
    static std::vector<std::string> outputOrder() { return {"dy"}; }
  }];
}

def OneFlow_AtanGradOp : OneFlow_BaseOp<"atan_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_AtanhOp : OneFlow_BaseOp<"atanh", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_AtanhGradOp : OneFlow_BaseOp<"atanh_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_CosOp : OneFlow_BaseOp<"cos", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_CosGradOp : OneFlow_BaseOp<"cos_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_CoshOp : OneFlow_BaseOp<"cosh", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_CoshGradOp : OneFlow_BaseOp<"cosh_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_HardtanhOp : OneFlow_BaseOp<"hardtanh", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

def OneFlow_HardtanhGradOp : OneFlow_BaseOp<"hardtanh_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"y", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_SinOp : OneFlow_BaseOp<"sin", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_SinGradOp : OneFlow_BaseOp<"sin_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_SinhOp : OneFlow_BaseOp<"sinh", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_SinhGradOp : OneFlow_BaseOp<"sinh_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_TanOp : OneFlow_BaseOp<"tan", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_TanGradOp : OneFlow_BaseOp<"tan_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_TanhOp : OneFlow_BaseOp<"tanh", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_TanhGradOp : OneFlow_BaseOp<"tanh_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x", "dy"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

#endif  // GET_ONEFLOW_TRIGONOMETRIC_OP_DEFINITIONS

// upsample, upsample_bicubic_2d, upsample_bicubic_2d_grad, upsample_bilinear_2d, upsample_bilinear_2d_grad, upsample_grad, upsample_linear_1d, upsample_linear_1d_grad, upsample_nearest_1d, upsample_nearest_1d_grad, upsample_nearest_2d, upsample_nearest_2d_grad, upsample_nearest_3d, upsample_nearest_3d_grad, upsample_trilinear_3d, upsample_trilinear_3d_grad
// Total: 16
#ifndef GET_ONEFLOW_UPSAMPLE_OP_DEFINITIONS
#define GET_ONEFLOW_UPSAMPLE_OP_DEFINITIONS

def OneFlow_UpsampleOp : OneFlow_BaseOp<"upsample", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_UpsampleBicubic2DOp : OneFlow_BaseOp<"upsample_bicubic_2d", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_UpsampleBicubic2DGradOp : OneFlow_BaseOp<"upsample_bicubic_2d_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"dy", "x"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_UpsampleBilinear2DOp : OneFlow_BaseOp<"upsample_bilinear_2d", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_UpsampleBilinear2DGradOp : OneFlow_BaseOp<"upsample_bilinear_2d_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"dy", "x"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_UpsampleGradOp : OneFlow_BaseOp<"upsample_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"dy", "x"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_UpsampleLinear1DOp : OneFlow_BaseOp<"upsample_linear_1d", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_UpsampleLinear1DGradOp : OneFlow_BaseOp<"upsample_linear_1d_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"dy", "x"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_UpsampleNearest1DOp : OneFlow_BaseOp<"upsample_nearest_1d", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_UpsampleNearest1DGradOp : OneFlow_BaseOp<"upsample_nearest_1d_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"dy", "x"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_UpsampleNearest2DOp : OneFlow_BaseOp<"upsample_nearest_2d", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_UpsampleNearest2DGradOp : OneFlow_BaseOp<"upsample_nearest_2d_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"dy", "x"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_UpsampleNearest3DOp : OneFlow_BaseOp<"upsample_nearest_3d", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_UpsampleNearest3DGradOp : OneFlow_BaseOp<"upsample_nearest_3d_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"dy", "x"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

def OneFlow_UpsampleTrilinear3DOp : OneFlow_BaseOp<"upsample_trilinear_3d", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"x"}; }
    static std::vector<std::string> outputOrder() { return {"y"}; }
  }];
}

def OneFlow_UpsampleTrilinear3DGradOp : OneFlow_BaseOp<"upsample_trilinear_3d_grad", [DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
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
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"dy", "x"}; }
    static std::vector<std::string> outputOrder() { return {"dx"}; }
  }];
}

#endif  // GET_ONEFLOW_UPSAMPLE_OP_DEFINITIONS


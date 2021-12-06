// ASSIGN;BASE;BINARY;BROADCAST;CONV;CROSS_ENTROPY;CUDA;DATASET;DETECTION;EAGER;FUSED;IDEMPOTENT;IDENTITY;IMAGE;INDICES;INVOLUTION;LOSS;MATH;MATMUL;MISC;NCCL;NORMALIZATION;OPTIMIZER;PADDING;PARALLEL_CAST;POOL;QUANTIZATION;REDUCE;RESHAPE;SCALAR;SOFTMAX;SUMMARY;TENSOR_BUFFER;TEST;TRIGONOMETRIC;UNARY;UPSAMPLE

/*
#define GET_OP_LIST
#include "OneFlow/OneFlow.assign_ops.cpp.inc"
,
#define GET_OP_LIST
#include "OneFlow/OneFlow.binary_ops.cpp.inc"
,
#define GET_OP_LIST
#include "OneFlow/OneFlow.broadcast_ops.cpp.inc"
,
#define GET_OP_LIST
#include "OneFlow/OneFlow.conv_ops.cpp.inc"
,
#define GET_OP_LIST
#include "OneFlow/OneFlow.cross_entropy_ops.cpp.inc"
,
#define GET_OP_LIST
#include "OneFlow/OneFlow.cuda_ops.cpp.inc"
,
#define GET_OP_LIST
#include "OneFlow/OneFlow.dataset_ops.cpp.inc"
,
#define GET_OP_LIST
#include "OneFlow/OneFlow.detection_ops.cpp.inc"
,
#define GET_OP_LIST
#include "OneFlow/OneFlow.eager_ops.cpp.inc"
,
#define GET_OP_LIST
#include "OneFlow/OneFlow.fused_ops.cpp.inc"
,
#define GET_OP_LIST
#include "OneFlow/OneFlow.idempotent_ops.cpp.inc"
,
#define GET_OP_LIST
#include "OneFlow/OneFlow.identity_ops.cpp.inc"
,
#define GET_OP_LIST
#include "OneFlow/OneFlow.image_ops.cpp.inc"
,
#define GET_OP_LIST
#include "OneFlow/OneFlow.indices_ops.cpp.inc"
,
#define GET_OP_LIST
#include "OneFlow/OneFlow.involution_ops.cpp.inc"
,
#define GET_OP_LIST
#include "OneFlow/OneFlow.loss_ops.cpp.inc"
,
#define GET_OP_LIST
#include "OneFlow/OneFlow.math_ops.cpp.inc"
,
#define GET_OP_LIST
#include "OneFlow/OneFlow.matmul_ops.cpp.inc"
,
#define GET_OP_LIST
#include "OneFlow/OneFlow.misc_ops.cpp.inc"
,
#define GET_OP_LIST
#include "OneFlow/OneFlow.nccl_ops.cpp.inc"
,
#define GET_OP_LIST
#include "OneFlow/OneFlow.normalization_ops.cpp.inc"
,
#define GET_OP_LIST
#include "OneFlow/OneFlow.optimizer_ops.cpp.inc"
,
#define GET_OP_LIST
#include "OneFlow/OneFlow.padding_ops.cpp.inc"
,
#define GET_OP_LIST
#include "OneFlow/OneFlow.parallel_cast_ops.cpp.inc"
,
#define GET_OP_LIST
#include "OneFlow/OneFlow.pool_ops.cpp.inc"
,
#define GET_OP_LIST
#include "OneFlow/OneFlow.quantization_ops.cpp.inc"
,
#define GET_OP_LIST
#include "OneFlow/OneFlow.reduce_ops.cpp.inc"
,
#define GET_OP_LIST
#include "OneFlow/OneFlow.reshape_ops.cpp.inc"
,
#define GET_OP_LIST
#include "OneFlow/OneFlow.scalar_ops.cpp.inc"
,
#define GET_OP_LIST
#include "OneFlow/OneFlow.softmax_ops.cpp.inc"
,
#define GET_OP_LIST
#include "OneFlow/OneFlow.summary_ops.cpp.inc"
,
#define GET_OP_LIST
#include "OneFlow/OneFlow.tensor_buffer_ops.cpp.inc"
,
#define GET_OP_LIST
#include "OneFlow/OneFlow.trigonometric_ops.cpp.inc"
,
#define GET_OP_LIST
#include "OneFlow/OneFlow.unary_ops.cpp.inc"
,
#define GET_OP_LIST
#include "OneFlow/OneFlow.upsample_ops.cpp.inc"
*/

// Group: ASSIGN
// assign, assign_if, assign_if_not, logical_slice_assign
// Total: 4

#ifdef GET_ONEFLOW_ASSIGN_OP_DEFINITIONS

def OneFlow_AssignOp : OneFlow_BaseOp<"assign", [DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$ref,
    AnyType:$value
  );
}

def OneFlow_AssignIfOp : OneFlow_BaseOp<"assign_if", [DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$ref,
    AnyType:$value,
    AnyType:$condition
  );
}

def OneFlow_AssignIfNotOp : OneFlow_BaseOp<"assign_if_not", [DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$ref,
    AnyType:$value,
    AnyType:$condition
  );
}

def OneFlow_LogicalSliceAssignOp : OneFlow_BaseOp<"logical_slice_assign", [DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

#endif // GET_ONEFLOW_ASSIGN_OP_DEFINITIONS

// Group: BASE
// normalization_add_relu
// Total: 1

#ifdef GET_ONEFLOW_BASE_OP_DEFINITIONS

class OneFlow_NormalizationAddReluBaseOp : OneFlow_BaseOp<"normalization_add_relu", [NoSideEffect, AttrSizedOperandSegments, AttrSizedResultSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
    DefaultValuedAttr<SI32Attr, "0">:$axis,
    DefaultValuedAttr<F32Attr, "0.">:$epsilon,
    DefaultValuedAttr<BoolAttr, "false">:$training,
    DefaultValuedAttr<F32Attr, "0.">:$momentum
  );
  let trait_attrs = (ins
    I32ElementsAttr:$operand_segment_sizes,
    I32ElementsAttr:$result_segment_sizes
  );
}

#endif // GET_ONEFLOW_BASE_OP_DEFINITIONS

// Group: BINARY
// bias_add, cast_like, celu_grad, diag_grad, dropout_grad, elementwise_maximum, elementwise_minimum, elu_grad, floordiv, gelu_grad, grid_sample, hardsigmoid_grad, hardswish_grad, l1_l2_regularize_gradient, leaky_relu_grad, masked_fill, mish_grad, multiply, narrow_grad, pow, prelu, relu_grad, selu_grad, sigmoid_grad, silu_grad, tf_prelu, unfold_tensor_grad, xdivy, xlogy
// Total: 29

#ifdef GET_ONEFLOW_BINARY_OP_DEFINITIONS

def OneFlow_BiasAddOp : OneFlow_BaseOp<"bias_add", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$a,
    AnyType:$b
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$axis
  );
}

def OneFlow_CastLikeOp : OneFlow_BaseOp<"cast_like", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in,
    AnyType:$dtype_like
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_CeluGradOp : OneFlow_BaseOp<"celu_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.">:$alpha
  );
}

def OneFlow_DiagGradOp : OneFlow_BaseOp<"diag_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$dy,
    AnyType:$in
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$diagonal
  );
}

def OneFlow_DropoutGradOp : OneFlow_BaseOp<"dropout_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$dy,
    AnyType:$mask
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$scale
  );
}

def OneFlow_ElementwiseMaximumOp : OneFlow_BaseOp<"elementwise_maximum", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
}

def OneFlow_ElementwiseMinimumOp : OneFlow_BaseOp<"elementwise_minimum", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
}

def OneFlow_EluGradOp : OneFlow_BaseOp<"elu_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.">:$alpha
  );
}

def OneFlow_FloordivOp : OneFlow_BaseOp<"floordiv", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
}

def OneFlow_GeluGradOp : OneFlow_BaseOp<"gelu_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_GridSampleOp : OneFlow_BaseOp<"grid_sample", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
    DefaultValuedAttr<BoolAttr, "false">:$align_corners
  );
}

def OneFlow_HardsigmoidGradOp : OneFlow_BaseOp<"hardsigmoid_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_HardswishGradOp : OneFlow_BaseOp<"hardswish_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_L1L2RegularizeGradientOp : OneFlow_BaseOp<"l1_l2_regularize_gradient", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$model,
    AnyType:$model_diff
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$l1,
    DefaultValuedAttr<F32Attr, "0.">:$l2
  );
}

def OneFlow_LeakyReluGradOp : OneFlow_BaseOp<"leaky_relu_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$alpha
  );
}

def OneFlow_MaskedFillOp : OneFlow_BaseOp<"masked_fill", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$mask
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$has_int_operand,
    DefaultValuedAttr<BoolAttr, "false">:$has_float_operand,
    DefaultValuedAttr<SI64Attr, "0">:$int_operand,
    DefaultValuedAttr<F64Attr, "0.">:$float_operand
  );
}

def OneFlow_MishGradOp : OneFlow_BaseOp<"mish_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_MultiplyOp : OneFlow_BaseOp<"multiply", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_NarrowGradOp : OneFlow_BaseOp<"narrow_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$dy,
    AnyType:$like
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$dim,
    DefaultValuedAttr<SI64Attr, "0">:$start,
    DefaultValuedAttr<SI64Attr, "0">:$length
  );
}

def OneFlow_PowOp : OneFlow_BaseOp<"pow", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
}

def OneFlow_PreluOp : OneFlow_BaseOp<"prelu", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$alpha
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_ReluGradOp : OneFlow_BaseOp<"relu_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$y,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_SeluGradOp : OneFlow_BaseOp<"selu_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_SigmoidGradOp : OneFlow_BaseOp<"sigmoid_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$y,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_SiluGradOp : OneFlow_BaseOp<"silu_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_TfPreluOp : OneFlow_BaseOp<"tf_prelu", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$alpha
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_UnfoldTensorGradOp : OneFlow_BaseOp<"unfold_tensor_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$dy,
    AnyType:$x
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$dimension,
    DefaultValuedAttr<SI32Attr, "0">:$size,
    DefaultValuedAttr<SI32Attr, "0">:$step
  );
}

def OneFlow_XdivyOp : OneFlow_BaseOp<"xdivy", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
}

def OneFlow_XlogyOp : OneFlow_BaseOp<"xlogy", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
}

#endif // GET_ONEFLOW_BINARY_OP_DEFINITIONS

// Group: BROADCAST
// broadcast_add, broadcast_div, broadcast_div_grad, broadcast_equal, broadcast_floor_mod, broadcast_fmod, broadcast_greater, broadcast_greater_equal, broadcast_less, broadcast_less_equal, broadcast_like, broadcast_logical_and, broadcast_logical_or, broadcast_logical_xor, broadcast_maximum, broadcast_minimum, broadcast_mul, broadcast_not_equal, broadcast_pow, broadcast_pow_x_grad, broadcast_pow_y_grad, broadcast_sub
// Total: 22

#ifdef GET_ONEFLOW_BROADCAST_OP_DEFINITIONS

def OneFlow_BroadcastAddOp : OneFlow_BaseOp<"broadcast_add", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
}

def OneFlow_BroadcastDivOp : OneFlow_BaseOp<"broadcast_div", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
}

def OneFlow_BroadcastDivGradOp : OneFlow_BaseOp<"broadcast_div_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$y,
    AnyType:$z,
    AnyType:$dz
  );
  let output = (outs
    AnyType:$dy
  );
}

def OneFlow_BroadcastEqualOp : OneFlow_BaseOp<"broadcast_equal", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
}

def OneFlow_BroadcastFloorModOp : OneFlow_BaseOp<"broadcast_floor_mod", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
}

def OneFlow_BroadcastFmodOp : OneFlow_BaseOp<"broadcast_fmod", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
}

def OneFlow_BroadcastGreaterOp : OneFlow_BaseOp<"broadcast_greater", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
}

def OneFlow_BroadcastGreaterEqualOp : OneFlow_BaseOp<"broadcast_greater_equal", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
}

def OneFlow_BroadcastLessOp : OneFlow_BaseOp<"broadcast_less", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
}

def OneFlow_BroadcastLessEqualOp : OneFlow_BaseOp<"broadcast_less_equal", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
}

def OneFlow_BroadcastLikeOp : OneFlow_BaseOp<"broadcast_like", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_BroadcastLogicalAndOp : OneFlow_BaseOp<"broadcast_logical_and", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
}

def OneFlow_BroadcastLogicalOrOp : OneFlow_BaseOp<"broadcast_logical_or", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
}

def OneFlow_BroadcastLogicalXorOp : OneFlow_BaseOp<"broadcast_logical_xor", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
}

def OneFlow_BroadcastMaximumOp : OneFlow_BaseOp<"broadcast_maximum", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
}

def OneFlow_BroadcastMinimumOp : OneFlow_BaseOp<"broadcast_minimum", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
}

def OneFlow_BroadcastMulOp : OneFlow_BaseOp<"broadcast_mul", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
}

def OneFlow_BroadcastNotEqualOp : OneFlow_BaseOp<"broadcast_not_equal", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
}

def OneFlow_BroadcastPowOp : OneFlow_BaseOp<"broadcast_pow", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
}

def OneFlow_BroadcastPowXGradOp : OneFlow_BaseOp<"broadcast_pow_x_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_BroadcastPowYGradOp : OneFlow_BaseOp<"broadcast_pow_y_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_BroadcastSubOp : OneFlow_BaseOp<"broadcast_sub", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
}

#endif // GET_ONEFLOW_BROADCAST_OP_DEFINITIONS

// Group: CONV
// conv1d, conv2d, conv3d, conv_bias_grad, conv_data_grad, conv_filter_grad, deconv1d, deconv2d, deconv3d
// Total: 9

#ifdef GET_ONEFLOW_CONV_OP_DEFINITIONS

def OneFlow_Conv1DOp : OneFlow_ConvolutionBaseOp<"conv1d", [NoSideEffect, AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_Conv2DOp : OneFlow_ConvolutionBaseOp<"conv2d", [NoSideEffect, AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_Conv3DOp : OneFlow_ConvolutionBaseOp<"conv3d", [NoSideEffect, AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_ConvBiasGradOp : OneFlow_BaseOp<"conv_bias_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$dy
  );
  let output = (outs
    AnyType:$bias_diff
  );
  let attrs = (ins
    StrAttr:$data_format,
    DefaultValuedAttr<SI32Attr, "0">:$num_spatial_dims
  );
}

def OneFlow_ConvDataGradOp : OneFlow_BaseOp<"conv_data_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
    DefaultValuedAttr<SI32Attr, "0">:$num_spatial_dims,
    SI32ArrayAttr:$padding_before,
    StrAttr:$data_format,
    SI32ArrayAttr:$kernel_size,
    SI32ArrayAttr:$strides,
    SI32ArrayAttr:$dilation_rate,
    DefaultValuedAttr<SI32Attr, "0">:$groups
  );
}

def OneFlow_ConvFilterGradOp : OneFlow_BaseOp<"conv_filter_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$dy,
    AnyType:$x
  );
  let output = (outs
    AnyType:$filter_diff
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
}

def OneFlow_Deconv1DOp : OneFlow_BaseOp<"deconv1d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in,
    AnyType:$weight
  );
  let output = (outs
    AnyType:$out
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
}

def OneFlow_Deconv2DOp : OneFlow_BaseOp<"deconv2d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in,
    AnyType:$weight
  );
  let output = (outs
    AnyType:$out
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
}

def OneFlow_Deconv3DOp : OneFlow_BaseOp<"deconv3d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in,
    AnyType:$weight
  );
  let output = (outs
    AnyType:$out
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
}

#endif // GET_ONEFLOW_CONV_OP_DEFINITIONS

// Group: CROSS_ENTROPY
// binary_cross_entropy, binary_cross_entropy_grad, binary_cross_entropy_with_logits, binary_cross_entropy_with_logits_grad, sigmoid_cross_entropy, sigmoid_cross_entropy_grad, sparse_cross_entropy, sparse_cross_entropy_grad, sparse_cross_entropy_ms, sparse_cross_entropy_ms_grad
// Total: 10

#ifdef GET_ONEFLOW_CROSS_ENTROPY_OP_DEFINITIONS

def OneFlow_BinaryCrossEntropyOp : OneFlow_BaseOp<"binary_cross_entropy", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
}

def OneFlow_BinaryCrossEntropyGradOp : OneFlow_BaseOp<"binary_cross_entropy_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
}

def OneFlow_BinaryCrossEntropyWithLogitsOp : OneFlow_BaseOp<"binary_cross_entropy_with_logits", [NoSideEffect, AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
    DefaultValuedAttr<BoolAttr, "false">:$has_pos_weight,
    StrAttr:$reduction
  );
  let trait_attrs = (ins
    I32ElementsAttr:$operand_segment_sizes
  );
}

def OneFlow_BinaryCrossEntropyWithLogitsGradOp : OneFlow_BaseOp<"binary_cross_entropy_with_logits_grad", [NoSideEffect, AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
    DefaultValuedAttr<BoolAttr, "false">:$has_pos_weight,
    StrAttr:$reduction
  );
  let trait_attrs = (ins
    I32ElementsAttr:$operand_segment_sizes
  );
}

def OneFlow_SigmoidCrossEntropyOp : OneFlow_BaseOp<"sigmoid_cross_entropy", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$prediction,
    AnyType:$label
  );
  let output = (outs
    AnyType:$loss
  );
}

def OneFlow_SigmoidCrossEntropyGradOp : OneFlow_BaseOp<"sigmoid_cross_entropy_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$prediction,
    AnyType:$loss_diff,
    AnyType:$label
  );
  let output = (outs
    AnyType:$prediction_diff
  );
}

def OneFlow_SparseCrossEntropyOp : OneFlow_BaseOp<"sparse_cross_entropy", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$prediction,
    AnyType:$label
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$depth
  );
}

def OneFlow_SparseCrossEntropyGradOp : OneFlow_BaseOp<"sparse_cross_entropy_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$prediction,
    AnyType:$label,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$prediction_diff
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$depth
  );
}

def OneFlow_SparseCrossEntropyMsOp : OneFlow_BaseOp<"sparse_cross_entropy_ms", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$prediction,
    AnyType:$label
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$depth
  );
}

def OneFlow_SparseCrossEntropyMsGradOp : OneFlow_BaseOp<"sparse_cross_entropy_ms_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$prediction,
    AnyType:$label,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$prediction_diff
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$depth
  );
}

#endif // GET_ONEFLOW_CROSS_ENTROPY_OP_DEFINITIONS

// Group: CUDA
// nvtx_end, nvtx_start
// Total: 2

#ifdef GET_ONEFLOW_CUDA_OP_DEFINITIONS

def OneFlow_NvtxEndOp : OneFlow_BaseOp<"nvtx_end", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_NvtxStartOp : OneFlow_BaseOp<"nvtx_start", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

#endif // GET_ONEFLOW_CUDA_OP_DEFINITIONS

// Group: DATASET
// COCOReader, OFRecordReader, OneRecReader, ctc_greedy_decoder, megatron_gpt_mmap_data_loader, ofrecord_bytes_decoder, ofrecord_image_classification_reader, ofrecord_image_decoder, ofrecord_image_decoder_random_crop, ofrecord_raw_decoder, onerec_decoder
// Total: 11

#ifdef GET_ONEFLOW_DATASET_OP_DEFINITIONS

def OneFlow_COCOReaderOp : OneFlow_BaseOp<"COCOReader", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
}

def OneFlow_OFRecordReaderOp : OneFlow_BaseOp<"OFRecordReader", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let output = (outs
    AnyType:$out
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
}

def OneFlow_OneRecReaderOp : OneFlow_BaseOp<"OneRecReader", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    StrArrayAttr:$files,
    DefaultValuedAttr<SI32Attr, "0">:$batch_size,
    DefaultValuedAttr<BoolAttr, "false">:$random_shuffle,
    DefaultValuedAttr<StrAttr, "\"instance\"">:$shuffle_mode,
    DefaultValuedAttr<SI64Attr, "-1">:$seed,
    DefaultValuedAttr<SI32Attr, "1024">:$shuffle_buffer_size,
    DefaultValuedAttr<BoolAttr, "false">:$shuffle_after_epoch,
    DefaultValuedAttr<BoolAttr, "true">:$verify_example
  );
}

def OneFlow_CtcGreedyDecoderOp : OneFlow_BaseOp<"ctc_greedy_decoder", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$log_probs,
    AnyType:$input_lengths
  );
  let output = (outs
    AnyType:$decoded,
    AnyType:$neg_sum_logits
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$merge_repeated
  );
}

def OneFlow_MegatronGptMmapDataLoaderOp : OneFlow_BaseOp<"megatron_gpt_mmap_data_loader", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    Optional<AnyType>:$iteration
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    StrAttr:$data_file_prefix,
    DefaultValuedAttr<SI64Attr, "0">:$seq_length,
    DefaultValuedAttr<SI64Attr, "1">:$label_length,
    DefaultValuedAttr<SI64Attr, "0">:$num_samples,
    DefaultValuedAttr<SI64Attr, "0">:$batch_size,
    StrAttr:$dtype,
    SI64ArrayAttr:$split_sizes,
    DefaultValuedAttr<SI64Attr, "0">:$split_index,
    DefaultValuedAttr<BoolAttr, "false">:$shuffle,
    DefaultValuedAttr<SI64Attr, "0">:$random_seed,
    StrArrayAttr:$nd_sbp
  );
}

def OneFlow_OfrecordBytesDecoderOp : OneFlow_BaseOp<"ofrecord_bytes_decoder", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_OfrecordImageClassificationReaderOp : OneFlow_BaseOp<"ofrecord_image_classification_reader", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let output = (outs
    AnyType:$image,
    AnyType:$label
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
}

def OneFlow_OfrecordImageDecoderOp : OneFlow_BaseOp<"ofrecord_image_decoder", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    StrAttr:$name,
    DefaultValuedAttr<StrAttr, "\"BGR\"">:$color_space
  );
}

def OneFlow_OfrecordImageDecoderRandomCropOp : OneFlow_BaseOp<"ofrecord_image_decoder_random_crop", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
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
}

def OneFlow_OfrecordRawDecoderOp : OneFlow_BaseOp<"ofrecord_raw_decoder", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
    DefaultValuedAttr<BoolAttr, "false">:$dim1_varying_length,
    DefaultValuedAttr<BoolAttr, "false">:$truncate
  );
}

def OneFlow_OnerecDecoderOp : OneFlow_BaseOp<"onerec_decoder", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
    DefaultValuedAttr<BoolAttr, "false">:$is_dynamic,
    DefaultValuedAttr<BoolAttr, "false">:$has_reshape,
    AnyI64ElementsAttr:$reshape,
    DefaultValuedAttr<BoolAttr, "false">:$has_batch_padding,
    AnyI64ElementsAttr:$batch_padding
  );
}

#endif // GET_ONEFLOW_DATASET_OP_DEFINITIONS

// Group: DETECTION
// in_top_k, nms, object_bbox_flip, object_bbox_scale, object_segmentation_polygon_flip, object_segmentation_polygon_scale, object_segmentation_polygon_to_mask, top_k
// Total: 8

#ifdef GET_ONEFLOW_DETECTION_OP_DEFINITIONS

def OneFlow_InTopKOp : OneFlow_BaseOp<"in_top_k", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$targets,
    AnyType:$predictions
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$k
  );
}

def OneFlow_NmsOp : OneFlow_BaseOp<"nms", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$iou_threshold,
    DefaultValuedAttr<SI32Attr, "0">:$keep_n
  );
}

def OneFlow_ObjectBboxFlipOp : OneFlow_BaseOp<"object_bbox_flip", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$bbox,
    AnyType:$image_size,
    AnyType:$flip_code
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_ObjectBboxScaleOp : OneFlow_BaseOp<"object_bbox_scale", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$bbox,
    AnyType:$scale
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_ObjectSegmentationPolygonFlipOp : OneFlow_BaseOp<"object_segmentation_polygon_flip", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$poly,
    AnyType:$image_size,
    AnyType:$flip_code
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_ObjectSegmentationPolygonScaleOp : OneFlow_BaseOp<"object_segmentation_polygon_scale", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$poly,
    AnyType:$scale
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_ObjectSegmentationPolygonToMaskOp : OneFlow_BaseOp<"object_segmentation_polygon_to_mask", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$poly,
    AnyType:$poly_index,
    AnyType:$image_size
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_TopKOp : OneFlow_BaseOp<"top_k", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$k,
    DefaultValuedAttr<BoolAttr, "false">:$sorted
  );
}

#endif // GET_ONEFLOW_DETECTION_OP_DEFINITIONS

// Group: EAGER
// eager_b_to_s, eager_naive_s_to_s, eager_nccl_all_gather, eager_nccl_all_reduce, eager_nccl_broadcast, eager_nccl_reduce, eager_nccl_reduce_scatter, eager_nccl_s2s, eager_p_to_b, eager_p_to_s, eager_s_to_b, eager_symmetric_s_to_p
// Total: 12

#ifdef GET_ONEFLOW_EAGER_OP_DEFINITIONS

def OneFlow_EagerBToSOp : OneFlow_BaseOp<"eager_b_to_s", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "-1">:$out_split_axis,
    StrAttr:$in_parallel_conf,
    StrAttr:$out_parallel_conf,
    AnyI64ElementsAttr:$shape
  );
}

def OneFlow_EagerNaiveSToSOp : OneFlow_BaseOp<"eager_naive_s_to_s", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "-1">:$in_split_axis,
    DefaultValuedAttr<SI64Attr, "-1">:$out_split_axis,
    StrAttr:$in_parallel_conf,
    StrAttr:$out_parallel_conf,
    AnyI64ElementsAttr:$shape
  );
}

def OneFlow_EagerNcclAllGatherOp : OneFlow_BaseOp<"eager_nccl_all_gather", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_EagerNcclAllReduceOp : OneFlow_BaseOp<"eager_nccl_all_reduce", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    StrAttr:$parallel_conf,
    DefaultValuedAttr<BoolAttr, "false">:$async_launch
  );
}

def OneFlow_EagerNcclBroadcastOp : OneFlow_BaseOp<"eager_nccl_broadcast", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    StrAttr:$parallel_conf,
    DefaultValuedAttr<SI64Attr, "0">:$root
  );
}

def OneFlow_EagerNcclReduceOp : OneFlow_BaseOp<"eager_nccl_reduce", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    StrAttr:$parallel_conf,
    DefaultValuedAttr<SI64Attr, "0">:$root
  );
}

def OneFlow_EagerNcclReduceScatterOp : OneFlow_BaseOp<"eager_nccl_reduce_scatter", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    StrAttr:$parallel_conf,
    DefaultValuedAttr<StrAttr, "\"sum\"">:$op_type
  );
}

def OneFlow_EagerNcclS2sOp : OneFlow_BaseOp<"eager_nccl_s2s", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "-1">:$in_split_axis,
    DefaultValuedAttr<SI64Attr, "-1">:$out_split_axis,
    StrAttr:$parallel_conf
  );
}

def OneFlow_EagerPToBOp : OneFlow_BaseOp<"eager_p_to_b", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_EagerPToSOp : OneFlow_BaseOp<"eager_p_to_s", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "-1">:$out_split_axis,
    StrAttr:$in_parallel_conf,
    StrAttr:$out_parallel_conf,
    AnyI64ElementsAttr:$shape
  );
}

def OneFlow_EagerSToBOp : OneFlow_BaseOp<"eager_s_to_b", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "-1">:$in_split_axis,
    StrAttr:$in_parallel_conf,
    StrAttr:$out_parallel_conf,
    AnyI64ElementsAttr:$shape
  );
}

def OneFlow_EagerSymmetricSToPOp : OneFlow_BaseOp<"eager_symmetric_s_to_p", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "-1">:$in_split_axis,
    StrAttr:$parallel_conf
  );
}

#endif // GET_ONEFLOW_EAGER_OP_DEFINITIONS

// Group: FUSED
// cudnn_fused_normalization_add_relu, cudnn_fused_normalization_add_relu_grad, fused_bias_add_gelu, fused_bias_add_gelu_grad, fused_bias_add_mask_scale, fused_cast_scale, fused_scale_mask_softmax, fused_scale_mask_softmax_dropout, fused_scale_mask_softmax_dropout_grad, fused_scale_mask_softmax_grad, fused_scale_tril, fused_self_attention_query_mul_key_and_value, fused_self_attention_query_mul_key_and_value_grad, fused_tril_scale_softmax_mask_scale, fused_tril_scale_softmax_mask_scale_grad, normalization_add_relu_grad
// Total: 16

#ifdef GET_ONEFLOW_FUSED_OP_DEFINITIONS

def OneFlow_CudnnFusedNormalizationAddReluOp : OneFlow_BaseOp<"cudnn_fused_normalization_add_relu", [NoSideEffect, AttrSizedOperandSegments, AttrSizedResultSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
    DefaultValuedAttr<SI32Attr, "0">:$axis,
    DefaultValuedAttr<F32Attr, "0.">:$epsilon,
    DefaultValuedAttr<F32Attr, "0.">:$momentum
  );
  let trait_attrs = (ins
    I32ElementsAttr:$operand_segment_sizes,
    I32ElementsAttr:$result_segment_sizes
  );
}

def OneFlow_CudnnFusedNormalizationAddReluGradOp : OneFlow_BaseOp<"cudnn_fused_normalization_add_relu_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
    DefaultValuedAttr<SI32Attr, "0">:$axis,
    DefaultValuedAttr<F32Attr, "0.">:$epsilon
  );
}

def OneFlow_FusedBiasAddGeluOp : OneFlow_BaseOp<"fused_bias_add_gelu", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$a,
    AnyType:$b
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$axis
  );
}

def OneFlow_FusedBiasAddGeluGradOp : OneFlow_BaseOp<"fused_bias_add_gelu_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$a,
    AnyType:$b,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$axis
  );
}

def OneFlow_FusedBiasAddMaskScaleOp : OneFlow_BaseOp<"fused_bias_add_mask_scale", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
    DefaultValuedAttr<SI32Attr, "0">:$axis,
    DefaultValuedAttr<F32Attr, "0.">:$scale
  );
}

def OneFlow_FusedCastScaleOp : OneFlow_BaseOp<"fused_cast_scale", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$scale_by_tensor
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "1.">:$scale
  );
}

def OneFlow_FusedScaleMaskSoftmaxOp : OneFlow_BaseOp<"fused_scale_mask_softmax", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$mask
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "1.">:$scale_value,
    DefaultValuedAttr<F32Attr, "0.">:$mask_fill_value
  );
}

def OneFlow_FusedScaleMaskSoftmaxDropoutOp : OneFlow_BaseOp<"fused_scale_mask_softmax_dropout", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$mask,
    AnyType:$dropout_mask
  );
  let output = (outs
    AnyType:$y,
    AnyType:$softmax_y
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "1.">:$scale_value,
    DefaultValuedAttr<F32Attr, "0.">:$mask_fill_value,
    DefaultValuedAttr<F32Attr, "1.">:$dropout_scale_value
  );
}

def OneFlow_FusedScaleMaskSoftmaxDropoutGradOp : OneFlow_BaseOp<"fused_scale_mask_softmax_dropout_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$softmax_y,
    AnyType:$dy,
    AnyType:$mask,
    AnyType:$dropout_mask
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$scale_value,
    DefaultValuedAttr<F32Attr, "0.">:$dropout_scale_value
  );
}

def OneFlow_FusedScaleMaskSoftmaxGradOp : OneFlow_BaseOp<"fused_scale_mask_softmax_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$y,
    AnyType:$dy,
    AnyType:$mask
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$scale_value
  );
}

def OneFlow_FusedScaleTrilOp : OneFlow_BaseOp<"fused_scale_tril", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
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
}

def OneFlow_FusedSelfAttentionQueryMulKeyAndValueOp : OneFlow_BaseOp<"fused_self_attention_query_mul_key_and_value", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$hidden_states
  );
  let output = (outs
    AnyType:$query_mul_key,
    AnyType:$value
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$head_size,
    DefaultValuedAttr<F32Attr, "0.">:$alpha
  );
}

def OneFlow_FusedSelfAttentionQueryMulKeyAndValueGradOp : OneFlow_BaseOp<"fused_self_attention_query_mul_key_and_value_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$query_mul_key_grad,
    AnyType:$value_grad,
    AnyType:$hidden_states
  );
  let output = (outs
    AnyType:$hidden_states_grad
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$alpha
  );
}

def OneFlow_FusedTrilScaleSoftmaxMaskScaleOp : OneFlow_BaseOp<"fused_tril_scale_softmax_mask_scale", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$mask
  );
  let output = (outs
    AnyType:$y,
    AnyType:$softmax_y
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$diagonal,
    DefaultValuedAttr<F32Attr, "0.">:$tril_fill_value,
    DefaultValuedAttr<F32Attr, "1.">:$tril_scale_value,
    DefaultValuedAttr<F32Attr, "1.">:$mask_scale_value
  );
}

def OneFlow_FusedTrilScaleSoftmaxMaskScaleGradOp : OneFlow_BaseOp<"fused_tril_scale_softmax_mask_scale_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$softmax_y,
    AnyType:$dy,
    AnyType:$mask
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$diagonal,
    DefaultValuedAttr<F32Attr, "0.">:$tril_scale_value,
    DefaultValuedAttr<F32Attr, "0.">:$mask_scale_value
  );
}

def OneFlow_NormalizationAddReluGradOp : OneFlow_BaseOp<"normalization_add_relu_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
    DefaultValuedAttr<SI32Attr, "0">:$axis,
    DefaultValuedAttr<F32Attr, "0.">:$epsilon
  );
}

#endif // GET_ONEFLOW_FUSED_OP_DEFINITIONS

// Group: IDEMPOTENT
// abs, ceil, floor, ones_like, relu, rint, round, sign
// Total: 8

#ifdef GET_ONEFLOW_IDEMPOTENT_OP_DEFINITIONS

def OneFlow_AbsOp : OneFlow_IdempotentBaseOp<"abs", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
}

def OneFlow_CeilOp : OneFlow_IdempotentBaseOp<"ceil", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
}

def OneFlow_FloorOp : OneFlow_IdempotentBaseOp<"floor", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
}

def OneFlow_OnesLikeOp : OneFlow_IdempotentBaseOp<"ones_like", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
}

def OneFlow_ReluOp : OneFlow_IdempotentBaseOp<"relu", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
}

def OneFlow_RintOp : OneFlow_IdempotentBaseOp<"rint", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
}

def OneFlow_RoundOp : OneFlow_IdempotentBaseOp<"round", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
}

def OneFlow_SignOp : OneFlow_IdempotentBaseOp<"sign", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
}

#endif // GET_ONEFLOW_IDEMPOTENT_OP_DEFINITIONS

// Group: IDENTITY
// amp_white_identity, identity, identity_buffer, tuple_identity
// Total: 4

#ifdef GET_ONEFLOW_IDENTITY_OP_DEFINITIONS

def OneFlow_AmpWhiteIdentityOp : OneFlow_BaseOp<"amp_white_identity", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_IdentityOp : OneFlow_BaseOp<"identity", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_IdentityBufferOp : OneFlow_BaseOp<"identity_buffer", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$buffer_size
  );
}

def OneFlow_TupleIdentityOp : OneFlow_BaseOp<"tuple_identity", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    Variadic<AnyType>:$in
  );
  let output = (outs
    Variadic<AnyType>:$out
  );
}

#endif // GET_ONEFLOW_IDENTITY_OP_DEFINITIONS

// Group: IMAGE
// image_batch_align, image_decode, image_flip, image_random_crop, image_resize_keep_aspect_ratio, image_resize_to_fixed
// Total: 6

#ifdef GET_ONEFLOW_IMAGE_OP_DEFINITIONS

def OneFlow_ImageBatchAlignOp : OneFlow_BaseOp<"image_batch_align", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    AnyI64ElementsAttr:$shape,
    StrAttr:$data_type,
    DefaultValuedAttr<SI32Attr, "0">:$alignment,
    DefaultValuedAttr<BoolAttr, "false">:$dynamic_out
  );
}

def OneFlow_ImageDecodeOp : OneFlow_BaseOp<"image_decode", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<StrAttr, "\"BGR\"">:$color_space,
    StrAttr:$data_type
  );
}

def OneFlow_ImageFlipOp : OneFlow_BaseOp<"image_flip", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in,
    AnyType:$flip_code
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_ImageRandomCropOp : OneFlow_BaseOp<"image_random_crop", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "10">:$num_attempts,
    DefaultValuedAttr<SI64Attr, "-1">:$seed,
    DefaultValuedAttr<BoolAttr, "false">:$has_seed,
    F32ArrayAttr:$random_area,
    F32ArrayAttr:$random_aspect_ratio
  );
}

def OneFlow_ImageResizeKeepAspectRatioOp : OneFlow_BaseOp<"image_resize_keep_aspect_ratio", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out,
    AnyType:$size,
    AnyType:$scale
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$target_size,
    DefaultValuedAttr<SI32Attr, "0">:$min_size,
    DefaultValuedAttr<SI32Attr, "0">:$max_size,
    DefaultValuedAttr<BoolAttr, "false">:$resize_longer,
    DefaultValuedAttr<StrAttr, "\"bilinear\"">:$interpolation_type
  );
}

def OneFlow_ImageResizeToFixedOp : OneFlow_BaseOp<"image_resize_to_fixed", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out,
    AnyType:$scale
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$target_width,
    DefaultValuedAttr<SI64Attr, "0">:$target_height,
    DefaultValuedAttr<SI64Attr, "3">:$channels,
    StrAttr:$data_type,
    DefaultValuedAttr<StrAttr, "\"bilinear\"">:$interpolation_type
  );
}

#endif // GET_ONEFLOW_IMAGE_OP_DEFINITIONS

// Group: INDICES
// arg_sort, argmax, argwhere, batch_gather, dim_gather, dim_scatter_add, dim_scatter_add_like, dim_scatter_add_scalar, dim_scatter_mul, dim_scatter_mul_scalar, dim_scatter_update, dim_scatter_update_scalar, gather, gather_nd, generate_random_batch_permutation_indices, image_target_resize, logical_slice, scatter_nd, scatter_nd_like, slice, slice_grad, tensor_scatter_nd_add, tensor_scatter_nd_update, unsorted_batch_segment_sum, unsorted_segment_sum, unsorted_segment_sum_like, where, where_scalar_x, where_scalar_xy, where_scalar_y
// Total: 30

#ifdef GET_ONEFLOW_INDICES_OP_DEFINITIONS

def OneFlow_ArgSortOp : OneFlow_BaseOp<"arg_sort", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ArgmaxOp : OneFlow_BaseOp<"argmax", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_ArgwhereOp : OneFlow_BaseOp<"argwhere", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_BatchGatherOp : OneFlow_BaseOp<"batch_gather", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in,
    AnyType:$indices
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_DimGatherOp : OneFlow_BaseOp<"dim_gather", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$input,
    AnyType:$index
  );
  let output = (outs
    AnyType:$output
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$dim
  );
}

def OneFlow_DimScatterAddOp : OneFlow_BaseOp<"dim_scatter_add", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$input,
    AnyType:$index,
    AnyType:$src
  );
  let output = (outs
    AnyType:$output
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$dim
  );
}

def OneFlow_DimScatterAddLikeOp : OneFlow_BaseOp<"dim_scatter_add_like", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$like,
    AnyType:$index,
    AnyType:$src
  );
  let output = (outs
    AnyType:$output
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$dim
  );
}

def OneFlow_DimScatterAddScalarOp : OneFlow_BaseOp<"dim_scatter_add_scalar", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$input,
    AnyType:$index
  );
  let output = (outs
    AnyType:$output
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$src_scalar,
    DefaultValuedAttr<SI32Attr, "0">:$dim
  );
}

def OneFlow_DimScatterMulOp : OneFlow_BaseOp<"dim_scatter_mul", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$input,
    AnyType:$index,
    AnyType:$src
  );
  let output = (outs
    AnyType:$output
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$dim
  );
}

def OneFlow_DimScatterMulScalarOp : OneFlow_BaseOp<"dim_scatter_mul_scalar", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$input,
    AnyType:$index
  );
  let output = (outs
    AnyType:$output
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$src_scalar,
    DefaultValuedAttr<SI32Attr, "0">:$dim
  );
}

def OneFlow_DimScatterUpdateOp : OneFlow_BaseOp<"dim_scatter_update", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$input,
    AnyType:$index,
    AnyType:$src
  );
  let output = (outs
    AnyType:$output
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$dim
  );
}

def OneFlow_DimScatterUpdateScalarOp : OneFlow_BaseOp<"dim_scatter_update_scalar", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$input,
    AnyType:$index
  );
  let output = (outs
    AnyType:$output
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$src_scalar,
    DefaultValuedAttr<SI32Attr, "0">:$dim
  );
}

def OneFlow_GatherOp : OneFlow_BaseOp<"gather", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in,
    AnyType:$indices
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$axis
  );
}

def OneFlow_GatherNdOp : OneFlow_BaseOp<"gather_nd", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$params,
    AnyType:$indices
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_GenerateRandomBatchPermutationIndicesOp : OneFlow_BaseOp<"generate_random_batch_permutation_indices", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$seed
  );
}

def OneFlow_ImageTargetResizeOp : OneFlow_BaseOp<"image_target_resize", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out,
    AnyType:$size,
    AnyType:$scale
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$target_size,
    DefaultValuedAttr<SI32Attr, "0">:$max_size
  );
}

def OneFlow_LogicalSliceOp : OneFlow_BaseOp<"logical_slice", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ScatterNdOp : OneFlow_BaseOp<"scatter_nd", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ScatterNdLikeOp : OneFlow_BaseOp<"scatter_nd_like", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$like,
    AnyType:$indices,
    AnyType:$updates
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_SliceOp : OneFlow_BaseOp<"slice", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_SliceGradOp : OneFlow_BaseOp<"slice_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_TensorScatterNdAddOp : OneFlow_BaseOp<"tensor_scatter_nd_add", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$params,
    AnyType:$updates,
    AnyType:$indices
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_TensorScatterNdUpdateOp : OneFlow_BaseOp<"tensor_scatter_nd_update", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$params,
    AnyType:$updates,
    AnyType:$indices
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_UnsortedBatchSegmentSumOp : OneFlow_BaseOp<"unsorted_batch_segment_sum", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$data,
    AnyType:$segment_ids
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$num_segments
  );
}

def OneFlow_UnsortedSegmentSumOp : OneFlow_BaseOp<"unsorted_segment_sum", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$data,
    AnyType:$segment_ids
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$axis,
    DefaultValuedAttr<SI64Attr, "0">:$num_segments
  );
}

def OneFlow_UnsortedSegmentSumLikeOp : OneFlow_BaseOp<"unsorted_segment_sum_like", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$data,
    AnyType:$segment_ids,
    AnyType:$like
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$axis
  );
}

def OneFlow_WhereOp : OneFlow_BaseOp<"where", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$condition,
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_WhereScalarXOp : OneFlow_BaseOp<"where_scalar_x", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$condition,
    AnyType:$y
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$has_int_operand,
    DefaultValuedAttr<BoolAttr, "false">:$has_float_operand,
    DefaultValuedAttr<SI64Attr, "0">:$int_operand,
    DefaultValuedAttr<F64Attr, "0.">:$float_operand
  );
}

def OneFlow_WhereScalarXyOp : OneFlow_BaseOp<"where_scalar_xy", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$condition
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$has_x_int_operand,
    DefaultValuedAttr<BoolAttr, "false">:$has_x_float_operand,
    DefaultValuedAttr<BoolAttr, "false">:$has_y_int_operand,
    DefaultValuedAttr<BoolAttr, "false">:$has_y_float_operand,
    DefaultValuedAttr<SI64Attr, "0">:$x_int_operand,
    DefaultValuedAttr<F64Attr, "0.">:$x_float_operand,
    DefaultValuedAttr<SI64Attr, "0">:$y_int_operand,
    DefaultValuedAttr<F64Attr, "0.">:$y_float_operand
  );
}

def OneFlow_WhereScalarYOp : OneFlow_BaseOp<"where_scalar_y", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$condition,
    AnyType:$x
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$has_int_operand,
    DefaultValuedAttr<BoolAttr, "false">:$has_float_operand,
    DefaultValuedAttr<SI64Attr, "0">:$int_operand,
    DefaultValuedAttr<F64Attr, "0.">:$float_operand
  );
}

#endif // GET_ONEFLOW_INDICES_OP_DEFINITIONS

// Group: INVOLUTION
// negative, reciprocal
// Total: 2

#ifdef GET_ONEFLOW_INVOLUTION_OP_DEFINITIONS

def OneFlow_NegativeOp : OneFlow_InvolutionBaseOp<"negative", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
}

def OneFlow_ReciprocalOp : OneFlow_InvolutionBaseOp<"reciprocal", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
}

#endif // GET_ONEFLOW_INVOLUTION_OP_DEFINITIONS

// Group: LOSS
// combined_margin_loss, combined_margin_loss_grad, ctc_loss, ctc_loss_grad, dynamic_loss_scale_schedule, kl_div_loss, kl_div_loss_grad, smooth_l1_loss, smooth_l1_loss_grad
// Total: 9

#ifdef GET_ONEFLOW_LOSS_OP_DEFINITIONS

def OneFlow_CombinedMarginLossOp : OneFlow_BaseOp<"combined_margin_loss", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$label
  );
  let output = (outs
    AnyType:$y,
    AnyType:$theta
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$m1,
    DefaultValuedAttr<F32Attr, "0.">:$m2,
    DefaultValuedAttr<F32Attr, "0.">:$m3,
    DefaultValuedAttr<SI64Attr, "0">:$depth
  );
}

def OneFlow_CombinedMarginLossGradOp : OneFlow_BaseOp<"combined_margin_loss_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$dy,
    AnyType:$label,
    AnyType:$theta
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$m1,
    DefaultValuedAttr<F32Attr, "0.">:$m2,
    DefaultValuedAttr<F32Attr, "0.">:$m3,
    DefaultValuedAttr<SI64Attr, "0">:$depth
  );
}

def OneFlow_CtcLossOp : OneFlow_BaseOp<"ctc_loss", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
    DefaultValuedAttr<SI64Attr, "0">:$max_target_length,
    DefaultValuedAttr<SI32Attr, "0">:$blank,
    DefaultValuedAttr<BoolAttr, "false">:$zero_infinity
  );
}

def OneFlow_CtcLossGradOp : OneFlow_BaseOp<"ctc_loss_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
    DefaultValuedAttr<SI64Attr, "0">:$max_target_length,
    DefaultValuedAttr<SI32Attr, "0">:$blank,
    DefaultValuedAttr<BoolAttr, "false">:$zero_infinity
  );
}

def OneFlow_DynamicLossScaleScheduleOp : OneFlow_BaseOp<"dynamic_loss_scale_schedule", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$count_not_finite,
    AnyType:$loss_scale,
    AnyType:$good_step_counter
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "2000">:$increment_period,
    DefaultValuedAttr<F32Attr, "2.">:$multiplier
  );
}

def OneFlow_KlDivLossOp : OneFlow_BaseOp<"kl_div_loss", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$input,
    AnyType:$target
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    StrAttr:$reduction,
    DefaultValuedAttr<BoolAttr, "false">:$log_target
  );
}

def OneFlow_KlDivLossGradOp : OneFlow_BaseOp<"kl_div_loss_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
    DefaultValuedAttr<BoolAttr, "false">:$log_target
  );
}

def OneFlow_SmoothL1LossOp : OneFlow_BaseOp<"smooth_l1_loss", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$input,
    AnyType:$target
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    StrAttr:$reduction,
    DefaultValuedAttr<F32Attr, "0.">:$beta
  );
}

def OneFlow_SmoothL1LossGradOp : OneFlow_BaseOp<"smooth_l1_loss_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
    DefaultValuedAttr<F32Attr, "0.">:$beta
  );
}

#endif // GET_ONEFLOW_LOSS_OP_DEFINITIONS

// Group: MATH
// abs_grad, ceil_grad, erf, erf_grad, exp, exp_grad, expand_grad, expm1, expm1_grad, floor_grad, floordiv_x_grad, floordiv_y_grad, lgamma, lgamma_grad, log, log1p, log1p_grad, log_grad, log_sigmoid, log_sigmoid_grad, negative_grad, reciprocal_grad, reciprocal_no_nan, reciprocal_no_nan_grad, rint_grad, round_grad, rsqrt, rsqrt_grad, sigmoid_v2, sigmoid_v2_grad, sign_grad, softplus, softplus_grad, softsign_grad, sqrt, sqrt_grad, square, square_grad, xlogy_x_grad, xlogy_y_grad
// Total: 40

#ifdef GET_ONEFLOW_MATH_OP_DEFINITIONS

def OneFlow_AbsGradOp : OneFlow_BaseOp<"abs_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_CeilGradOp : OneFlow_BaseOp<"ceil_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_ErfOp : OneFlow_BaseOp<"erf", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_ErfGradOp : OneFlow_BaseOp<"erf_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_ExpOp : OneFlow_BaseOp<"exp", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_ExpGradOp : OneFlow_BaseOp<"exp_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_ExpandGradOp : OneFlow_BaseOp<"expand_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_Expm1Op : OneFlow_BaseOp<"expm1", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_Expm1GradOp : OneFlow_BaseOp<"expm1_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_FloorGradOp : OneFlow_BaseOp<"floor_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_FloordivXGradOp : OneFlow_BaseOp<"floordiv_x_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y,
    AnyType:$dz
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_FloordivYGradOp : OneFlow_BaseOp<"floordiv_y_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y,
    AnyType:$dz
  );
  let output = (outs
    AnyType:$dy
  );
}

def OneFlow_LgammaOp : OneFlow_BaseOp<"lgamma", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_LgammaGradOp : OneFlow_BaseOp<"lgamma_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_LogOp : OneFlow_BaseOp<"log", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_Log1pOp : OneFlow_BaseOp<"log1p", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_Log1pGradOp : OneFlow_BaseOp<"log1p_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_LogGradOp : OneFlow_BaseOp<"log_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_LogSigmoidOp : OneFlow_BaseOp<"log_sigmoid", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_LogSigmoidGradOp : OneFlow_BaseOp<"log_sigmoid_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_NegativeGradOp : OneFlow_BaseOp<"negative_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_ReciprocalGradOp : OneFlow_BaseOp<"reciprocal_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_ReciprocalNoNanOp : OneFlow_BaseOp<"reciprocal_no_nan", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_ReciprocalNoNanGradOp : OneFlow_BaseOp<"reciprocal_no_nan_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_RintGradOp : OneFlow_BaseOp<"rint_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_RoundGradOp : OneFlow_BaseOp<"round_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_RsqrtOp : OneFlow_BaseOp<"rsqrt", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_RsqrtGradOp : OneFlow_BaseOp<"rsqrt_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_SigmoidV2Op : OneFlow_BaseOp<"sigmoid_v2", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_SigmoidV2GradOp : OneFlow_BaseOp<"sigmoid_v2_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_SignGradOp : OneFlow_BaseOp<"sign_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_SoftplusOp : OneFlow_BaseOp<"softplus", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_SoftplusGradOp : OneFlow_BaseOp<"softplus_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_SoftsignGradOp : OneFlow_BaseOp<"softsign_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_SqrtOp : OneFlow_BaseOp<"sqrt", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_SqrtGradOp : OneFlow_BaseOp<"sqrt_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_SquareOp : OneFlow_BaseOp<"square", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_SquareGradOp : OneFlow_BaseOp<"square_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_XlogyXGradOp : OneFlow_BaseOp<"xlogy_x_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y,
    AnyType:$dz
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_XlogyYGradOp : OneFlow_BaseOp<"xlogy_y_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y,
    AnyType:$dz
  );
  let output = (outs
    AnyType:$dy
  );
}

#endif // GET_ONEFLOW_MATH_OP_DEFINITIONS

// Group: MATMUL
// batch_matmul, broadcast_matmul, broadcast_matmul_grad_b, distributed_partial_fc_sample, distributed_partial_fc_sample_disable_boxing, erfc, erfc_grad, matmul
// Total: 8

#ifdef GET_ONEFLOW_MATMUL_OP_DEFINITIONS

def OneFlow_BatchMatmulOp : OneFlow_BaseOp<"batch_matmul", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$a,
    AnyType:$b,
    Optional<AnyType>:$_add_to_output
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$transpose_a,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_b,
    DefaultValuedAttr<F64Attr, "1.">:$alpha
  );
}

def OneFlow_BroadcastMatmulOp : OneFlow_BaseOp<"broadcast_matmul", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$a,
    AnyType:$b,
    Optional<AnyType>:$_add_to_output
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$transpose_a,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_b,
    DefaultValuedAttr<F64Attr, "1.">:$alpha
  );
}

def OneFlow_BroadcastMatmulGradBOp : OneFlow_BaseOp<"broadcast_matmul_grad_b", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$a,
    AnyType:$b,
    Optional<AnyType>:$_add_to_output
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "1.">:$alpha
  );
}

def OneFlow_DistributedPartialFcSampleOp : OneFlow_BaseOp<"distributed_partial_fc_sample", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
    DefaultValuedAttr<SI64Attr, "0">:$num_sample,
    DefaultValuedAttr<SI64Attr, "-1">:$seed
  );
}

def OneFlow_DistributedPartialFcSampleDisableBoxingOp : OneFlow_BaseOp<"distributed_partial_fc_sample_disable_boxing", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$sampled_weight_diff,
    AnyType:$sampled_label
  );
  let output = (outs
    AnyType:$boxing_disabled_sampled_weight_diff,
    AnyType:$boxing_disabled_sampled_label
  );
}

def OneFlow_ErfcOp : OneFlow_BaseOp<"erfc", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_ErfcGradOp : OneFlow_BaseOp<"erfc_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_MatmulOp : OneFlow_BaseOp<"matmul", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$a,
    AnyType:$b,
    Optional<AnyType>:$_add_to_output
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$transpose_a,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_b,
    DefaultValuedAttr<F64Attr, "1.">:$alpha
  );
}

#endif // GET_ONEFLOW_MATMUL_OP_DEFINITIONS

// Group: MISC
// CategoricalOrdinalEncode, add_n, arange, coin_flip, concat, constant, dropout, elementwise_maximum_backward, elementwise_minimum_backward, empty, eye, grid_sample_grad, multi_count_not_finite, multi_square_sum, nll, nll_grad, pow_x_grad, pow_y_grad, prelu_grad, randperm, recv, send, split_like, ssp_variable_proxy, tf_prelu_grad, uniform, uniform_int, unique_with_counts, xdivy_x_grad, xdivy_y_grad
// Total: 30

#ifdef GET_ONEFLOW_MISC_OP_DEFINITIONS

def OneFlow_CategoricalOrdinalEncodeOp : OneFlow_BaseOp<"CategoricalOrdinalEncode", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$table,
    AnyType:$size,
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$hash_precomputed
  );
}

def OneFlow_AddNOp : OneFlow_BaseOp<"add_n", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    Variadic<AnyType>:$in
  );
  let output = (outs
    AnyType:$out
  );
  let hasCanonicalizer = 1;
}

def OneFlow_ArangeOp : OneFlow_BaseOp<"arange", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$integer_start,
    DefaultValuedAttr<SI64Attr, "0">:$integer_delta,
    DefaultValuedAttr<SI64Attr, "0">:$integer_limit,
    DefaultValuedAttr<F64Attr, "0.">:$float_start,
    DefaultValuedAttr<F64Attr, "0.">:$float_delta,
    DefaultValuedAttr<F64Attr, "0.">:$float_limit,
    StrAttr:$dtype,
    StrArrayAttr:$nd_sbp
  );
}

def OneFlow_CoinFlipOp : OneFlow_BaseOp<"coin_flip", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.5">:$probability,
    DefaultValuedAttr<SI64Attr, "0">:$batch_size,
    DefaultValuedAttr<SI64Attr, "-1">:$seed,
    DefaultValuedAttr<BoolAttr, "false">:$has_seed,
    StrArrayAttr:$nd_sbp
  );
}

def OneFlow_ConcatOp : OneFlow_BaseOp<"concat", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    Variadic<AnyType>:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$axis,
    DefaultValuedAttr<SI64Attr, "0">:$max_dim_size
  );
}

def OneFlow_ConstantOp : OneFlow_BaseOp<"constant", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.">:$floating_value,
    DefaultValuedAttr<SI64Attr, "0">:$integer_value,
    DefaultValuedAttr<BoolAttr, "false">:$is_floating_value,
    StrAttr:$dtype,
    AnyI64ElementsAttr:$shape,
    StrArrayAttr:$nd_sbp
  );
}

def OneFlow_DropoutOp : OneFlow_BaseOp<"dropout", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in,
    AnyType:$mask,
    Optional<AnyType>:$_add_to_output
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$scale
  );
}

def OneFlow_ElementwiseMaximumBackwardOp : OneFlow_BaseOp<"elementwise_maximum_backward", [NoSideEffect, AttrSizedResultSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
}

def OneFlow_ElementwiseMinimumBackwardOp : OneFlow_BaseOp<"elementwise_minimum_backward", [NoSideEffect, AttrSizedResultSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
}

def OneFlow_EmptyOp : OneFlow_BaseOp<"empty", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    StrAttr:$dtype,
    AnyI64ElementsAttr:$shape,
    StrArrayAttr:$nd_sbp
  );
}

def OneFlow_EyeOp : OneFlow_BaseOp<"eye", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$rows,
    DefaultValuedAttr<SI64Attr, "0">:$cols,
    StrAttr:$dtype,
    StrArrayAttr:$nd_sbp
  );
}

def OneFlow_GridSampleGradOp : OneFlow_BaseOp<"grid_sample_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
    DefaultValuedAttr<BoolAttr, "false">:$align_corners
  );
}

def OneFlow_MultiCountNotFiniteOp : OneFlow_BaseOp<"multi_count_not_finite", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    Variadic<AnyType>:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_MultiSquareSumOp : OneFlow_BaseOp<"multi_square_sum", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    Variadic<AnyType>:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_NllOp : OneFlow_BaseOp<"nll", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
    DefaultValuedAttr<SI64Attr, "0">:$ignore_index,
    StrAttr:$reduction
  );
}

def OneFlow_NllGradOp : OneFlow_BaseOp<"nll_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
    DefaultValuedAttr<SI64Attr, "0">:$ignore_index,
    StrAttr:$reduction
  );
}

def OneFlow_PowXGradOp : OneFlow_BaseOp<"pow_x_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y,
    AnyType:$dz
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_PowYGradOp : OneFlow_BaseOp<"pow_y_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y,
    AnyType:$dz
  );
  let output = (outs
    AnyType:$dy
  );
}

def OneFlow_PreluGradOp : OneFlow_BaseOp<"prelu_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_RandpermOp : OneFlow_BaseOp<"randperm", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$n,
    DefaultValuedAttr<SI64Attr, "0">:$seed,
    StrAttr:$nd_sbp
  );
}

def OneFlow_RecvOp : OneFlow_BaseOp<"recv", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$src_process_id,
    StrAttr:$dtype,
    AnyI64ElementsAttr:$shape,
    StrAttr:$device_type,
    DefaultValuedAttr<SI64Attr, "0">:$device_id
  );
}

def OneFlow_SendOp : OneFlow_BaseOp<"send", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$dst_process_id
  );
}

def OneFlow_SplitLikeOp : OneFlow_BaseOp<"split_like", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in,
    Variadic<AnyType>:$like
  );
  let output = (outs
    Variadic<AnyType>:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$axis
  );
}

def OneFlow_SspVariableProxyOp : OneFlow_BaseOp<"ssp_variable_proxy", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$var
  );
  let output = (outs
    AnyType:$ref,
    AnyType:$value
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "1">:$buffer_size
  );
}

def OneFlow_TfPreluGradOp : OneFlow_BaseOp<"tf_prelu_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_UniformOp : OneFlow_BaseOp<"uniform", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.">:$from,
    DefaultValuedAttr<F64Attr, "1.">:$to,
    DefaultValuedAttr<SI64Attr, "0">:$seed,
    StrAttr:$dtype,
    AnyI64ElementsAttr:$shape,
    StrAttr:$nd_sbp
  );
}

def OneFlow_UniformIntOp : OneFlow_BaseOp<"uniform_int", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$from,
    DefaultValuedAttr<SI64Attr, "1">:$to,
    DefaultValuedAttr<SI64Attr, "0">:$seed,
    StrAttr:$dtype,
    AnyI64ElementsAttr:$shape,
    StrAttr:$nd_sbp
  );
}

def OneFlow_UniqueWithCountsOp : OneFlow_BaseOp<"unique_with_counts", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_XdivyXGradOp : OneFlow_BaseOp<"xdivy_x_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y,
    AnyType:$dz
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_XdivyYGradOp : OneFlow_BaseOp<"xdivy_y_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y,
    AnyType:$dz
  );
  let output = (outs
    AnyType:$dy
  );
}

#endif // GET_ONEFLOW_MISC_OP_DEFINITIONS

// Group: NCCL
// _nccl_logical_2D_same_dim0_all2all, _nccl_logical_2D_same_dim0_all_gather, _nccl_logical_2D_same_dim0_all_gather_noncontinuous, _nccl_logical_2D_same_dim0_all_reduce, _nccl_logical_2D_same_dim1_all_reduce, _nccl_logical_all_gather, _nccl_logical_all_gather_noncontinuous, _nccl_logical_all_reduce, _nccl_logical_reduce_scatter, _nccl_logical_s2s
// Total: 10

#ifdef GET_ONEFLOW_NCCL_OP_DEFINITIONS

def OneFlow__ncclLogical_2DSameDim0All2allOp : OneFlow_BaseOp<"_nccl_logical_2D_same_dim0_all2all", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "-1">:$in_dim1_split_axis,
    DefaultValuedAttr<SI64Attr, "-1">:$out_dim1_split_axis
  );
}

def OneFlow__ncclLogical_2DSameDim0AllGatherOp : OneFlow_BaseOp<"_nccl_logical_2D_same_dim0_all_gather", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow__ncclLogical_2DSameDim0AllGatherNoncontinuousOp : OneFlow_BaseOp<"_nccl_logical_2D_same_dim0_all_gather_noncontinuous", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "-1">:$in_dim1_split_axis
  );
}

def OneFlow__ncclLogical_2DSameDim0AllReduceOp : OneFlow_BaseOp<"_nccl_logical_2D_same_dim0_all_reduce", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow__ncclLogical_2DSameDim1AllReduceOp : OneFlow_BaseOp<"_nccl_logical_2D_same_dim1_all_reduce", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow__ncclLogicalAllGatherOp : OneFlow_BaseOp<"_nccl_logical_all_gather", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow__ncclLogicalAllGatherNoncontinuousOp : OneFlow_BaseOp<"_nccl_logical_all_gather_noncontinuous", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "-1">:$in_split_axis
  );
}

def OneFlow__ncclLogicalAllReduceOp : OneFlow_BaseOp<"_nccl_logical_all_reduce", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow__ncclLogicalReduceScatterOp : OneFlow_BaseOp<"_nccl_logical_reduce_scatter", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow__ncclLogicalS2sOp : OneFlow_BaseOp<"_nccl_logical_s2s", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "-1">:$in_split_axis,
    DefaultValuedAttr<SI64Attr, "-1">:$out_split_axis
  );
}

#endif // GET_ONEFLOW_NCCL_OP_DEFINITIONS

// Group: NORMALIZATION
// crop_mirror_normalize_from_tensorbuffer, crop_mirror_normalize_from_uint8, image_normalize, l2_normalize, l2_normalize_grad, layer_norm, layer_norm_grad, layer_norm_param_grad, normal, normalization, normalization_grad
// Total: 11

#ifdef GET_ONEFLOW_NORMALIZATION_OP_DEFINITIONS

def OneFlow_CropMirrorNormalizeFromTensorbufferOp : OneFlow_BaseOp<"crop_mirror_normalize_from_tensorbuffer", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in,
    Optional<AnyType>:$mirror
  );
  let output = (outs
    AnyType:$out
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
    StrAttr:$output_dtype
  );
}

def OneFlow_CropMirrorNormalizeFromUint8Op : OneFlow_BaseOp<"crop_mirror_normalize_from_uint8", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in,
    Optional<AnyType>:$mirror
  );
  let output = (outs
    AnyType:$out
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
    StrAttr:$output_dtype
  );
}

def OneFlow_ImageNormalizeOp : OneFlow_BaseOp<"image_normalize", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_L2NormalizeOp : OneFlow_BaseOp<"l2_normalize", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y,
    AnyType:$square_x_sum
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$axis,
    DefaultValuedAttr<F32Attr, "0.">:$epsilon
  );
}

def OneFlow_L2NormalizeGradOp : OneFlow_BaseOp<"l2_normalize_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$dy,
    AnyType:$y,
    AnyType:$square_x_sum
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$axis,
    DefaultValuedAttr<F32Attr, "0.">:$epsilon
  );
}

def OneFlow_LayerNormOp : OneFlow_BaseOp<"layer_norm", [NoSideEffect, AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
    DefaultValuedAttr<BoolAttr, "false">:$center,
    DefaultValuedAttr<BoolAttr, "false">:$scale,
    DefaultValuedAttr<SI64Attr, "0">:$begin_norm_axis,
    DefaultValuedAttr<SI64Attr, "0">:$begin_params_axis,
    DefaultValuedAttr<F64Attr, "0.">:$epsilon
  );
  let trait_attrs = (ins
    I32ElementsAttr:$operand_segment_sizes
  );
}

def OneFlow_LayerNormGradOp : OneFlow_BaseOp<"layer_norm_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
    DefaultValuedAttr<SI64Attr, "0">:$begin_norm_axis,
    DefaultValuedAttr<F64Attr, "0.">:$epsilon
  );
}

def OneFlow_LayerNormParamGradOp : OneFlow_BaseOp<"layer_norm_param_grad", [NoSideEffect, AttrSizedOperandSegments, AttrSizedResultSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
    DefaultValuedAttr<SI64Attr, "0">:$begin_params_axis
  );
  let trait_attrs = (ins
    I32ElementsAttr:$operand_segment_sizes,
    I32ElementsAttr:$result_segment_sizes
  );
}

def OneFlow_NormalOp : OneFlow_BaseOp<"normal", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.">:$mean,
    DefaultValuedAttr<F64Attr, "1.">:$std,
    DefaultValuedAttr<SI64Attr, "0">:$seed,
    StrAttr:$dtype,
    AnyI64ElementsAttr:$shape,
    StrAttr:$nd_sbp
  );
}

def OneFlow_NormalizationOp : OneFlow_BaseOp<"normalization", [NoSideEffect, AttrSizedOperandSegments, AttrSizedResultSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
    DefaultValuedAttr<SI32Attr, "0">:$axis,
    DefaultValuedAttr<F32Attr, "0.">:$epsilon,
    DefaultValuedAttr<BoolAttr, "false">:$training,
    DefaultValuedAttr<F32Attr, "0.">:$momentum
  );
  let trait_attrs = (ins
    I32ElementsAttr:$operand_segment_sizes,
    I32ElementsAttr:$result_segment_sizes
  );
}

def OneFlow_NormalizationGradOp : OneFlow_BaseOp<"normalization_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
    DefaultValuedAttr<SI32Attr, "0">:$axis,
    DefaultValuedAttr<F32Attr, "0.">:$epsilon
  );
}

#endif // GET_ONEFLOW_NORMALIZATION_OP_DEFINITIONS

// Group: OPTIMIZER
// adagrad_update, adam_bias_correction_factor, adam_update, indexed_slices_adam_update, indexed_slices_momentum_update, indexed_slices_sgd_update, lamb_update, lars_update, momentum_update, rmsprop_update, sgd_update, slice_update
// Total: 12

#ifdef GET_ONEFLOW_OPTIMIZER_OP_DEFINITIONS

def OneFlow_AdagradUpdateOp : OneFlow_BaseOp<"adagrad_update", [AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
    DefaultValuedAttr<SI32Attr, "0">:$train_step_val,
    DefaultValuedAttr<F32Attr, "0.">:$learning_rate_val,
    DefaultValuedAttr<F64Attr, "1.">:$scale,
    DefaultValuedAttr<F32Attr, "0.">:$l1,
    DefaultValuedAttr<F32Attr, "0.">:$l2,
    DefaultValuedAttr<F32Attr, "0.">:$lr_decay,
    DefaultValuedAttr<F32Attr, "0.">:$weight_decay,
    DefaultValuedAttr<F32Attr, "0.">:$epsilon
  );
  let trait_attrs = (ins
    I32ElementsAttr:$operand_segment_sizes
  );
}

def OneFlow_AdamBiasCorrectionFactorOp : OneFlow_BaseOp<"adam_bias_correction_factor", [DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$train_step
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.9">:$beta
  );
}

def OneFlow_AdamUpdateOp : OneFlow_BaseOp<"adam_update", [AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
    DefaultValuedAttr<F32Attr, "0.">:$learning_rate_val,
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
    I32ElementsAttr:$operand_segment_sizes
  );
}

def OneFlow_IndexedSlicesAdamUpdateOp : OneFlow_BaseOp<"indexed_slices_adam_update", [AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
    DefaultValuedAttr<F32Attr, "0.">:$learning_rate_val,
    DefaultValuedAttr<F32Attr, "0.9">:$beta1,
    DefaultValuedAttr<F32Attr, "0.999">:$beta2,
    DefaultValuedAttr<F32Attr, "0.">:$epsilon,
    DefaultValuedAttr<F32Attr, "0.">:$weight_decay,
    DefaultValuedAttr<BoolAttr, "false">:$amsgrad,
    DefaultValuedAttr<BoolAttr, "true">:$do_bias_correction
  );
  let trait_attrs = (ins
    I32ElementsAttr:$operand_segment_sizes
  );
}

def OneFlow_IndexedSlicesMomentumUpdateOp : OneFlow_BaseOp<"indexed_slices_momentum_update", [DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$model,
    AnyType:$model_diff_indices,
    AnyType:$model_diff_values,
    AnyType:$learning_rate,
    AnyType:$momentum
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.9">:$beta,
    DefaultValuedAttr<F32Attr, "0.">:$weight_decay
  );
}

def OneFlow_IndexedSlicesSgdUpdateOp : OneFlow_BaseOp<"indexed_slices_sgd_update", [DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$model,
    AnyType:$model_diff_indices,
    AnyType:$model_diff_values,
    AnyType:$learning_rate
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$weight_decay
  );
}

def OneFlow_LambUpdateOp : OneFlow_BaseOp<"lamb_update", [AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
    DefaultValuedAttr<F32Attr, "0.">:$beta1,
    DefaultValuedAttr<F32Attr, "0.">:$beta2,
    DefaultValuedAttr<F32Attr, "0.">:$epsilon,
    DefaultValuedAttr<F64Attr, "1.">:$scale,
    DefaultValuedAttr<F32Attr, "0.">:$l1,
    DefaultValuedAttr<F32Attr, "0.">:$l2,
    DefaultValuedAttr<F32Attr, "0.">:$weight_decay
  );
  let trait_attrs = (ins
    I32ElementsAttr:$operand_segment_sizes
  );
}

def OneFlow_LarsUpdateOp : OneFlow_BaseOp<"lars_update", [AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$model,
    AnyType:$model_diff,
    AnyType:$learning_rate,
    AnyType:$momentum,
    Optional<AnyType>:$scale_by_tensor,
    Optional<AnyType>:$skip_if
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "1.">:$scale,
    DefaultValuedAttr<F32Attr, "0.">:$l1,
    DefaultValuedAttr<F32Attr, "0.">:$l2,
    DefaultValuedAttr<F32Attr, "0.9">:$momentum_beta,
    DefaultValuedAttr<F32Attr, "0.">:$epsilon,
    DefaultValuedAttr<F32Attr, "0.0001">:$lars_coefficient,
    DefaultValuedAttr<F32Attr, "0.">:$weight_decay
  );
  let trait_attrs = (ins
    I32ElementsAttr:$operand_segment_sizes
  );
}

def OneFlow_MomentumUpdateOp : OneFlow_BaseOp<"momentum_update", [AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$model,
    AnyType:$model_diff,
    AnyType:$momentum,
    Optional<AnyType>:$learning_rate,
    Optional<AnyType>:$scale_by_tensor,
    Optional<AnyType>:$skip_if
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$learning_rate_val,
    DefaultValuedAttr<F64Attr, "1.">:$scale,
    DefaultValuedAttr<F32Attr, "0.">:$l1,
    DefaultValuedAttr<F32Attr, "0.">:$l2,
    DefaultValuedAttr<F32Attr, "0.9">:$beta,
    DefaultValuedAttr<F32Attr, "0.">:$weight_decay
  );
  let trait_attrs = (ins
    I32ElementsAttr:$operand_segment_sizes
  );
}

def OneFlow_RmspropUpdateOp : OneFlow_BaseOp<"rmsprop_update", [AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
    DefaultValuedAttr<F32Attr, "0.">:$learning_rate_val,
    DefaultValuedAttr<F64Attr, "1.">:$scale,
    DefaultValuedAttr<F32Attr, "0.">:$l1,
    DefaultValuedAttr<F32Attr, "0.">:$l2,
    DefaultValuedAttr<BoolAttr, "false">:$centered,
    DefaultValuedAttr<F32Attr, "0.">:$epsilon,
    DefaultValuedAttr<F32Attr, "0.99">:$decay_rate,
    DefaultValuedAttr<F32Attr, "0.">:$weight_decay
  );
  let trait_attrs = (ins
    I32ElementsAttr:$operand_segment_sizes
  );
}

def OneFlow_SgdUpdateOp : OneFlow_BaseOp<"sgd_update", [AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$model,
    AnyType:$model_diff,
    Optional<AnyType>:$learning_rate,
    Optional<AnyType>:$scale_by_tensor,
    Optional<AnyType>:$skip_if
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$learning_rate_val,
    DefaultValuedAttr<F64Attr, "1.">:$scale,
    DefaultValuedAttr<F32Attr, "0.">:$l1,
    DefaultValuedAttr<F32Attr, "0.">:$l2,
    DefaultValuedAttr<F32Attr, "0.">:$weight_decay
  );
  let trait_attrs = (ins
    I32ElementsAttr:$operand_segment_sizes
  );
}

def OneFlow_SliceUpdateOp : OneFlow_BaseOp<"slice_update", [DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

#endif // GET_ONEFLOW_OPTIMIZER_OP_DEFINITIONS

// Group: PADDING
// constant_pad1d, constant_pad1d_grad, constant_pad2d, constant_pad2d_grad, constant_pad3d, constant_pad3d_grad, pad, pad_grad, reflection_pad2d, reflection_pad2d_grad, replication_pad2d, replication_pad2d_grad, same_padding, same_padding_grad
// Total: 14

#ifdef GET_ONEFLOW_PADDING_OP_DEFINITIONS

def OneFlow_ConstantPad1DOp : OneFlow_BaseOp<"constant_pad1d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    SI64ArrayAttr:$padding,
    DefaultValuedAttr<F64Attr, "0.">:$floating_value,
    DefaultValuedAttr<SI64Attr, "0">:$integral_value
  );
}

def OneFlow_ConstantPad1DGradOp : OneFlow_BaseOp<"constant_pad1d_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    SI64ArrayAttr:$padding,
    DefaultValuedAttr<F64Attr, "0.">:$floating_value,
    DefaultValuedAttr<SI64Attr, "0">:$integral_value
  );
}

def OneFlow_ConstantPad2DOp : OneFlow_BaseOp<"constant_pad2d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    SI64ArrayAttr:$padding,
    DefaultValuedAttr<F64Attr, "0.">:$floating_value,
    DefaultValuedAttr<SI64Attr, "0">:$integral_value
  );
}

def OneFlow_ConstantPad2DGradOp : OneFlow_BaseOp<"constant_pad2d_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    SI64ArrayAttr:$padding,
    DefaultValuedAttr<F64Attr, "0.">:$floating_value,
    DefaultValuedAttr<SI64Attr, "0">:$integral_value
  );
}

def OneFlow_ConstantPad3DOp : OneFlow_BaseOp<"constant_pad3d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    SI64ArrayAttr:$padding,
    DefaultValuedAttr<F64Attr, "0.">:$floating_value,
    DefaultValuedAttr<SI64Attr, "0">:$integral_value
  );
}

def OneFlow_ConstantPad3DGradOp : OneFlow_BaseOp<"constant_pad3d_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    SI64ArrayAttr:$padding,
    DefaultValuedAttr<F64Attr, "0.">:$floating_value,
    DefaultValuedAttr<SI64Attr, "0">:$integral_value
  );
}

def OneFlow_PadOp : OneFlow_BaseOp<"pad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    SI64ArrayAttr:$padding_before,
    SI64ArrayAttr:$padding_after,
    DefaultValuedAttr<F64Attr, "0.">:$floating_constant_value,
    DefaultValuedAttr<SI64Attr, "0">:$integral_constant_value
  );
}

def OneFlow_PadGradOp : OneFlow_BaseOp<"pad_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    SI64ArrayAttr:$padding_before,
    SI64ArrayAttr:$padding_after,
    DefaultValuedAttr<F64Attr, "0.">:$floating_constant_value,
    DefaultValuedAttr<SI64Attr, "0">:$integral_constant_value
  );
}

def OneFlow_ReflectionPad2DOp : OneFlow_BaseOp<"reflection_pad2d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ReflectionPad2DGradOp : OneFlow_BaseOp<"reflection_pad2d_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ReplicationPad2DOp : OneFlow_BaseOp<"replication_pad2d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ReplicationPad2DGradOp : OneFlow_BaseOp<"replication_pad2d_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_SamePaddingOp : OneFlow_BaseOp<"same_padding", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_SamePaddingGradOp : OneFlow_BaseOp<"same_padding_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

#endif // GET_ONEFLOW_PADDING_OP_DEFINITIONS

// Group: PARALLEL_CAST
// hierarchical_parallel_cast, hierarchical_parallel_cast_like, parallel_cast
// Total: 3

#ifdef GET_ONEFLOW_PARALLEL_CAST_OP_DEFINITIONS

def OneFlow_HierarchicalParallelCastOp : OneFlow_BaseOp<"hierarchical_parallel_cast", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_HierarchicalParallelCastLikeOp : OneFlow_BaseOp<"hierarchical_parallel_cast_like", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in,
    AnyType:$like
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_ParallelCastOp : OneFlow_BaseOp<"parallel_cast", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

#endif // GET_ONEFLOW_PARALLEL_CAST_OP_DEFINITIONS

// Group: POOL
// adaptive_avg_pool1d, adaptive_avg_pool1d_grad, adaptive_avg_pool2d, adaptive_avg_pool2d_grad, adaptive_avg_pool3d, adaptive_avg_pool3d_grad, avgpool_1d, avgpool_1d_grad, avgpool_2d, avgpool_2d_grad, avgpool_3d, avgpool_3d_grad, maxpool_1d, maxpool_1d_grad, maxpool_2d, maxpool_2d_grad, maxpool_3d, maxpool_3d_grad, tf_avg_pool_1d, tf_avg_pool_1d_grad, tf_avg_pool_2d, tf_avg_pool_2d_grad, tf_avg_pool_3d, tf_avg_pool_3d_grad, tf_max_pool_1d, tf_max_pool_1d_grad, tf_max_pool_2d, tf_max_pool_2d_grad, tf_max_pool_3d, tf_max_pool_3d_grad
// Total: 30

#ifdef GET_ONEFLOW_POOL_OP_DEFINITIONS

def OneFlow_AdaptiveAvgPool1DOp : OneFlow_AdaptivePoolBaseOp<"adaptive_avg_pool1d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_AdaptiveAvgPoolGrad1DOp : OneFlow_AdaptivePoolGradBaseOp<"adaptive_avg_pool1d_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_AdaptiveAvgPool2DOp : OneFlow_AdaptivePoolBaseOp<"adaptive_avg_pool2d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_AdaptiveAvgPoolGrad2DOp : OneFlow_AdaptivePoolGradBaseOp<"adaptive_avg_pool2d_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_AdaptiveAvgPool3DOp : OneFlow_AdaptivePoolBaseOp<"adaptive_avg_pool3d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_AdaptiveAvgPoolGrad3DOp : OneFlow_AdaptivePoolGradBaseOp<"adaptive_avg_pool3d_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_AvgPool1DOp : OneFlow_PoolBaseOp<"avgpool_1d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_AvgPool1DGradOp : OneFlow_PoolGradBaseOp<"avgpool_1d_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_AvgPool2DOp : OneFlow_PoolBaseOp<"avgpool_2d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_AvgPool2DGradOp : OneFlow_PoolGradBaseOp<"avgpool_2d_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_AvgPool3DOp : OneFlow_PoolBaseOp<"avgpool_3d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_AvgPool3DGradOp : OneFlow_PoolGradBaseOp<"avgpool_3d_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_MaxPool1DOp : OneFlow_PoolBaseOp<"maxpool_1d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_MaxPool1DGradOp : OneFlow_PoolGradBaseOp<"maxpool_1d_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_MaxPool2DOp : OneFlow_PoolBaseOp<"maxpool_2d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_MaxPool2DGradOp : OneFlow_PoolGradBaseOp<"maxpool_2d_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_MaxPool3DOp : OneFlow_PoolBaseOp<"maxpool_3d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_MaxPool3DGradOp : OneFlow_PoolGradBaseOp<"maxpool_3d_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_TfAvgPool1DOp : OneFlow_TFPoolBaseOp<"tf_avg_pool_1d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_TfAvgPool1DGradOp : OneFlow_TFPoolGradBaseOp<"tf_avg_pool_1d_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_TfAvgPool2DOp : OneFlow_TFPoolBaseOp<"tf_avg_pool_2d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_TfAvgPool2DGradOp : OneFlow_TFPoolGradBaseOp<"tf_avg_pool_2d_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_TfAvgPool3DOp : OneFlow_TFPoolBaseOp<"tf_avg_pool_3d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_TfAvgPool3DGradOp : OneFlow_TFPoolGradBaseOp<"tf_avg_pool_3d_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_TfMaxPool1DOp : OneFlow_TFPoolBaseOp<"tf_max_pool_1d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_TfMaxPool1DGradOp : OneFlow_TFPoolGradBaseOp<"tf_max_pool_1d_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_TfMaxPool2DOp : OneFlow_TFPoolBaseOp<"tf_max_pool_2d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_TfMaxPool2DGradOp : OneFlow_TFPoolGradBaseOp<"tf_max_pool_2d_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_TfMaxPool3DOp : OneFlow_TFPoolBaseOp<"tf_max_pool_3d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_TfMaxPool3DGradOp : OneFlow_TFPoolGradBaseOp<"tf_max_pool_3d_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

#endif // GET_ONEFLOW_POOL_OP_DEFINITIONS

// Group: QUANTIZATION
// fake_quantization, min_max_observer, moving_average_min_max_observer, quantization
// Total: 4

#ifdef GET_ONEFLOW_QUANTIZATION_OP_DEFINITIONS

def OneFlow_FakeQuantizationOp : OneFlow_BaseOp<"fake_quantization", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in,
    AnyType:$scale,
    AnyType:$zero_point
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<StrAttr, "\"google\"">:$quantization_formula,
    DefaultValuedAttr<SI32Attr, "8">:$quantization_bit,
    DefaultValuedAttr<StrAttr, "\"symmetric\"">:$quantization_scheme
  );
}

def OneFlow_MinMaxObserverOp : OneFlow_BaseOp<"min_max_observer", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$scale,
    AnyType:$zero_point
  );
  let attrs = (ins
    DefaultValuedAttr<StrAttr, "\"google\"">:$quantization_formula,
    DefaultValuedAttr<SI32Attr, "8">:$quantization_bit,
    DefaultValuedAttr<StrAttr, "\"symmetric\"">:$quantization_scheme,
    DefaultValuedAttr<BoolAttr, "true">:$per_layer_quantization
  );
}

def OneFlow_MovingAverageMinMaxObserverOp : OneFlow_BaseOp<"moving_average_min_max_observer", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
    DefaultValuedAttr<BoolAttr, "false">:$training,
    DefaultValuedAttr<StrAttr, "\"google\"">:$quantization_formula,
    DefaultValuedAttr<SI64Attr, "0">:$stop_update_after_iters,
    DefaultValuedAttr<SI32Attr, "8">:$quantization_bit,
    DefaultValuedAttr<StrAttr, "\"symmetric\"">:$quantization_scheme,
    DefaultValuedAttr<F32Attr, "0.95">:$momentum
  );
}

def OneFlow_QuantizationOp : OneFlow_BaseOp<"quantization", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in,
    AnyType:$scale,
    AnyType:$zero_point
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<StrAttr, "\"google\"">:$quantization_formula,
    DefaultValuedAttr<SI32Attr, "8">:$quantization_bit,
    DefaultValuedAttr<StrAttr, "\"symmetric\"">:$quantization_scheme
  );
}

#endif // GET_ONEFLOW_QUANTIZATION_OP_DEFINITIONS

// Group: REDUCE
// indexed_slices_reduce_sum, reduce_all, reduce_any, reduce_max, reduce_max_device_stage, reduce_max_device_stage_grad, reduce_max_global_stage, reduce_max_global_stage_grad, reduce_min, reduce_min_device_stage, reduce_min_device_stage_grad, reduce_min_global_stage, reduce_min_global_stage_grad, reduce_prod, reduce_sum, reduce_sum_like
// Total: 16

#ifdef GET_ONEFLOW_REDUCE_OP_DEFINITIONS

def OneFlow_IndexedSlicesReduceSumOp : OneFlow_BaseOp<"indexed_slices_reduce_sum", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ReduceAllOp : OneFlow_BaseOp<"reduce_all", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$input_tensor
  );
  let output = (outs
    AnyType:$output_tensor
  );
  let attrs = (ins
    SI32ArrayAttr:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$keepdims
  );
}

def OneFlow_ReduceAnyOp : OneFlow_BaseOp<"reduce_any", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$input_tensor
  );
  let output = (outs
    AnyType:$output_tensor
  );
  let attrs = (ins
    SI32ArrayAttr:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$keepdims
  );
}

def OneFlow_ReduceMaxOp : OneFlow_BaseOp<"reduce_max", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$input_tensor
  );
  let output = (outs
    AnyType:$output_tensor
  );
  let attrs = (ins
    SI32ArrayAttr:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$keepdims
  );
}

def OneFlow_ReduceMaxDeviceStageOp : OneFlow_BaseOp<"reduce_max_device_stage", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ReduceMaxDeviceStageGradOp : OneFlow_BaseOp<"reduce_max_device_stage_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ReduceMaxGlobalStageOp : OneFlow_BaseOp<"reduce_max_global_stage", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
    DefaultValuedAttr<BoolAttr, "false">:$keepdims
  );
}

def OneFlow_ReduceMaxGlobalStageGradOp : OneFlow_BaseOp<"reduce_max_global_stage_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
    DefaultValuedAttr<BoolAttr, "false">:$keepdims
  );
}

def OneFlow_ReduceMinOp : OneFlow_BaseOp<"reduce_min", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$input_tensor
  );
  let output = (outs
    AnyType:$output_tensor
  );
  let attrs = (ins
    SI32ArrayAttr:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$keepdims
  );
}

def OneFlow_ReduceMinDeviceStageOp : OneFlow_BaseOp<"reduce_min_device_stage", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ReduceMinDeviceStageGradOp : OneFlow_BaseOp<"reduce_min_device_stage_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ReduceMinGlobalStageOp : OneFlow_BaseOp<"reduce_min_global_stage", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
    DefaultValuedAttr<BoolAttr, "false">:$keepdims
  );
}

def OneFlow_ReduceMinGlobalStageGradOp : OneFlow_BaseOp<"reduce_min_global_stage_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
    DefaultValuedAttr<BoolAttr, "false">:$keepdims
  );
}

def OneFlow_ReduceProdOp : OneFlow_BaseOp<"reduce_prod", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$input_tensor
  );
  let output = (outs
    AnyType:$output_tensor
  );
  let attrs = (ins
    SI32ArrayAttr:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$keepdims
  );
}

def OneFlow_ReduceSumOp : OneFlow_BaseOp<"reduce_sum", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$input_tensor
  );
  let output = (outs
    AnyType:$output_tensor
  );
  let attrs = (ins
    SI32ArrayAttr:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$keepdims
  );
}

def OneFlow_ReduceSumLikeOp : OneFlow_BaseOp<"reduce_sum_like", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

#endif // GET_ONEFLOW_REDUCE_OP_DEFINITIONS

// Group: RESHAPE
// reshape, reshape_like
// Total: 2

#ifdef GET_ONEFLOW_RESHAPE_OP_DEFINITIONS

def OneFlow_ReshapeOp : OneFlow_BaseOp<"reshape", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ReshapeLikeOp : OneFlow_BaseOp<"reshape_like", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in,
    AnyType:$like
  );
  let output = (outs
    AnyType:$out
  );
}

#endif // GET_ONEFLOW_RESHAPE_OP_DEFINITIONS

// Group: SCALAR
// clip_by_scalar, clip_by_scalar_grad, clip_by_scalar_max, clip_by_scalar_max_grad, clip_by_scalar_min, clip_by_scalar_min_grad, scalar_add, scalar_add_by_tensor, scalar_div_by_tensor, scalar_floordiv, scalar_fmod, scalar_logical_and, scalar_logical_equal, scalar_logical_greater, scalar_logical_greater_equal, scalar_logical_less, scalar_logical_less_equal, scalar_logical_not_equal, scalar_logical_or, scalar_logical_xor, scalar_mul, scalar_mul_by_tensor, scalar_pow, scalar_pow_grad, scalar_sub_by_tensor
// Total: 25

#ifdef GET_ONEFLOW_SCALAR_OP_DEFINITIONS

def OneFlow_ClipByScalarOp : OneFlow_BaseOp<"clip_by_scalar", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.">:$floating_min,
    DefaultValuedAttr<SI64Attr, "0">:$integral_min,
    DefaultValuedAttr<F64Attr, "0.">:$floating_max,
    DefaultValuedAttr<SI64Attr, "0">:$integral_max
  );
}

def OneFlow_ClipByScalarGradOp : OneFlow_BaseOp<"clip_by_scalar_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$dy,
    AnyType:$x
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.">:$floating_min,
    DefaultValuedAttr<SI64Attr, "0">:$integral_min,
    DefaultValuedAttr<F64Attr, "0.">:$floating_max,
    DefaultValuedAttr<SI64Attr, "0">:$integral_max
  );
}

def OneFlow_ClipByScalarMaxOp : OneFlow_BaseOp<"clip_by_scalar_max", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.">:$floating_max,
    DefaultValuedAttr<SI64Attr, "0">:$integral_max
  );
}

def OneFlow_ClipByScalarMaxGradOp : OneFlow_BaseOp<"clip_by_scalar_max_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$dy,
    AnyType:$x
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.">:$floating_max,
    DefaultValuedAttr<SI64Attr, "0">:$integral_max
  );
}

def OneFlow_ClipByScalarMinOp : OneFlow_BaseOp<"clip_by_scalar_min", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.">:$floating_min,
    DefaultValuedAttr<SI64Attr, "0">:$integral_min
  );
}

def OneFlow_ClipByScalarMinGradOp : OneFlow_BaseOp<"clip_by_scalar_min_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$dy,
    AnyType:$x
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.">:$floating_min,
    DefaultValuedAttr<SI64Attr, "0">:$integral_min
  );
}

def OneFlow_ScalarAddOp : OneFlow_BaseOp<"scalar_add", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$has_int_operand,
    DefaultValuedAttr<BoolAttr, "false">:$has_float_operand,
    DefaultValuedAttr<SI64Attr, "0">:$int_operand,
    DefaultValuedAttr<F64Attr, "0.">:$float_operand
  );
}

def OneFlow_ScalarAddByTensorOp : OneFlow_BaseOp<"scalar_add_by_tensor", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$scalar
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_ScalarDivByTensorOp : OneFlow_BaseOp<"scalar_div_by_tensor", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$scalar
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_ScalarFloordivOp : OneFlow_BaseOp<"scalar_floordiv", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$has_int_operand,
    DefaultValuedAttr<BoolAttr, "false">:$has_float_operand,
    DefaultValuedAttr<SI64Attr, "0">:$int_operand,
    DefaultValuedAttr<F64Attr, "0.">:$float_operand
  );
}

def OneFlow_ScalarFmodOp : OneFlow_BaseOp<"scalar_fmod", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$has_int_operand,
    DefaultValuedAttr<BoolAttr, "false">:$has_float_operand,
    DefaultValuedAttr<SI64Attr, "0">:$int_operand,
    DefaultValuedAttr<F64Attr, "0.">:$float_operand
  );
}

def OneFlow_ScalarLogicalAndOp : OneFlow_BaseOp<"scalar_logical_and", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$has_int_operand,
    DefaultValuedAttr<BoolAttr, "false">:$has_float_operand,
    DefaultValuedAttr<SI64Attr, "0">:$int_operand,
    DefaultValuedAttr<F64Attr, "0.">:$float_operand
  );
}

def OneFlow_ScalarLogicalEqualOp : OneFlow_BaseOp<"scalar_logical_equal", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$has_int_operand,
    DefaultValuedAttr<BoolAttr, "false">:$has_float_operand,
    DefaultValuedAttr<SI64Attr, "0">:$int_operand,
    DefaultValuedAttr<F64Attr, "0.">:$float_operand
  );
}

def OneFlow_ScalarLogicalGreaterOp : OneFlow_BaseOp<"scalar_logical_greater", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$has_int_operand,
    DefaultValuedAttr<BoolAttr, "false">:$has_float_operand,
    DefaultValuedAttr<SI64Attr, "0">:$int_operand,
    DefaultValuedAttr<F64Attr, "0.">:$float_operand
  );
}

def OneFlow_ScalarLogicalGreaterEqualOp : OneFlow_BaseOp<"scalar_logical_greater_equal", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$has_int_operand,
    DefaultValuedAttr<BoolAttr, "false">:$has_float_operand,
    DefaultValuedAttr<SI64Attr, "0">:$int_operand,
    DefaultValuedAttr<F64Attr, "0.">:$float_operand
  );
}

def OneFlow_ScalarLogicalLessOp : OneFlow_BaseOp<"scalar_logical_less", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$has_int_operand,
    DefaultValuedAttr<BoolAttr, "false">:$has_float_operand,
    DefaultValuedAttr<SI64Attr, "0">:$int_operand,
    DefaultValuedAttr<F64Attr, "0.">:$float_operand
  );
}

def OneFlow_ScalarLogicalLessEqualOp : OneFlow_BaseOp<"scalar_logical_less_equal", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$has_int_operand,
    DefaultValuedAttr<BoolAttr, "false">:$has_float_operand,
    DefaultValuedAttr<SI64Attr, "0">:$int_operand,
    DefaultValuedAttr<F64Attr, "0.">:$float_operand
  );
}

def OneFlow_ScalarLogicalNotEqualOp : OneFlow_BaseOp<"scalar_logical_not_equal", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$has_int_operand,
    DefaultValuedAttr<BoolAttr, "false">:$has_float_operand,
    DefaultValuedAttr<SI64Attr, "0">:$int_operand,
    DefaultValuedAttr<F64Attr, "0.">:$float_operand
  );
}

def OneFlow_ScalarLogicalOrOp : OneFlow_BaseOp<"scalar_logical_or", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$has_int_operand,
    DefaultValuedAttr<BoolAttr, "false">:$has_float_operand,
    DefaultValuedAttr<SI64Attr, "0">:$int_operand,
    DefaultValuedAttr<F64Attr, "0.">:$float_operand
  );
}

def OneFlow_ScalarLogicalXorOp : OneFlow_BaseOp<"scalar_logical_xor", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$has_int_operand,
    DefaultValuedAttr<BoolAttr, "false">:$has_float_operand,
    DefaultValuedAttr<SI64Attr, "0">:$int_operand,
    DefaultValuedAttr<F64Attr, "0.">:$float_operand
  );
}

def OneFlow_ScalarMulOp : OneFlow_BaseOp<"scalar_mul", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$has_int_operand,
    DefaultValuedAttr<BoolAttr, "false">:$has_float_operand,
    DefaultValuedAttr<SI64Attr, "0">:$int_operand,
    DefaultValuedAttr<F64Attr, "0.">:$float_operand
  );
}

def OneFlow_ScalarMulByTensorOp : OneFlow_BaseOp<"scalar_mul_by_tensor", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$scalar
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_ScalarPowOp : OneFlow_BaseOp<"scalar_pow", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$has_int_operand,
    DefaultValuedAttr<BoolAttr, "false">:$has_float_operand,
    DefaultValuedAttr<SI64Attr, "0">:$int_operand,
    DefaultValuedAttr<F64Attr, "0.">:$float_operand
  );
}

def OneFlow_ScalarPowGradOp : OneFlow_BaseOp<"scalar_pow_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<BoolAttr, "false">:$has_int_operand,
    DefaultValuedAttr<BoolAttr, "false">:$has_float_operand,
    DefaultValuedAttr<SI64Attr, "0">:$int_operand,
    DefaultValuedAttr<F64Attr, "0.">:$float_operand
  );
}

def OneFlow_ScalarSubByTensorOp : OneFlow_BaseOp<"scalar_sub_by_tensor", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$scalar
  );
  let output = (outs
    AnyType:$y
  );
}

#endif // GET_ONEFLOW_SCALAR_OP_DEFINITIONS

// Group: SOFTMAX
// log_softmax, log_softmax_grad, softmax, softmax_cross_entropy, softmax_cross_entropy_grad, softmax_grad, sparse_softmax_cross_entropy, sparse_softmax_cross_entropy_grad, sparse_softmax_cross_entropy_ms, sparse_softmax_cross_entropy_ms_grad
// Total: 10

#ifdef GET_ONEFLOW_SOFTMAX_OP_DEFINITIONS

def OneFlow_LogSoftmaxOp : OneFlow_BaseOp<"log_softmax", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$prob
  );
}

def OneFlow_LogSoftmaxGradOp : OneFlow_BaseOp<"log_softmax_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$prob,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_SoftmaxOp : OneFlow_BaseOp<"softmax", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_SoftmaxCrossEntropyOp : OneFlow_BaseOp<"softmax_cross_entropy", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$prediction,
    AnyType:$label
  );
  let output = (outs
    AnyType:$prob,
    AnyType:$out
  );
}

def OneFlow_SoftmaxCrossEntropyGradOp : OneFlow_BaseOp<"softmax_cross_entropy_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$dy,
    AnyType:$label,
    AnyType:$prob
  );
  let output = (outs
    AnyType:$prediction_diff
  );
}

def OneFlow_SoftmaxGradOp : OneFlow_BaseOp<"softmax_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$y,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_SparseSoftmaxCrossEntropyOp : OneFlow_BaseOp<"sparse_softmax_cross_entropy", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$prediction,
    AnyType:$label
  );
  let output = (outs
    AnyType:$prob,
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$depth
  );
}

def OneFlow_SparseSoftmaxCrossEntropyGradOp : OneFlow_BaseOp<"sparse_softmax_cross_entropy_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$label,
    AnyType:$dy,
    AnyType:$prob
  );
  let output = (outs
    AnyType:$prediction_diff
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$depth
  );
}

def OneFlow_SparseSoftmaxCrossEntropyMsOp : OneFlow_BaseOp<"sparse_softmax_cross_entropy_ms", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$prediction,
    AnyType:$label
  );
  let output = (outs
    AnyType:$prob,
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$depth
  );
}

def OneFlow_SparseSoftmaxCrossEntropyMsGradOp : OneFlow_BaseOp<"sparse_softmax_cross_entropy_ms_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$label,
    AnyType:$dy,
    AnyType:$prob
  );
  let output = (outs
    AnyType:$prediction_diff
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$depth
  );
}

#endif // GET_ONEFLOW_SOFTMAX_OP_DEFINITIONS

// Group: SUMMARY
// create_summary_writer, flush_summary_writer, summary_write_histogram, summary_write_image, summary_write_pb, summary_write_scalar
// Total: 6

#ifdef GET_ONEFLOW_SUMMARY_OP_DEFINITIONS

def OneFlow_CreateSummaryWriterOp : OneFlow_BaseOp<"create_summary_writer", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let attrs = (ins
    StrAttr:$logdir
  );
}

def OneFlow_FlushSummaryWriterOp : OneFlow_BaseOp<"flush_summary_writer", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
}

def OneFlow_SummaryWriteHistogramOp : OneFlow_BaseOp<"summary_write_histogram", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in,
    AnyType:$step,
    AnyType:$tag
  );
}

def OneFlow_SummaryWriteImageOp : OneFlow_BaseOp<"summary_write_image", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in,
    AnyType:$step,
    AnyType:$tag
  );
}

def OneFlow_SummaryWritePbOp : OneFlow_BaseOp<"summary_write_pb", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in,
    AnyType:$step
  );
}

def OneFlow_SummaryWriteScalarOp : OneFlow_BaseOp<"summary_write_scalar", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in,
    AnyType:$step,
    AnyType:$tag
  );
}

#endif // GET_ONEFLOW_SUMMARY_OP_DEFINITIONS

// Group: TENSOR_BUFFER
// gen_tensor_buffer, tensor_buffer_to_list_of_tensors, tensor_buffer_to_list_of_tensors_v2, tensor_buffer_to_tensor, tensor_to_tensor_buffer
// Total: 5

#ifdef GET_ONEFLOW_TENSOR_BUFFER_OP_DEFINITIONS

def OneFlow_GenTensorBufferOp : OneFlow_BaseOp<"gen_tensor_buffer", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    AnyI64ElementsAttr:$shape,
    ShapeArrayAttr:$shape_list,
    F32ArrayAttr:$value_list,
    StrAttr:$data_type,
    DefaultValuedAttr<BoolAttr, "false">:$dynamic_out
  );
}

def OneFlow_TensorBufferToListOfTensorsOp : OneFlow_BaseOp<"tensor_buffer_to_list_of_tensors", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    Variadic<AnyType>:$out
  );
  let attrs = (ins
    AnyI64ElementsAttr:$out_shape,
    StrAttr:$out_dtype,
    DefaultValuedAttr<BoolAttr, "false">:$dynamic_out
  );
}

def OneFlow_TensorBufferToListOfTensorsV2Op : OneFlow_BaseOp<"tensor_buffer_to_list_of_tensors_v2", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    Variadic<AnyType>:$out
  );
  let attrs = (ins
    ShapeArrayAttr:$out_shapes,
    DTArrayAttr:$out_dtypes,
    DefaultValuedAttr<BoolAttr, "false">:$dynamic_out
  );
}

def OneFlow_TensorBufferToTensorOp : OneFlow_BaseOp<"tensor_buffer_to_tensor", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_TensorToTensorBufferOp : OneFlow_BaseOp<"tensor_to_tensor_buffer", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$instance_dims
  );
}

#endif // GET_ONEFLOW_TENSOR_BUFFER_OP_DEFINITIONS

// Group: TEST
// TestDataTypeAttr, TestDynamicSource, TestListDataTypeAndListShapeAndListStringAttr, TestMultiInput, TestMultiInputGrad, TestMultiOutputOrder, TestRandomSource, TestReshape, TestSource, TestSourceMultiGpuFixedOutNum, ccrelu, ccrelu_grad, cpu_only_relu_test, test_user_op_attr_auto_type
// Total: 14

#ifdef GET_ONEFLOW_TEST_OP_DEFINITIONS

def OneFlow_TestDataTypeAttrOp : OneFlow_BaseOp<"TestDataTypeAttr", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_TestDynamicSourceOp : OneFlow_BaseOp<"TestDynamicSource", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_TestListDataTypeAndListShapeAndListStringAttrOp : OneFlow_BaseOp<"TestListDataTypeAndListShapeAndListStringAttr", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_TestMultiInputOp : OneFlow_BaseOp<"TestMultiInput", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x1,
    AnyType:$x2
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_TestMultiInputGradOp : OneFlow_BaseOp<"TestMultiInputGrad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_TestMultiOutputOrderOp : OneFlow_BaseOp<"TestMultiOutputOrder", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out1,
    AnyType:$out2
  );
}

def OneFlow_TestRandomSourceOp : OneFlow_BaseOp<"TestRandomSource", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$seed
  );
}

def OneFlow_TestReshapeOp : OneFlow_BaseOp<"TestReshape", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_TestSourceOp : OneFlow_BaseOp<"TestSource", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_TestSourceMultiGpuFixedOutNumOp : OneFlow_BaseOp<"TestSourceMultiGpuFixedOutNum", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$out_num
  );
}

def OneFlow_CcreluOp : OneFlow_BaseOp<"ccrelu", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_CcreluGradOp : OneFlow_BaseOp<"ccrelu_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$y,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_CpuOnlyReluTestOp : OneFlow_BaseOp<"cpu_only_relu_test", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_TestUserOpAttrAutoTypeOp : OneFlow_BaseOp<"test_user_op_attr_auto_type", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$int1,
    DefaultValuedAttr<SI32Attr, "0">:$int2
  );
}

#endif // GET_ONEFLOW_TEST_OP_DEFINITIONS

// Group: TRIGONOMETRIC
// acos, acos_grad, acosh, acosh_grad, asin, asin_grad, asinh, asinh_grad, atan, atan2, atan2_x_grad, atan2_y_grad, atan_grad, atanh, atanh_grad, cos, cos_grad, cosh, cosh_grad, hardtanh, hardtanh_grad, sin, sin_grad, sinh, sinh_grad, tan, tan_grad, tanh, tanh_grad
// Total: 29

#ifdef GET_ONEFLOW_TRIGONOMETRIC_OP_DEFINITIONS

def OneFlow_AcosOp : OneFlow_BaseOp<"acos", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_AcosGradOp : OneFlow_BaseOp<"acos_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_AcoshOp : OneFlow_BaseOp<"acosh", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_AcoshGradOp : OneFlow_BaseOp<"acosh_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_AsinOp : OneFlow_BaseOp<"asin", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_AsinGradOp : OneFlow_BaseOp<"asin_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_AsinhOp : OneFlow_BaseOp<"asinh", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_AsinhGradOp : OneFlow_BaseOp<"asinh_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_AtanOp : OneFlow_BaseOp<"atan", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_Atan2Op : OneFlow_BaseOp<"atan2", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y
  );
  let output = (outs
    AnyType:$z
  );
}

def OneFlow_Atan2XGradOp : OneFlow_BaseOp<"atan2_x_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y,
    AnyType:$dz
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_Atan2YGradOp : OneFlow_BaseOp<"atan2_y_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$y,
    AnyType:$dz
  );
  let output = (outs
    AnyType:$dy
  );
}

def OneFlow_AtanGradOp : OneFlow_BaseOp<"atan_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_AtanhOp : OneFlow_BaseOp<"atanh", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_AtanhGradOp : OneFlow_BaseOp<"atanh_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_CosOp : OneFlow_BaseOp<"cos", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_CosGradOp : OneFlow_BaseOp<"cos_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_CoshOp : OneFlow_BaseOp<"cosh", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_CoshGradOp : OneFlow_BaseOp<"cosh_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_HardtanhOp : OneFlow_BaseOp<"hardtanh", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.">:$min_val,
    DefaultValuedAttr<F64Attr, "0.">:$max_val
  );
}

def OneFlow_HardtanhGradOp : OneFlow_BaseOp<"hardtanh_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$y,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.">:$min_val,
    DefaultValuedAttr<F64Attr, "0.">:$max_val
  );
}

def OneFlow_SinOp : OneFlow_BaseOp<"sin", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_SinGradOp : OneFlow_BaseOp<"sin_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_SinhOp : OneFlow_BaseOp<"sinh", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_SinhGradOp : OneFlow_BaseOp<"sinh_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_TanOp : OneFlow_BaseOp<"tan", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_TanGradOp : OneFlow_BaseOp<"tan_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

def OneFlow_TanhOp : OneFlow_BaseOp<"tanh", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_TanhGradOp : OneFlow_BaseOp<"tanh_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
}

#endif // GET_ONEFLOW_TRIGONOMETRIC_OP_DEFINITIONS

// Group: UNARY
// acc, affine_grid, affine_grid_grad, bernoulli, cast, cast_to_static_shape, cast_to_tick, celu, copy, count_not_finite, diag, elu, expand, expand_dims, flatten, flip, flip_grad, fold, gelu, hardsigmoid, hardswish, leaky_relu, logical_not, mish, narrow, one_hot, pack, random_mask_like, repeat, roll, selu, sigmoid, silu, softsign, sort, square_sum, squeeze, transpose, tril, triu, unfold, unfold_tensor, unpack, zero_like
// Total: 44

#ifdef GET_ONEFLOW_UNARY_OP_DEFINITIONS

def OneFlow_AccOp : OneFlow_BaseOp<"acc", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$max_acc_num
  );
}

def OneFlow_AffineGridOp : OneFlow_BaseOp<"affine_grid", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$theta
  );
  let output = (outs
    AnyType:$grid
  );
  let attrs = (ins
    AnyI64ElementsAttr:$size,
    DefaultValuedAttr<BoolAttr, "false">:$align_corners
  );
}

def OneFlow_AffineGridGradOp : OneFlow_BaseOp<"affine_grid_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$dgrid
  );
  let output = (outs
    AnyType:$dtheta
  );
  let attrs = (ins
    AnyI64ElementsAttr:$size,
    DefaultValuedAttr<BoolAttr, "false">:$align_corners
  );
}

def OneFlow_BernoulliOp : OneFlow_BaseOp<"bernoulli", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "-1">:$seed,
    DefaultValuedAttr<BoolAttr, "false">:$has_seed,
    StrAttr:$dtype
  );
}

def OneFlow_CastOp : OneFlow_BaseOp<"cast", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_CastToStaticShapeOp : OneFlow_BaseOp<"cast_to_static_shape", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$input
  );
  let output = (outs
    AnyType:$output
  );
}

def OneFlow_CastToTickOp : OneFlow_BaseOp<"cast_to_tick", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_CeluOp : OneFlow_BaseOp<"celu", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.">:$alpha
  );
}

def OneFlow_CopyOp : OneFlow_BaseOp<"copy", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    StrAttr:$device_type,
    DefaultValuedAttr<SI64Attr, "0">:$device_id
  );
}

def OneFlow_CountNotFiniteOp : OneFlow_BaseOp<"count_not_finite", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_DiagOp : OneFlow_BaseOp<"diag", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$diagonal
  );
}

def OneFlow_EluOp : OneFlow_BaseOp<"elu", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.">:$alpha
  );
}

def OneFlow_ExpandOp : OneFlow_BaseOp<"expand", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ExpandDimsOp : OneFlow_BaseOp<"expand_dims", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$axis
  );
}

def OneFlow_FlattenOp : OneFlow_BaseOp<"flatten", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$start_dim,
    DefaultValuedAttr<SI32Attr, "-1">:$end_dim
  );
}

def OneFlow_FlipOp : OneFlow_BaseOp<"flip", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_FlipGradOp : OneFlow_BaseOp<"flip_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_FoldOp : OneFlow_BaseOp<"fold", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_GeluOp : OneFlow_BaseOp<"gelu", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_HardsigmoidOp : OneFlow_BaseOp<"hardsigmoid", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_HardswishOp : OneFlow_BaseOp<"hardswish", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_LeakyReluOp : OneFlow_BaseOp<"leaky_relu", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$alpha
  );
}

def OneFlow_LogicalNotOp : OneFlow_BaseOp<"logical_not", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_MishOp : OneFlow_BaseOp<"mish", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_NarrowOp : OneFlow_BaseOp<"narrow", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$dim,
    DefaultValuedAttr<SI64Attr, "0">:$start,
    DefaultValuedAttr<SI64Attr, "0">:$length
  );
}

def OneFlow_OneHotOp : OneFlow_BaseOp<"one_hot", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$indices
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$depth,
    DefaultValuedAttr<F64Attr, "0.">:$floating_on_value,
    DefaultValuedAttr<SI64Attr, "0">:$integer_on_value,
    DefaultValuedAttr<F64Attr, "0.">:$floating_off_value,
    DefaultValuedAttr<SI64Attr, "0">:$integer_off_value,
    StrAttr:$dtype
  );
}

def OneFlow_PackOp : OneFlow_BaseOp<"pack", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$pack_num
  );
}

def OneFlow_RandomMaskLikeOp : OneFlow_BaseOp<"random_mask_like", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$like
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$rate,
    DefaultValuedAttr<SI64Attr, "0">:$seed
  );
}

def OneFlow_RepeatOp : OneFlow_BaseOp<"repeat", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$repeat_num
  );
}

def OneFlow_RollOp : OneFlow_BaseOp<"roll", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_SeluOp : OneFlow_BaseOp<"selu", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_SigmoidOp : OneFlow_BaseOp<"sigmoid", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_SiluOp : OneFlow_BaseOp<"silu", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_SoftsignOp : OneFlow_BaseOp<"softsign", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
}

def OneFlow_SortOp : OneFlow_BaseOp<"sort", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_SquareSumOp : OneFlow_BaseOp<"square_sum", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
}

def OneFlow_SqueezeOp : OneFlow_BaseOp<"squeeze", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_TransposeOp : OneFlow_BaseOp<"transpose", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_TrilOp : OneFlow_BaseOp<"tril", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$diagonal,
    DefaultValuedAttr<F64Attr, "0.">:$floating_fill_value,
    DefaultValuedAttr<SI64Attr, "0">:$integer_fill_value,
    DefaultValuedAttr<BoolAttr, "false">:$is_floating_fill_value
  );
}

def OneFlow_TriuOp : OneFlow_BaseOp<"triu", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$diagonal
  );
}

def OneFlow_UnfoldOp : OneFlow_BaseOp<"unfold", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_UnfoldTensorOp : OneFlow_BaseOp<"unfold_tensor", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$dimension,
    DefaultValuedAttr<SI32Attr, "0">:$size,
    DefaultValuedAttr<SI32Attr, "0">:$step
  );
}

def OneFlow_UnpackOp : OneFlow_BaseOp<"unpack", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$unpack_num
  );
}

def OneFlow_ZeroLikeOp : OneFlow_BaseOp<"zero_like", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$like
  );
  let output = (outs
    AnyType:$out
  );
}

#endif // GET_ONEFLOW_UNARY_OP_DEFINITIONS

// Group: UPSAMPLE
// upsample, upsample_bicubic_2d, upsample_bicubic_2d_grad, upsample_bilinear_2d, upsample_bilinear_2d_grad, upsample_grad, upsample_linear_1d, upsample_linear_1d_grad, upsample_nearest_1d, upsample_nearest_1d_grad, upsample_nearest_2d, upsample_nearest_2d_grad, upsample_nearest_3d, upsample_nearest_3d_grad, upsample_trilinear_3d, upsample_trilinear_3d_grad
// Total: 16

#ifdef GET_ONEFLOW_UPSAMPLE_OP_DEFINITIONS

def OneFlow_UpsampleOp : OneFlow_BaseOp<"upsample", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$height_scale,
    DefaultValuedAttr<F32Attr, "0.">:$width_scale,
    DefaultValuedAttr<BoolAttr, "false">:$align_corners,
    StrAttr:$data_format,
    StrAttr:$interpolation
  );
}

def OneFlow_UpsampleBicubic2DOp : OneFlow_BaseOp<"upsample_bicubic_2d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$height_scale,
    DefaultValuedAttr<F32Attr, "0.">:$width_scale,
    DefaultValuedAttr<BoolAttr, "false">:$align_corners,
    StrAttr:$data_format
  );
}

def OneFlow_UpsampleBicubic2DGradOp : OneFlow_BaseOp<"upsample_bicubic_2d_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$dy,
    AnyType:$x
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$height_scale,
    DefaultValuedAttr<F32Attr, "0.">:$width_scale,
    DefaultValuedAttr<BoolAttr, "false">:$align_corners,
    StrAttr:$data_format
  );
}

def OneFlow_UpsampleBilinear2DOp : OneFlow_BaseOp<"upsample_bilinear_2d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$height_scale,
    DefaultValuedAttr<F32Attr, "0.">:$width_scale,
    DefaultValuedAttr<BoolAttr, "false">:$align_corners,
    StrAttr:$data_format
  );
}

def OneFlow_UpsampleBilinear2DGradOp : OneFlow_BaseOp<"upsample_bilinear_2d_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$dy,
    AnyType:$x
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$height_scale,
    DefaultValuedAttr<F32Attr, "0.">:$width_scale,
    DefaultValuedAttr<BoolAttr, "false">:$align_corners,
    StrAttr:$data_format
  );
}

def OneFlow_UpsampleGradOp : OneFlow_BaseOp<"upsample_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$dy,
    AnyType:$x
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$height_scale,
    DefaultValuedAttr<F32Attr, "0.">:$width_scale,
    DefaultValuedAttr<BoolAttr, "false">:$align_corners,
    StrAttr:$data_format,
    StrAttr:$interpolation
  );
}

def OneFlow_UpsampleLinear1DOp : OneFlow_BaseOp<"upsample_linear_1d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$scale_factor,
    DefaultValuedAttr<BoolAttr, "false">:$align_corners,
    StrAttr:$data_format
  );
}

def OneFlow_UpsampleLinear1DGradOp : OneFlow_BaseOp<"upsample_linear_1d_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$dy,
    AnyType:$x
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$scale_factor,
    DefaultValuedAttr<BoolAttr, "false">:$align_corners,
    StrAttr:$data_format
  );
}

def OneFlow_UpsampleNearest1DOp : OneFlow_BaseOp<"upsample_nearest_1d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$scale_factor,
    StrAttr:$data_format
  );
}

def OneFlow_UpsampleNearest1DGradOp : OneFlow_BaseOp<"upsample_nearest_1d_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$dy,
    AnyType:$x
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$scale_factor,
    StrAttr:$data_format
  );
}

def OneFlow_UpsampleNearest2DOp : OneFlow_BaseOp<"upsample_nearest_2d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$height_scale,
    DefaultValuedAttr<F32Attr, "0.">:$width_scale,
    StrAttr:$data_format
  );
}

def OneFlow_UpsampleNearest2DGradOp : OneFlow_BaseOp<"upsample_nearest_2d_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$dy,
    AnyType:$x
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$height_scale,
    DefaultValuedAttr<F32Attr, "0.">:$width_scale,
    StrAttr:$data_format
  );
}

def OneFlow_UpsampleNearest3DOp : OneFlow_BaseOp<"upsample_nearest_3d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$depth_scale,
    DefaultValuedAttr<F32Attr, "0.">:$height_scale,
    DefaultValuedAttr<F32Attr, "0.">:$width_scale,
    StrAttr:$data_format
  );
}

def OneFlow_UpsampleNearest3DGradOp : OneFlow_BaseOp<"upsample_nearest_3d_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$dy,
    AnyType:$x
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$depth_scale,
    DefaultValuedAttr<F32Attr, "0.">:$height_scale,
    DefaultValuedAttr<F32Attr, "0.">:$width_scale,
    StrAttr:$data_format
  );
}

def OneFlow_UpsampleTrilinear3DOp : OneFlow_BaseOp<"upsample_trilinear_3d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$depth_scale,
    DefaultValuedAttr<F32Attr, "0.">:$height_scale,
    DefaultValuedAttr<F32Attr, "0.">:$width_scale,
    DefaultValuedAttr<BoolAttr, "false">:$align_corners,
    StrAttr:$data_format
  );
}

def OneFlow_UpsampleTrilinear3DGradOp : OneFlow_BaseOp<"upsample_trilinear_3d_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$dy,
    AnyType:$x
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$depth_scale,
    DefaultValuedAttr<F32Attr, "0.">:$height_scale,
    DefaultValuedAttr<F32Attr, "0.">:$width_scale,
    DefaultValuedAttr<BoolAttr, "false">:$align_corners,
    StrAttr:$data_format
  );
}

#endif // GET_ONEFLOW_UPSAMPLE_OP_DEFINITIONS


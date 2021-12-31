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

def OneFlow_LogicalSliceAssignOp : OneFlow_BaseOp<"logical_slice_assign", [DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$ref,
    OneFlow_Tensor:$value
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
  let has_input_arg_modify_fn = 1;
}

#endif // GET_ONEFLOW_ASSIGN_OP_DEFINITIONS

// Group: BASE
// normalization_add_relu
// Total: 1

#ifdef GET_ONEFLOW_BASE_OP_DEFINITIONS

class OneFlow_NormalizationAddReluBaseOp : OneFlow_BaseOp<"normalization_add_relu", [NoSideEffect, AttrSizedOperandSegments, AttrSizedResultSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
    I32ElementsAttr:$operand_segment_sizes,
    I32ElementsAttr:$result_segment_sizes
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

#endif // GET_ONEFLOW_BASE_OP_DEFINITIONS

// Group: BINARY
// bias_add, cast_like, celu_grad, diag_grad, diagonal_grad, dot, dropout_grad, elementwise_maximum, elementwise_minimum, elu_grad, floordiv, gelu_grad, grid_sample, hardsigmoid_grad, hardswish_grad, l1_l2_regularize_gradient, leaky_relu_grad, masked_fill, mish_grad, multiply, narrow_grad, pow, prelu, relu_grad, selu_grad, sigmoid_grad, silu_grad, tf_prelu, unfold_tensor_grad, xdivy, xlogy
// Total: 31

#ifdef GET_ONEFLOW_BINARY_OP_DEFINITIONS

def OneFlow_BiasAddOp : OneFlow_BaseOp<"bias_add", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_CastLikeOp : OneFlow_BaseOp<"cast_like", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_CeluGradOp : OneFlow_BaseOp<"celu_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_DiagGradOp : OneFlow_BaseOp<"diag_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_DiagonalGradOp : OneFlow_BaseOp<"diagonal_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_DotOp : OneFlow_BaseOp<"dot", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_DropoutGradOp : OneFlow_BaseOp<"dropout_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ElementwiseMaximumOp : OneFlow_BaseOp<"elementwise_maximum", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ElementwiseMinimumOp : OneFlow_BaseOp<"elementwise_minimum", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_EluGradOp : OneFlow_BaseOp<"elu_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_FloordivOp : OneFlow_BaseOp<"floordiv", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_GeluGradOp : OneFlow_BaseOp<"gelu_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_GridSampleOp : OneFlow_BaseOp<"grid_sample", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_HardsigmoidGradOp : OneFlow_BaseOp<"hardsigmoid_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_HardswishGradOp : OneFlow_BaseOp<"hardswish_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_L1L2RegularizeGradientOp : OneFlow_BaseOp<"l1_l2_regularize_gradient", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_LeakyReluGradOp : OneFlow_BaseOp<"leaky_relu_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_MaskedFillOp : OneFlow_BaseOp<"masked_fill", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
    DefaultValuedAttr<SI64Attr, "0">:$int_operand,
    DefaultValuedAttr<F64Attr, "0.">:$float_operand
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_MishGradOp : OneFlow_BaseOp<"mish_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_MultiplyOp : OneFlow_BaseOp<"multiply", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_NarrowGradOp : OneFlow_BaseOp<"narrow_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_PowOp : OneFlow_BaseOp<"pow", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_PreluOp : OneFlow_BaseOp<"prelu", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ReluGradOp : OneFlow_BaseOp<"relu_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_SeluGradOp : OneFlow_BaseOp<"selu_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_SigmoidGradOp : OneFlow_BaseOp<"sigmoid_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_SiluGradOp : OneFlow_BaseOp<"silu_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_TfPreluOp : OneFlow_BaseOp<"tf_prelu", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_UnfoldTensorGradOp : OneFlow_BaseOp<"unfold_tensor_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_XdivyOp : OneFlow_BaseOp<"xdivy", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_XlogyOp : OneFlow_BaseOp<"xlogy", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

// Group: BROADCAST
// broadcast_add, broadcast_div, broadcast_div_grad, broadcast_equal, broadcast_floor_mod, broadcast_fmod, broadcast_greater, broadcast_greater_equal, broadcast_less, broadcast_less_equal, broadcast_like, broadcast_logical_and, broadcast_logical_or, broadcast_logical_xor, broadcast_maximum, broadcast_minimum, broadcast_mul, broadcast_not_equal, broadcast_pow, broadcast_pow_x_grad, broadcast_pow_y_grad, broadcast_sub
// Total: 22

#ifdef GET_ONEFLOW_BROADCAST_OP_DEFINITIONS

def OneFlow_BroadcastAddOp : OneFlow_BaseOp<"broadcast_add", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_BroadcastDivOp : OneFlow_BaseOp<"broadcast_div", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_BroadcastDivGradOp : OneFlow_BaseOp<"broadcast_div_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_BroadcastEqualOp : OneFlow_BaseOp<"broadcast_equal", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_BroadcastFloorModOp : OneFlow_BaseOp<"broadcast_floor_mod", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_BroadcastFmodOp : OneFlow_BaseOp<"broadcast_fmod", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_BroadcastGreaterOp : OneFlow_BaseOp<"broadcast_greater", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_BroadcastGreaterEqualOp : OneFlow_BaseOp<"broadcast_greater_equal", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_BroadcastLessOp : OneFlow_BaseOp<"broadcast_less", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_BroadcastLessEqualOp : OneFlow_BaseOp<"broadcast_less_equal", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_BroadcastLikeOp : OneFlow_BaseOp<"broadcast_like", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_BroadcastLogicalAndOp : OneFlow_BaseOp<"broadcast_logical_and", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_BroadcastLogicalOrOp : OneFlow_BaseOp<"broadcast_logical_or", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_BroadcastLogicalXorOp : OneFlow_BaseOp<"broadcast_logical_xor", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_BroadcastMaximumOp : OneFlow_BaseOp<"broadcast_maximum", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_BroadcastMinimumOp : OneFlow_BaseOp<"broadcast_minimum", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_BroadcastMulOp : OneFlow_BaseOp<"broadcast_mul", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_BroadcastNotEqualOp : OneFlow_BaseOp<"broadcast_not_equal", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_BroadcastPowOp : OneFlow_BaseOp<"broadcast_pow", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_BroadcastPowXGradOp : OneFlow_BaseOp<"broadcast_pow_x_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$y,
    OneFlow_Tensor:$z,
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

def OneFlow_BroadcastPowYGradOp : OneFlow_BaseOp<"broadcast_pow_y_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
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

def OneFlow_BroadcastSubOp : OneFlow_BaseOp<"broadcast_sub", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

// Group: CONV
// conv1d, conv2d, conv3d, conv_bias_grad, conv_data_grad, conv_filter_grad, deconv1d, deconv2d, deconv3d
// Total: 9

#ifdef GET_ONEFLOW_CONV_OP_DEFINITIONS

def OneFlow_Conv1DOp : OneFlow_ConvolutionBaseOp<"conv1d", [NoSideEffect, AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_Conv2DOp : OneFlow_ConvolutionBaseOp<"conv2d", [NoSideEffect, AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_Conv3DOp : OneFlow_ConvolutionBaseOp<"conv3d", [NoSideEffect, AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_ConvBiasGradOp : OneFlow_BaseOp<"conv_bias_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
}

def OneFlow_ConvDataGradOp : OneFlow_BaseOp<"conv_data_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
}

def OneFlow_ConvFilterGradOp : OneFlow_BaseOp<"conv_filter_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
}

def OneFlow_Deconv1DOp : OneFlow_BaseOp<"deconv1d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_Deconv2DOp : OneFlow_BaseOp<"deconv2d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_Deconv3DOp : OneFlow_BaseOp<"deconv3d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

#endif // GET_ONEFLOW_CONV_OP_DEFINITIONS

// Group: CROSS_ENTROPY
// binary_cross_entropy, binary_cross_entropy_grad, binary_cross_entropy_with_logits, binary_cross_entropy_with_logits_grad, sigmoid_cross_entropy, sigmoid_cross_entropy_grad, sparse_cross_entropy, sparse_cross_entropy_grad, sparse_cross_entropy_ms, sparse_cross_entropy_ms_grad
// Total: 10

#ifdef GET_ONEFLOW_CROSS_ENTROPY_OP_DEFINITIONS

def OneFlow_BinaryCrossEntropyOp : OneFlow_BaseOp<"binary_cross_entropy", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_BinaryCrossEntropyGradOp : OneFlow_BaseOp<"binary_cross_entropy_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_BinaryCrossEntropyWithLogitsOp : OneFlow_BaseOp<"binary_cross_entropy_with_logits", [NoSideEffect, AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
    I32ElementsAttr:$operand_segment_sizes
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_BinaryCrossEntropyWithLogitsGradOp : OneFlow_BaseOp<"binary_cross_entropy_with_logits_grad", [NoSideEffect, AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
    I32ElementsAttr:$operand_segment_sizes
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_SigmoidCrossEntropyOp : OneFlow_BaseOp<"sigmoid_cross_entropy", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_SigmoidCrossEntropyGradOp : OneFlow_BaseOp<"sigmoid_cross_entropy_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_SparseCrossEntropyOp : OneFlow_BaseOp<"sparse_cross_entropy", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_SparseCrossEntropyGradOp : OneFlow_BaseOp<"sparse_cross_entropy_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_SparseCrossEntropyMsOp : OneFlow_BaseOp<"sparse_cross_entropy_ms", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_SparseCrossEntropyMsGradOp : OneFlow_BaseOp<"sparse_cross_entropy_ms_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

// Group: CUDA
// nvtx_end, nvtx_start
// Total: 2

#ifdef GET_ONEFLOW_CUDA_OP_DEFINITIONS

def OneFlow_NvtxEndOp : OneFlow_BaseOp<"nvtx_end", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_NvtxStartOp : OneFlow_BaseOp<"nvtx_start", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

// Group: DATASET
// COCOReader, OFRecordReader, OneRecReader, ctc_greedy_decoder, megatron_gpt_mmap_data_loader, ofrecord_bytes_decoder, ofrecord_image_classification_reader, ofrecord_image_decoder, ofrecord_image_decoder_random_crop, ofrecord_raw_decoder, onerec_decoder
// Total: 11

#ifdef GET_ONEFLOW_DATASET_OP_DEFINITIONS

def OneFlow_COCOReaderOp : OneFlow_BaseOp<"COCOReader", [NoSideEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let output = (outs
    OneFlow_Tensor:$image,
    OneFlow_Tensor:$image_id,
    OneFlow_Tensor:$image_size,
    OneFlow_Tensor:$gt_bbox,
    OneFlow_Tensor:$gt_label,
    OneFlow_Tensor:$gt_segm,
    OneFlow_Tensor:$gt_segm_index
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

def OneFlow_OFRecordReaderOp : OneFlow_BaseOp<"OFRecordReader", [NoSideEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
}

def OneFlow_OneRecReaderOp : OneFlow_BaseOp<"OneRecReader", [NoSideEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let output = (outs
    OneFlow_Tensor:$out
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
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_CtcGreedyDecoderOp : OneFlow_BaseOp<"ctc_greedy_decoder", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_MegatronGptMmapDataLoaderOp : OneFlow_BaseOp<"megatron_gpt_mmap_data_loader", [NoSideEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_OfrecordBytesDecoderOp : OneFlow_BaseOp<"ofrecord_bytes_decoder", [NoSideEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_OfrecordImageClassificationReaderOp : OneFlow_BaseOp<"ofrecord_image_classification_reader", [NoSideEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_OfrecordImageDecoderOp : OneFlow_BaseOp<"ofrecord_image_decoder", [NoSideEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_OfrecordImageDecoderRandomCropOp : OneFlow_BaseOp<"ofrecord_image_decoder_random_crop", [NoSideEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_OfrecordRawDecoderOp : OneFlow_BaseOp<"ofrecord_raw_decoder", [NoSideEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_OnerecDecoderOp : OneFlow_BaseOp<"onerec_decoder", [NoSideEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    StrAttr:$key,
    OneFlow_DataType:$data_type,
    ShapeAttr:$static_shape,
    DefaultValuedAttr<BoolAttr, "false">:$is_dynamic,
    DefaultValuedAttr<BoolAttr, "false">:$has_reshape,
    ShapeAttr:$reshape,
    DefaultValuedAttr<BoolAttr, "false">:$has_batch_padding,
    ShapeAttr:$batch_padding
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
  let has_output_arg_modify_fn = 1;
}

#endif // GET_ONEFLOW_DATASET_OP_DEFINITIONS

// Group: DETECTION
// in_top_k, nms, object_bbox_flip, object_bbox_scale, object_segmentation_polygon_flip, object_segmentation_polygon_scale, object_segmentation_polygon_to_mask, roi_align, roi_align_grad, top_k
// Total: 10

#ifdef GET_ONEFLOW_DETECTION_OP_DEFINITIONS

def OneFlow_InTopKOp : OneFlow_BaseOp<"in_top_k", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_NmsOp : OneFlow_BaseOp<"nms", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ObjectBboxFlipOp : OneFlow_BaseOp<"object_bbox_flip", [NoSideEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ObjectBboxScaleOp : OneFlow_BaseOp<"object_bbox_scale", [NoSideEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ObjectSegmentationPolygonFlipOp : OneFlow_BaseOp<"object_segmentation_polygon_flip", [NoSideEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ObjectSegmentationPolygonScaleOp : OneFlow_BaseOp<"object_segmentation_polygon_scale", [NoSideEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ObjectSegmentationPolygonToMaskOp : OneFlow_BaseOp<"object_segmentation_polygon_to_mask", [NoSideEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_RoiAlignOp : OneFlow_BaseOp<"roi_align", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_RoiAlignGradOp : OneFlow_BaseOp<"roi_align_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_TopKOp : OneFlow_BaseOp<"top_k", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

// Group: EAGER
// eager_b_to_s, eager_naive_s_to_s, eager_nccl_all_gather, eager_nccl_all_reduce, eager_nccl_broadcast, eager_nccl_reduce, eager_nccl_reduce_scatter, eager_nccl_s2s, eager_p_to_b, eager_p_to_s, eager_s_to_b, eager_symmetric_s_to_p
// Total: 12

#ifdef GET_ONEFLOW_EAGER_OP_DEFINITIONS

def OneFlow_EagerBToSOp : OneFlow_BaseOp<"eager_b_to_s", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
  let has_device_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

def OneFlow_EagerNaiveSToSOp : OneFlow_BaseOp<"eager_naive_s_to_s", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
  let has_device_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

def OneFlow_EagerNcclAllGatherOp : OneFlow_BaseOp<"eager_nccl_all_gather", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
  let has_device_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

def OneFlow_EagerNcclAllReduceOp : OneFlow_BaseOp<"eager_nccl_all_reduce", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    StrAttr:$parallel_conf,
    DefaultValuedAttr<BoolAttr, "false">:$async_launch
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_device_infer_fn = 1;
}

def OneFlow_EagerNcclBroadcastOp : OneFlow_BaseOp<"eager_nccl_broadcast", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
  let has_device_infer_fn = 1;
}

def OneFlow_EagerNcclReduceOp : OneFlow_BaseOp<"eager_nccl_reduce", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
  let has_device_infer_fn = 1;
}

def OneFlow_EagerNcclReduceScatterOp : OneFlow_BaseOp<"eager_nccl_reduce_scatter", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    StrAttr:$parallel_conf,
    DefaultValuedAttr<StrAttr, "\"sum\"">:$op_type
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_device_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

def OneFlow_EagerNcclS2sOp : OneFlow_BaseOp<"eager_nccl_s2s", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
  let has_device_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

def OneFlow_EagerPToBOp : OneFlow_BaseOp<"eager_p_to_b", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
  let has_device_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

def OneFlow_EagerPToSOp : OneFlow_BaseOp<"eager_p_to_s", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
  let has_device_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

def OneFlow_EagerSToBOp : OneFlow_BaseOp<"eager_s_to_b", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
  let has_device_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

def OneFlow_EagerSymmetricSToPOp : OneFlow_BaseOp<"eager_symmetric_s_to_p", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
  let has_device_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

#endif // GET_ONEFLOW_EAGER_OP_DEFINITIONS

// Group: FUSED
// cudnn_fused_normalization_add_relu, cudnn_fused_normalization_add_relu_grad, fused_bias_add_gelu, fused_bias_add_gelu_grad, fused_bias_add_mask_scale, fused_cast_scale, fused_scale_mask_softmax, fused_scale_mask_softmax_dropout, fused_scale_mask_softmax_dropout_grad, fused_scale_mask_softmax_grad, fused_scale_tril, fused_self_attention_query_mul_key_and_value, fused_self_attention_query_mul_key_and_value_grad, fused_tril_scale_softmax_mask_scale, fused_tril_scale_softmax_mask_scale_grad, normalization_add_relu_grad
// Total: 16

#ifdef GET_ONEFLOW_FUSED_OP_DEFINITIONS

def OneFlow_CudnnFusedNormalizationAddReluOp : OneFlow_BaseOp<"cudnn_fused_normalization_add_relu", [NoSideEffect, AttrSizedOperandSegments, AttrSizedResultSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
    I32ElementsAttr:$operand_segment_sizes,
    I32ElementsAttr:$result_segment_sizes
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_CudnnFusedNormalizationAddReluGradOp : OneFlow_BaseOp<"cudnn_fused_normalization_add_relu_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_FusedBiasAddGeluOp : OneFlow_BaseOp<"fused_bias_add_gelu", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_FusedBiasAddGeluGradOp : OneFlow_BaseOp<"fused_bias_add_gelu_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_FusedBiasAddMaskScaleOp : OneFlow_BaseOp<"fused_bias_add_mask_scale", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
    DefaultValuedAttr<F32Attr, "0.">:$scale
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_FusedCastScaleOp : OneFlow_BaseOp<"fused_cast_scale", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_FusedScaleMaskSoftmaxOp : OneFlow_BaseOp<"fused_scale_mask_softmax", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_FusedScaleMaskSoftmaxDropoutOp : OneFlow_BaseOp<"fused_scale_mask_softmax_dropout", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_FusedScaleMaskSoftmaxDropoutGradOp : OneFlow_BaseOp<"fused_scale_mask_softmax_dropout_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_FusedScaleMaskSoftmaxGradOp : OneFlow_BaseOp<"fused_scale_mask_softmax_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_FusedScaleTrilOp : OneFlow_BaseOp<"fused_scale_tril", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_FusedSelfAttentionQueryMulKeyAndValueOp : OneFlow_BaseOp<"fused_self_attention_query_mul_key_and_value", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_FusedSelfAttentionQueryMulKeyAndValueGradOp : OneFlow_BaseOp<"fused_self_attention_query_mul_key_and_value_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_FusedTrilScaleSoftmaxMaskScaleOp : OneFlow_BaseOp<"fused_tril_scale_softmax_mask_scale", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_FusedTrilScaleSoftmaxMaskScaleGradOp : OneFlow_BaseOp<"fused_tril_scale_softmax_mask_scale_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_NormalizationAddReluGradOp : OneFlow_BaseOp<"normalization_add_relu_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

#endif // GET_ONEFLOW_FUSED_OP_DEFINITIONS

// Group: IDEMPOTENT
// abs, ceil, floor, ones_like, relu, rint, round, sign
// Total: 8

#ifdef GET_ONEFLOW_IDEMPOTENT_OP_DEFINITIONS

def OneFlow_AbsOp : OneFlow_IdempotentBaseOp<"abs", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_CeilOp : OneFlow_IdempotentBaseOp<"ceil", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_FloorOp : OneFlow_IdempotentBaseOp<"floor", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_OnesLikeOp : OneFlow_IdempotentBaseOp<"ones_like", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let same_output_regst_num = 1;
}

def OneFlow_ReluOp : OneFlow_IdempotentBaseOp<"relu", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_RintOp : OneFlow_IdempotentBaseOp<"rint", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_RoundOp : OneFlow_IdempotentBaseOp<"round", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_SignOp : OneFlow_IdempotentBaseOp<"sign", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

#endif // GET_ONEFLOW_IDEMPOTENT_OP_DEFINITIONS

// Group: IDENTITY
// amp_white_identity, identity, identity_buffer, tuple_identity
// Total: 4

#ifdef GET_ONEFLOW_IDENTITY_OP_DEFINITIONS

def OneFlow_AmpWhiteIdentityOp : OneFlow_BaseOp<"amp_white_identity", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_IdentityOp : OneFlow_BaseOp<"identity", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_IdentityBufferOp : OneFlow_BaseOp<"identity_buffer", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_TupleIdentityOp : OneFlow_BaseOp<"tuple_identity", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

#endif // GET_ONEFLOW_IDENTITY_OP_DEFINITIONS

// Group: IMAGE
// image_batch_align, image_decode, image_flip, image_random_crop, image_resize_keep_aspect_ratio, image_resize_to_fixed
// Total: 6

#ifdef GET_ONEFLOW_IMAGE_OP_DEFINITIONS

def OneFlow_ImageBatchAlignOp : OneFlow_BaseOp<"image_batch_align", [NoSideEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ImageDecodeOp : OneFlow_BaseOp<"image_decode", [NoSideEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ImageFlipOp : OneFlow_BaseOp<"image_flip", [NoSideEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ImageRandomCropOp : OneFlow_BaseOp<"image_random_crop", [NoSideEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ImageResizeKeepAspectRatioOp : OneFlow_BaseOp<"image_resize_keep_aspect_ratio", [NoSideEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ImageResizeToFixedOp : OneFlow_BaseOp<"image_resize_to_fixed", [NoSideEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

// Group: INDICES
// arg_sort, argmax, argwhere, batch_gather, dim_gather, dim_scatter_add, dim_scatter_add_like, dim_scatter_add_scalar, dim_scatter_mul, dim_scatter_mul_scalar, dim_scatter_update, dim_scatter_update_scalar, gather, gather_nd, generate_random_batch_permutation_indices, image_target_resize, logical_slice, scatter_nd, scatter_nd_like, slice, slice_grad, tensor_scatter_nd_add, tensor_scatter_nd_update, unsorted_batch_segment_sum, unsorted_segment_sum, unsorted_segment_sum_like, where, where_scalar_x, where_scalar_xy, where_scalar_y
// Total: 30

#ifdef GET_ONEFLOW_INDICES_OP_DEFINITIONS

def OneFlow_ArgSortOp : OneFlow_BaseOp<"arg_sort", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ArgmaxOp : OneFlow_BaseOp<"argmax", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ArgwhereOp : OneFlow_BaseOp<"argwhere", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_BatchGatherOp : OneFlow_BaseOp<"batch_gather", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_DimGatherOp : OneFlow_BaseOp<"dim_gather", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_DimScatterAddOp : OneFlow_BaseOp<"dim_scatter_add", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_DimScatterAddLikeOp : OneFlow_BaseOp<"dim_scatter_add_like", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_DimScatterAddScalarOp : OneFlow_BaseOp<"dim_scatter_add_scalar", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_DimScatterMulOp : OneFlow_BaseOp<"dim_scatter_mul", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_DimScatterMulScalarOp : OneFlow_BaseOp<"dim_scatter_mul_scalar", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_DimScatterUpdateOp : OneFlow_BaseOp<"dim_scatter_update", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_DimScatterUpdateScalarOp : OneFlow_BaseOp<"dim_scatter_update_scalar", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_GatherOp : OneFlow_BaseOp<"gather", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_GatherNdOp : OneFlow_BaseOp<"gather_nd", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_GenerateRandomBatchPermutationIndicesOp : OneFlow_BaseOp<"generate_random_batch_permutation_indices", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ImageTargetResizeOp : OneFlow_BaseOp<"image_target_resize", [NoSideEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_LogicalSliceOp : OneFlow_BaseOp<"logical_slice", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ScatterNdOp : OneFlow_BaseOp<"scatter_nd", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ScatterNdLikeOp : OneFlow_BaseOp<"scatter_nd_like", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_SliceOp : OneFlow_BaseOp<"slice", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_SliceGradOp : OneFlow_BaseOp<"slice_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_TensorScatterNdAddOp : OneFlow_BaseOp<"tensor_scatter_nd_add", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_TensorScatterNdUpdateOp : OneFlow_BaseOp<"tensor_scatter_nd_update", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_UnsortedBatchSegmentSumOp : OneFlow_BaseOp<"unsorted_batch_segment_sum", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_UnsortedSegmentSumOp : OneFlow_BaseOp<"unsorted_segment_sum", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_UnsortedSegmentSumLikeOp : OneFlow_BaseOp<"unsorted_segment_sum_like", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_WhereOp : OneFlow_BaseOp<"where", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_WhereScalarXOp : OneFlow_BaseOp<"where_scalar_x", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$condition,
    OneFlow_Tensor:$y
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
  let has_input_arg_modify_fn = 1;
}

def OneFlow_WhereScalarXyOp : OneFlow_BaseOp<"where_scalar_xy", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$condition
  );
  let output = (outs
    OneFlow_Tensor:$out
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
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_WhereScalarYOp : OneFlow_BaseOp<"where_scalar_y", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$condition,
    OneFlow_Tensor:$x
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
  let has_input_arg_modify_fn = 1;
}

#endif // GET_ONEFLOW_INDICES_OP_DEFINITIONS

// Group: INVOLUTION
// negative, reciprocal
// Total: 2

#ifdef GET_ONEFLOW_INVOLUTION_OP_DEFINITIONS

def OneFlow_NegativeOp : OneFlow_InvolutionBaseOp<"negative", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_ReciprocalOp : OneFlow_InvolutionBaseOp<"reciprocal", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

#endif // GET_ONEFLOW_INVOLUTION_OP_DEFINITIONS

// Group: LOSS
// combined_margin_loss, combined_margin_loss_grad, ctc_loss, ctc_loss_grad, dynamic_loss_scale_schedule, kl_div_loss, kl_div_loss_grad, smooth_l1_loss, smooth_l1_loss_grad
// Total: 9

#ifdef GET_ONEFLOW_LOSS_OP_DEFINITIONS

def OneFlow_CombinedMarginLossOp : OneFlow_BaseOp<"combined_margin_loss", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_CombinedMarginLossGradOp : OneFlow_BaseOp<"combined_margin_loss_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_CtcLossOp : OneFlow_BaseOp<"ctc_loss", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_CtcLossGradOp : OneFlow_BaseOp<"ctc_loss_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_DynamicLossScaleScheduleOp : OneFlow_BaseOp<"dynamic_loss_scale_schedule", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_KlDivLossOp : OneFlow_BaseOp<"kl_div_loss", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_KlDivLossGradOp : OneFlow_BaseOp<"kl_div_loss_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_SmoothL1LossOp : OneFlow_BaseOp<"smooth_l1_loss", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_SmoothL1LossGradOp : OneFlow_BaseOp<"smooth_l1_loss_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

// Group: MATH
// abs_grad, ceil_grad, erf, erf_grad, exp, exp_grad, expand_grad, expm1, expm1_grad, floor_grad, floordiv_x_grad, floordiv_y_grad, lgamma, lgamma_grad, log, log1p, log1p_grad, log2_grad, log_grad, log_sigmoid, log_sigmoid_grad, negative_grad, reciprocal_grad, reciprocal_no_nan, reciprocal_no_nan_grad, rint_grad, round_grad, rsqrt, rsqrt_grad, sigmoid_v2, sigmoid_v2_grad, sign_grad, softplus, softplus_grad, softsign_grad, sqrt, sqrt_grad, square, square_grad, xlogy_x_grad, xlogy_y_grad
// Total: 41

#ifdef GET_ONEFLOW_MATH_OP_DEFINITIONS

def OneFlow_AbsGradOp : OneFlow_BaseOp<"abs_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_CeilGradOp : OneFlow_BaseOp<"ceil_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ErfOp : OneFlow_BaseOp<"erf", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ErfGradOp : OneFlow_BaseOp<"erf_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ExpOp : OneFlow_BaseOp<"exp", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ExpGradOp : OneFlow_BaseOp<"exp_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ExpandGradOp : OneFlow_BaseOp<"expand_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    SI32ArrayAttr:$logical_out_shape,
    SI32ArrayAttr:$logical_expand_shape
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_Expm1Op : OneFlow_BaseOp<"expm1", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_Expm1GradOp : OneFlow_BaseOp<"expm1_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_FloorGradOp : OneFlow_BaseOp<"floor_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_FloordivXGradOp : OneFlow_BaseOp<"floordiv_x_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_FloordivYGradOp : OneFlow_BaseOp<"floordiv_y_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_LgammaOp : OneFlow_BaseOp<"lgamma", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_LgammaGradOp : OneFlow_BaseOp<"lgamma_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_LogOp : OneFlow_BaseOp<"log", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_Log1pOp : OneFlow_BaseOp<"log1p", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_Log1pGradOp : OneFlow_BaseOp<"log1p_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_Log2GradOp : OneFlow_BaseOp<"log2_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_LogGradOp : OneFlow_BaseOp<"log_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_LogSigmoidOp : OneFlow_BaseOp<"log_sigmoid", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_LogSigmoidGradOp : OneFlow_BaseOp<"log_sigmoid_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_NegativeGradOp : OneFlow_BaseOp<"negative_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ReciprocalGradOp : OneFlow_BaseOp<"reciprocal_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ReciprocalNoNanOp : OneFlow_BaseOp<"reciprocal_no_nan", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ReciprocalNoNanGradOp : OneFlow_BaseOp<"reciprocal_no_nan_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_RintGradOp : OneFlow_BaseOp<"rint_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_RoundGradOp : OneFlow_BaseOp<"round_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_RsqrtOp : OneFlow_BaseOp<"rsqrt", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_RsqrtGradOp : OneFlow_BaseOp<"rsqrt_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_SigmoidV2Op : OneFlow_BaseOp<"sigmoid_v2", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_SigmoidV2GradOp : OneFlow_BaseOp<"sigmoid_v2_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_SignGradOp : OneFlow_BaseOp<"sign_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_SoftplusOp : OneFlow_BaseOp<"softplus", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_SoftplusGradOp : OneFlow_BaseOp<"softplus_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_SoftsignGradOp : OneFlow_BaseOp<"softsign_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_SqrtOp : OneFlow_BaseOp<"sqrt", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_SqrtGradOp : OneFlow_BaseOp<"sqrt_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_SquareOp : OneFlow_BaseOp<"square", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_SquareGradOp : OneFlow_BaseOp<"square_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_XlogyXGradOp : OneFlow_BaseOp<"xlogy_x_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_XlogyYGradOp : OneFlow_BaseOp<"xlogy_y_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

#endif // GET_ONEFLOW_MATH_OP_DEFINITIONS

// Group: MATMUL
// batch_matmul, broadcast_matmul, broadcast_matmul_grad_b, distributed_partial_fc_sample, distributed_partial_fc_sample_disable_boxing, erfc, erfc_grad, matmul
// Total: 8

#ifdef GET_ONEFLOW_MATMUL_OP_DEFINITIONS

def OneFlow_BatchMatmulOp : OneFlow_BaseOp<"batch_matmul", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
}

def OneFlow_BroadcastMatmulOp : OneFlow_BaseOp<"broadcast_matmul", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
}

def OneFlow_BroadcastMatmulGradBOp : OneFlow_BaseOp<"broadcast_matmul_grad_b", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
}

def OneFlow_DistributedPartialFcSampleOp : OneFlow_BaseOp<"distributed_partial_fc_sample", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_DistributedPartialFcSampleDisableBoxingOp : OneFlow_BaseOp<"distributed_partial_fc_sample_disable_boxing", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ErfcOp : OneFlow_BaseOp<"erfc", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ErfcGradOp : OneFlow_BaseOp<"erfc_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_MatmulOp : OneFlow_BaseOp<"matmul", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
}

#endif // GET_ONEFLOW_MATMUL_OP_DEFINITIONS

// Group: MISC
// CategoricalOrdinalEncode, add_n, arange, coin_flip, concat, constant, dropout, elementwise_maximum_backward, elementwise_minimum_backward, empty, eye, grid_sample_grad, multi_count_not_finite, multi_square_sum, nll, nll_grad, pow_x_grad, pow_y_grad, prelu_grad, randperm, recv, send, split_like, ssp_variable_proxy, tf_prelu_grad, uniform, uniform_int, unique_with_counts, xdivy_x_grad, xdivy_y_grad
// Total: 30

#ifdef GET_ONEFLOW_MISC_OP_DEFINITIONS

def OneFlow_CategoricalOrdinalEncodeOp : OneFlow_BaseOp<"CategoricalOrdinalEncode", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_AddNOp : OneFlow_BaseOp<"add_n", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ArangeOp : OneFlow_BaseOp<"arange", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_CoinFlipOp : OneFlow_BaseOp<"coin_flip", [NoSideEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ConcatOp : OneFlow_BaseOp<"concat", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ConstantOp : OneFlow_BaseOp<"constant", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.">:$floating_value,
    DefaultValuedAttr<SI64Attr, "0">:$integer_value,
    DefaultValuedAttr<BoolAttr, "false">:$is_floating_value,
    OneFlow_DataType:$dtype,
    ShapeAttr:$shape,
    StrArrayAttr:$nd_sbp
  );
  let same_output_regst_num = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

def OneFlow_DropoutOp : OneFlow_BaseOp<"dropout", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in,
    Optional<OneFlow_Tensor>:$_add_to_output
  );
  let output = (outs
    OneFlow_Tensor:$out,
    OneFlow_Tensor:$mask
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$rate
  );
  let has_check_fn = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ElementwiseMaximumBackwardOp : OneFlow_BaseOp<"elementwise_maximum_backward", [NoSideEffect, AttrSizedResultSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
    I32ElementsAttr:$result_segment_sizes
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ElementwiseMinimumBackwardOp : OneFlow_BaseOp<"elementwise_minimum_backward", [NoSideEffect, AttrSizedResultSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
    I32ElementsAttr:$result_segment_sizes
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_EmptyOp : OneFlow_BaseOp<"empty", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_EyeOp : OneFlow_BaseOp<"eye", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_GridSampleGradOp : OneFlow_BaseOp<"grid_sample_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_MultiCountNotFiniteOp : OneFlow_BaseOp<"multi_count_not_finite", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_MultiSquareSumOp : OneFlow_BaseOp<"multi_square_sum", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_NllOp : OneFlow_BaseOp<"nll", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$input,
    OneFlow_Tensor:$target,
    Optional<OneFlow_Tensor>:$weight
  );
  let output = (outs
    OneFlow_Tensor:$out,
    OneFlow_Tensor:$total_weight
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$ignore_index
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_NllGradOp : OneFlow_BaseOp<"nll_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$input,
    OneFlow_Tensor:$target,
    OneFlow_Tensor:$total_weight,
    Optional<OneFlow_Tensor>:$weight,
    OneFlow_Tensor:$dy
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$ignore_index
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_PowXGradOp : OneFlow_BaseOp<"pow_x_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_PowYGradOp : OneFlow_BaseOp<"pow_y_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_PreluGradOp : OneFlow_BaseOp<"prelu_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_RandpermOp : OneFlow_BaseOp<"randperm", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_RecvOp : OneFlow_BaseOp<"recv", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
  let has_device_infer_fn = 1;
}

def OneFlow_SendOp : OneFlow_BaseOp<"send", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
  let has_device_infer_fn = 1;
}

def OneFlow_SplitLikeOp : OneFlow_BaseOp<"split_like", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_SspVariableProxyOp : OneFlow_BaseOp<"ssp_variable_proxy", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_TfPreluGradOp : OneFlow_BaseOp<"tf_prelu_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_UniformOp : OneFlow_BaseOp<"uniform", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
}

def OneFlow_UniformIntOp : OneFlow_BaseOp<"uniform_int", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_UniqueWithCountsOp : OneFlow_BaseOp<"unique_with_counts", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y,
    OneFlow_Tensor:$idx,
    OneFlow_Tensor:$count,
    OneFlow_Tensor:$num_unique
  );
  let attrs = (ins
    OneFlow_DataType:$out_idx
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_XdivyXGradOp : OneFlow_BaseOp<"xdivy_x_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_XdivyYGradOp : OneFlow_BaseOp<"xdivy_y_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

#endif // GET_ONEFLOW_MISC_OP_DEFINITIONS

// Group: NCCL
// _nccl_logical_2D_same_dim0_all2all, _nccl_logical_2D_same_dim0_all_gather, _nccl_logical_2D_same_dim0_all_gather_noncontinuous, _nccl_logical_2D_same_dim0_all_reduce, _nccl_logical_2D_same_dim1_all_reduce, _nccl_logical_all_gather, _nccl_logical_all_gather_noncontinuous, _nccl_logical_all_reduce, _nccl_logical_reduce_scatter, _nccl_logical_s2s
// Total: 10

#ifdef GET_ONEFLOW_NCCL_OP_DEFINITIONS

def OneFlow__ncclLogical_2DSameDim0All2allOp : OneFlow_BaseOp<"_nccl_logical_2D_same_dim0_all2all", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "-1">:$in_dim1_split_axis,
    DefaultValuedAttr<SI64Attr, "-1">:$out_dim1_split_axis
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_device_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

def OneFlow__ncclLogical_2DSameDim0AllGatherOp : OneFlow_BaseOp<"_nccl_logical_2D_same_dim0_all_gather", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_device_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

def OneFlow__ncclLogical_2DSameDim0AllGatherNoncontinuousOp : OneFlow_BaseOp<"_nccl_logical_2D_same_dim0_all_gather_noncontinuous", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "-1">:$in_dim1_split_axis
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_device_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

def OneFlow__ncclLogical_2DSameDim0AllReduceOp : OneFlow_BaseOp<"_nccl_logical_2D_same_dim0_all_reduce", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_device_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

def OneFlow__ncclLogical_2DSameDim1AllReduceOp : OneFlow_BaseOp<"_nccl_logical_2D_same_dim1_all_reduce", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_device_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

def OneFlow__ncclLogicalAllGatherOp : OneFlow_BaseOp<"_nccl_logical_all_gather", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_device_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

def OneFlow__ncclLogicalAllGatherNoncontinuousOp : OneFlow_BaseOp<"_nccl_logical_all_gather_noncontinuous", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "-1">:$in_split_axis
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_device_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

def OneFlow__ncclLogicalAllReduceOp : OneFlow_BaseOp<"_nccl_logical_all_reduce", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_device_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

def OneFlow__ncclLogicalReduceScatterOp : OneFlow_BaseOp<"_nccl_logical_reduce_scatter", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_device_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

def OneFlow__ncclLogicalS2sOp : OneFlow_BaseOp<"_nccl_logical_s2s", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "-1">:$in_split_axis,
    DefaultValuedAttr<SI64Attr, "-1">:$out_split_axis
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_device_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

#endif // GET_ONEFLOW_NCCL_OP_DEFINITIONS

// Group: NORMALIZATION
// crop_mirror_normalize_from_tensorbuffer, crop_mirror_normalize_from_uint8, image_normalize, l2_normalize, l2_normalize_grad, layer_norm, layer_norm_grad, layer_norm_param_grad, normal, normalization, normalization_grad
// Total: 11

#ifdef GET_ONEFLOW_NORMALIZATION_OP_DEFINITIONS

def OneFlow_CropMirrorNormalizeFromTensorbufferOp : OneFlow_BaseOp<"crop_mirror_normalize_from_tensorbuffer", [NoSideEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_CropMirrorNormalizeFromUint8Op : OneFlow_BaseOp<"crop_mirror_normalize_from_uint8", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ImageNormalizeOp : OneFlow_BaseOp<"image_normalize", [NoSideEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_L2NormalizeOp : OneFlow_BaseOp<"l2_normalize", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_L2NormalizeGradOp : OneFlow_BaseOp<"l2_normalize_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_LayerNormOp : OneFlow_BaseOp<"layer_norm", [NoSideEffect, AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    Optional<OneFlow_Tensor>:$beta,
    Optional<OneFlow_Tensor>:$gamma
  );
  let output = (outs
    OneFlow_Tensor:$y,
    OneFlow_Tensor:$mean,
    OneFlow_Tensor:$inv_variance,
    Optional<OneFlow_Tensor>:$normalized
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
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_LayerNormGradOp : OneFlow_BaseOp<"layer_norm_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$mean,
    OneFlow_Tensor:$inv_variance,
    Optional<OneFlow_Tensor>:$_add_to_output
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$begin_norm_axis,
    DefaultValuedAttr<F64Attr, "0.">:$epsilon
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_LayerNormParamGradOp : OneFlow_BaseOp<"layer_norm_param_grad", [NoSideEffect, AttrSizedOperandSegments, AttrSizedResultSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    Optional<OneFlow_Tensor>:$normalized,
    Optional<OneFlow_Tensor>:$gamma
  );
  let output = (outs
    Optional<OneFlow_Tensor>:$normalized_diff,
    Optional<OneFlow_Tensor>:$beta_diff,
    Optional<OneFlow_Tensor>:$gamma_diff,
    Optional<OneFlow_Tensor>:$reduce_buf
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$begin_params_axis
  );
  let trait_attrs = (ins
    I32ElementsAttr:$operand_segment_sizes,
    I32ElementsAttr:$result_segment_sizes
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_NormalOp : OneFlow_BaseOp<"normal", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_NormalizationOp : OneFlow_BaseOp<"normalization", [NoSideEffect, AttrSizedOperandSegments, AttrSizedResultSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
    I32ElementsAttr:$operand_segment_sizes,
    I32ElementsAttr:$result_segment_sizes
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_NormalizationGradOp : OneFlow_BaseOp<"normalization_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

#endif // GET_ONEFLOW_NORMALIZATION_OP_DEFINITIONS

// Group: OPTIMIZER
// adagrad_update, adam_bias_correction_factor, adam_update, indexed_slices_adam_update, indexed_slices_momentum_update, indexed_slices_sgd_update, lamb_update, lars_update, momentum_update, rmsprop_update, sgd_update, slice_update
// Total: 12

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
    Optional<OneFlow_Tensor>:$learning_rate,
    Optional<OneFlow_Tensor>:$scale_by_tensor,
    Optional<OneFlow_Tensor>:$skip_if,
    Optional<OneFlow_Tensor>:$bias_correction1,
    Optional<OneFlow_Tensor>:$bias_correction2,
    OneFlow_Tensor:$m,
    OneFlow_Tensor:$v,
    OneFlow_Tensor:$max_v
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
    OneFlow_Tensor:$max_v
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
    DefaultValuedAttr<F32Attr, "0.9">:$beta,
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
    OneFlow_Tensor:$m,
    OneFlow_Tensor:$v,
    OneFlow_Tensor:$beta1_t,
    OneFlow_Tensor:$beta2_t,
    OneFlow_Tensor:$model,
    OneFlow_Tensor:$model_diff,
    OneFlow_Tensor:$learning_rate,
    Optional<OneFlow_Tensor>:$scale_by_tensor,
    Optional<OneFlow_Tensor>:$skip_if
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
    DefaultValuedAttr<F64Attr, "1.">:$scale,
    DefaultValuedAttr<F32Attr, "0.">:$l1,
    DefaultValuedAttr<F32Attr, "0.">:$l2,
    DefaultValuedAttr<F32Attr, "0.9">:$beta,
    DefaultValuedAttr<F32Attr, "0.">:$weight_decay
  );
  let trait_attrs = (ins
    I32ElementsAttr:$operand_segment_sizes
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
    Optional<OneFlow_Tensor>:$learning_rate,
    Optional<OneFlow_Tensor>:$scale_by_tensor,
    Optional<OneFlow_Tensor>:$skip_if
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
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_SliceUpdateOp : OneFlow_BaseOp<"slice_update", [DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x,
    OneFlow_Tensor:$update
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

#endif // GET_ONEFLOW_OPTIMIZER_OP_DEFINITIONS

// Group: PADDING
// constant_pad1d, constant_pad1d_grad, constant_pad2d, constant_pad2d_grad, constant_pad3d, constant_pad3d_grad, pad, pad_grad, reflection_pad2d, reflection_pad2d_grad, replication_pad2d, replication_pad2d_grad, same_padding, same_padding_grad
// Total: 14

#ifdef GET_ONEFLOW_PADDING_OP_DEFINITIONS

def OneFlow_ConstantPad1DOp : OneFlow_BaseOp<"constant_pad1d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let attrs = (ins
    SI64ArrayAttr:$padding,
    DefaultValuedAttr<F64Attr, "0.">:$floating_value,
    DefaultValuedAttr<SI64Attr, "0">:$integral_value
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_ConstantPad1DGradOp : OneFlow_BaseOp<"constant_pad1d_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    SI64ArrayAttr:$padding,
    DefaultValuedAttr<F64Attr, "0.">:$floating_value,
    DefaultValuedAttr<SI64Attr, "0">:$integral_value
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ConstantPad2DOp : OneFlow_BaseOp<"constant_pad2d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let attrs = (ins
    SI64ArrayAttr:$padding,
    DefaultValuedAttr<F64Attr, "0.">:$floating_value,
    DefaultValuedAttr<SI64Attr, "0">:$integral_value
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_ConstantPad2DGradOp : OneFlow_BaseOp<"constant_pad2d_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    SI64ArrayAttr:$padding,
    DefaultValuedAttr<F64Attr, "0.">:$floating_value,
    DefaultValuedAttr<SI64Attr, "0">:$integral_value
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ConstantPad3DOp : OneFlow_BaseOp<"constant_pad3d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let attrs = (ins
    SI64ArrayAttr:$padding,
    DefaultValuedAttr<F64Attr, "0.">:$floating_value,
    DefaultValuedAttr<SI64Attr, "0">:$integral_value
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_input_arg_modify_fn = 1;
}

def OneFlow_ConstantPad3DGradOp : OneFlow_BaseOp<"constant_pad3d_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    SI64ArrayAttr:$padding,
    DefaultValuedAttr<F64Attr, "0.">:$floating_value,
    DefaultValuedAttr<SI64Attr, "0">:$integral_value
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_PadOp : OneFlow_BaseOp<"pad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let attrs = (ins
    SI64ArrayAttr:$padding_before,
    SI64ArrayAttr:$padding_after,
    DefaultValuedAttr<F64Attr, "0.">:$floating_constant_value,
    DefaultValuedAttr<SI64Attr, "0">:$integral_constant_value
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_PadGradOp : OneFlow_BaseOp<"pad_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    SI64ArrayAttr:$padding_before,
    SI64ArrayAttr:$padding_after,
    DefaultValuedAttr<F64Attr, "0.">:$floating_constant_value,
    DefaultValuedAttr<SI64Attr, "0">:$integral_constant_value
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ReflectionPad2DOp : OneFlow_BaseOp<"reflection_pad2d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ReflectionPad2DGradOp : OneFlow_BaseOp<"reflection_pad2d_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ReplicationPad2DOp : OneFlow_BaseOp<"replication_pad2d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ReplicationPad2DGradOp : OneFlow_BaseOp<"replication_pad2d_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_SamePaddingOp : OneFlow_BaseOp<"same_padding", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_SamePaddingGradOp : OneFlow_BaseOp<"same_padding_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

// Group: PARALLEL_CAST
// hierarchical_parallel_cast, hierarchical_parallel_cast_like, parallel_cast
// Total: 3

#ifdef GET_ONEFLOW_PARALLEL_CAST_OP_DEFINITIONS

def OneFlow_HierarchicalParallelCastOp : OneFlow_BaseOp<"hierarchical_parallel_cast", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
}

def OneFlow_HierarchicalParallelCastLikeOp : OneFlow_BaseOp<"hierarchical_parallel_cast_like", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ParallelCastOp : OneFlow_BaseOp<"parallel_cast", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

// Group: POOL
// adaptive_avg_pool1d, adaptive_avg_pool1d_grad, adaptive_avg_pool2d, adaptive_avg_pool2d_grad, adaptive_avg_pool3d, adaptive_avg_pool3d_grad, avgpool_1d, avgpool_1d_grad, avgpool_2d, avgpool_2d_grad, avgpool_3d, avgpool_3d_grad, maxpool_1d, maxpool_1d_grad, maxpool_2d, maxpool_2d_grad, maxpool_3d, maxpool_3d_grad, tf_avg_pool_1d, tf_avg_pool_1d_grad, tf_avg_pool_2d, tf_avg_pool_2d_grad, tf_avg_pool_3d, tf_avg_pool_3d_grad, tf_max_pool_1d, tf_max_pool_1d_grad, tf_max_pool_2d, tf_max_pool_2d_grad, tf_max_pool_3d, tf_max_pool_3d_grad
// Total: 30

#ifdef GET_ONEFLOW_POOL_OP_DEFINITIONS

def OneFlow_AdaptiveAvgPool1DOp : OneFlow_AdaptivePoolBaseOp<"adaptive_avg_pool1d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_AdaptiveAvgPool1DGradOp : OneFlow_AdaptivePoolGradBaseOp<"adaptive_avg_pool1d_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_AdaptiveAvgPool2DOp : OneFlow_AdaptivePoolBaseOp<"adaptive_avg_pool2d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_AdaptiveAvgPool2DGradOp : OneFlow_AdaptivePoolGradBaseOp<"adaptive_avg_pool2d_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_AdaptiveAvgPool3DOp : OneFlow_AdaptivePoolBaseOp<"adaptive_avg_pool3d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_AdaptiveAvgPool3DGradOp : OneFlow_AdaptivePoolGradBaseOp<"adaptive_avg_pool3d_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_AvgPool1DOp : OneFlow_AvgPoolBaseOp<"avgpool_1d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_AvgPool1DGradOp : OneFlow_AvgPoolGradBaseOp<"avgpool_1d_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_AvgPool2DOp : OneFlow_AvgPoolBaseOp<"avgpool_2d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_AvgPool2DGradOp : OneFlow_AvgPoolGradBaseOp<"avgpool_2d_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_AvgPool3DOp : OneFlow_AvgPoolBaseOp<"avgpool_3d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_AvgPool3DGradOp : OneFlow_AvgPoolGradBaseOp<"avgpool_3d_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_MaxPool1DOp : OneFlow_MaxPoolBaseOp<"maxpool_1d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_MaxPool1DGradOp : OneFlow_MaxPoolGradBaseOp<"maxpool_1d_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_MaxPool2DOp : OneFlow_MaxPoolBaseOp<"maxpool_2d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_MaxPool2DGradOp : OneFlow_MaxPoolGradBaseOp<"maxpool_2d_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_MaxPool3DOp : OneFlow_MaxPoolBaseOp<"maxpool_3d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

def OneFlow_MaxPool3DGradOp : OneFlow_MaxPoolGradBaseOp<"maxpool_3d_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {}

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

def OneFlow_MinMaxObserverOp : OneFlow_BaseOp<"min_max_observer", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_MovingAverageMinMaxObserverOp : OneFlow_BaseOp<"moving_average_min_max_observer", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_QuantizationOp : OneFlow_BaseOp<"quantization", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

#endif // GET_ONEFLOW_QUANTIZATION_OP_DEFINITIONS

// Group: REDUCE
// indexed_slices_reduce_sum, reduce_all, reduce_any, reduce_max, reduce_max_device_stage, reduce_max_device_stage_grad, reduce_max_global_stage, reduce_max_global_stage_grad, reduce_min, reduce_min_device_stage, reduce_min_device_stage_grad, reduce_min_global_stage, reduce_min_global_stage_grad, reduce_prod, reduce_sum, reduce_sum_like
// Total: 16

#ifdef GET_ONEFLOW_REDUCE_OP_DEFINITIONS

def OneFlow_IndexedSlicesReduceSumOp : OneFlow_BaseOp<"indexed_slices_reduce_sum", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ReduceAllOp : OneFlow_BaseOp<"reduce_all", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ReduceAnyOp : OneFlow_BaseOp<"reduce_any", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ReduceMaxOp : OneFlow_BaseOp<"reduce_max", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ReduceMaxDeviceStageOp : OneFlow_BaseOp<"reduce_max_device_stage", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ReduceMaxDeviceStageGradOp : OneFlow_BaseOp<"reduce_max_device_stage_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ReduceMaxGlobalStageOp : OneFlow_BaseOp<"reduce_max_global_stage", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ReduceMaxGlobalStageGradOp : OneFlow_BaseOp<"reduce_max_global_stage_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ReduceMinOp : OneFlow_BaseOp<"reduce_min", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ReduceMinDeviceStageOp : OneFlow_BaseOp<"reduce_min_device_stage", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ReduceMinDeviceStageGradOp : OneFlow_BaseOp<"reduce_min_device_stage_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ReduceMinGlobalStageOp : OneFlow_BaseOp<"reduce_min_global_stage", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ReduceMinGlobalStageGradOp : OneFlow_BaseOp<"reduce_min_global_stage_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ReduceProdOp : OneFlow_BaseOp<"reduce_prod", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ReduceSumOp : OneFlow_BaseOp<"reduce_sum", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ReduceSumLikeOp : OneFlow_BaseOp<"reduce_sum_like", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

// Group: RESHAPE
// reshape, reshape_like
// Total: 2

#ifdef GET_ONEFLOW_RESHAPE_OP_DEFINITIONS

def OneFlow_ReshapeOp : OneFlow_BaseOp<"reshape", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
  let has_data_type_infer_fn = 1;
  let has_nd_sbp_infer_fn = 1;
}

def OneFlow_ReshapeLikeOp : OneFlow_BaseOp<"reshape_like", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
  let has_nd_sbp_infer_fn = 1;
}

#endif // GET_ONEFLOW_RESHAPE_OP_DEFINITIONS

// Group: SCALAR
// clip_by_scalar, clip_by_scalar_grad, clip_by_scalar_max, clip_by_scalar_max_grad, clip_by_scalar_min, clip_by_scalar_min_grad, scalar_add, scalar_add_by_tensor, scalar_div_by_tensor, scalar_floordiv, scalar_fmod, scalar_logical_and, scalar_logical_equal, scalar_logical_greater, scalar_logical_greater_equal, scalar_logical_less, scalar_logical_less_equal, scalar_logical_not_equal, scalar_logical_or, scalar_logical_xor, scalar_mul, scalar_mul_by_tensor, scalar_pow, scalar_pow_grad, scalar_sub_by_tensor
// Total: 25

#ifdef GET_ONEFLOW_SCALAR_OP_DEFINITIONS

def OneFlow_ClipByScalarOp : OneFlow_BaseOp<"clip_by_scalar", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ClipByScalarGradOp : OneFlow_BaseOp<"clip_by_scalar_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ClipByScalarMaxOp : OneFlow_BaseOp<"clip_by_scalar_max", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ClipByScalarMaxGradOp : OneFlow_BaseOp<"clip_by_scalar_max_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ClipByScalarMinOp : OneFlow_BaseOp<"clip_by_scalar_min", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ClipByScalarMinGradOp : OneFlow_BaseOp<"clip_by_scalar_min_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ScalarAddOp : OneFlow_BaseOp<"scalar_add", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ScalarAddByTensorOp : OneFlow_BaseOp<"scalar_add_by_tensor", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ScalarDivByTensorOp : OneFlow_BaseOp<"scalar_div_by_tensor", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ScalarFloordivOp : OneFlow_BaseOp<"scalar_floordiv", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ScalarFmodOp : OneFlow_BaseOp<"scalar_fmod", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ScalarLogicalAndOp : OneFlow_BaseOp<"scalar_logical_and", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ScalarLogicalEqualOp : OneFlow_BaseOp<"scalar_logical_equal", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ScalarLogicalGreaterOp : OneFlow_BaseOp<"scalar_logical_greater", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ScalarLogicalGreaterEqualOp : OneFlow_BaseOp<"scalar_logical_greater_equal", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ScalarLogicalLessOp : OneFlow_BaseOp<"scalar_logical_less", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ScalarLogicalLessEqualOp : OneFlow_BaseOp<"scalar_logical_less_equal", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ScalarLogicalNotEqualOp : OneFlow_BaseOp<"scalar_logical_not_equal", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ScalarLogicalOrOp : OneFlow_BaseOp<"scalar_logical_or", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ScalarLogicalXorOp : OneFlow_BaseOp<"scalar_logical_xor", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ScalarMulOp : OneFlow_BaseOp<"scalar_mul", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ScalarMulByTensorOp : OneFlow_BaseOp<"scalar_mul_by_tensor", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ScalarPowOp : OneFlow_BaseOp<"scalar_pow", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ScalarPowGradOp : OneFlow_BaseOp<"scalar_pow_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ScalarSubByTensorOp : OneFlow_BaseOp<"scalar_sub_by_tensor", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

#endif // GET_ONEFLOW_SCALAR_OP_DEFINITIONS

// Group: SOFTMAX
// log_softmax, log_softmax_grad, softmax, softmax_cross_entropy, softmax_cross_entropy_grad, softmax_grad, sparse_softmax_cross_entropy, sparse_softmax_cross_entropy_grad, sparse_softmax_cross_entropy_ms, sparse_softmax_cross_entropy_ms_grad
// Total: 10

#ifdef GET_ONEFLOW_SOFTMAX_OP_DEFINITIONS

def OneFlow_LogSoftmaxOp : OneFlow_BaseOp<"log_softmax", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_LogSoftmaxGradOp : OneFlow_BaseOp<"log_softmax_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_SoftmaxOp : OneFlow_BaseOp<"softmax", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_SoftmaxCrossEntropyOp : OneFlow_BaseOp<"softmax_cross_entropy", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_SoftmaxCrossEntropyGradOp : OneFlow_BaseOp<"softmax_cross_entropy_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_SoftmaxGradOp : OneFlow_BaseOp<"softmax_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_SparseSoftmaxCrossEntropyOp : OneFlow_BaseOp<"sparse_softmax_cross_entropy", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_SparseSoftmaxCrossEntropyGradOp : OneFlow_BaseOp<"sparse_softmax_cross_entropy_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_SparseSoftmaxCrossEntropyMsOp : OneFlow_BaseOp<"sparse_softmax_cross_entropy_ms", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_SparseSoftmaxCrossEntropyMsGradOp : OneFlow_BaseOp<"sparse_softmax_cross_entropy_ms_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

// Group: SUMMARY
// create_summary_writer, flush_summary_writer, summary_write_histogram, summary_write_image, summary_write_pb, summary_write_scalar
// Total: 6

#ifdef GET_ONEFLOW_SUMMARY_OP_DEFINITIONS

def OneFlow_CreateSummaryWriterOp : OneFlow_BaseOp<"create_summary_writer", [NoSideEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let attrs = (ins
    StrAttr:$logdir
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FlushSummaryWriterOp : OneFlow_BaseOp<"flush_summary_writer", [NoSideEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_SummaryWriteHistogramOp : OneFlow_BaseOp<"summary_write_histogram", [NoSideEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_SummaryWriteImageOp : OneFlow_BaseOp<"summary_write_image", [NoSideEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_SummaryWritePbOp : OneFlow_BaseOp<"summary_write_pb", [NoSideEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in,
    OneFlow_Tensor:$step
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_SummaryWriteScalarOp : OneFlow_BaseOp<"summary_write_scalar", [NoSideEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

// Group: TENSOR_BUFFER
// gen_tensor_buffer, tensor_buffer_to_list_of_tensors, tensor_buffer_to_list_of_tensors_v2, tensor_buffer_to_tensor, tensor_to_tensor_buffer
// Total: 5

#ifdef GET_ONEFLOW_TENSOR_BUFFER_OP_DEFINITIONS

def OneFlow_GenTensorBufferOp : OneFlow_BaseOp<"gen_tensor_buffer", [NoSideEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_TensorBufferToListOfTensorsOp : OneFlow_BaseOp<"tensor_buffer_to_list_of_tensors", [NoSideEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_TensorBufferToListOfTensorsV2Op : OneFlow_BaseOp<"tensor_buffer_to_list_of_tensors_v2", [NoSideEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_TensorBufferToTensorOp : OneFlow_BaseOp<"tensor_buffer_to_tensor", [NoSideEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_TensorToTensorBufferOp : OneFlow_BaseOp<"tensor_to_tensor_buffer", [NoSideEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

// Group: TEST
// TestDataTypeAttr, TestDynamicSource, TestListDataTypeAndListShapeAndListStringAttr, TestMultiInput, TestMultiInputGrad, TestMultiOutputOrder, TestRandomSource, TestReshape, TestSource, TestSourceMultiGpuFixedOutNum, ccrelu, ccrelu_grad, cpu_only_relu_test, test_user_op_attr_auto_type
// Total: 14

#ifdef GET_ONEFLOW_TEST_OP_DEFINITIONS

def OneFlow_TestDataTypeAttrOp : OneFlow_BaseOp<"TestDataTypeAttr", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    OneFlow_DataType:$output_type
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_TestDynamicSourceOp : OneFlow_BaseOp<"TestDynamicSource", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_output_arg_modify_fn = 1;
}

def OneFlow_TestListDataTypeAndListShapeAndListStringAttrOp : OneFlow_BaseOp<"TestListDataTypeAndListShapeAndListStringAttr", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    ShapeArrayAttr:$out_shapes,
    DTArrayAttr:$out_types,
    StrArrayAttr:$string_list
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_TestMultiInputOp : OneFlow_BaseOp<"TestMultiInput", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x1,
    OneFlow_Tensor:$x2
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_TestMultiInputGradOp : OneFlow_BaseOp<"TestMultiInputGrad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x1,
    OneFlow_Tensor:$x2,
    OneFlow_Tensor:$y_diff
  );
  let output = (outs
    OneFlow_Tensor:$x1_diff,
    OneFlow_Tensor:$x2_diff
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_TestMultiOutputOrderOp : OneFlow_BaseOp<"TestMultiOutputOrder", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out1,
    OneFlow_Tensor:$out2
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_TestRandomSourceOp : OneFlow_BaseOp<"TestRandomSource", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$seed
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_TestReshapeOp : OneFlow_BaseOp<"TestReshape", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
  let has_data_type_infer_fn = 1;
}

def OneFlow_TestSourceOp : OneFlow_BaseOp<"TestSource", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let output = (outs
    OneFlow_Tensor:$out
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_TestSourceMultiGpuFixedOutNumOp : OneFlow_BaseOp<"TestSourceMultiGpuFixedOutNum", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "0">:$out_num
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_CcreluOp : OneFlow_BaseOp<"ccrelu", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_CcreluGradOp : OneFlow_BaseOp<"ccrelu_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_CpuOnlyReluTestOp : OneFlow_BaseOp<"cpu_only_relu_test", [NoSideEffect, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_TestUserOpAttrAutoTypeOp : OneFlow_BaseOp<"test_user_op_attr_auto_type", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$int1,
    DefaultValuedAttr<SI32Attr, "0">:$int2
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

#endif // GET_ONEFLOW_TEST_OP_DEFINITIONS

// Group: TRIGONOMETRIC
// acos, acos_grad, acosh, acosh_grad, asin, asin_grad, asinh, asinh_grad, atan, atan2, atan2_x_grad, atan2_y_grad, atan_grad, atanh, atanh_grad, cos, cos_grad, cosh, cosh_grad, hardtanh, hardtanh_grad, sin, sin_grad, sinh, sinh_grad, tan, tan_grad, tanh, tanh_grad
// Total: 29

#ifdef GET_ONEFLOW_TRIGONOMETRIC_OP_DEFINITIONS

def OneFlow_AcosOp : OneFlow_BaseOp<"acos", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_AcosGradOp : OneFlow_BaseOp<"acos_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_AcoshOp : OneFlow_BaseOp<"acosh", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_AcoshGradOp : OneFlow_BaseOp<"acosh_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_AsinOp : OneFlow_BaseOp<"asin", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_AsinGradOp : OneFlow_BaseOp<"asin_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_AsinhOp : OneFlow_BaseOp<"asinh", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_AsinhGradOp : OneFlow_BaseOp<"asinh_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_AtanOp : OneFlow_BaseOp<"atan", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_Atan2Op : OneFlow_BaseOp<"atan2", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_Atan2XGradOp : OneFlow_BaseOp<"atan2_x_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_Atan2YGradOp : OneFlow_BaseOp<"atan2_y_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_AtanGradOp : OneFlow_BaseOp<"atan_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_AtanhOp : OneFlow_BaseOp<"atanh", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_AtanhGradOp : OneFlow_BaseOp<"atanh_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_CosOp : OneFlow_BaseOp<"cos", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_CosGradOp : OneFlow_BaseOp<"cos_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_CoshOp : OneFlow_BaseOp<"cosh", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_CoshGradOp : OneFlow_BaseOp<"cosh_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_HardtanhOp : OneFlow_BaseOp<"hardtanh", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_HardtanhGradOp : OneFlow_BaseOp<"hardtanh_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_SinOp : OneFlow_BaseOp<"sin", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_SinGradOp : OneFlow_BaseOp<"sin_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_SinhOp : OneFlow_BaseOp<"sinh", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_SinhGradOp : OneFlow_BaseOp<"sinh_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_TanOp : OneFlow_BaseOp<"tan", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_TanGradOp : OneFlow_BaseOp<"tan_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_TanhOp : OneFlow_BaseOp<"tanh", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_TanhGradOp : OneFlow_BaseOp<"tanh_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

#endif // GET_ONEFLOW_TRIGONOMETRIC_OP_DEFINITIONS

// Group: UNARY
// acc, affine_grid, affine_grid_grad, bernoulli, cast, cast_to_static_shape, cast_to_tick, celu, copy, count_not_finite, diag, diagonal, elu, expand, expand_dims, flatten, flip, flip_grad, fold, gelu, hardsigmoid, hardswish, leaky_relu, log2, logical_not, mish, narrow, one_hot, pack, random_mask_like, repeat, roll, selu, sigmoid, silu, softsign, sort, square_sum, squeeze, transpose, tril, triu, unfold, unfold_tensor, unpack, zero_like
// Total: 46

#ifdef GET_ONEFLOW_UNARY_OP_DEFINITIONS

def OneFlow_AccOp : OneFlow_BaseOp<"acc", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_AffineGridOp : OneFlow_BaseOp<"affine_grid", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_AffineGridGradOp : OneFlow_BaseOp<"affine_grid_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_BernoulliOp : OneFlow_BaseOp<"bernoulli", [NoSideEffect, NoGrad, CpuOnly, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI64Attr, "-1">:$seed,
    DefaultValuedAttr<BoolAttr, "false">:$has_seed,
    OneFlow_DataType:$dtype
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_CastOp : OneFlow_BaseOp<"cast", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
}

def OneFlow_CastToStaticShapeOp : OneFlow_BaseOp<"cast_to_static_shape", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_CastToTickOp : OneFlow_BaseOp<"cast_to_tick", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_CeluOp : OneFlow_BaseOp<"celu", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_CopyOp : OneFlow_BaseOp<"copy", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    StrAttr:$device_type,
    DefaultValuedAttr<SI64Attr, "0">:$device_id
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_device_infer_fn = 1;
}

def OneFlow_CountNotFiniteOp : OneFlow_BaseOp<"count_not_finite", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_DiagOp : OneFlow_BaseOp<"diag", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_DiagonalOp : OneFlow_BaseOp<"diagonal", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_EluOp : OneFlow_BaseOp<"elu", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ExpandOp : OneFlow_BaseOp<"expand", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    SI32ArrayAttr:$logical_in_shape,
    SI32ArrayAttr:$logical_expand_shape
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_ExpandDimsOp : OneFlow_BaseOp<"expand_dims", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_FlattenOp : OneFlow_BaseOp<"flatten", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$in
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<SI32Attr, "0">:$start_dim,
    DefaultValuedAttr<SI32Attr, "-1">:$end_dim
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FlipOp : OneFlow_BaseOp<"flip", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_FlipGradOp : OneFlow_BaseOp<"flip_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    SI32ArrayAttr:$dims
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_FoldOp : OneFlow_BaseOp<"fold", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_GeluOp : OneFlow_BaseOp<"gelu", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_HardsigmoidOp : OneFlow_BaseOp<"hardsigmoid", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_HardswishOp : OneFlow_BaseOp<"hardswish", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_LeakyReluOp : OneFlow_BaseOp<"leaky_relu", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_Log2Op : OneFlow_BaseOp<"log2", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_LogicalNotOp : OneFlow_BaseOp<"logical_not", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_MishOp : OneFlow_BaseOp<"mish", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_NarrowOp : OneFlow_BaseOp<"narrow", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_OneHotOp : OneFlow_BaseOp<"one_hot", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_PackOp : OneFlow_BaseOp<"pack", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_RandomMaskLikeOp : OneFlow_BaseOp<"random_mask_like", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
}

def OneFlow_RepeatOp : OneFlow_BaseOp<"repeat", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_RollOp : OneFlow_BaseOp<"roll", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_SeluOp : OneFlow_BaseOp<"selu", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_SigmoidOp : OneFlow_BaseOp<"sigmoid", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_SiluOp : OneFlow_BaseOp<"silu", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_SoftsignOp : OneFlow_BaseOp<"softsign", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_SortOp : OneFlow_BaseOp<"sort", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_SquareSumOp : OneFlow_BaseOp<"square_sum", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_SqueezeOp : OneFlow_BaseOp<"squeeze", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_TransposeOp : OneFlow_BaseOp<"transpose", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
}

def OneFlow_TrilOp : OneFlow_BaseOp<"tril", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_TriuOp : OneFlow_BaseOp<"triu", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_UnfoldOp : OneFlow_BaseOp<"unfold", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_UnfoldTensorOp : OneFlow_BaseOp<"unfold_tensor", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_UnpackOp : OneFlow_BaseOp<"unpack", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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

def OneFlow_ZeroLikeOp : OneFlow_BaseOp<"zero_like", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
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
}

#endif // GET_ONEFLOW_UNARY_OP_DEFINITIONS

// Group: UPSAMPLE
// upsample, upsample_bicubic_2d, upsample_bicubic_2d_grad, upsample_bilinear_2d, upsample_bilinear_2d_grad, upsample_grad, upsample_linear_1d, upsample_linear_1d_grad, upsample_nearest_1d, upsample_nearest_1d_grad, upsample_nearest_2d, upsample_nearest_2d_grad, upsample_nearest_3d, upsample_nearest_3d_grad, upsample_trilinear_3d, upsample_trilinear_3d_grad
// Total: 16

#ifdef GET_ONEFLOW_UPSAMPLE_OP_DEFINITIONS

def OneFlow_UpsampleOp : OneFlow_BaseOp<"upsample", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$height_scale,
    DefaultValuedAttr<F32Attr, "0.">:$width_scale,
    DefaultValuedAttr<BoolAttr, "false">:$align_corners,
    StrAttr:$data_format,
    StrAttr:$interpolation
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_UpsampleBicubic2DOp : OneFlow_BaseOp<"upsample_bicubic_2d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$height_scale,
    DefaultValuedAttr<F32Attr, "0.">:$width_scale,
    DefaultValuedAttr<BoolAttr, "false">:$align_corners,
    StrAttr:$data_format
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_UpsampleBicubic2DGradOp : OneFlow_BaseOp<"upsample_bicubic_2d_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$height_scale,
    DefaultValuedAttr<F32Attr, "0.">:$width_scale,
    DefaultValuedAttr<BoolAttr, "false">:$align_corners,
    StrAttr:$data_format
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_UpsampleBilinear2DOp : OneFlow_BaseOp<"upsample_bilinear_2d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$height_scale,
    DefaultValuedAttr<F32Attr, "0.">:$width_scale,
    DefaultValuedAttr<BoolAttr, "false">:$align_corners,
    StrAttr:$data_format
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_UpsampleBilinear2DGradOp : OneFlow_BaseOp<"upsample_bilinear_2d_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$height_scale,
    DefaultValuedAttr<F32Attr, "0.">:$width_scale,
    DefaultValuedAttr<BoolAttr, "false">:$align_corners,
    StrAttr:$data_format
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_UpsampleGradOp : OneFlow_BaseOp<"upsample_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$height_scale,
    DefaultValuedAttr<F32Attr, "0.">:$width_scale,
    DefaultValuedAttr<BoolAttr, "false">:$align_corners,
    StrAttr:$data_format,
    StrAttr:$interpolation
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_UpsampleLinear1DOp : OneFlow_BaseOp<"upsample_linear_1d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$scale_factor,
    DefaultValuedAttr<BoolAttr, "false">:$align_corners,
    StrAttr:$data_format
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_UpsampleLinear1DGradOp : OneFlow_BaseOp<"upsample_linear_1d_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$scale_factor,
    DefaultValuedAttr<BoolAttr, "false">:$align_corners,
    StrAttr:$data_format
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_UpsampleNearest1DOp : OneFlow_BaseOp<"upsample_nearest_1d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$scale_factor,
    StrAttr:$data_format
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_UpsampleNearest1DGradOp : OneFlow_BaseOp<"upsample_nearest_1d_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$scale_factor,
    StrAttr:$data_format
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_UpsampleNearest2DOp : OneFlow_BaseOp<"upsample_nearest_2d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$height_scale,
    DefaultValuedAttr<F32Attr, "0.">:$width_scale,
    StrAttr:$data_format
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_UpsampleNearest2DGradOp : OneFlow_BaseOp<"upsample_nearest_2d_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$height_scale,
    DefaultValuedAttr<F32Attr, "0.">:$width_scale,
    StrAttr:$data_format
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_UpsampleNearest3DOp : OneFlow_BaseOp<"upsample_nearest_3d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$depth_scale,
    DefaultValuedAttr<F32Attr, "0.">:$height_scale,
    DefaultValuedAttr<F32Attr, "0.">:$width_scale,
    StrAttr:$data_format
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_UpsampleNearest3DGradOp : OneFlow_BaseOp<"upsample_nearest_3d_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$depth_scale,
    DefaultValuedAttr<F32Attr, "0.">:$height_scale,
    DefaultValuedAttr<F32Attr, "0.">:$width_scale,
    StrAttr:$data_format
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_UpsampleTrilinear3DOp : OneFlow_BaseOp<"upsample_trilinear_3d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$depth_scale,
    DefaultValuedAttr<F32Attr, "0.">:$height_scale,
    DefaultValuedAttr<F32Attr, "0.">:$width_scale,
    DefaultValuedAttr<BoolAttr, "false">:$align_corners,
    StrAttr:$data_format
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

def OneFlow_UpsampleTrilinear3DGradOp : OneFlow_BaseOp<"upsample_trilinear_3d_grad", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$dy,
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$dx
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$depth_scale,
    DefaultValuedAttr<F32Attr, "0.">:$height_scale,
    DefaultValuedAttr<F32Attr, "0.">:$width_scale,
    DefaultValuedAttr<BoolAttr, "false">:$align_corners,
    StrAttr:$data_format
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}

#endif // GET_ONEFLOW_UPSAMPLE_OP_DEFINITIONS


#ifndef ONEFLOW_PATTERNS
#define ONEFLOW_PATTERNS

include "mlir/IR/PatternBase.td"
include "OneFlow/OneFlowOps.td"
include "mlir/Dialect/MemRef/IR/MemRefOps.td"
include "mlir/Dialect/GPU/IR/GPUOps.td"

def IsNotNestedInJit: Constraint<CPred<"($0.getDefiningOp()->getParentOfType<::mlir::oneflow::Job>())">, "">;
def IsScalarTensor: Constraint<CPred<"::mlir::oneflow::IsScalarTensor($0)">, "">;
def OutlineMulCast : NativeCodeCall<"::mlir::oneflow::OutlineMulCast($_builder, $0, $1)">;
// TODO: remove attr binding if possible
def MulCastPattern : Pat<
  (
    OneFlow_ScalarMulByTensorOp : $mul_op
    (
      OneFlow_CastOp : $cast_op
        $cast_x,
        $cast_op_name,
        $cast_device_tag,
        $cast_device_name,
        $cast_scope_symbol_id,
        $cast_hierarchy,
        $cast_dtype
    ),
    $scalar,
    $mul_op_name,
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

def BroadcastMulToScalarMulPattern : Pat<
  (
    OneFlow_BroadcastMulOp
    $x,
    $y,
    $mul_op_name,
    $mul_device_tag,
    $mul_device_name,
    $mul_scope_symbol_id,
    $mul_hierarchy
  ),
  (
    OneFlow_ScalarMulByTensorOp
    $x,
    $y,
    $mul_op_name,
    $mul_device_tag,
    $mul_device_name,
    $mul_scope_symbol_id,
    $mul_hierarchy
  ),
  [
    (IsScalarTensor $y)
  ]
>;

def GetFirstValue :
  NativeCodeCall<"*$0.begin()">;

def IsGPU: Constraint<CPred<"$0.getValue().equals(\"cuda\")">, "is GPU device">;

def FusedScaleTrilPattern : Pat<
  (
    OneFlow_TrilOp
    (
      OneFlow_ScalarMulOp
        $x,
        $scale_op_name,
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

def FusePadAndConv2d :
  NativeCodeCall<"mlir::oneflow::CreateConv2dAndErasePad($_builder, $0, $1)">;

def HasIdenticalPadding: Constraint<CPred<"mlir::oneflow::IsPaddingCouldBeAssimilatedIntoConv($0, $1, $2)">, "">;

def FusedPadConv2DPattern : Pat<
  (
    OneFlow_Conv2DOp : $conv_2d_res
    (
      OneFlow_PadOp : $pad_res
        $x,
        $pad_op_name,
        $pad_device_tag,
        $pad_device_name,
        $pad_scope_symbol_id,
        $pad_hierarchy,
        $pad_op_padding_before,
        $pad_op_padding_after,
        $pad_op_padding,
        $floating_constant_value,
        $integral_constant_value
    ),
    $weight,
    $bias,
    $bias_multiplier,
    $conv2d_op_name,
    $conv2d_device_tag,
    $conv2d_device_name,
    $conv2d_scope_symbol_id,
    $conv2d_hierarchy,
    $operand_segment_sizes,
    $filters,
    $padding_before,
    $data_format,
    $kernel_size,
    $strides,
    $dilation_rate,
    $groups
  ),
  (FusePadAndConv2d $conv_2d_res, $pad_res),
  [
    (HasIdenticalPadding $pad_op_padding_before, $pad_op_padding_after, $data_format)
  ]
>;

def GetDefaultSeed :
  NativeCodeCall<"mlir::oneflow::GetDefaultSeed($_builder)">;

def FusedBiasAddMaskScale :
  NativeCodeCall<"mlir::oneflow::CreateFusedBiasAddMaskScale($_builder, $0, $1, $2)">;

def IsAddToOutputNone: Constraint<CPred<"mlir::oneflow::IsAddToOutputNone($0)">, "">;

def FusedBiasAddDropoutPattern : Pattern<
  (
    OneFlow_DropoutOp: $dropout_res
    (
      OneFlow_BiasAddOp: $bias_add_res
        $a,
        $b,
        $bias_add_op_name,
        $bias_add_device_tag,
        $bias_add_device_name,
        $bias_add_scope_symbol_id,
        $bias_add_hierarchy,
        $bias_add_op_axis
    ),
    $_add_to_output,
    $dropout_op_name,
    $dropout_device_tag,
    $dropout_device_name,
    $dropout_scope_symbol_id,
    $dropout_hierarchy,
    $dropout_op_rate
  ),
  [
    (
      FusedBiasAddMaskScale
      $dropout_res__0,
      $bias_add_res,
      (
        OneFlow_RandomMaskLikeOp : $mask
          $a,
          $bias_add_op_name,
          $dropout_device_tag,
          $dropout_device_name,
          $dropout_scope_symbol_id,
          $dropout_hierarchy,
          $dropout_op_rate,
          (GetDefaultSeed)
      )
    ),
    (replaceWithValue $mask)
  ],
  [(IsAddToOutputNone $_add_to_output)]
>;

// TODO: use either to merge two patterns
def FusedScaleTrilPattern2 : Pat<
  (
    OneFlow_ScalarMulOp
    (
      OneFlow_TrilOp
      $x,
      $tril_op_name,
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

def FusedBiasAddGeluPattern : Pat<
  (
    OneFlow_GeluOp : $gelu_op
    (
      OneFlow_BiasAddOp
        $a,
        $b,
        $bias_add_op_name,
        $bias_add_device_tag,
        $bias_add_device_name,
        $bias_add_scope_symbol_id,
        $bias_add_hierarchy,
        $axis
    ),
    $gelu_op_name,
    $gelu_device_tag,
    $gelu_device_name,
    $gelu_scope_symbol_id,
    $gelu_hierarchy
  ),
  (OneFlow_FusedBiasAddGeluOp $a, $b,
    $gelu_op_name,
    $gelu_device_tag,
    $gelu_device_name,
    $gelu_scope_symbol_id,
    $gelu_hierarchy,
    $axis
  ),
  []
>;

def IsTraingTrue: Constraint<CPred<"$0.getValue()">, "">;
// TODO: check mean and inv_variance are not used
def NormalizationAddReluPattern : Pattern<
  /* match */ (
    OneFlow_ReluOp (
      OneFlow_Add2Op (
          OneFlow_NormalizationOp:$results
            $bn_x,
            $bn_moving_mean,
            $bn_moving_variance,
            $bn_gamma,
            $bn_beta,
            $bn__add_to_output, // TODO: check this is none
            $bn_op_name,
            $bn_device_tag,
            $bn_device_name,
            $bn_scope_symbol_id,
            $bn_hierarchy,
            $operand_segment_sizes,
            $result_segment_sizes,
            $bn_axis,
            $bn_epsilon,
            $bn_training,
            $bn_momentum
      ),
      $addend,
      $add_op_name,
      $add_device_tag,
      $add_device_name,
      $add_scope_symbol_id,
      $add_hierarchy
    ),
    $relu_op_name,
    $relu_device_tag,
    $relu_device_name,
    $relu_scope_symbol_id,
    $relu_hierarchy
  ),
  /* replace */ [(
    OneFlow_NormalizationAddReluOp: $fuse_results
    $bn_x,
    $addend,
    (GetFirstValue $bn_moving_mean),
    (GetFirstValue $bn_moving_variance),
    $bn_gamma,
    $bn_beta,
    $bn_op_name,
    $bn_device_tag,
    $bn_device_name,
    $bn_scope_symbol_id,
    $bn_hierarchy,
    /* not used */ $operand_segment_sizes,
    /* not used */ $result_segment_sizes,
    $bn_axis,
    $bn_epsilon,
    $bn_training,
    $bn_momentum
  ),
  (replaceWithValue $fuse_results__0),
  ],
  [(IsTraingTrue $bn_training)]
>;

def IsArg: Constraint<CPred<"$0.dyn_cast<::mlir::BlockArgument>()">, "">;
def getResultTypes : NativeCodeCall<"$0.getResultTypes()">;
def CreateGPUMemcpyOpFromMemrefCopy : NativeCodeCall<"::mlir::oneflow::CreateGPUMemcpyOpFromMemrefCopy($_builder, $0)">;
def ReplaceCopyWithGPUPattern : Pat<
  (
    CopyOp:$results
    $src,
    $dst
  ),
  (
    CreateGPUMemcpyOpFromMemrefCopy $results
  ),
  [(IsArg $dst)]
>;

def FuseConv2DBatchNorm : NativeCodeCall<"mlir::oneflow::FuseConv2DBatchNorm($_builder, $0, $1)">;
def FuseConv2DBatchNormPattern : Pat<
  (
    OneFlow_NormalizationInferenceOp: $batch_norm_op
      (
        OneFlow_Conv2DOp : $conv_2d_op
          $x,
          (
            OneFlow_FrozenVariableOp : $variable_conv_2d_weight
              $value,
              $v_op_name,
              $v_device_tag,
              $v_device_name,
              $v_scope_symbol_id,
              $v_hierarchy,
              $v_nd_sbp
          ),
          $bias,
          $bias_multiplier,
          $conv2d_op_name,
          $conv2d_device_tag,
          $conv2d_device_name,
          $conv2d_scope_symbol_id,
          $conv2d_hierarchy,
          $conv2d_operand_segment_sizes,
          $filters,
          $padding_before,
          $data_format,
          $kernel_size,
          $strides,
          $dilation_rate,
          $groups
      ),
      $bn_moving_mean,
      $bn_moving_variance,
      $bn_gamma,
      $bn_beta,
      $bn__add_to_output,
      $bn_op_name,
      $bn_device_tag,
      $bn_device_name,
      $bn_scope_symbol_id,
      $bn_hierarchy,
      $bn_operand_segment_sizes,
      $bn_result_segment_sizes,
      $bn_axis,
      $bn_epsilon,
      $bn_training,
      $bn_momentum
  ),
  (
    FuseConv2DBatchNorm $conv_2d_op, $batch_norm_op__0
  ),
  []
>;

def IsSameDtype : Constraint<CPred<"mlir::oneflow::IsSameDtype($0, $1)">>;

def DeleteSameDtypeCastOpPattern : Pat<
  (
    OneFlow_CastOp : $res
     $x,
     $cast_op_name,
     $cast_device_tag,
     $cast_device_name,
     $cast_scope_symbol_id,
     $cast_hierarchy,
     $dtype
  ),
  (
    replaceWithValue $x
  ),
  [(IsSameDtype $res, $x)]
>;

#endif // ONEFLOW_PATTERNS

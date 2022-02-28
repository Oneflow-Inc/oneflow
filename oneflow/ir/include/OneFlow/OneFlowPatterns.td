
#ifndef ONEFLOW_PATTERNS
#define ONEFLOW_PATTERNS

include "OneFlow/OneFlowOps.td"
include "mlir/Dialect/MemRef/IR/MemRefOps.td"
include "mlir/Dialect/GPU/GPUOps.td"

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

def OutlineBatchMatMul : NativeCodeCall<"::mlir::oneflow::OutlineBatchMatMul($_builder, $0)">;
def BatchMatmulPattern : Pat<
  (
    OneFlow_BatchMatmulOp : $batch_matmul_op
    $a,
    $b,
    $_add_to_output,
    $transpose_a,
    $transpose_b,
    $alpha,
    $batch_matmul_op_name,
    $matmul_device_tag,
    $matmul_device_name,
    $matmul_scope_symbol_id,
    $matmul_hierarchy
  ),
  (OutlineBatchMatMul $batch_matmul_op),
  [
    (IsNotNestedInJit $batch_matmul_op)
  ]
>;

def GetFirstValue :
  NativeCodeCall<"*$0.begin()">;

def IsGPU: Constraint<CPred<"$0.getValue().equals(\"gpu\")">, "is GPU device">;

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

#endif // ONEFLOW_PATTERNS

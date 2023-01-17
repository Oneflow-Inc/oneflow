
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
        $cast_dtype,
        $cast_pin_memory
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


def IsAddToOutputNone: Constraint<CPred<"mlir::oneflow::IsAddToOutputNone($0)">, "">;

def IsTraingTrue: Constraint<CPred<"$0.getValue()">, "">;

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

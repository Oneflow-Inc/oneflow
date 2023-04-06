
#ifndef ONEFLOW_PATTERNS
#define ONEFLOW_PATTERNS

include "mlir/IR/PatternBase.td"
include "OneFlow/OneFlowOps.td"
include "mlir/Dialect/MemRef/IR/MemRefOps.td"
include "mlir/Dialect/GPU/IR/GPUOps.td"

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

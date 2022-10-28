#ifndef ONEFLOW_IR_INCLUDE_ONEFLOW_TRANSFORM_TRANSPOSEHELPERS_H_
#define ONEFLOW_IR_INCLUDE_ONEFLOW_TRANSFORM_TRANSPOSEHELPERS_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "OneFlow/OneFlowOps.h"

namespace mlir {

namespace oneflow {

RankedTensorType getNHWCType(RankedTensorType t);
RankedTensorType getNHWCType(Type t);
RankedTensorType getNHWCType(Value v);
RankedTensorType getNCHWType(RankedTensorType t);
RankedTensorType getNCHWType(Type t);
RankedTensorType getNCHWType(Value v);
llvm::SmallVector<Type, 4> getNHWCResultTypes(NCHWCompatible op);

}  // namespace oneflow

}  // namespace mlir

#endif  // ONEFLOW_IR_INCLUDE_ONEFLOW_TRANSFORM_TRANSPOSEHELPERS_H_

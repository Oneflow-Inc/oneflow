#include "OneFlow/OneFlowOps.h"

namespace mlir {

namespace oneflow {

template<typename OpTy>
bool isLinearMatmulOp(OpTy op) {
  const bool isAlphaOne = op.alpha().convertToDouble() == 1.0;
  const bool isLinear = op.transpose_a() == false && op.transpose_b() == true;
  const bool hasNoAddToOutput = !op._add_to_output();
  const bool isCUDA = op.device_tag() == "cuda";
  return isAlphaOne && isLinear && hasNoAddToOutput && isCUDA;
}

bool MatmulOp::isLinear() { return isLinearMatmulOp(*this); }

bool BroadcastMatmulOp::isLinear() { return isLinearMatmulOp(*this); }

bool BiasAddOp::isLastDim() {
  return axis() == -1 || axis() == out().getType().cast<ShapedType>().getRank() - 1;
}

Value BroadcastAddOp::b() { return y(); }

Value BroadcastAddOp::out() { return z(); }

bool BroadcastAddOp::isLastDim() { return true; }

}  // namespace oneflow

}  // namespace mlir

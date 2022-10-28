#include <oneflow/ir/include/OneFlow/Transform/TransposeHelpers.h>

namespace mlir {

namespace oneflow {

RankedTensorType getNHWCType(RankedTensorType t) {
  return RankedTensorType::get({t.getShape()[0], t.getShape()[2], t.getShape()[3], t.getShape()[1]},
                               t.getElementType());
}

RankedTensorType getNHWCType(Type t) { return getNHWCType(t.cast<RankedTensorType>()); }
RankedTensorType getNHWCType(Value v) { return getNHWCType(v.getType()); }

RankedTensorType getNCHWType(RankedTensorType t) {
  return RankedTensorType::get({t.getShape()[0], t.getShape()[3], t.getShape()[1], t.getShape()[2]},
                               t.getElementType());
}
RankedTensorType getNCHWType(Type t) { return getNCHWType(t.cast<RankedTensorType>()); }
RankedTensorType getNCHWType(Value v) { return getNCHWType(v.getType()); }

}  // namespace oneflow

}  // namespace mlir

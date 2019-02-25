#include "oneflow/core/ndarray/ndarray_apply_broadcast_unary_core.h"

namespace oneflow {

template<typename T, int NDIMS, const T (*unary_func)(const T)>
struct NdarrayApplyBroadcastUnaryCoreWrapper<DeviceType::kCPU, T, NDIMS, unary_func> final {
  static void Apply(DeviceCtx* ctx, const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x) {
    NdarrayApplyBroadcastUnaryCore<T, NDIMS, unary_func>::Apply(y, x);
  }
};

#define INSTANTIATE_BROADCAST_UNARY_FUNC(dtype_pair, NDIMS, unary_func) \
  template struct NdarrayApplyBroadcastUnaryCoreWrapper<                \
      DeviceType::kCPU, OF_PP_PAIR_FIRST(dtype_pair), NDIMS, unary_func>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_BROADCAST_UNARY_FUNC, ARITHMETIC_DATA_TYPE_SEQ,
                                 DIM_SEQ, ARITHMETIC_UNARY_FUNC_SEQ)
}  // namespace oneflow

#include "oneflow/core/ndarray/ndarray_apply_broadcast_binary_core.h"

namespace oneflow {

template<typename T, int NDIMS, template<typename> class binary_func>
struct NdarrayApplyBroadcastBinaryCoreWrapper<DeviceType::kCPU, T, NDIMS, binary_func> final {
  static void Apply(DeviceCtx* ctx,
                    const XpuVarNdarray<typename BinaryFuncTrait<binary_func, T>::return_type>& y,
                    const XpuVarNdarray<const T>& a, const XpuVarNdarray<const T>& b) {
    NdarrayApplyBroadcastBinaryCore<T, NDIMS, binary_func>::Apply(y, a, b);
  }
  static void InplaceApply(DeviceCtx* ctx, const XpuVarNdarray<T>& y,
                           const XpuVarNdarray<const T>& x) {
    NdarrayApplyBroadcastBinaryCore<T, NDIMS, binary_func>::InplaceApply(y, x);
  }
};

#define INSTANTIATE_BROADCAST_BINARY_FUNC(dtype_pair, NDIMS, binary_func) \
  template struct NdarrayApplyBroadcastBinaryCoreWrapper<                 \
      DeviceType::kCPU, OF_PP_PAIR_FIRST(dtype_pair), NDIMS, binary_func>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_BROADCAST_BINARY_FUNC,
                                 ARITHMETIC_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ, DIM_SEQ,
                                 BINARY_FUNC_SEQ)
}  // namespace oneflow

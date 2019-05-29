#include "oneflow/core/ndarray/ndarray_apply_broadcast_unary_core.h"

namespace oneflow {

namespace {

template<typename T, int NDIMS, template<typename> class unary_func>
__global__ void GpuBroadcastUnaryFunc(const XpuVarNdarray<T> y, const XpuVarNdarray<const T> x) {
  NdarrayApplyBroadcastUnaryCore<T, NDIMS, unary_func>::Apply(y, x);
}

}  // namespace

template<typename T, int NDIMS, template<typename> class unary_func>
struct NdarrayApplyBroadcastUnaryCoreWrapper<DeviceType::kGPU, T, NDIMS, unary_func> final {
  static void Apply(DeviceCtx* ctx, const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x) {
    size_t n = y.host_shape().HostElemNum();
    RUN_CUDA_KERNEL((GpuBroadcastUnaryFunc<T, NDIMS, unary_func>), ctx, n, y, x);
  }
};

#define INSTANTIATE_BROADCAST_UNARY_FUNC(dtype_pair, NDIMS, unary_func) \
  template struct NdarrayApplyBroadcastUnaryCoreWrapper<                \
      DeviceType::kGPU, OF_PP_PAIR_FIRST(dtype_pair), NDIMS, unary_func>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_BROADCAST_UNARY_FUNC,
                                 ARITHMETIC_DATA_TYPE_SEQ HALF_DATA_TYPE_SEQ, DIM_SEQ,
                                 ARITHMETIC_UNARY_FUNC_SEQ)
}  // namespace oneflow

#include "oneflow/core/kernel/broadcast_binary_kernel.h"
#include "oneflow/core/kernel/broadcast_binary_xpu_util.h"
#include "oneflow/core/ndarray/binary_func.h"

namespace oneflow {

namespace {

template<typename T, int NDIMS, const T (*binary_func)(const T, const T)>
__global__ void GpuBroadcastBinaryFunc(XpuVarNdarray<T> y, const XpuVarNdarray<const T> a,
                                       const XpuVarNdarray<const T> b) {
  BroadcastBinaryXpuUtil<T, NDIMS, binary_func>::Forward(&y, a, b);
}

}  // namespace

template<typename T, int NDIMS, const T (*binary_func)(const T, const T)>
struct BroadcastBinaryKernelHelper<DeviceType::kGPU, T, NDIMS, binary_func> final {
  static void Forward(DeviceCtx* ctx, XpuVarNdarray<T>&& y, const XpuVarNdarray<const T>& a,
                      const XpuVarNdarray<const T>& b) {
    size_t n = y.host_shape().HostElemNum();
    GpuBroadcastBinaryFunc<T, NDIMS, binary_func>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(y, a, b);
  }
};

#define INSTANTIATE_BROADCAST_BINARY_FUNC(dtype_pair, NDIMS, binary_func)                     \
  template struct BroadcastBinaryKernelHelper<DeviceType::kGPU, OF_PP_PAIR_FIRST(dtype_pair), \
                                              NDIMS, binary_func>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_BROADCAST_BINARY_FUNC, ARITHMETIC_DATA_TYPE_SEQ,
                                 DIM_SEQ, ARITHMETIC_BINARY_FUNC_SEQ)
}  // namespace oneflow

#include "oneflow/core/ndarray/ndarray_reduce_core.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

namespace {

template<typename T, int NDIMS>
__global__ void NdArrayReduceGpuImplaceReduceAxis(const XpuReducedNdarray<T, NDIMS> dst_reduced,
                                                  const XpuReducedNdarray<T, NDIMS> x, int axis) {
  NdArrayReduceCore<T, NDIMS>::ReduceAxis(dst_reduced, x, axis);
}

}  // namespace

template<typename T, int NDIMS>
struct NdArrayReduceCoreWrapper<DeviceType::kGPU, T, NDIMS> final {
  static void ReduceAxis(DeviceCtx* ctx, const XpuReducedNdarray<T, NDIMS>& dst_reduced,
                         const XpuReducedNdarray<T, NDIMS>& x, int axis) {
    size_t n = x.host_shape().HostElemNum();
    NdArrayReduceGpuImplaceReduceAxis<T, NDIMS> WITH_CUDA_PARAM(ctx, n, dst_reduced, x, axis);
  }
};

#define INSTANTIATE_NDARRAY_REDUCE(dtype_pair, NDIMS) \
  template struct NdArrayReduceCoreWrapper<DeviceType::kGPU, OF_PP_PAIR_FIRST(dtype_pair), NDIMS>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_NDARRAY_REDUCE, ARITHMETIC_DATA_TYPE_SEQ, DIM_SEQ);

}  // namespace oneflow

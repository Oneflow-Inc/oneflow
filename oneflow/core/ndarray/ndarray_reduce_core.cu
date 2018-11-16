#include "oneflow/core/ndarray/ndarray_reduce_core.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

namespace {

template<typename T, int NDIMS>
__global__ void NdArrayReduceGpuReduceAxis(T* dst_ptr, const XpuVarNdarray<const T> x, int axis,
                                           int64_t new_dim_value) {
  NdArrayReduceCore<T, NDIMS>::ReduceAxis(dst_ptr, x, axis, new_dim_value);
}

template<typename T, int NDIMS>
__global__ void NdArrayReduceGpuImplaceReduceAxis(const XpuReducedNdarray<T, NDIMS> x, int axis,
                                                  int64_t new_dim_value) {
  NdArrayReduceCore<T, NDIMS>::ImplaceReduceAxis(x, axis, new_dim_value);
}

}  // namespace

template<typename T, int NDIMS>
struct NdArrayReduceCoreWrapper<DeviceType::kGPU, T, NDIMS> final {
  static void ReduceAxis(DeviceCtx* ctx, T* dst_ptr, const XpuVarNdarray<const T>& x, int axis,
                         int64_t new_dim_value) {
    size_t n = x.host_shape().HostElemNum();
    NdArrayReduceGpuReduceAxis<T, NDIMS> WITH_CUDA_PARAM(ctx, n, dst_ptr, x, axis, new_dim_value);
  }
  static void ImplaceReduceAxis(DeviceCtx* ctx, const XpuReducedNdarray<T, NDIMS>& x, int axis,
                                int64_t new_dim_value) {
    size_t n = x.host_shape().HostElemNum();
    NdArrayReduceGpuImplaceReduceAxis<T, NDIMS> WITH_CUDA_PARAM(ctx, n, x, axis, new_dim_value);
  }
};

#define INSTANTIATE_NDARRAY_REDUCE(dtype_pair, NDIMS) \
  template struct NdArrayReduceCoreWrapper<DeviceType::kGPU, OF_PP_PAIR_FIRST(dtype_pair), NDIMS>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_NDARRAY_REDUCE, ARITHMETIC_DATA_TYPE_SEQ, DIM_SEQ);

}  // namespace oneflow

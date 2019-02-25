#include "oneflow/core/ndarray/ndarray_assign_core.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

namespace {

template<typename T, int NDIMS>
__global__ void NdarrayAssignGpu(XpuVarNdarray<T> y, const XpuReducedNdarray<T, NDIMS> reduced) {
  NdarrayAssignCore<T, NDIMS>::Assign(y, reduced);
}

}  // namespace

template<typename T, int NDIMS>
struct NdarrayAssignCoreWrapper<DeviceType::kGPU, T, NDIMS> final {
  static void Assign(DeviceCtx* ctx, const XpuVarNdarray<T>& y,
                     const XpuReducedNdarray<T, NDIMS>& reduced) {
    size_t n = y.host_shape().HostElemNum();
    RUN_CUDA_KERNEL((NdarrayAssignGpu<T, NDIMS>), ctx, n, y, reduced);
  }
};

#define INSTANTIATE_NDARRAY_ASSIGN(dtype_pair, NDIMS) \
  template struct NdarrayAssignCoreWrapper<DeviceType::kGPU, OF_PP_PAIR_FIRST(dtype_pair), NDIMS>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_NDARRAY_ASSIGN, ARITHMETIC_DATA_TYPE_SEQ, DIM_SEQ);

}  // namespace oneflow

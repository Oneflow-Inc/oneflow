#include "oneflow/core/kernel/broadcast_add_kernel.h"
#include "oneflow/core/kernel/broadcast_add_xpu_util.h"

namespace oneflow {

namespace {

template<typename T, int NDIMS>
__global__ void GpuBackwardAddOpInputDiff(XpuVarNdarray<T> in_diff,
                                          const XpuVarNdarray<const T> out_diff,
                                          XpuVarNdarray<T> tmp_storage) {
  BroadcastAddXpuUtil<T, NDIMS>::BackwardInputDiff(&in_diff, out_diff, &tmp_storage);
}

}  // namespace

template<typename T, int NDIMS>
struct BroadcastAddKernelUtil<DeviceType::kGPU, T, NDIMS> final {
  static void BackwardInputDiff(DeviceCtx* ctx, XpuVarNdarray<T>&& in_diff_a,
                                const XpuVarNdarray<const T>& out_diff, XpuVarNdarray<T>&& bw_buf) {
    size_t n = out_diff.host_shape().HostElemNum();
    GpuBackwardAddOpInputDiff<T, NDIMS> WITH_CUDA_PARAM(ctx, n, in_diff_a, out_diff, bw_buf);
  }
};

#define INSTANTIATE_BROADCAST_ADD(dtype_pair, NDIMS) \
  template struct BroadcastAddKernelUtil<DeviceType::kGPU, OF_PP_PAIR_FIRST(dtype_pair), NDIMS>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_BROADCAST_ADD, ARITHMETIC_DATA_TYPE_SEQ, DIM_SEQ)

}  // namespace oneflow

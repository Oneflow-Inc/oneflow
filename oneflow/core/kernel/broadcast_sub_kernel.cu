#include "oneflow/core/kernel/broadcast_sub_kernel.h"
#include "oneflow/core/kernel/broadcast_sub_xpu_util.h"

namespace oneflow {

namespace {

template<typename T, int NDIMS>
__global__ void GpuBackwardSubOpInputDiffA(XpuVarNdarray<T> in_diff,
                                           const XpuVarNdarray<const T> out_diff,
                                           XpuVarNdarray<T> tmp_storage) {
  BroadcastSubXpuUtil<T, NDIMS>::BackwardInputDiffA(&in_diff, out_diff, &tmp_storage);
}

template<typename T, int NDIMS>
__global__ void GpuBackwardSubOpInputDiffB(XpuVarNdarray<T> in_diff,
                                           const XpuVarNdarray<const T> out_diff,
                                           XpuVarNdarray<T> tmp_storage) {
  BroadcastSubXpuUtil<T, NDIMS>::BackwardInputDiffB(&in_diff, out_diff, &tmp_storage);
}

}  // namespace

template<typename T, int NDIMS>
struct BroadcastSubKernelUtil<DeviceType::kGPU, T, NDIMS> final {
  static void BackwardInputDiffA(DeviceCtx* ctx, XpuVarNdarray<T>&& input_diff,
                                 const XpuVarNdarray<const T>& out_diff,
                                 XpuVarNdarray<T>&& bw_buf) {
    size_t n = out_diff.host_shape().HostElemNum();
    GpuBackwardSubOpInputDiffA<T, NDIMS> WITH_CUDA_PARAM(ctx, n, input_diff, out_diff, bw_buf);
  }
  static void BackwardInputDiffB(DeviceCtx* ctx, XpuVarNdarray<T>&& input_diff,
                                 const XpuVarNdarray<const T>& out_diff,
                                 XpuVarNdarray<T>&& bw_buf) {
    size_t n = out_diff.host_shape().HostElemNum();
    GpuBackwardSubOpInputDiffB<T, NDIMS> WITH_CUDA_PARAM(ctx, n, input_diff, out_diff, bw_buf);
  }
};

#define INSTANTIATE_BROADCAST_SUB(dtype_pair, NDIMS) \
  template struct BroadcastSubKernelUtil<DeviceType::kGPU, OF_PP_PAIR_FIRST(dtype_pair), NDIMS>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_BROADCAST_SUB, ARITHMETIC_DATA_TYPE_SEQ, DIM_SEQ)

}  // namespace oneflow

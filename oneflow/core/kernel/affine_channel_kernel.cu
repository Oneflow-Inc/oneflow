#include "oneflow/core/kernel/affine_channel_kernel.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/kernel/transpose_kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void ScaleBiasForward(const int64_t n, const T* in, const T* scale, const T* bias,
                                 const int64_t channel_dim, const int64_t hxw_dim, T* out) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    const int64_t channel_i = (i / hxw_dim) % channel_dim;
    out[i] = in[i] * scale[channel_i] + bias[channel_i];
  }
}

template<typename T>
__global__ void ScaleBackward(const int64_t n, const T* out_diff, const T* scale,
                              const int64_t channel_dim, const int64_t hxw_dim, T* in_diff) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    const int64_t channel_i = (i / hxw_dim) % channel_dim;
    in_diff[i] = out_diff[i] * scale[channel_i];
  }
}

}  // namespace

template<typename T>
class AffineChannelKernelUtil<DeviceType::kGPU, T> final {
 public:
  static void Forward(DeviceCtx* ctx, int64_t n, int64_t channel_dim, int64_t per_channel_dim,
                      const T* in, const T* scale, const T* bias, T* out) {
    ScaleBiasForward<T>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            n, in, scale, bias, channel_dim, per_channel_dim, out);
  }

  static void Backward(DeviceCtx* ctx, int64_t n, int64_t channel_dim, int64_t per_channel_dim,
                       const T* out_diff, const T* scale, T* in_diff) {
    TODO();
  }
};

#define INSTANTIATE_GPU_KERNEL_UTIL(type_cpp, type_proto) \
  template class AffineChannelKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_GPU_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow

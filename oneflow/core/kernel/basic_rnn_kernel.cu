#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/basic_rnn_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void ComputeTanHDiffGpu(const int64_t n, const T* out, const T* out_diff,
                                   const T* rec_out_diff, T* plus_out_diff) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    plus_out_diff[i] = (1 - out[i] * out[i]) * (out_diff[i] + rec_out_diff[i]);
  }
}

template<typename T>
__global__ void ComputeSigmoidDiffGpu(const int64_t n, const T* out, const T* out_diff,
                                      const T* rec_out_diff, T* plus_out_diff) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    plus_out_diff[i] = out[i] * (1 - out[i]) * (out_diff[i] + rec_out_diff[i]);
  }
}

template<typename T>
__global__ void ComputeReluDiffGpu(const int64_t n, const T* out, const T* out_diff,
                                   const T* rec_out_diff, T* plus_out_diff) {
  CUDA_1D_KERNEL_LOOP(i, n) { plus_out_diff[i] = out[i] * (out_diff[i] + rec_out_diff[i]); }
}
}  // namespace

template<typename T>
struct BasicRnnKernelUtil<DeviceType::kGPU, T> {
  static void ComputeTanHDiff(DeviceCtx* ctx, int64_t n, const T* out, const T* out_diff,
                              const T* rec_out_diff, T* plus_out_diff) {
    ComputeTanHDiffGpu<T>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            n, out, out_diff, rec_out_diff, plus_out_diff);
  }
  static void ComputeSigmoidDiff(DeviceCtx* ctx, int64_t n, const T* out, const T* out_diff,
                                 const T* rec_out_diff, T* plus_out_diff) {
    ComputeSigmoidDiffGpu<T>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            n, out, out_diff, rec_out_diff, plus_out_diff);
  }
  static void ComputeReluDiff(DeviceCtx* ctx, int64_t n, const T* out, const T* out_diff,
                              const T* rec_out_diff, T* plus_out_diff) {
    ComputeReluDiffGpu<T>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            n, out, out_diff, rec_out_diff, plus_out_diff);
  }
};

#define INSTANTIATE_KERNEL_UTIL(type_cpp, type_proto) \
  template struct BasicRnnKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow

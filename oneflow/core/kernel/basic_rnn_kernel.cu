#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/basic_rnn_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void SigmoidGpu(const int64_t n, const T* x, T* y) {
  T one = static_cast<T>(1);
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = one / (one + std::exp(-x[i])); }
}

template<typename T>
__global__ void TanHGpu(const int64_t n, const T* x, T* y) {
  T one = static_cast<T>(1);
  T two = static_cast<T>(2);
  CUDA_1D_KERNEL_LOOP(i, n) {
    y[i] = two / (one + std::exp(-two * x[i])) - one;
  }
}

template<typename T>
__global__ void ComputeTanHDiffGpu(const int64_t n, const T* ht,
                                   const T* ht_diff, const T* rec_ht_diff,
                                   T* plus_out_diff) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    plus_out_diff[i] = (1 - ht[i] * ht[i]) * (ht_diff[i] + rec_ht_diff[i]);
  }
}

template<typename T>
__global__ void ComputeSigmoidDiffGpu(const int64_t n, const T* ht,
                                      const T* ht_diff, const T* rec_ht_diff,
                                      T* plus_out_diff) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    plus_out_diff[i] = ht[i] * (1 - ht[i]) * (ht_diff[i] + rec_ht_diff[i]);
  }
}

}  // namespace

template<typename T>
class BasicRnnKernelUtil<DeviceType::kGPU, T> final {
 public:
  static void Sigmoid(DeviceCtx* ctx, int64_t n, const T* x, T* y) {
    SigmoidGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
                    ctx->cuda_stream()>>>(n, x, y);
  }
  static void TanH(DeviceCtx* ctx, int64_t n, const T* x, T* y) {
    TanHGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
                 ctx->cuda_stream()>>>(n, x, y);
  }
  static void ComputeTanHDiff(DeviceCtx* ctx, int64_t n, const T* ht,
                              const T* ht_diff, const T* rec_ht_diff,
                              T* plus_out_diff) {
    ComputeTanHDiffGpu<T>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
           ctx->cuda_stream()>>>(n, ht, ht_diff, rec_ht_diff, plus_out_diff);
  }
  static void ComputeSigmoidDiff(DeviceCtx* ctx, int64_t n, const T* ht,
                                 const T* ht_diff, const T* rec_ht_diff,
                                 T* plus_out_diff) {
    ComputeSigmoidDiffGpu<T>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
           ctx->cuda_stream()>>>(n, ht, ht_diff, rec_ht_diff, plus_out_diff);
  }
};

template class BasicRnnKernelUtil<DeviceType::kGPU, float>;
template class BasicRnnKernelUtil<DeviceType::kGPU, double>;

}  // namespace oneflow

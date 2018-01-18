#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/basic_rnn_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void AddGpu(const int64_t n, const T* x, const T* y, T* z) {
  CUDA_1D_KERNEL_LOOP(i, n) { z[i] = x[i] + y[i]; }
}

template<typename T>
__global__ void TanhGpu(const int64_t n, const T* x, T* y) {
  auto sigmoid = [](T x) {
    return static_cast<T>(1) / (static_cast<T>(1) + std::exp(-x));
  };
  CUDA_1D_KERNEL_LOOP(i, n) {
    y[i] = static_cast<T>(2) * sigmoid(2 * x[i]) - static_cast<T>(1);
  }
}

template<typename T>
__global__ void ComputePlusOutDiffGpu(const int64_t n, const T* ht,
                                      const T* ht_diff, T* plus_out_diff) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    plus_out_diff[i] = (1 - ht[i] * ht[i]) * ht_diff[i];
  }
}

}  // namespace

template<typename T>
class BasicRnnKernelUtil<DeviceType::kGPU, T> final {
 public:
  static void Add(DeviceCtx* ctx, int64_t n, const T* x, const T* y, T* z) {
    AddGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
                ctx->cuda_stream()>>>(n, x, y, z);
  }
  static void Tanh(DeviceCtx* ctx, int64_t n, const T* x, T* y) {
    TanhGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
                 ctx->cuda_stream()>>>(n, x, y);
  }
  static void ComputePlusOutDiff(DeviceCtx* ctx, int64_t n, const T* ht,
                                 const T* ht_diff, T* plus_out_diff) {
    ComputePlusOutDiffGpu<T>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
           ctx->cuda_stream()>>>(n, ht, ht_diff, plus_out_diff);
  }
};

template class BasicRnnKernelUtil<DeviceType::kGPU, float>;
template class BasicRnnKernelUtil<DeviceType::kGPU, double>;

}  // namespace oneflow

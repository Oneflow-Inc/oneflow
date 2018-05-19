#include "oneflow/core/kernel/softmax_kernel.h"
#include "oneflow/core/kernel/kernel_util.cuh"

namespace oneflow {

namespace {

template<typename T>
__global__ void SoftmaxForwardMaxGpu(const int64_t n, const int64_t w, const T* x, T* y) {
  const int32_t tid = threadIdx.x;
  if (tid < n) { y[tid] = x[tid * w]; }
  __syncthreads();
  CUDA_1D_KERNEL_LOOP(i, n * w) { gpu_atomic_max(y + i / w, x[i]); }
}

template<typename T>
__global__ void SoftmaxSubGpu(const int64_t n, const int64_t w, T* matrix, const T* vector) {
  CUDA_1D_KERNEL_LOOP(i, n * w) { matrix[i] -= vector[i / w]; }
}

template<typename T>
__global__ void SoftmaxDivGpu(const int64_t n, const int64_t w, T* matrix, const T* vector) {
  CUDA_1D_KERNEL_LOOP(i, n * w) { matrix[i] /= vector[i / w]; }
}

}  // namespace

template<typename T>
struct SoftmaxKernelUtil<DeviceType::kGPU, T> {
  static void ForwardMax(DeviceCtx* ctx, const int64_t n, const int64_t w, const T* out, T* tmp) {
    SoftmaxForwardMaxGpu<T>
        <<<BlocksNum4ThreadsNum(n * w), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, w, out,
                                                                                          tmp);
  }

  static void RowSum(DeviceCtx* ctx, const int64_t n, const int64_t w, const T* matrix, T* sum_vec,
                     const T* sum_multiplier) {
    KernelUtil<DeviceType::kGPU, T>::Gemv(ctx, CblasTrans, n, w, 1, matrix, w, sum_multiplier, 1, 0,
                                          sum_vec, 1);
  }

  static void Sub(DeviceCtx* ctx, const int64_t n, const int64_t w, T* matrix, const T* vector) {
    SoftmaxSubGpu<T>
        <<<BlocksNum4ThreadsNum(n * w), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            n, w, matrix, vector);
  }

  static void Div(DeviceCtx* ctx, const int64_t n, const int64_t w, T* matrix, const T* vector) {
    SoftmaxDivGpu<T>
        <<<BlocksNum4ThreadsNum(n * w), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            n, w, matrix, vector);
  }
};

#define INSTANTIATE_SOFTMAX_KERNEL_UTIL(type_cpp, type_proto) \
  template struct SoftmaxKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_SOFTMAX_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow

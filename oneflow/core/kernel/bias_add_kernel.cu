#include "oneflow/core/kernel/bias_add_kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void BiasAddGpu(int64_t elem_cnt, int64_t bias_size, int64_t inner_size, const T* x,
                           const T* bias, T* y) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    y[i] = x[i] + bias[i % (inner_size * bias_size) / inner_size];
  }
}

}  // namespace

template<typename T>
struct BiasAddUtil<DeviceType::kGPU, T> {
  static void BiasAdd(DeviceCtx* ctx, int64_t outer_size, int64_t bias_size, int64_t inner_size,
                      const T* x, const T* bias, T* y) {
    const int64_t elem_cnt = outer_size * bias_size * inner_size;
    BiasAddGpu<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        elem_cnt, bias_size, inner_size, x, bias, y);
  }
};

template struct BiasAddUtil<DeviceType::kGPU, float>;
template struct BiasAddUtil<DeviceType::kGPU, double>;

}  // namespace oneflow

#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/customized/kernels/bias_add_kernel.h"

namespace oneflow {

namespace {

template<typename T, typename Index>
__global__ void BiasAddGpu(const Index elem_cnt, const Index bias_size, const Index inner_size,
                           const T* x, const T* bias, T* y) {
  const Index block_size = bias_size * inner_size;
  CUDA_1D_KERNEL_LOOP_T(Index, i, elem_cnt) { y[i] = x[i] + bias[(i % block_size) / inner_size]; }
}

template<typename Index>
__global__ void BiasAddGpuHalf(const Index elem_cnt, const Index bias_size, const Index inner_size,
                               const half* x, const half* bias, half* y) {
  const Index block_size = bias_size * inner_size;
  CUDA_1D_KERNEL_LOOP_T(Index, i, elem_cnt) {
    y[i] = __hadd(x[i], bias[(i % block_size) / inner_size]);
  }
}

template<typename T, typename Index>
__global__ void InplaceBiasAddGpu(const Index elem_cnt, const Index bias_size,
                                  const Index inner_size, const T* bias, T* y) {
  const Index block_size = bias_size * inner_size;
  CUDA_1D_KERNEL_LOOP_T(Index, i, elem_cnt) { y[i] += bias[(i % block_size) / inner_size]; }
}

}  // namespace

template<typename T, typename Index>
struct BiasAddCalculation<DeviceType::kGPU, T, Index> {
  static void Invoke(DeviceCtx* ctx, Index outer_size, Index bias_size, Index inner_size,
                     const T* x, const T* bias, T* y) {
    const Index elem_cnt = outer_size * bias_size * inner_size;
    if (x == y) {
      RUN_CUDA_KERNEL((InplaceBiasAddGpu<T, Index>), ctx, elem_cnt, elem_cnt, bias_size, inner_size,
                      bias, y);
    } else {
      RUN_CUDA_KERNEL((BiasAddGpu<T, Index>), ctx, elem_cnt, elem_cnt, bias_size, inner_size, x,
                      bias, y);
    }
  }
};

template<typename Index>
struct BiasAddCalculation<DeviceType::kGPU, float16, Index> {
  static void Invoke(DeviceCtx* ctx, Index outer_size, Index bias_size, Index inner_size,
                     const float16* x, const float16* bias, float16* y) {
    const Index elem_cnt = outer_size * bias_size * inner_size;
    RUN_CUDA_KERNEL((BiasAddGpuHalf<Index>), ctx, elem_cnt, elem_cnt, bias_size, inner_size,
                    reinterpret_cast<const half*>(x), reinterpret_cast<const half*>(bias),
                    reinterpret_cast<half*>(y));
  }
};

REGISTER_BIAS_ADD_USER_KERNEL(GPU, float16)
REGISTER_BIAS_ADD_USER_KERNEL(GPU, float)
REGISTER_BIAS_ADD_USER_KERNEL(GPU, double)
REGISTER_BIAS_ADD_USER_KERNEL(GPU, int8_t)
REGISTER_BIAS_ADD_USER_KERNEL(GPU, int32_t)
REGISTER_BIAS_ADD_USER_KERNEL(GPU, int64_t)

}  // namespace oneflow

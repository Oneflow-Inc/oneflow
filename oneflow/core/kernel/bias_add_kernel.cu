#include "oneflow/core/kernel/bias_add_kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void BiasAddNCXKernel(int32_t nthreads, const T* input, const T* bias, T* output,
                                 int32_t bias_size, int32_t x_size) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int32_t index2 = index / x_size;
    int32_t bias_offset = index2 % bias_size;
    output[index] = input[index] + bias[bias_offset];
  }
}

__global__ void HalfBiasAddNCXKernel(int32_t nthreads, const half* input, const half* bias,
                                     half* output, int32_t bias_size, int32_t x_size) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int32_t index2 = index / x_size;
    int32_t bias_offset = index2 % bias_size;
    output[index] = __hadd(input[index], bias[bias_offset]);
  }
}

}  // namespace

template<typename T>
struct BiasAddUtil<DeviceType::kGPU, T> {
  static void BiasAddNCX(DeviceCtx* ctx, const Shape& shape, const int32_t bias_axis,
                         const T* input, const T* bias, T* output) {
    BiasAddNCXKernel<<<BlocksNum4ThreadsNum(shape.At(0)), kCudaThreadsNumPerBlock, 0,
                       ctx->cuda_stream()>>>(shape.At(0), input, bias, output, shape.At(bias_axis),
                                             shape.Count(bias_axis + 1));
  }
};

void BiasAddUtil<DeviceType::kGPU, float16>::BiasAddNCX(DeviceCtx* ctx, const Shape& shape,
                                                        const int32_t bias_axis,
                                                        const float16* input, const float16* bias,
                                                        float16* output) {
  HalfBiasAddNCXKernel<<<BlocksNum4ThreadsNum(shape.At(0)), kCudaThreadsNumPerBlock, 0,
                         ctx->cuda_stream()>>>(
      shape.At(0), reinterpret_cast<const half*>(input), reinterpret_cast<const half*>(bias),
      reinterpret_cast<half*>(output), shape.At(bias_axis), shape.Count(bias_axis + 1));
}

template struct BiasAddUtil<DeviceType::kGPU, float16>;
template struct BiasAddUtil<DeviceType::kGPU, float>;
template struct BiasAddUtil<DeviceType::kGPU, double>;

}  // namespace oneflow

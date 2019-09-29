#include "oneflow/core/kernel/bias_add_kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void BiasAddGpu(const int64_t elem_cnt, const int64_t bias_size,
                           const int64_t inner_size, const T* x, const T* bias, T* y) {
  const int64_t block_size = bias_size * inner_size;
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) { y[i] = x[i] + bias[i % block_size / inner_size]; }
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

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
template<>
struct BiasAddUtil<DeviceType::kGPU, float16> final {
  static void BiasAdd(DeviceCtx* ctx, int64_t outer_size, int64_t bias_size, int64_t inner_size,
                      const float16* x, const float16* bias, float16* y) {
    BiasAddUtil<DeviceType::kGPU, half>::BiasAdd(
        ctx, outer_size, bias_size, inner_size, reinterpret_cast<const half*>(x),
        reinterpret_cast<const half*>(bias), reinterpret_cast<half*>(y));
  }
};
#endif

#define INITIATE_BIAS_ADD_KERNEL_UTIL_GPU_IMPL(type_cpp, type_proto) \
  template struct BiasAddUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INITIATE_BIAS_ADD_KERNEL_UTIL_GPU_IMPL,
                     ARITHMETIC_DATA_TYPE_SEQ
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
                         FLOAT16_DATA_TYPE_SEQ HALF_DATA_TYPE_SEQ
#endif
);
#undef INITIATE_BIAS_ADD_KERNEL_UTIL_GPU_IMPL

}  // namespace oneflow

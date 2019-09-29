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

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
template<>
struct BiasAddUtil<DeviceType::kGPU, float16> final {
  static void BiasAdd(DeviceCtx* ctx, int64_t outer_size, int64_t bias_size, int64_t inner_size,
                      const float16* x, const float16* bias, float16* y) {
    BiasAddUtil<DeviceType::kGPU, half>::BiasAdd(ctx, outer_size, bias_size, inner_size, x, bias,
                                                 y);
  }
};
#endif

#define INITIATE_BIAS_ADD_KERNEL_UTIL_GPU_IMPL(in_type_pair) \
  template struct BiasAddUtil<DeviceType::kGPU, OF_PP_PAIR_FIRST(in_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INITIATE_BIAS_ADD_KERNEL_UTIL_GPU_IMPL, ARITHMETIC_DATA_TYPE_SEQ
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
                                                                             FLOAT16_DATA_TYPE_SEQ
#endif
);
#undef INITIATE_BIAS_ADD_KERNEL_UTIL_GPU_IMPL

}  // namespace oneflow

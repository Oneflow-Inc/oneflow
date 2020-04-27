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
__global__ void BiasAddForwardGpuHalf(const Index elem_cnt, const Index bias_size,
                                      const Index inner_size, const half* x, const half* bias,
                                      half* y) {
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

template<typename T, typename Index>
struct BiasAddGpuHelper final {
  static void BiasAdd(DeviceCtx* ctx, const Index elem_cnt, const Index bias_size,
                      const Index inner_size, const T* x, const T* bias, T* y) {
    if (x == y) {
      InplaceBiasAddGpu<T, Index>
          <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
              elem_cnt, bias_size, inner_size, bias, y);
    } else {
      BiasAddGpu<T, Index>
          <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
              elem_cnt, bias_size, inner_size, x, bias, y);
    }
  }
};

template<typename Index>
struct BiasAddGpuHelper<float16, Index> final {
  static void BiasAdd(DeviceCtx* ctx, const Index elem_cnt, const Index bias_size,
                      const Index inner_size, const float16* x, const float16* bias, float16* y) {
    BiasAddForwardGpuHalf<Index>
        <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            elem_cnt, bias_size, inner_size, reinterpret_cast<const half*>(x),
            reinterpret_cast<const half*>(bias), reinterpret_cast<half*>(y));
  }
};

}  // namespace

template<typename T>
struct BiasAddUtil<DeviceType::kGPU, T> {
  static void BiasAdd(DeviceCtx* ctx, int64_t outer_size, int64_t bias_size, int64_t inner_size,
                      const T* x, const T* bias, T* y) {
    const int64_t elem_cnt = outer_size * bias_size * inner_size;
    if (IsKernelSafeInt32(elem_cnt)) {
      BiasAddGpuHelper<T, int32_t>::BiasAdd(ctx, elem_cnt, bias_size, inner_size, x, bias, y);
    } else {
      BiasAddGpuHelper<T, int64_t>::BiasAdd(ctx, elem_cnt, bias_size, inner_size, x, bias, y);
    }
  }
};

#define INITIATE_BIAS_ADD_KERNEL_UTIL_GPU_IMPL(type_cpp, type_proto) \
  template struct BiasAddUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INITIATE_BIAS_ADD_KERNEL_UTIL_GPU_IMPL, ARITHMETIC_DATA_TYPE_SEQ);
#undef INITIATE_BIAS_ADD_KERNEL_UTIL_GPU_IMPL

template struct BiasAddUtil<DeviceType::kGPU, float16>;

}  // namespace oneflow
